# MIND Multi-Stage News Recommender [Retrieval -> Ranker(DLRM+knowledge graph) -> Re-ranker]

This project implements a realistic recommender stack on the **Microsoft News Dataset (MIND)**:
- Preprocessing: prepare train/val/test data, click-count features, cold/new flags, impression-level eval data, and map IDs to indices.
- Train **Teacher retrieval encoders** (text-based item encoder + history-based user encoder).
- Build two Faiss ANN retrieval indexes: (1) for item embeddings from teacher, and (2) for item-base features. These indexes are used for hybrid candidate retrieval.
- Train **Student ranker**, a lean DLRM-style sparse+dense model that narrows **hybrid retrieval** returned `topk` items to more relevant `pool_size` items. Alongside the usual DLRM-style ID embeddings, category/subcategory embeddings, and dense features, this student uses semantic and **Knowledge Graph embeddings**: it encodes the candidate's text-plus-KG item-base embedding and pools the clicked-history text-plus-KG item-base embeddings. Teacher **distillation** (logit + representation) from the text/history-based teacher helps those semantic branches learn a stronger user/item matching space than click labels alone, **improving cold user/item behavior where ID memorization is weak**.
- An **optional post-ranking layer** supports diversity, coverage, and exposure-fairness experiments. It is isolated from the competition submission path.
- Re-ranking utilities can search weights and penalties under configurable product guardrails, then report the selected trade-off on held-out test data.
- Extensive **evaluation**: ranking metrics, calibration, diversity, exposure fairness, and cold/new slices.

👉 For production recommendations:

The system has three inference steps for each user/impression.

1. Retrieval narrows the full catalog (potentially thousands or millions of items) to `topk=200` candidates.
2. The student ranker scores those candidates and keeps the top `pool_size=50` candidates.
3. The reranker takes that pool and produces the final `k_out=10` items.

👉 For MIND competition submission:

1. No retrieval is needed at submission time because each test impression already provides a small candidate set. The trained teacher retriever is still used to supervise the student ranker's training.
2. The student ranker scores and ranks every candidate supplied in each test impression. It does not truncate the candidate set because the submission must contain all candidates.
3. Reranking is not needed for the competition.

MIND is widely used as a benchmark for news recommendation, with impression logs and rich news metadata.

Production design note: if a fresh batch of news arrives, for example from the last 15 minutes, the system would need fresh item-side representations. For each new article:
- Run the sentence-transformer over text/title/abstract to produce the text-only retrieval `item_base` embedding.
- Pass the text-only `item_base` through the teacher item encoder to produce `item_teacher_emb`.
- Parse linked Wikidata entities and aggregate entity/relation/one-hop context features to produce the KG-enhanced `item_ranker_base` embedding.
- Mark as cold/new with `is_new_item = 1`, `item_clicks_log1p = 0`.
- Add to both Faiss retrieval indexes: the teacher index (`item_teacher_emb`) and the text-only fallback index (`item_base`).

The current repo/CLI does not implement online index updates; it builds and writes both Faiss indexes in batch via `build_index`.

## Teacher model
- The teacher is a learned two-tower retrieval model:
  - a **news/item encoder** that starts from frozen sentence-transformer news embeddings and applies a trainable projection
  - a **user encoder** that attention-pools clicked-history item embeddings (before the current impression) into a user vector
- In the `behaviors.tsv` dataset, `history` is a list of previously clicked news IDs for that user before the current impression time. It is not a list of previous impressions.
- Training uses clicked positives plus in-impression negatives.
- Retrieval evaluation encodes each validation/test impression with that impression's own user history (not from some later history for the same user), so the query is temporally aligned with the impression being scored.
- `encode_user_from_item_vectors(history_z, mask)` expects:
  - `history_z`: shape `[B, T, D]` where `B` is batch size, `T` is history length, and `D` is the teacher hidden dimension
  - `mask`: shape `[B, T]` with `True` for valid history positions and `False` for padding
  - output: shape `[B, D]`
- Item embedding path:
  - `item_proj`
  - `normalize`
- User embedding path:
  - `item_proj` on each clicked history item
  - `normalize` on each clicked history item
  - `HistoryAttentionPool` (`MultiheadAttention` + learned query)
  - `normalize` on the pooled user vector
- The learned query vector in `HistoryAttentionPool` is a trainable vector used to emphasize certain click history over other click history. This query vector is global, not per-user.
- During teacher training, the trainable parts are:
  - `item_proj`
  - the multi-head attention parameters inside `HistoryAttentionPool`
  - the learned query vector
- The training objective pushes each user vector closer to its clicked positive item and farther from sampled in-impression negatives and other in-batch positives.
- At the end of `train_teacher`, the pipeline writes:
  - `item_base_emb.npy`: frozen text-only sentence-transformer feature for every news item, used by teacher training and retrieval.
  - `item_ranker_base_emb.npy`: separate `[sentence-transformer text embedding ; KG entity/context embedding]` feature used by the student ranker
  - `item_teacher_emb.npy`: the final teacher embedding for every news item
  - `user_teacher_emb.npy`: a cached teacher embedding for each training user history, retained as an artifact for inspection/backward compatibility
- These files are then reused by later stages:
  - `build_index` builds the Faiss retrieval index from `item_teacher_emb.npy`
  - `eval_retrieval` uses `item_teacher_emb.npy` and the saved teacher model to encode held-out histories and search that index
  - `train_ranker` loads `item_ranker_base_emb.npy` as student semantic input and `item_teacher_emb.npy` as item supervision. It does **not** consume `user_teacher_emb.npy`; instead, it dynamically encodes each pair's impression-time history with the saved teacher model.

## Hybrid retrieval scoring

During `eval_retrieval`, the teacher retrieval query is built by taking the impression's clicked history news, looking up their teacher item embeddings, and passing those vectors through `TeacherTwoTower.encode_user_from_item_vectors()`. The base retrieval query is built by averaging the impression's clicked history news' raw sentence-transformer `item_base_emb.npy` embeddings.

Faiss returns scalar similarity scores:
- `teacher_score`: similarity between the teacher user query and a candidate `item_teacher_emb.npy` vector
- `base_score`: similarity between the averaged of the clicked-history text-only `item_base_emb.npy` vectors and a candidate `item_base_emb.npy` vector

The retrieval code searches both Faiss indexes, merges the oversampled candidate lists, and keeps the final `topk` items by hybrid score. (`hybrid_oversample` controls how many candidates are fetched from each index before the merge.) With the current config, candidates are ranked mostly by teacher retrieval with a small text-only semantic contribution. KG is intentionally not used during retrieval.
```text
hybrid_score = (1 - retrieval.hybrid_base_weight) * teacher_score
             + retrieval.hybrid_base_weight * base_score
```

Retrieval evaluation currently skips users with no history.

History-length slices show how much the retrieval and ranking stages depend on having enough prior clicks to describe the user:

![images/AUC_by_user_history_len_bucket.png](images/AUC_by_user_history_len_bucket.png)

![images/nDCG_by_user_history_len_bucket.png](images/nDCG_by_user_history_len_bucket.png)

## Student model

The student keeps a lighter semantic core than the teacher, but combines it with classic DLRM signals:
- projected `user_id` / `news_id` branches
- category and subcategory embeddings
- lightweight dense features
- DLRM-style feature interactions

#### DLRM-style semantic additions
- A classic DLRM usually combines sparse ID/category embeddings, dense numerical features, pairwise feature interactions, and a final top MLP. It usually does not include item text embeddings or user-history semantic embeddings directly; those are project-specific additions here.
- In `DLRMStudent`, the candidate item's `item_ranker_base_emb.npy` vector is formed by concatenating its sentence-transformer text embedding with a KG feature vector. That KG vector aggregates the article's linked entity embeddings together with relation-aware one-hop neighbor context from the configured triples file. The candidate item's vector is projected into `item_sem`, and the corresponding clicked-history vectors are pooled/projected into `user_sem`. Both are mapped to the same `emb_dim` length as the other DLRM feature vectors. `ranker.dlrm.history_pooling` selects the original candidate-independent `mean` or the opt-in `candidate_attention` path, where the current candidate queries the clicked-history states before `user_sem` is formed.
- These semantic vectors, plus `sem_fused`, are added to the DLRM feature list as extra feature vectors. They are then included in the pairwise dot-product interaction layer and concatenated into the final top MLP input.

#### Distillation representation
- In `DLRMStudent.forward()`, the student representation used for distillation is `rep = [user_sem, item_sem, sem_fused]`.
- `user_sem`: student semantic user vector from pooled click-history KG-enhanced item-base features
- `item_sem`: student semantic item vector from the candidate item's KG-enhanced item-base feature
- `sem_fused`: a lightweight attention-fusion summary that mixes the semantic user/item states with the structured query context
- The teacher target is `concat(teacher_user_emb, teacher_item_emb)`. A projection head maps the student representation into the teacher space for representation distillation.
- The teacher is semantic/history-based rather than mostly `user_id`/`news_id` memorization. Representation distillation therefore **encourages the student's semantic branch to learn a useful user/item space, especially when ID signals are weak for cold users or new items**.
- If the ranker were trained without distillation, its loss would reduce to the supervised click-label objective:
```python
loss = binary_cross_entropy_with_logits(student_logits, click_label)
```
- Without distillation, retrieval can still be designed in two ways:
  - Use only the frozen text-only `item_base_emb.npy` sentence-transformer embeddings for retrieval
  - Train a retrieval model, but do not use that model as a teacher to supervise the ranker

#### Teacher -> student distillation map
- Distillation in this project is not copying the teacher into a smaller clone.
- The student is trained to imitate the teacher's semantic user/item representations and semantic matching behavior, while still having its additional features.

| Teacher block | What it does | Student replacement | Key difference |
|---|---|---|---|
| Frozen text-only item-base input | Base text semantics for each news item | KG-enhanced `item_ranker_base_emb.npy` input | Retrieval stays text-only; the ranker receives extra entity/relation/neighbor context |
| Teacher semantic item encoder | Refines each item into the teacher semantic space | Smaller student semantic item encoder | Student semantic dimension is much smaller than the teacher space |
| Teacher sequence-aware user encoder | Contextualizes clicked history items with attention before pooling | Cheaper history aggregation path | Student uses a lighter history refinement and aggregation path |
| Teacher attention pooling over history | Builds one semantic user vector from clicked history | Mean pooling by default; optional candidate-aware multi-head attention | Candidate-aware pooling recomputes `user_sem` for each candidate |
| Teacher item embedding | Semantic representation of the candidate item | `item_sem` | Student item representation is trained to approximate teacher behavior |
| Teacher user embedding | Semantic representation of the current user history | `user_sem` | Student user representation is cheaper to compute |
| Teacher cosine-style user-item scorer | Measures semantic compatibility between user and item | Student top MLP ranker | Student scoring uses richer ranking signals beyond pure semantic similarity |
| Teacher representation target | Provides a semantic supervision target for distillation | `rep = [user_sem, item_sem, sem_fused]` plus projection head | Student and teacher representations have different shapes and meanings |
| No direct teacher counterpart | None | `sem_fused` | Student adds an attention-fusion summary conditioned on query context |
| No direct teacher counterpart | None | `user_id` / `news_id` branches | Student adds collaborative-style memorization signals |
| No direct teacher counterpart | None | category / subcategory embeddings | Student adds structured metadata signals |
| No direct teacher counterpart | None | dense features such as `history_len` and `item_clicks_log1p` | Student adds non-semantic ranking features |
| No direct teacher counterpart | None | DLRM interaction terms + top MLP | Student is a broader ranker, not just a semantic retriever |

Category and clicked-item-popularity slices help check whether the model is robust across content verticals and between new, low-click, and high-click items:

![images/AUC_by_clicked_category.png](images/AUC_by_clicked_category.png)

![images/nDCG_by_clicked_category.png](images/nDCG_by_clicked_category.png)

![images/AUC_by_clicked_item_popularity.png](images/AUC_by_clicked_item_popularity.png)

![images/nDCG_by_clicked_item_popularity.png](images/nDCG_by_clicked_item_popularity.png)

#### Where the student ranker is simplified
- teacher semantic item encoder -> smaller student semantic item encoder
- teacher sequence-aware user encoder -> cheaper history aggregation path
- teacher cosine scorer -> student MLP ranker with many extra inputs

## Re-ranking (Optional; not used by the competition submission)

- The focused workflow and configuration reference live in [docs/reranking.md](docs/reranking.md); this section is only a conceptual overview.
- In this project, re-ranking is a **deterministic optimization layer** on top of ranker scores.
- It is controlled by hyperparameters/constraints (relevance, coverage, fairness).
- **No training loop is required** for this re-ranking stage.
- `rerank_search` uses the "Nov 14" data to choose an operating point; `rerank_eval` reports that selected setting on the "Nov 15" data (Large Temporal Val is Nov 14 + 15. See the dataset timeline below.)
- Both commands first obtain the ordinary student-ranker scores and then apply the optional greedy reranker. The reranker experiment does not replace or retrain the student model.

#### Re-ranking process

- `greedy_rerank()` takes the top `pool_size` candidates by ranker score, then builds the final top-`k_out` list one item at a time.
- With the recommended `relevance_normalization: minmax`, ranker logits inside each pool are mapped to `[0, 1]`. This keeps relevance and coverage weights interpretable when ranker checkpoints have different logit scales. Set it to `none` only to reproduce legacy raw-logit experiments.
- At each step it scores every remaining candidate with:
- `relevance_weight * relevance`
- `+ coverage_weight * coverage`
- `- fairness.penalty_weight * fairness_penalty`
- It then picks the candidate with the highest total value, adds it to the list, updates the running coverage/fairness state, and repeats until `k_out` items are selected.

#### Exposure fairness

- Let `p` be the **actual exposure distribution** of the current ranked list across groups such as categories.
- Let `q` be the **target distribution** we want to match.
- In the current code:
- `p` is built from the selected top-K list after applying position weights.
- `q` is derived from the reference candidate set:
  - `fairness_kl_pool` compares top-K exposure against the reranker's top-`pool_size` candidate mix.
  - `fairness_kl_full` compares top-K exposure against the full impression candidate set.
  - If `category_target: uniform`, then `q` is uniform over categories present in the reference set (not used in this project).
  - If `category_target: catalog`, then `q` is the category distribution of the impression candidate pool (the candidates available for that user/impression), not the selected top-K list and not the global corpus-wide catalog.
- `rerank_eval` and `rerank_search` report both `fairness_kl_pool` and `fairness_kl_full`. Fairness Gini includes zero-exposure categories present in the pool target.
- Product constraints in reranker search use `fairness_kl_pool`, because that matches the reranker's actual optimization target.

#### Worked examples for coverage, fairness, and diagnostic semantic diversity

These examples use dummy articles and the current scoring definitions. The frozen weights are relevance `0.965`, coverage `0.035`, and KL penalty `0`. We show three positions to keep the arithmetic short; the real reranker continues to ten articles, then reports metrics on the completed list.

Suppose the selected prefix is `[A, B]`: `A` is Sports, `B` is Politics, and their entities together are `{Messi, Real Madrid}`. Neither article is flagged as new/rare. Two candidates compete for the next position:

| Candidate | Normalized relevance | Category | Entities | New/rare flag |
| --- | ---: | --- | --- | --- |
| `C` | 0.80 | Sports | Messi, Inter Miami | Yes |
| `D` | 0.78 | Health | WHO, vaccine, pandemic, hospital | No |

- **Coverage example**:
  - The category bonus is `1.0`, the bonus per newly covered entity is `0.3`, and at most three new entities contribute per article.
  - `C` adds no category because Sports is already covered. Only Inter Miami is a new entity, so `coverage(C) = 0 + 0.3 * 1 = 0.3`.
  - `D` adds Health and four new entities. The entity cap limits the rewarded count to three, so `coverage(D) = 1.0 + 0.3 * 3 = 1.9`.
  - After selecting `D`, Health and all four of its entities enter the covered sets. The cap limits the reward for that selection; it does not leave the fourth entity eligible for a later new-entity bonus.

- **Choosing the next article with the frozen weights**:
  - `score(C) = 0.965 * 0.80 + 0.035 * 0.3 = 0.7825`.
  - `score(D) = 0.965 * 0.78 + 0.035 * 1.9 = 0.8192`.
  - `D` wins despite slightly lower predicted relevance because its coverage contribution more than compensates. The KL contribution is zero at the frozen penalty weight.
  - After adding `D`, the reranker recomputes scores for the remaining candidates using the updated coverage and exposure state. For example, another Health article no longer receives a new-category bonus. Weights stay fixed throughout list construction.
  - No click labels enter this calculation. Offline nDCG evaluates the resulting complete lists; the greedy step itself does not calculate actual nDCG.

- **Category-exposure fairness example**:
  - Suppose the accessible candidate pool contains 50% Sports, 20% Politics, and 30% Health. This gives the target `q = {Sports: 0.50, Politics: 0.20, Health: 0.30}`.
  - Log position weights for the first three positions are approximately `[1.0000, 0.6309, 0.5000]`, with total exposure `2.1309`.
  - Adding `C` gives `[Sports, Politics, Sports]`. Normalized exposure is approximately `pC = {Sports: 0.7039, Politics: 0.2961, Health: 0}`, and `KL(pC || q) = 0.3569`.
  - Adding `D` gives `[Sports, Politics, Health]`. Normalized exposure is approximately `pD = {Sports: 0.4693, Politics: 0.2961, Health: 0.2346}`, and `KL(pD || q) = 0.0287`.
  - KL uses `sum(p[c] * ln(p[c] / q[c]))`, with zero-exposure terms contributing zero and numerical smoothing in the implementation. Here, adding `D` produces exposure closer to the pool mix.
  - A trial penalty weight of `0.025` would subtract approximately `0.0089` from `C` and `0.0007` from `D`. The frozen weight is `0`, so these KL values do not affect the current selection score; completed-list pool KL still determines whether an offline setting satisfies its aggregate guardrail.

- **Diagnostic semantic diversity example**:
  - Suppose teacher-cosine similarities are `sim(A, B) = 0.60`, `sim(A, C) = 0.90`, and `sim(B, C) = 0.35`. The illustrative list `[A, B, C]` has `ILD = 1 - (0.60 + 0.90 + 0.35) / 3 = 0.3833`.
  - Suppose instead `sim(A, D) = 0.20` and `sim(B, D) = 0.10`. The illustrative list `[A, B, D]` has `ILD = 1 - (0.60 + 0.20 + 0.10) / 3 = 0.7000`.
  - The second list is more semantically diverse under this diagnostic. These similarities do not enter the current reranking score, guardrails, or selection tie-breaker. For actual top-ten lists, ILD averages all 45 distinct article pairs.

- **New-item exposure metric example**:
  - If `C` occupies position three and only `C` is new/rare, new-item exposure for the illustrative three-item list is `0.5000 / 2.1309 = 0.2346`, or about 23.46%.
  - If `D` occupies position three, none of these three articles is new/rare, so exposure is zero. An individual list can gain coverage while losing new-item exposure.
  - Search compares mean position-weighted new-item exposure against the baseline across completed top-ten lists. The non-decrease requirement applies to that aggregate, not to each prefix or impression. There is no new-item bonus, shortfall penalty, or per-list quota in the current formula.
  - Here, new/rare is defined by training click counts; it does not necessarily mean recently published.


#### Proposed alternative: relevance-first reranking with swaps

This is a design idea, **not implemented**. Instead of searching offline for coverage and KL weights, define per-list requirements such as a minimum category count, a minimum new-item exposure, and a maximum category KL. The intuition is: keep the most relevant news, then change only what is needed to meet those requirements.

1. Start with the ranker's top 10 articles from its top-50 pool. If all requirements pass, return them unchanged.
2. Try replacing one selected article with one of the 40 unselected articles. Sort each proposed list by relevance and recheck its coverage and position-weighted exposure metrics.
3. Prefer repairs according to explicit constraint priorities; among equally effective repairs, keep the most predicted relevance. Cap total relevance loss against the original top 10.
4. Repeat until the requirements pass, no improving swap is found, or an iteration limit is reached. Record unresolved violations and use a predefined fallback.

There are only **400 possible swaps per iteration**. Cached metadata and an early return can keep this lightweight; checking fewer candidates is a faster approximation that may miss useful repairs.

Per-list requirements differ from the current aggregate guardrails. Some pools cannot satisfy them, and greedy swaps can get stuck. Offline evaluation is still needed: preserving predicted relevance does not guarantee preserving actual nDCG.

## Evaluation
- **Ranking quality**: AUC, MRR, nDCG@K, MAP@K, Recall@K
- **Calibration**: ECE (expected calibration error), Brier score
- **Diversity**: intra-list diversity (ILD), category coverage@K, category entropy@K
- **Exposure fairness**: position-weighted exposure, disparity vs target distribution (KL / Gini), new-item exposure fraction

The official MIND leaderboard reports `AUC`, `MRR`, `nDCG@5`, and `nDCG@10` (see each `ranker_eval_*.json`). The official leaderboard uses the full/large hidden test set. Local metrics are split-dependent, so compare runs only when they use the same validation protocol.

The split protocols and current result locations are tracked in [docs/experiment_registry.md](docs/experiment_registry.md). Check that registry before comparing metrics across runs.

# Quickstart

## 0) Hardware target

This repo is designed to run on a powerful Windows laptop:
- trains and scores the neural models on a CUDA GPU through PyTorch
- uses **faiss-cpu only for ANN index construction and search**; this avoids a separate GPU-Faiss dependency and does not imply that model training runs on CPU
- uses the promoted MPNet text backbone with memory-bounded encoding, mixed-precision adaptation, and cached item representations
- supports full MIND-large processing plus optional config-driven subsampling

---

## 1) Setup (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2) Get the dataset (MIND)

Place files under:
```
data/raw/MINDlarge_train/
data/raw/MINDlarge_dev/
data/raw/MINDlarge_test/
```
Each folder should contain its MIND `behaviors.tsv` and `news.tsv` files. The hidden-test behaviors have no click labels. The reranker may use the entity annotation columns already present in `news.tsv` for coverage when `rerank.coverage.entity_bonus > 0`. When `knowledge_graph.enabled: true`, the model also needs to use `entity_embedding.vec` and `relation_embedding.vec`.

---

### 2.1) Quick terminology: entities in MIND

- In MIND, an **entity** is a named entity extracted from a news article (person, organization, location, etc.) and linked to a knowledge graph (the MIND paper references Wikidata).
- `entity_embedding.vec`: embedding vector for each entity ID.
- `relation_embedding.vec`: embedding vector for each relation type between entities.
- `WikidataId` is the bridge between `news.tsv` and `entity_embedding.vec`. During the `train_teacher` command, if an article's title or abstract entity annotation contains a `WikidataId`, the KG feature builder looks up the row with that ID in `entity_embedding.vec`.
- A **knowledge-graph triple** is a directed fact written as `(head entity, relation, tail entity)`. For example, `(Q76, P31, Q5)` means that entity `Q76` has relation `P31` to entity `Q5`. In a one-hop lookup, an entity mentioned by an article is the head or tail of a triple, and the entity at the other end is its neighbor.

The implemented approach follows the classic KG recommendation pattern:
- Parse linked `WikidataId` values from each article's title/abstract entity columns.
- Fetch those entity vectors from `entity_embedding.vec`.
- Fetch one-hop neighbors from the required triples file using triples `(entity, relation, entity)`, combine neighbor entity vectors with the relation vector from `relation_embedding.vec`, and aggregate the messages into a context vector.
- Concatenate the final KG vector with the sentence-transformer text vector to form `item_ranker_base_emb.npy`.
- Feed this KG-enhanced item base into the student ranker.

#### Building a triples file

A triples file we can use is **Wikidata5M**, a compact Wikidata-derived KG with the exact ID style this project needs: entities use `Q...`, relations use `P...`, and triples are stored as rows like `Q22686 P39 Q11696`. It is not guaranteed to be the exact MIND TransE training subgraph, but it is a reasonable Wikidata subgraph for one-hop neighbor expansion.

Download one of the Wikidata5M KG files from:
- https://deepgraphlearning.github.io/project/wikidata5m

Then filter it down to MIND-mentioned entities:

```powershell
python scripts/build_mind_wikidata5m_triples.py `
  --kg-path data/raw/wikidata5m/wikidata5m_transductive_train.txt `
  --raw-root data/raw `
  --train-dir MINDlarge_train `
  --dev-dir MINDlarge_dev `
  --entity-embedding data/raw/MINDlarge_train/entity_embedding.vec `
  --entity-embedding data/raw/MINDlarge_dev/entity_embedding.vec `
  --entity-embedding data/raw/MINDlarge_test/entity_embedding.vec `
  --relation-embedding data/raw/MINDlarge_train/relation_embedding.vec `
  --relation-embedding data/raw/MINDlarge_dev/relation_embedding.vec `
  --relation-embedding data/raw/MINDlarge_test/relation_embedding.vec `
  --output data/processed/MINDlarge/kg_triples.tsv
```

How this repo uses MIND entity annotations:
- `news.tsv` contains `title_entities` and `abstract_entities` columns.
- During greedy reranking, the reranker tracks covered entities for the entity coverage bonus.
- Reranker entity coverage is separate from the neural KG feature path: coverage decides list diversity, while KG features affect the learned ranker representation.

The accepted design is **text-only retrieval plus a KG-enhanced ranker**. Rejected retrieval variants are recorded in the [experiment registry](docs/experiment_registry.md#rejected-historical-retrieval-experiments).

---

### 2.2) What `run_preprocess()` does

`run_preprocess()` converts raw MIND TSV files into model-ready parquet/json files.

Main steps for Large temporal configs such as `configs/mind_large_temporal_mpnet.yaml`:
- Read `news.tsv` and `behaviors.tsv` from train/dev.
- Build ID mappings (`user_id/news_id/category/subcategory -> integer index`).
- Move the final day of `train_dir` into validation, then append all of `dev_dir` to that same validation split.
- Build reproducible random training/validation pairs. The random training pairs are kept because the same processed dataset supports the random-negative baseline; the promoted hard-negative ranker instead constructs temporary examples from grouped behaviors when ranker training begins.
- Build impression-level validation data plus separate chronological views for reranker tuning and reporting.

How pairs are created:

- For an impression with `P` positives and `N` negatives, `P * (1 + min(data.ranker_negatives_per_positive, N))` pairs are generated.
- Each positive is paired with up to `data.ranker_negatives_per_positive` sampled negatives; the current config uses `4`.
- These negatives are labeled non-clicked candidates from the same impression. This is the fallback ranker pair-building behavior when hard-negative sampling is disabled.

#### Negative selection by training stage

- **Text adaptation (Phases 1 and 3):** samples are generated lazily each epoch. For a cold user with usable history, a frozen snapshot of the text encoder at the start of that phase scores a random pool of up to 20 same-impression negatives; one hard and three random negatives are retained. Phase 1 uses the base backbone snapshot, while Phase 3 uses the selected Phase 1 checkpoint. Warm users receive four random negatives, and impressions without usable history are omitted because the adaptation objective requires a history representation.
- **Teacher training:** this is independent of the hard-negative machinery. It randomly samples up to eight non-clicked candidates from the same impression for each positive and also uses in-batch positives as contrastive negatives.
- **Student-ranker training (Phases 2 and 3):** before ranker optimization, the frozen trained teacher scores a random same-impression pool of up to 20. Cold users with 1–4 usable history items retain one teacher-hard and three random negatives; warm and zero-history users retain four random negatives. These temporary rows are built from grouped training behaviors, independently of the retained baseline `train_pairs.parquet` artifact.
- To avoid teacher/student label conflict, teacher-hard candidates are selected only from negatives that the teacher scores no higher than the clicked positive in the same impression. Ranker distillation is disabled on those teacher-mined rows by default. Text adaptation applies the analogous positive-consistency filter using its frozen encoder snapshot.

Why there is no `train_impressions.parquet`:
- Ranker training uses pairwise rows, either persisted in `train_pairs.parquet` for random sampling or built dynamically from `train_behaviors.parquet` for hard mining.
- Impression-grouped data is mainly needed for ranking evaluation. Temporal-tune configs generate combined `val`; Large temporal configs additionally generate reranker-only `rerank_tune` and `rerank_test` views.

#### Large data timeline

| Calendar window | Raw source | Role during model selection | Reranker role | Impressions |
| --- | --- | --- | --- | ---: |
| Nov 9–13 | `MINDlarge_train` before the cutoff | Large Temporal Train | Not used | 1,801,231 |
| Nov 14 | Tail of `MINDlarge_train` | Part of Large Temporal Val | Tune priorities and policy | 431,517 |
| Nov 15 | `MINDlarge_dev` | Part of Large Temporal Val | Frozen-setting follow-up report | 376,471 |
| Nov 16–22 | `MINDlarge_test` | Hidden competition test; labels unavailable | Not used | 2,370,727 |

Large Temporal Val therefore contains 807,988 impressions across the middle two rows. Upstream encoder, teacher, and ranker selection used that combined validation set. Reranking uses two chronological views, but November 15 was also reused while refining requirements. Its results are follow-up measurements, not an independent test.

In Phase 3, the selected encoder continues on Large Temporal Val, and the teacher/ranker fit uses all labeled Train + Dev impressions. Hidden Test click labels are never available or used.

Evaluation also divides each holdout into chronological `time_period__...` slices so regressions can be checked against impression order:

![images/AUC_over_time.png](images/AUC_over_time.png)

![images/nDCG_over_time.png](images/nDCG_over_time.png)

---

## 3) End-to-end Large MPNet workflow

Sections 3.1--3.4 reproduce the original MPNet (`lr=2e-5`, Large Test AUC `0.6948`). The best reported single model uses `lr=1e-5` and achieves `0.6960`; its complete training workflow is retained at commit `31ab682` on `feature/lr_sweep_2`. The phase runner below reproduces the original model.

The promoted workflow first selects the model on the chronological Large temporal split, then performs a maximum-data fit for the hidden leaderboard test.

| Phase | Work performed | Data and output |
| --- | --- | --- |
| **1: Select the text encoder** | Adapt MPNet and select its checkpoint using temporal validation. | Train on Nov 9–13; validate on Nov 14–15. Output: the selected update-9,000 encoder. |
| **2: Validate the complete model** | Freeze the selected encoder, train the teacher and student ranker, and evaluate the complete temporal model. | Train on Nov 9–13; validate on Nov 14–15. Output: temporal checkpoints and evaluation reports. |
| **3: Train on all labeled data and submit** | Continue the encoder, train the teacher and ranker with fixed schedules, and rank every hidden-test candidate. | Continue the encoder on Nov 14–15; train the teacher/ranker on all Train + Dev. Output: the Nov 16–22 hidden-test submission. |

These are training and submission phases. Optional ANN retrieval evaluation and reranking experiments follow in Sections 3.5 and 3.6.

### 3.1 Prepare Large temporal data

After placing the Large raw files and building the KG triples described above, run:

```powershell
python -m mindrec.cli preprocess --config configs/mind_large_temporal_mpnet.yaml
```

This creates the training, validation, and reranker views shown in the timeline above. The promoted phase runner performs this step automatically when the compatible processed data is not already present.

### 3.2 Phase 1: select the adapted MPNet checkpoint

```powershell
.\scripts\run_mpnet_backbone.ps1 -Phase phase1
```

Phase 1 adapts `all-mpnet-base-v2` on Large Temporal Train for at most 10,000 successful optimizer updates. Every 1,000 updates it evaluates the text objective on Large Temporal Val; early stopping selected update 9,000. This phase selects only the text encoder—it does not yet train the final two-tower teacher or student ranker.

For each training impression, MPNet encodes every article as `title [SEP] abstract`; the mean of up to 10 clicked-history embeddings becomes the temporary user vector. A temperature-scaled contrastive loss trains the encoder to score the clicked candidate above four same-impression negatives and also separates mismatched user/positive pairs within the batch. AdamW updates the encoder itself (`lr=2e-5`, weight decay `0.01`) using batches of 16 with four-step gradient accumulation. Phase 3 later continues the same objective from the selected checkpoint.

### 3.3 Phase 2: train and evaluate the complete temporal model

```powershell
.\scripts\run_mpnet_backbone.ps1 -Phase phase2
```

Phase 2 loads the selected update-9,000 encoder, trains the two-tower teacher, then trains the candidate-attention student with the hard-negative and distillation policies described above. It finally evaluates the student on all 807,988 Large Temporal Val impressions. The completed model reached AUC `0.688880`, MRR `0.336397`, nDCG@5 `0.371215`, and nDCG@10 `0.431291`.

The ranker evaluation also reports chronological, history-length, cold/new-item, popularity, category, and subcategory slices. A slice such as `impressions_with_clicked_new_item` evaluates whole impressions containing at least one clicked new item; it does not evaluate only the new candidates.

![images/AUC_by_cold_warm_user.png](images/AUC_by_cold_warm_user.png) ![images/nDCG_by_cold_warm_user.png](images/nDCG_by_cold_warm_user.png)

#### Architecture validated in Phase 2

Phase 2 trains and validates the following architecture, which Phase 3 carries into the maximum-data fit:

- The **text backbone** encodes article text. The promoted model uses `all-mpnet-base-v2` (768 dimensions); the earlier MiniLM experiments are retained in the [experiment registry](docs/experiment_registry.md).
- Text adaptation contrasts each clicked candidate with impression negatives against the mean of up to 10 recent-history article embeddings. Cold users with usable history receive one hard and three random negatives; warm users use random negatives.
- The **two-tower teacher** projects item text into a 384-dimensional retrieval space and uses multi-head attention to pool the clicked history. It supplies retrieval embeddings plus logit and representation targets for distillation.
- The **student DLRM-style ranker** combines learned ID/category features, dense behavioral features, text-plus-KG item semantics, and candidate-aware attention over text-plus-KG history semantics.
- The competition submission scores every supplied impression candidate with the student ranker. The selected submission adds only the label-free recency tiebreaker (`alpha=0.02`); it does not run ANN retrieval or the optional diversity/fairness reranker.

##### Candidate-aware history attention

Mean pooling gives every candidate in an impression the same semantic user vector. Candidate-aware pooling instead projects the current candidate as the query in four-head attention over the clicked-history item states as keys and values. The resulting user vector is therefore different for, say, a sports candidate and a health candidate shown to the same user. Empty histories map to a zero semantic user vector, leaving the ID, taxonomy, item-semantic, and dense branches to score the candidate. This change improved the matched MiniLM temporal AUC from `0.664328` to `0.671593` and was retained in the promoted MPNet ranker.

Implementation details and completed temporal results are recorded in the [MPNet experiment reference](docs/experiment_registry.md#controlled-mpnet-backbone-experiment).

### 3.4 Phase 3: train on all labeled data and build the leaderboard submission

Phase 3 uses all labeled data to build the final leaderboard model. It continues the selected Phase 1 encoder on Large Temporal Val, then trains the teacher and ranker on all Large Train + Dev impressions using the architecture validated in Phase 2. Finally, it scores every supplied hidden-test candidate and writes the submission ranks:

```powershell
.\scripts\run_mpnet_backbone.ps1 -Phase phase3
```

The orchestration script implements all three phases and checks artifact provenance before reuse. It directly uses these configs:

- `mind_large_temporal_mpnet.yaml` selects MPNet and candidate attention in Phases 1–2 and configures the reranker search and selection.
- `mind_large_submission_mpnet_text_continue.yaml` continues the selected update-9,000 encoder for exactly 2,000 successful optimizer updates on Large Temporal Val, without another early-stopping decision.
- `mind_large_submission_mpnet.yaml` and `mind_large_submission_mpnet_candidate_attention.yaml` train the teacher for four complete epochs and the candidate-attention ranker for two complete epochs on Large Train + Dev, with early stopping disabled.
- `mind_large_submission_mpnet_candidate_attention_recency_alpha_002.yaml` applies the constant, non-learned recency coefficient `alpha=0.02` and writes the submission.

Here the schedules are **locked before maximum-data training**: the encoder update count and teacher/ranker epoch counts are no longer selected in Phase 3, because no labeled local holdout remains. The teacher and ranker parameters are still trained normally; “locked” describes their training schedules, not frozen model weights. Likewise, `alpha=0.02` is a configured post-hoc constant rather than a learned parameter.

Inherited defaults and configs for reproducing other baselines are documented in the [configuration background](docs/experiment_registry.md#configuration-background).

#### Submission execution and completed result

Before training, Phase 3 verifies the Phase 2 metrics and selected update-9,000 checkpoint. After the fixed training schedule, it builds or reuses the item-age index and scores the hidden candidate sets with `alpha=0.02` recency. Compatible completed stages are reused; incompatible metadata is rejected instead of silently mixing runs.

The original MPNet submission achieved Large Test AUC **`0.6948`**. This is `+0.0079` over the selected MiniLM candidate-attention submission (`0.6869`), `+0.0100` over text-adapt v1 (`0.6848`), and `+0.0224` over frozen MiniLM (`0.6724`). Its 2,370,727 impression rankings passed sequential-ID, rank-permutation, ZIP-integrity, and content-hash checks.

To write the candidate-attention model without the recency tiebreaker, use `python -m mindrec.cli write_submission --config configs/mind_large_submission_mpnet_candidate_attention.yaml`; that path does not require `build_item_age`.

MIND has no publication timestamps, so the recency adjustment uses an exposure-age proxy: time since an article first appeared as a candidate in the observed dataset. The clock is evaluated separately for each impression and uses no click labels. The [submission reference](docs/submission.md) explains the age calculation, missing-value behavior, and scoring formula; the [metrics guide](docs/metrics.md#mind-submission-evaluation) explains the evaluator format.

Optional submission ensembling is documented separately in [Ensembling: MPNet and MiniLM](docs/ensembling.md), including results and CLI commands.

### 3.5 Optional ANN retrieval evaluation

ANN retrieval is useful for a production-style full-catalog recommender but is not needed when writing a MIND submission, because each test impression already provides its candidate set. To evaluate retrieval from the Phase 2 teacher:

```powershell
python -m mindrec.cli build_index --config configs/mind_large_temporal_mpnet.yaml
python -m mindrec.cli eval_retrieval --config configs/mind_large_temporal_mpnet.yaml
python -m mindrec.cli eval_retrieval_sweep --config configs/mind_large_temporal_mpnet.yaml
```

The retrieval reports include chronological, history-length, popularity, category, and subcategory slices. The configured sweep compares the text-only fallback weight and oversampling choices, then selects the best held-out `recall@K` setting.

### 3.6 Reranking: search offline, use fixed weights in production

**Offline search finds the weight combination with the highest mean nDCG@10 among the evaluated combinations meeting every aggregate guardrail.** Each combination builds actual top-10 lists on tuning impressions; labels are used afterward to evaluate those lists. **The weights are found offline and reused during production reranking.** At each step, the greedy list builder selects the highest-scoring remaining article, updates the list state, and repeats. This same list-building procedure is used during search and serving; production does not search for weights again.

For candidate article `i` and already-selected list `S`:

```text
score(i | S) = wR * R(i) + wC * C(i, S) - lambda * KL(p || q)
wR = 1 - wC
```

- `R`: predicted relevance, min-max normalized within the ranker's top-50 pool.
- `C`: `1.0 * new_category + 0.3 * min(new_entity_count, 3)`.
- `p`: prospective prefix's position-weighted category exposure after adding `i`; `q`: the accessible candidate pool's category mix.

Semantic novelty is absent from the score and search. Teacher-cosine ILD is reported on completed lists for diagnosis only. There is no L1 or new-item shortfall penalty; new-item exposure remains an aggregate guardrail.

| Reranking choice | How it is determined |
| --- | --- |
| Coverage weight `wC` and pool-KL penalty weight `lambda` | Search the parameter grid and select the highest-nDCG eligible combination. |
| Relevance weight `wR` | Derive as `1 - wC` for each combination. |
| Coverage bonuses (category 1.0, entity 0.3) and cap (3 new entities) | Set the score definition before comparing weights. |
| Relevance normalization, position weights, and metric definitions | Use the same definitions for every combination and during serving where applicable. |
| Aggregate guardrail thresholds | Agree on acceptable outcomes before comparing weights. |

Relevance weight is derived automatically as `1 - coverage_weight`; do not configure it separately. Bonuses and caps are technical modeling choices; business stakeholders specify acceptable outcomes. The implementation uses logarithmic position weights and min-max relevance normalization. Trials reuse the same ranker predictions and tuning impressions so that their differences reflect the reranking weights.

| Aggregate requirement relative to the original ranker | Threshold |
| --- | ---: |
| Maximum relative nDCG@10 loss | 2% |
| Minimum new-item exposure gain | 0 (non-regression) |
| Minimum mean category-coverage gain | +0.25 categories/list |
| Minimum mean pool-KL reduction | 0.03 |

#### Choosing the initial parameter grid

There is no universally correct weight range: it depends on the sizes of the score components and the relevance differences between candidates. Business stakeholders set acceptable outcomes; inspecting scores and running tuning comparisons determines useful weight ranges.

1. **Look at articles with similar relevance scores.** Choose trial weights large enough for better coverage or category balance to change some close decisions. For example, a coverage bonus of 0.025 can outweigh a weighted relevance gap of 0.02 when the other score terms are equal.
2. **Try a few weights, including zero.** Start with coverage weights `[0, 0.025, 0.05, 0.10]` and KL penalties `[0, 0.025, 0.05, 0.10, 0.20]`, testing every combination. Zero turns a component off, helping show whether it is useful.
3. **Inspect feasibility and relevance together.** If lists barely change, inspect whether weights are too small or components favor the same candidates. If nDCG degrades sharply, test smaller weights. If coverage remains insufficient, check whether the candidate pools contain enough additional categories. Larger weights cannot create missing candidates, and coverage or KL weights cannot reliably solve a new-item exposure shortfall.
4. **Expand promising boundaries, then refine locally.** If the best eligible setting lies at the largest tested weight and nearby results remain promising, test a larger value; a boundary winner alone does not prove expansion will help. Add intermediate values around the best region, retain the previous winner, and compare all finalists on full tuning data with unchanged guardrails. For example, a coverage winner at 0.05 between 0.025 and 0.10 suggests testing 0.0375 and 0.075. Keep useful zero controls when expanding the comparison.

#### Offline search procedure

1. Score the labeled November 14 tuning impressions with the frozen ranker. Its original top-ten lists form the comparison baseline.
2. Build lists for the current 15-setting local grid on a deterministic 5,000-impression sample: `wC` in `[0.025, 0.03, 0.035, 0.04, 0.05]`, and `lambda` in `[0, 0.025, 0.05]`. It retains the best feasible tuning setting (`wC=0.035`, `lambda=0`) and includes nearby coverage adjustments.
3. Evaluate **all 15 combinations on full tuning data** (`shortlist_size: 15`). Screening supplies diagnostics and cannot exclude a grid point. Remove settings failing any guardrail; choose the remaining setting with the **highest full-tuning nDCG@10**. If none qualifies, select none.
4. Inspect `grid_diagnostics` in `rerank_search.json` or the readable `pareto_frontier.md`: relevance-gap scales, candidate-pool opportunity, failures by guardrail, feasibility across parameter values, and boundary winners. These profiles and flags use full-tuning results when available; partial grids are labeled, with untested points left empty. Pool-opportunity statistics remain sample-based. Extend an upper range only when the profile supports it. A boundary winner alone does not prove larger weights help.
5. Refine around the full-tuning winner using the report's suggested midpoint values. Edit this same config and make `shortlist_size` large enough to fully evaluate the local grid. Keep the previous winning combination. Use full-tuning results when they disagree with the sample; increase evaluation coverage if an expanded grid was only partially evaluated. Keep thresholds fixed during these comparisons.
6. Copy the winning coverage and KL penalty weights into the serving configuration, record the selection provenance, freeze, then report on November 15.

This solves `argmax mean_nDCG@10(theta)` subject to the guardrails over the fully evaluated grid. It does not establish a global optimum beyond that grid. See the [grid-range process](docs/reranking.md#choosing-score-definitions-and-a-search-grid) for scale examples, failure diagnosis, expansion and stopping guidance.

**Inspect feasibility before interpreting the winner:**

A useful pattern to look for as reranking becomes stronger is:

```text
Too weak: coverage or KL improvement fails
    ↓
Useful region: all guardrails pass
    ↓
Too strong: nDCG loss exceeds the allowed limit
```

| Observation | Interpretation and next action |
| --- | --- |
| Some settings pass every guardrail | Compare their full-tuning nDCG and inspect the region around the best. |
| Low weights fail coverage, moderate weights pass, high weights lose too much nDCG | A useful region may lie between insufficient change and excessive relevance loss. |
| Rankings barely change | Weights may be weak, but components might also be constant or favor the same articles as relevance. Inspect before increasing them. |
| Coverage fails throughout the grid | Check whether pools contain enough additional categories and whether exposing them fits the nDCG budget. Larger weights cannot create missing candidates. |
| New-item exposure consistently declines | Increasing coverage or KL weights may not solve it; neither directly targets new items. |

Individual improvements do not prove joint feasibility: one setting might pass coverage while another passes exposure, with neither passing both. Reported parameter profiles take the best results across other parameter choices; they help locate promising regions but do not isolate one weight's causal effect.

#### Serving behavior

**Production uses the same winning parameters and list builder as offline evaluation, without searching again.** Take the ranker's top 50 candidates, start with an empty list, and repeatedly select the highest-scoring remaining article. Update covered categories/entities and exposure state after each selection. Stop at ten articles or when the pool is exhausted. Component values change as the list grows; the selected weights stay fixed.

**Why do guardrails not apply "up to this item"?** They concern average changes in completed lists across many impressions. One prefix is neither a completed list nor that population. The first position always has at most one category; requiring an immediate +0.25 category gain could reject a useful eventual list. Prefix KL guides selection without enforcing aggregate guarantees. Production outcomes need monitoring over traffic windows.

**Why not maximize actual nDCG at each step?** Serving has no ground-truth click or relevance labels. The algorithm maximizes its available predicted score; offline labels let us compare the complete lists generated by different weights.

Production monitoring should compare completed-list metrics over traffic windows, with an agreed fallback to the original ranker when aggregate outcomes fail. A single list missing an aggregate target is not itself such a failure. This repo implements the algorithm and offline evaluation; serving, monitoring, and rollback services are not implemented.

#### Current results and reproduction

**Current status: evaluation complete; three of four reporting guardrails pass.** The frozen setting uses coverage **0.035**, KL penalty **0**, and derived relevance **0.965**. It was the highest-nDCG eligible setting among 15 fully evaluated combinations on 431,517 tuning impressions. All tuning requirements passed, but the pool-KL requirement did not hold on the 376,471 November 15 reporting impressions:

| Metric | Reporting baseline | Reranked | Guardrail outcome |
| --- | ---: | ---: | --- |
| nDCG@10 | 0.439457 | 0.438302 | Pass: 0.263% relative loss, within 2% |
| New-item exposure | 89.1707% | 89.2423% | Pass: +0.0716 percentage points |
| Categories/list | 4.955561 | 5.236411 | Pass: +0.280850, above +0.25 |
| Pool KL | 0.442187 | 0.413450 | **Fail:** reduction 0.028736, below 0.03 |

To reproduce the search and frozen-setting evaluation:

```powershell
python -m mindrec.cli rerank_search --config configs/mind_large_temporal_mpnet.yaml
python -m mindrec.cli rerank_eval --config configs/mind_large_temporal_mpnet.yaml
```

November 15 was reused while refining requirements; this is a follow-up report, not an independent test. The [reranking guide](docs/reranking.md) records the selection decision, tuning and reporting metrics, and the unmet requirement.

---

## 4) Repo layout

- `src/mindrec/`
  - `cli.py` and `config.py`: command entry points and inherited YAML loading.
  - `data/`: MIND parsing, datasets, feature/KG construction, item-age indexing, and the recency adjustment.
  - `models/`: the two-tower teacher, DLRM-style student, distillation, and calibration modules.
  - `pipeline/`: preprocessing, text adaptation, teacher/ranker training, hard-negative mining, retrieval, evaluation, reranking, and submission.
  - `rerank/`: the deterministic greedy relevance/diversity/fairness policy.
  - `metrics/`: ranking, calibration, diversity, fairness, and slice benchmarks.
- `configs/`: composable Large temporal, submission, MPNet, and reranker experiment configs.
- `scripts/`: the promoted MPNet phase runner, KG-triples builder, and GPU check.
- `tests/`: focused tests for adaptation, hard negatives, candidate attention, taxonomy handling, reranking, recency, age, and submission integrity.
- `docs/`: experiment registry, metric definitions, submission reference, and the detailed reranking and ensembling workflows.
- `notebooks/` and `images/`: evaluation-slice visualization and README figures.
- `data/` and `runs/`: local raw/processed datasets and generated experiment artifacts; neither contains the implementation itself.

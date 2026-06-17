# MIND Multi-Stage News Recommender (Retrieval → DLRM+Knowledge_Graph Ranker → Diversity+Coverage+Fairness Re-ranker)

This project implements a realistic recommender stack on the **Microsoft News Dataset (MIND)**:
- Preprocessing: prepare train/val/test data, click-count features, cold/new flags, impression-level eval data, and map IDs to indices.
- Train **Teacher retrieval encoders** (text-based item encoder + history-based user encoder).
- Build two Faiss ANN retrieval indexes: (1) for item embeddings from teacher, and (2) for item-base features. These indexes are used for hybrid candidate retrieval.
- Train **Student ranker**, a lean DLRM-style sparse+dense model that narrows **hybrid retrieval** returned `topk` items to more relevant `pool_size` items. Alongside the usual DLRM-style ID embeddings, category/subcategory embeddings, and dense features, this student has separate text and **Knowledge Graph** semantic branches. Representation distillation targets only the text branch, while final-logit distillation regularizes the complete ranker score.
- **Re-ranking** enforcing a certain degree of item diversity, category/named-entity coverage, and position-weighted exposure fairness for categories/new-items. It further narrows `pool_size` items to `k_out` items.
- Find the best set of reranking weights/penalties given constraints on metrics.
- Extensive **evaluation**: ranking metrics, calibration, diversity, exposure fairness, and cold/new slices.

In production, for each user we have three inference steps:
1. Retrieval narrows millions/thousands of items to `topk=200`
2. Student ranker scores the topk, and narrows down to `pool_size=50`
3. Reranker takes the `pool_size`, and outputs final `k_out=10` items

MIND is widely used as a benchmark for news recommendation, with impression logs and rich news metadata.

Production design note: if a fresh batch of news arrives, for example from the last 15 minutes, the system would need fresh item-side representations. For each new article:
- Run the sentence-transformer over text/title/abstract to produce the text-only retrieval `item_base` embedding.
- Pass the text-only `item_base` through the teacher item encoder to produce `item_teacher_emb`.
- Parse linked Wikidata entities and build relation-aware, one-hop-enriched entity slots for the separate KG ranker branch.
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
  - `item_kg_base_emb.npy`: structured `[news, entity_slot, KG_dim]` relation-aware entity features used by the student ranker
  - `item_teacher_emb.npy`: the final teacher embedding for every news item
  - `user_teacher_emb.npy`: the final teacher embedding for each training user history
- These files are then reused by later stages:
  - `build_index` builds the Faiss retrieval index from `item_teacher_emb.npy`
  - `eval_retrieval` uses `item_teacher_emb.npy` and the saved teacher model to encode held-out histories and search that index
  - `train_ranker` loads text features from `item_base_emb.npy` and KG features from `item_kg_base_emb.npy`
  - representation distillation targets only the text semantic branch using `item_teacher_emb.npy` and teacher user vectors computed from each batch's history

## Hybrid retrieval scoring

During `eval_retrieval`, the teacher retrieval query is built by taking the impression's clicked history news, looking up their teacher item embeddings, and passing those vectors through `TeacherTwoTower.encode_user_from_item_vectors()`.
The base retrieval query is built by averaging the impression's clicked history news' raw sentence-transformer `item_base_emb.npy` embeddings.

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

![Recall@200 by user history length bucket](images/recall_by_user_history_len_bucket.png)

![nDCG@10 by user history length bucket](images/nDCG_by_user_history_len_bucket.png)

## Student model

The student keeps a lighter semantic core than the teacher, but combines it with classic DLRM signals:
- projected `user_id` / `news_id` branches
- category and subcategory embeddings
- lightweight dense features
- DLRM-style feature interactions

#### DLRM-style semantic additions
- A classic DLRM usually combines sparse ID/category embeddings, dense numerical features, pairwise feature interactions, and a final top MLP. It usually does not include item text embeddings or user-history semantic embeddings directly; those are project-specific additions here.
- In `DLRMStudent`, the candidate item's sentence-transformer text vector is projected into `item_text_sem`, and the clicked-history text vectors are pooled/projected into `user_text_sem`.
- Separately, each candidate article keeps multiple relation-aware entity slots from `item_kg_base_emb.npy`. A simplified KRED-style encoder projects those slots and uses the article's own text semantic vector as an attention query to select relevant entities before producing `item_kg_sem`.
- The same text-conditioned entity attention is applied independently to every clicked-history article before its article-level KG vectors are pooled into `user_kg_sem`. Explicit entity-slot and article masks keep missing KG representations exactly zero.
- A bounded gate, configured by `ranker.dlrm.kg_gate_init`, weakens the normalized KG vectors before they enter the DLRM feature interactions. The current setting fixes it at `0.15`.
- Set `ranker.dlrm.kg_gate_trainable = false` to hold that gate fixed. A fixed gate of `0.0` gives a controlled text-only ranker baseline while preserving the same DLRM architecture and input dimensions.
- `sem_fused` uses only the text user/item semantic states. The gated KG vectors participate only as ordinary DLRM feature vectors in the pairwise interaction layer and final top MLP.

#### Distillation representation
- In `DLRMStudent.forward()`, the student representation used for distillation is `rep = [user_text_sem, item_text_sem]`.
- `user_text_sem`: student semantic user vector from pooled click-history text features
- `item_text_sem`: student semantic item vector from the candidate item's text feature
- `user_kg_sem` / `item_kg_sem`: KG semantic vectors used by the ranker, but not included in representation distillation
- `sem_fused`: a lightweight attention-fusion summary that mixes only the text user/item semantic states with the structured query context
- The teacher target is `concat(teacher_user_emb, teacher_item_emb)`. A projection head maps only the text-branch student representation into the teacher space for representation distillation.
- The teacher is semantic/history-based rather than mostly `user_id`/`news_id` memorization. Representation distillation therefore encourages the text branch to learn a useful user/item space.
- Separately, final-logit distillation compares the teacher's soft semantic user-item score with the student's complete final score. Because that student score includes text, KG, IDs, categories, and dense features, this acts as a general scorer regularizer rather than requiring the KG representation itself to imitate the text-only teacher.
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
| Frozen text-only item-base input | Base text semantics for each news item | Text branch from `item_base_emb.npy` plus separate KG branch from `item_kg_base_emb.npy` | Retrieval stays text-only; KG is a separate ranker feature group |
| Teacher semantic item encoder | Refines each item into the teacher semantic space | Smaller student semantic item encoder | Student semantic dimension is much smaller than the teacher space |
| Teacher sequence-aware user encoder | Contextualizes clicked history items with attention before pooling | Cheaper history aggregation path | Student uses a lighter history refinement and aggregation path |
| Teacher attention pooling over history | Builds one semantic user vector from clicked history | Mean-style pooled semantic user path | Student pooling is cheaper and less sequence-aware |
| Teacher item embedding | Semantic representation of the candidate item | `item_text_sem` | Text item representation is trained to approximate teacher behavior |
| Teacher user embedding | Semantic representation of the current user history | `user_text_sem` | Text user representation is cheaper to compute |
| Teacher cosine-style user-item scorer | Measures semantic compatibility between user and item | Student top MLP ranker | Student scoring uses richer ranking signals beyond pure semantic similarity |
| Teacher soft score | Provides general final-score regularization | Complete student final logit | The gradient reaches the complete scorer, but no individual KG representation is forced to match a teacher representation |
| Teacher representation target | Provides a semantic supervision target for distillation | `rep = [user_text_sem, item_text_sem]` plus projection head | KG semantic vectors are not representation-distilled |
| No direct teacher counterpart | None | gated `user_kg_sem` / `item_kg_sem` | Available KG is learned through click labels and general final-score distillation; it is not representation-distilled |
| No direct teacher counterpart | None | `sem_fused` | Student adds an attention-fusion summary conditioned on query context |
| No direct teacher counterpart | None | `user_id` / `news_id` branches | Student adds collaborative-style memorization signals |
| No direct teacher counterpart | None | category / subcategory embeddings | Student adds structured metadata signals |
| No direct teacher counterpart | None | dense features such as `history_len` and `item_clicks_log1p` | Student adds non-semantic ranking features |
| No direct teacher counterpart | None | DLRM interaction terms + top MLP | Student is a broader ranker, not just a semantic retriever |

Category and clicked-item-popularity slices help check whether the model is robust across content verticals and between new, low-click, and high-click items:

![Recall@200 by clicked category](images/recall_by_clicked_category.png)

![nDCG@10 by clicked category](images/nDCG_by_clicked_category.png)

![Recall@200 by clicked item popularity](images/recall_by_clicked_item_popularity.png)

![nDCG@10 by clicked item popularity](images/nDCG_by_clicked_item_popularity.png)

#### Where the student ranker is simplified
- teacher semantic item encoder -> smaller student semantic item encoder
- teacher sequence-aware user encoder -> cheaper history aggregation path
- teacher cosine scorer -> student MLP ranker with many extra inputs

## Re-ranking
- In this project, re-ranking is a **deterministic optimization layer** on top of ranker scores.
- It is controlled by hyperparameters/constraints (relevance, novelty, coverage, fairness).
- **No training loop is required** for this re-ranking stage.

#### Re-ranking process
- `greedy_rerank()` takes the top `pool_size` candidates by ranker score, then builds the final top-`k_out` list one item at a time.
- At each step it scores every remaining candidate with:
- `relevance_weight * relevance`
- `+ novelty_weight * novelty`
- `+ coverage_weight * coverage`
- `- fairness.penalty_weight * fairness_penalty`
- It then picks the candidate with the highest total value, adds it to the list, updates the running novelty/coverage/fairness state, and repeats until `k_out` items are selected.

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
- `rerank_eval` and `rerank_search` report both `fairness_kl_pool` and `fairness_kl_full`.
- Product constraints in reranker search use `fairness_kl_pool`, because that matches the reranker's actual optimization target.

#### Worked examples for novelty, coverage, and fairness
- Suppose the reranker has already selected two items: `A` and `B`.
- Candidate `C` has ranker relevance score `0.80`, category `Sports`, entities `{Messi, Inter Miami}`, and is marked as a new item.
- Candidate `D` has relevance score `0.78`, category `Health`, entities `{WHO, vaccine, pandemic, hospital}`, and is not new.

- **Novelty example with `teacher_cosine`**:
  - If `C` has teacher-embedding cosine similarities `0.90` to `A` and `0.35` to `B`, then `novelty(C) = -max(0.90, 0.35) = -0.90`.
  - If `D` has similarities `0.20` to `A` and `0.10` to `B`, then `novelty(D) = -0.20`.
  - Because `-0.20 > -0.90`, `D` is treated as more novel than `C`.

- **Coverage example**:
  - Suppose the selected list has already covered categories `{Sports, Politics}` and entities `{Messi, Real Madrid}`.
  - If `coverage.category_bonus = 1.0`, `coverage.entity_bonus = 0.3`, and `max_new_entities_per_item = 3`:
  - `C` is in `Sports`, which is already covered, so it gets no category bonus. It adds one new entity, `Inter Miami`, so `coverage(C) = 0.3`.
  - `D` is in `Health`, which is new, so it gets `1.0` category bonus. Its four entities are all new, but only the first `max_new_entities_per_item = 3` new entities can contribute, so its entity bonus is `3 * 0.3 = 0.9` and `coverage(D) = 1.9`.

- **Exposure fairness example**:
  - Suppose the current top-3 list has categories `[Sports, Sports, Health]`.
  - With log position weights, a typical exposure pattern is roughly `[1.00, 0.63, 0.50]`.
  - Then the actual category exposure map is:
  - `Sports: 1.00 + 0.63 = 1.63`
  - `Health: 0.50`
  - After normalization, this becomes `p`, the actual exposure share by category.
  - If the reference candidate pool category mix is `Sports: 50%`, `Health: 30%`, `Politics: 20%`, that normalized mix is `q`.
  - The fairness penalty compares `p` and `q` using both KL divergence and L1 distance.

- **New-item exposure penalty example**:
  - Suppose `new_item_floor = 0.20`.
  - If only the rank-3 item is new, then new-item exposure is `0.50`.
  - Total exposure is `1.00 + 0.63 + 0.50 = 2.13`.
  - So `new_item_exposure_frac = 0.50 / 2.13 = 0.235`, which is above the floor, so no extra penalty is added.
  - If no selected item is new, then `new_item_exposure_frac = 0.0`, which is below `0.20`, so the fairness penalty is increased.

## Evaluation
- **Ranking quality**: AUC, MRR, nDCG@K, MAP@K, Recall@K
- **Calibration**: ECE (expected calibration error), Brier score
- **Diversity**: intra-list diversity (ILD), category coverage@K, category entropy@K
- **Exposure fairness**: position-weighted exposure, disparity vs target distribution (KL / L1 / Gini), new-item exposure floor

The official MIND leaderboard reports `AUC`, `MRR`, `nDCG@5`, and `nDCG@10` (see each `ranker_eval_*.json`). The official leaderboard uses the full/large hidden test set. These metrics are not directly comparable to this repo's metrics (from `MINDsmall_dev`'s time-based val-test splits).

<br>
<br>

# Quickstart

## 0) Hardware target

This repo is designed to run on a powerful Windows laptop:
- uses **faiss-cpu**
- uses a small, strong sentence-transformer as teacher by default (MiniLM family), and caches item embeddings
- supports sub-sampling MIND-large, or using **MIND-small** first

---

## 1) Setup (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

## 2) Get the dataset (MIND)

Place files under:
```
data/raw/MINDsmall_train/
data/raw/MINDsmall_dev/
```
Each folder should contain `behaviors.tsv` and `news.tsv`. The reranker may use the entity annotation columns already present in `news.tsv` for coverage when `rerank.coverage.entity_bonus > 0`.
When `knowledge_graph.enabled: true`, the model also needs to use `entity_embedding.vec` and `relation_embedding.vec`.


---

### 2.1) Quick terminology: entities in MIND

- In MIND, an **entity** is a named entity extracted from a news article (person, organization, location, etc.) and linked to a knowledge graph (the MIND paper references Wikidata).
- `entity_embedding.vec`: embedding vector for each entity ID.
- `relation_embedding.vec`: embedding vector for each relation type between entities.
- `WikidataId` is the bridge between `news.tsv` and `entity_embedding.vec`. During the `train_teacher` command, if an article's title or abstract entity annotation contains a `WikidataId`, the KG feature builder looks up the row with that ID in `entity_embedding.vec`.
- A **knowledge-graph triple** is a directed fact written as `(head entity, relation, tail entity)`. For example, `(Q76, P31, Q5)` means that entity `Q76` has relation `P31` to entity `Q5`. In a one-hop lookup, an entity mentioned by an article is the head or tail of a triple, and the entity at the other end is its neighbor.

The implemented approach is a simplified KRED-style KG encoder:
- Parse linked `WikidataId` values from each article's title/abstract entity columns.
- Fetch those entity vectors from `entity_embedding.vec`.
- Fetch one-hop neighbors from the required triples file using triples `(head, relation, tail)`. Each one-hop message combines the neighbor embedding with a signed relation embedding so incoming and outgoing uses of the same relation remain distinguishable.
- For each directly mentioned entity, aggregate its relation-aware one-hop messages into that entity's own enriched slot instead of immediately averaging the whole article's graph context.
- Save the structured entity-slot tensor as `item_kg_base_emb.npy`.
- In the ranker, use the article text representation to attend over its enriched entity slots. The resulting article-level KG representation is trained through click labels and general final-score distillation as a separate DLRM feature group, but it is not representation-distilled.

After changing only KG feature construction, rebuild the ranker KG matrix without retraining the text-only teacher or retrieval indexes:

```powershell
python -m mindrec.cli build_ranker_kg --config configs/mind_small.yaml
```

The ranker validates the KG artifact metadata against the active KG configuration. If entity limits, neighbor limits, reverse-edge behavior, weights, or normalization change, rebuild the KG artifact before training.

#### Building a triples file

A triples file we can use is **Wikidata5M**, a compact Wikidata-derived KG with the exact ID style this project needs: entities use `Q...`, relations use `P...`, and triples are stored as rows like `Q22686 P39 Q11696`. It is not guaranteed to be the exact MIND TransE training subgraph, but it is a reasonable Wikidata subgraph for one-hop neighbor expansion.

Download one of the Wikidata5M KG files from:
- https://deepgraphlearning.github.io/project/wikidata5m

Then filter it down to MIND-mentioned entities:

```
python scripts/build_mind_wikidata5m_triples.py
  --kg-path path/to/wikidata5m_transductive_train.txt
  --raw-root data/raw
  --train-dir MINDsmall_train
  --dev-dir MINDsmall_dev
  --entity-embedding data/raw/MINDsmall_train/entity_embedding.vec
  --entity-embedding data/raw/MINDsmall_dev/entity_embedding.vec
  --relation-embedding data/raw/MINDsmall_train/relation_embedding.vec
  --relation-embedding data/raw/MINDsmall_dev/relation_embedding.vec
  --output data/processed/MINDsmall/kg_triples.tsv
```

How this repo uses MIND entity annotations:
- `news.tsv` contains `title_entities` and `abstract_entities` columns.
- During greedy reranking, the reranker tracks covered entities for the entity coverage bonus.
- Reranker entity coverage is separate from the neural KG feature path: coverage decides list diversity, while KG features affect the learned ranker representation.

#### Rejected KG retrieval experiments

KG is kept in the downstream ranker because the retrieval-stage KG experiments did not generalize reliably:

1. **Full-strength KG in retrieval/item base** concatenated a strongly weighted KG vector with the text vector used by the teacher and base retrieval index. This allowed entity embeddings, one-hop neighbors, and relation messages to directly influence nearest-neighbor retrieval.
2. **Weakened KG in retrieval** kept the same text-plus-KG retrieval design but reduced the KG, neighbor, and relation weights so text remained the dominant retrieval signal.
3. **Reserved KG candidate slots** generated candidates from entities and their one-hop neighbors, then forced a fixed quota of those candidates into the final top-`K` retrieval set.
4. **KG-aware retrieval score bonus** left candidate generation unchanged, but added a small score bonus to candidates whose linked entities matched or neighbored entities found in the user's clicked history.

The accepted design is therefore **text-only retrieval plus a KG-enhanced ranker**.

---

### 2.2) What `run_preprocess()` does

`run_preprocess()` converts raw MIND TSV files into model-ready parquet/json files.

Main steps:
- Read `news.tsv` and `behaviors.tsv` from train/dev.
- Build ID mappings (`user_id/news_id/category/subcategory -> integer index`).
- Keep `train_dir` as training data. Split `dev_dir` into validation and test, by timestamp.
- The current config uses an `80% / 20%` time split inside `MINDsmall_dev`: the earlier window becomes `val`, the later window becomes `test`.
- Build pairwise rows for training and held-out evaluation (`train_pairs.parquet`, `val_pairs.parquet`, `test_pairs.parquet`).
- Build impression-level validation/test data (`val_impressions.parquet`, `test_impressions.parquet`).

How pairs are created:
- For an impression with `P` positives and `N` negatives, `P * (1 + min(data.ranker_negatives_per_positive, N))` pairs are generated.
- Each positive is paired with up to `data.ranker_negatives_per_positive` sampled negatives; the current config uses `4`.
- These negatives are labeled non-clicked candidates from the same impression.
  This in-impression sampling is the ranker pair-building behavior.
- Generated rows are used to train the **DLRM ranker (student)**.

What about the negative sampling for teacher?
- `teacher.negatives_per_positive` controls the contrastive sample of the teacher retriever.
- The current config uses `8`, so each clicked item can be paired with up to 8 non-clicked candidates from the same impression when training the teacher. If an impression only has 5 negatives, it uses 5.
- Samples are generated in-memory. Sampling does not change the ranker pair parquet size.

Why there is no `train_impressions.parquet`:
- Training uses pairwise rows (`train_pairs.parquet`), not full impression-grouped rows.
- Impression-grouped data is mainly needed for ranking evaluation, so only generated for val and test.

Validation/test split note:
- Training, early stopping, and calibration use `val`.
- `eval_retrieval` and `evaluate` evaluate on both `val` and `test` data and report every split listed in `eval.report_splits`.
- Reranker search only uses `val` so the chosen operating point can still be reported fairly on `test`.
- `rerank_eval` is the final reranker report and only uses `test`.

The evaluation JSONs also split each evaluated holdout window into chronological `time_period__...` slices, so regressions can be checked against the actual temporal order of impressions:

![Recall@200 over time](images/recall_over_time.png)

![nDCG@10 over time](images/nDCG_over_time.png)

Current MINDsmall split sizes in this repo:
- Training source (`MINDsmall_train`): `156,965` impressions
- Raw holdout source (`MINDsmall_dev`, split into `val` and `test` during preprocessing): `73,152` impressions
- Validation split: `58,521` impressions
- Test split: `14,631` impressions
- Training pairs: `1,135,225`
- Validation pairs: `436,424`
- Test pairs: `105,355`

---

## 3) End-to-end

### 3.1 Preprocess TSV → Parquet + feature maps
```bash
python -m mindrec.cli preprocess --config configs/mind_small.yaml
```

### 3.2 Train teacher retriever + build ANN index
```bash
python -m mindrec.cli train_teacher --config configs/mind_small.yaml
python -m mindrec.cli build_index --config configs/mind_small.yaml
python -m mindrec.cli eval_retrieval --config configs/mind_small.yaml
python -m mindrec.cli eval_retrieval_sweep --config configs/mind_small.yaml
```

With the current config, `eval_retrieval` writes:
- `runs/<run_name>/retrieval/eval_val.json`
- `runs/<run_name>/retrieval/eval_test.json`

The retrieval evaluation includes additional slice families:
- chronological `time_period__...` slices within each evaluated split
- `history_len_bucket__...` slices
- `impressions_with_clicked_popularity_bucket__...` slices
- `impressions_with_clicked_category__...` slices
- `impressions_with_clicked_subcategory__...` slices

`eval_retrieval_sweep` evaluates the hybrid grid defined in `retrieval.sweep`:
- `hybrid_base_weights: [0.05, 0.0625, 0.075]`
- `hybrid_oversamples: [18, 22, 26]`

It writes `runs/<run_name>/retrieval/sweep.json` with all tested settings plus the best one by held-out `recall@K`.

### 3.3 Train student DLRM ranker with distillation
```bash
python -m mindrec.cli preprocess --config configs/mind_small.yaml
python -m mindrec.cli train_ranker --config configs/mind_small.yaml
```

After each epoch, `train_ranker` evaluates the ranker on the full validation impressions and selects `best.pt` using mean per-impression validation AUC. This matches the official MIND evaluator's primary metric more closely than sampled pair-level AUC. The sampled pair AUC is still logged as a diagnostic because it is cheaper and useful for spotting training collapse.

After selecting the best ranker checkpoint, `train_ranker` fits a temperature scaler on the sampled `val_pairs.parquet`. It tunes a single positive scalar `T` in `sigmoid(logit / T)` without changing ranking order. Because full-impression evaluation has a different candidate/label distribution, always check the reported raw and calibrated Brier/ECE values; calibration fitted on sampled pairs is not guaranteed to improve full-impression calibration.

`ranker.score_batch_size` independently bounds memory use during ranker evaluation and reranker scoring. Structured entity-slot histories are substantially larger than the previous pooled KG vectors, so this should usually remain close to the training batch size.

To compare controlled fixed KG strengths:

```powershell
python -m mindrec.cli ranker_kg_gate_sweep --config configs/mind_small.yaml
```

The command trains and evaluates every value in `ranker.kg_gate_sweep.values`, keeps each run under `runs/<run_name>/tuning/kg_gate_fixed/gate_<value>/`, and writes `sweep.json`. The best gate is selected using validation `nDCG@10`; test metrics are reported but are not used for selection.

To tune the strength and softness of general final-logit distillation while
keeping the accepted KG architecture and gate fixed:

```powershell
python -m mindrec.cli ranker_distill_sweep --config configs/mind_small.yaml
```

The command evaluates the Cartesian product of
`ranker.distill_sweep.lambda_logits` and
`ranker.distill_sweep.temperatures`, while holding
`ranker.distill.lambda_repr` fixed. Each trial is stored under
`runs/<run_name>/tuning/distill_final_logit/`. The best setting is selected
strictly by validation `nDCG@10`; test metrics are included only as a
generalization diagnostic.

### 3.4 Evaluate ranker + reranker (metrics + slices)
```bash
python -m mindrec.cli evaluate --config configs/mind_small.yaml
python -m mindrec.cli rerank_eval --config configs/mind_small.yaml
```

With the current config, `evaluate` writes:
- `runs/<run_name>/eval/ranker_eval_val.json`
- `runs/<run_name>/eval/ranker_eval_test.json`

The ranker evaluation includes additional slice families:
- chronological `time_period__...` slices within each evaluated split
- `history_len_bucket__...` slices
- `impressions_with_clicked_new_item` and `impressions_with_clicked_warm_item`
- `impressions_with_clicked_popularity_bucket__...` slices
- `impressions_with_clicked_category__...` slices
- `impressions_with_clicked_subcategory__...` slices

For new/cold item ranking evaluation, each impression contains many candidate items, and ranking metrics are calculated for the whole impression. Therefore `impressions_with_clicked_new_item` means: evaluate whole impressions where at least one clicked positive item is new. It does not mean evaluating only the new candidate items inside all impressions.

### 3.5 Search reranker hyperparameters under a product constraint
```bash
python -m mindrec.cli rerank_search --config configs/mind_small.yaml
```

The reranking score and scalar utility are used at different levels:
- The reranking score answers: with this fixed hyperparameter setting, which candidate item should be placed next in this user's top-`k_out` list?
- `scalar_utility` answers: after evaluating a full candidate setting, which hyperparameter setting gives the best product tradeoff under the predetermined utility coefficients?

In practice, it is usually easier for business/product stakeholders to decide global scalar utility coefficients or guardrails than local reranker weights. That is why we search the local reranker hyperparameters instead of determine them.

Before searching for the best local reranker weights/penalties, we need to determine the following values that define the search problem:
- `coverage.category_bonus`, `coverage.entity_bonus`
- Guardrail thresholds: acceptable nDCG drop, required coverage gain, required fairness improvement, and new-item exposure constraint
- Utility coefficients: the relative importance of retained relevance, new-item exposure, category coverage, and fairness improvement when ranking candidate settings
- A practical workflow to determine these values is:
1. Set the maximum acceptable nDCG drop.
2. Agree on guardrails (minimum improvements) for the other metrics by reviewing mocked or held-out impressions and feeling what the metric changes look like.
3. Review sample impressions as baseline list vs. reranked list to choose utility coefficients/weights:
  - Did the list feel repetitive?
  - Did new items appear in reasonable positions?
  - Did relevance visibly degrade?
  - Did category coverage feel diverse but relevant, or completely random?

The scalar utility is computed from baseline-relative **normalized** units:
- `ndcg_retention_units = 1 - relative_ndcg_drop`
- `new_item_exposure_gain_units = new_item_exposure_gain`
- `category_coverage_gain_units = category_coverage_gain / min_category_coverage_gain`
- `fairness_kl_pool_improvement_units = fairness_kl_pool_improvement / min_fairness_kl_pool_improvement`

with coefficients:
- `4.0 * ndcg_retention_units`
- `0.5 * new_item_exposure_gain_units`
- `1.5 * category_coverage_gain_units`
- `1.5 * fairness_kl_pool_improvement_units`

The search uses baseline-relative guardrails. We need guardrails because guardrails define what "acceptable" means, and then we can choose the setting with the highest utility among acceptable tradeoffs. Also, if the coefficients underweight relevance or overweight coverage/fairness, then a setting can look "best" by utility while still being a bad product choice; guardrails can prevent that. Current guardrails:
- `nDCG@10` drop must be at most `2.1%` relative to the baseline ranker (initially 2%; changed to 2.1% to allow a good hyperparameter set).
- `new_item_exposure_frac` must not decrease relative to baseline.
- `category_coverage@10` must improve by at least `0.25`.
- `fairness_kl_pool` must improve by at least `0.04`.

Typical workflow:
- run `rerank_search`
- inspect `best_feasible` or another chosen operating point in `rerank_search.json`
- update `configs/mind_small.yaml` with the selected rerank parameters
- run `rerank_eval` again to evaluate that chosen setting as the new default reranker

The current reranker search reports three views of the tradeoff surface on validation:
- `best_feasible`: maximize scalar utility among settings that satisfy the guardrails
- `best_scalar_utility`: maximize a normalized scalar utility
- `pareto_frontier`: nondominated settings across ranking/diversity/fairness axes

`rerank_search` evaluates candidates in two passes:
- First pass: fast screening. All candidate settings in the search grid are evaluated on a sample of up to 500 validation impressions.
- Second pass: full validation. Only a shortlist of promising candidates is evaluated on the full validation split.
- `pareto_frontier.md` is written from the full-pass shortlisted results, not from every sample-screened candidate.

The current selected setting is:
- `relevance_weight=0.875`
- `novelty_weight=0.05`
- `coverage_weight=0.075`
- `novelty_sim=teacher_cosine`
- `fairness.penalty_weight=0.30`
- `fairness.new_item_floor=0.20`

The search writes its summary to `runs/<run_name>/eval/rerank_search.json`.

Artifacts go to `runs/<run_name>/`.
Training logs are written to `runs/<run_name>/teacher/epochs.json` and `runs/<run_name>/ranker/epochs.json`.

### 3.6 Last completed demo results (`runs/mind_small_demo`)

The numbers below are from the current text-only retrieval plus KRED-style KG ranker pipeline.

Teacher retrieval:
- Current retrieval setup is text-only hybrid retrieval:
  - teacher retrieval from `item_teacher_emb.npy`
  - text-only item-base fallback from `item_base_emb.npy`
  - KG is not used by the teacher or retrieval indexes
  - `teacher.text.include_category_prefix = false`
  - `hybrid_base_weight = 0.075`
  - `hybrid_oversample = 26`
- Teacher retrieval validation `Recall@200 = 0.044892`
- Teacher retrieval test `Recall@200 = 0.042396`
- Early stopping monitor: `retrieval_recall@200`
- Best teacher epoch: `2`
- Rejected experiment: adding category/subcategory prefixes to the teacher text input improved Teacher retrieval validation `Recall@200`, but hurt downstream Student ranker quality, so the default remains `teacher.text.include_category_prefix = false`.

Student ranker:
- Current semantic input uses two separate matrices:
  - `item_base_emb.npy`: `384`-dimensional sentence-transformer text input for the distilled text branch
  - `item_kg_base_emb.npy`: `[max_entities_per_news, 100]` relation-aware entity slots for each article; text-conditioned attention produces the article KG vector used by the complete ranker scorer
  - `knowledge_graph.enabled = true`
  - the teacher and retrieval indexes remain text-only
- Current semantic settings:
  - `semantic_ff_mult=3`
  - `semantic_dropout=0.20`
  - `dropout=0.15`
  - `weight_decay=3.0e-5`
  - `kg_gate_init=0.15`
  - `distill.temperature=0.75`
  - `distill.lambda_logit=0.15`
  - `distill.lambda_repr=0.05`
  - `news_id_warm_scale=1.0`
  - `news_id_cold_scale=0.0`
- Current ranker training/selection settings:
  - `lr=1.0e-3`
  - `max_grad_norm=5.0`
  - `ranker.early_stopping.monitor=impression_auc`
  - `ranker.lr_scheduler.enabled=false`
- The `news_id_*_scale` settings control only the learned `news_id` embedding branch in the DLRM ranker.
  Warm items keep their ID embedding contribution, while **new/cold items get that branch zeroed out** and are scored relying on semantic, category/subcategory, user, and dense features instead.
  new/cold items are found during preprocessing from training (click counts under the `min_item_train_clicks_for_warm` limit).
- Best ranker epoch: `1`
- Best validation impression AUC: `0.661867`
- Student ranker validation:
  - `AUC = 0.661867`
  - `MRR = 0.370356`
  - `nDCG@5 = 0.353482`
  - `nDCG@10 = 0.415184`
  - `Recall@10 = 0.674316`
- Student ranker test:
  - `AUC = 0.654816`
  - `MRR = 0.348304`
  - `nDCG@5 = 0.332940`
  - `nDCG@10 = 0.392341`
  - `Recall@10 = 0.640532`
- calibration on test changed:
  - `Brier: 0.075886 -> 0.062743`
  - `ECE@15: 0.184654 -> 0.142282`

Search summary:
- `best_feasible` is the setting that has the highest-utility and is also feasible
- Current validation baseline ranker before reranking:
  - `nDCG@10 = 0.415184`
  - `Recall@10 = 0.674316`
  - `category_coverage@10 = 4.989030`
  - `fairness_kl_pool = 0.442832`
  - `new_item_exposure_frac = 0.567963`
- Current validation `best_feasible`:
  - `nDCG@10 = 0.411044`
  - `Recall@10 = 0.671606`
  - `category_coverage@10 = 5.388322`
  - `fairness_kl_pool = 0.378925`
  - `new_item_exposure_frac = 0.580714`
  - relative `nDCG@10` drop: `0.997%`
- `n_feasible = 23` under the current guardrails

Final reranker test report:
- Baseline ranker:
  - `nDCG@10 = 0.392341`
  - `Recall@10 = 0.640532`
  - `category_coverage@10 = 4.857563`
  - `fairness_kl_pool = 0.441752`
  - `new_item_exposure_frac = 0.774823`
- Reranked output:
  - `nDCG@10 = 0.390285`
  - `Recall@10 = 0.637306`
  - `category_coverage@10 = 5.275784`
  - `fairness_kl_pool = 0.369262`
  - `new_item_exposure_frac = 0.783522`
- The reranker trades approximately `0.52%` relative `nDCG@10` for better category coverage, fairness KL, and new-item exposure.

---

## 4) Repo layout

- `src/mindrec/`
  - `data/`: parsing + feature building
  - `models/teacher.py`: teacher-side embedding utilities
  - `models/dlrm.py`: student DLRM ranker + attention fusion
  - `rerank/`: diversity + coverage + exposure fairness reranking
  - `metrics/`: ranking, calibration, diversity, fairness
  - `cli.py`: entrypoint for scripts
- `configs/`: YAML configs

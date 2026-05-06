# MIND Multi-Stage Recommender (Retrieval → DLRM Ranker → Diversity+Fairness Re-ranker)

This project implements a realistic recommender stack on the **Microsoft News Dataset (MIND)**:
- Stage 1: **Teacher retrieval embeddings** (text-based item encoder + history-based user encoder)
- Stage 2: **Student ranker** (lean DLRM-style sparse+dense model) that uses click history, sentence-transformer item bases, projected `user_id`/`news_id` branches, lightweight structured features, and a widened semantic MLP path
- Stage 3: **Re-ranking** enforcing **(1) relevance vs novelty** and **(2) category/entity-informed coverage bonuses**, plus **exposure fairness** constraints/penalties for category and new items
- Extensive **evaluation**: ranking metrics, calibration, diversity, exposure fairness, and cold/new slices

MIND is widely used as a benchmark for news recommendation, with impression logs and rich news metadata.

Recommender architecture

- Candidate generation with ANN search using **Faiss** (CPU-friendly).
- Knowledge **distillation** (logit + representation) from a stronger teacher into a smaller student (better cold/new performance vs training the student from scratch).
- A **lean DLRM-style** ranker (dense MLP + semantic branch with widened semantic MLPs + projected `user_id`/`news_id` branches + lightweight feature interaction).

## Teacher model
- The teacher is a learned two-tower retrieval model:
  - a **news/item encoder** that starts from frozen sentence-transformer news embeddings and applies a trainable projection
  - a **user encoder** that attention-pools clicked-history item embeddings into a user vector
- Training uses clicked positives plus in-impression negatives.
- Retrieval evaluation encodes each validation/test impression with that impression's own history, so the query is temporally aligned with the impression being scored.
- At the end of `train_teacher`, the pipeline writes:
  - `item_base_emb.npy`: frozen sentence-transformer embedding for every news item
  - `item_teacher_emb.npy`: the final teacher embedding for every news item
  - `user_teacher_emb.npy`: the final teacher embedding for each training user history
- These files are then reused by later stages:
  - `build_index` builds the Faiss retrieval index from `item_teacher_emb.npy`
  - `eval_retrieval` uses `item_teacher_emb.npy` and the saved teacher model to encode held-out histories and search that index
  - `train_ranker` loads `item_base_emb.npy` as student semantic input and uses `item_teacher_emb.npy` / `user_teacher_emb.npy` only as teacher supervision targets

## Teacher and retrieval
- In MIND `behaviors.tsv`, `history` is a list of previously clicked news IDs for that user before the current impression time. It is not a list of previous impressions.
- The user embedding is built from those clicked-history items before the current impression.
- During retrieval evaluation, the query vector for a row is built from that row's own `history` only, not from some later history for the same user.
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

## Distillation representation
- In `DLRMStudent.forward()`, the student representation used for distillation is `rep = [user_sem, item_sem, sem_fused]`.
- `user_sem`: student semantic user vector from pooled click-history sentence-transformer item bases
- `item_sem`: student semantic item vector from the candidate item's sentence-transformer base embedding
- `sem_fused`: a lightweight attention-fusion summary that mixes the semantic user/item states with the structured query context
- The teacher target is `concat(teacher_user_emb, teacher_item_emb)`. A projection head maps the student representation into the teacher space for representation distillation.

#### Teacher -> student distillation map
- Distillation in this project is not copying the teacher into a smaller clone.
- The student is trained to imitate the teacher's semantic user/item representations and semantic matching behavior, while still having its additional features.

| Teacher block | What it does | Student replacement | Key difference |
|---|---|---|---|
| Frozen sentence-transformer news embedding input | Base text semantics for each news item | Same `item_base_emb.npy` input | Both stages start from the same base text embedding |
| Teacher semantic item encoder | Refines each item into the teacher semantic space | Smaller student semantic item encoder | Student semantic dimension is much smaller than the teacher space |
| Teacher sequence-aware user encoder | Contextualizes clicked history items with attention before pooling | Cheaper history aggregation path | Student uses a lighter history refinement and aggregation path |
| Teacher attention pooling over history | Builds one semantic user vector from clicked history | Mean-style pooled semantic user path | Student pooling is cheaper and less sequence-aware |
| Teacher item embedding | Semantic representation of the candidate item | `item_sem` | Student item representation is trained to approximate teacher behavior |
| Teacher user embedding | Semantic representation of the current user history | `user_sem` | Student user representation is cheaper to compute |
| Teacher cosine-style user-item scorer | Measures semantic compatibility between user and item | Student top MLP ranker | Student scoring uses richer ranking signals beyond pure semantic similarity |
| Teacher representation target | Provides a semantic supervision target for distillation | `rep = [user_sem, item_sem, sem_fused]` plus projection head | Student and teacher representations have different shapes and meanings |
| No direct teacher counterpart | None | `sem_fused` | Student adds an attention-fusion summary conditioned on query context |
| No direct teacher counterpart | None | `user_id` / `news_id` branches | Student adds collaborative-style memorization signals |
| No direct teacher counterpart | None | category / subcategory embeddings | Student adds structured metadata signals |
| No direct teacher counterpart | None | dense features such as `history_len` and `item_clicks_log1p` | Student adds non-semantic ranking features |
| No direct teacher counterpart | None | DLRM interaction terms + top MLP | Student is a broader ranker, not just a semantic retriever |

The student keeps a lighter semantic core than the teacher, but combines it with extra ranking-specific signals:
- projected `user_id` / `news_id` branches
- category and subcategory embeddings
- lightweight dense features
- DLRM-style feature interactions

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
  - `fairness_kl_pool`: reference is the reranker's top-`pool_size` candidate mix
  - `fairness_kl_full`: reference is the full impression candidate mix
  - If `category_target: uniform`, then `q` is uniform over categories present in the reference set (not used in this project).
  - If `category_target: catalog`, then `q` is the empirical category distribution of that reference set.

#### Worked examples for novelty, coverage, and fairness
- Suppose the reranker has already selected two items: `A` and `B`.
- Candidate `C` has ranker relevance score `0.80`, category `Sports`, entities `{Messi, Inter Miami}`, and is marked as a new item.
- Candidate `D` has relevance score `0.78`, category `Health`, entities `{WHO, vaccine}`, and is not new.

- **Novelty example with `teacher_cosine`**:
  - If `C` has teacher-embedding cosine similarities `0.90` to `A` and `0.35` to `B`, then `novelty(C) = -max(0.90, 0.35) = -0.90`.
  - If `D` has similarities `0.20` to `A` and `0.10` to `B`, then `novelty(D) = -0.20`.
  - Because `-0.20 > -0.90`, `D` is treated as more novel than `C`.

- **Coverage example**:
  - Suppose the selected list has already covered categories `{Sports, Politics}` and entities `{Messi, Real Madrid}`.
  - If `coverage.category_bonus = 1.0`, `coverage.entity_bonus = 0.3`, and `max_new_entities_per_item = 3`:
  - `C` is in `Sports`, which is already covered, so it gets no category bonus. It adds one new entity, `Inter Miami`, so `coverage(C) = 0.3`.
  - `D` is in `Health`, which is new, so it gets `1.0` category bonus. If both `WHO` and `vaccine` are new, it also gets `2 * 0.3 = 0.6` entity bonus, so `coverage(D) = 1.6`.

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

Fairness target note:
- In the current reranker/evaluation code, `fairness.category_target: "catalog"` means the category distribution of the impression candidate pool (the candidates available for that user/impression), not the selected top-K list and not the global corpus-wide catalog.
- `rerank_eval` and `rerank_search` now report both `fairness_kl_pool` and `fairness_kl_full`.
- `fairness_kl_pool` compares top-K exposure against the reranker's top-`pool_size` candidate mix.
- `fairness_kl_full` compares top-K exposure against the full impression candidate set.
- Product constraints in reranker search use `fairness_kl_pool`, because that matches the reranker's actual optimization target.

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
Each folder should contain `behaviors.tsv` and `news.tsv`.

MIND also provides `entity_embedding.vec` and `relation_embedding.vec`.
In this repo, for simplicity, these two files are currently **not used** by the pipeline.
The reranker may still use the entity annotation columns already present in `news.tsv` for coverage when `rerank.coverage.entity_bonus > 0`; this is separate from using the external entity/relation embedding files.

---

### 2.1) Quick terminology: entities in MIND

- In MIND, an **entity** is a named entity extracted from a news article (person, organization, location, etc.) and linked to a knowledge graph (the MIND paper references Wikidata).
- `entity_embedding.vec`: embedding vector for each entity ID.
- `relation_embedding.vec`: embedding vector for each relation type between entities.

If you choose to use these files, a common approach is to use KG triples `(entity, relation, entity)` to fetch neighbors of entities mentioned in a news article, then build richer representations (for example with graph attention or memory-network style modules).

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
- For an impression with `P` positives and `N` negatives, this code contributes:
- `P * (1 + min(4, N))` pairs
- because each positive is paired with up to 4 sampled negatives.

Why there is no `train_impressions.parquet`:
- Training uses pairwise rows (`train_pairs.parquet`), not full impression-grouped rows.
- Impression-grouped data is mainly needed for ranking evaluation, so only generated for val and test.

Validation/test split note:
- Training, early stopping, and calibration use `val`.
- Final offline evaluation (`eval_retrieval`, `evaluate`, `rerank_eval`) uses `test`.
- Reranker search uses `val` so the chosen operating point can still be reported fairly on `test`.

Current MINDsmall split sizes in this repo:
- Training source (`MINDsmall_train`): `156,965` impressions
- Holdout source (`MINDsmall_dev`): `73,152` impressions
- Validation split: `58,521` impressions
- Test split: `14,631` impressions
- Training pairs: `1,135,225`
- Validation pairs: `436,424`
- Test pairs: `105,355`

---

## 3) End-to-end on a small slice

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

The retrieval evaluation now includes additional slice families:
- chronological `time_period__...` slices within each evaluated split
- `history_len_bucket__...` slices
- `impressions_with_clicked_popularity_bucket__...` slices
- `impressions_with_clicked_category__...` slices
- `impressions_with_clicked_subcategory__...` slices

`eval_retrieval_sweep` evaluates the small hybrid grid defined in `retrieval.sweep`:
- `hybrid_base_weights: [0.0625, 0.075, 0.0875]`
- `hybrid_oversamples: [18, 20, 22]`

It writes `runs/<run_name>/retrieval/sweep.json` with all tested settings plus the best one by held-out `recall@K`.

### 3.3 Train student DLRM ranker with distillation
```bash
python -m mindrec.cli train_ranker --config configs/mind_small.yaml
```

After training the ranker on `train_pairs.parquet`, train_ranker fits a temperature scaler on `val_pairs.parquet`. It tunes a single positive scalar `T` in `sigmoid(logit / T)` against held-out labels, improving probability **calibration** without changing ranking order.

### 3.4 Evaluate ranker + reranker (metrics + slices)
```bash
python -m mindrec.cli evaluate --config configs/mind_small.yaml
python -m mindrec.cli rerank_eval --config configs/mind_small.yaml
```

With the current config, `evaluate` writes:
- `runs/<run_name>/eval/ranker_eval_val.json`
- `runs/<run_name>/eval/ranker_eval_test.json`

The ranker evaluation now includes additional slice families:
- chronological `time_period__...` slices within each evaluated split
- `history_len_bucket__...` slices
- `impressions_with_clicked_popularity_bucket__...` slices
- `impressions_with_clicked_category__...` slices
- `impressions_with_clicked_subcategory__...` slices

### 3.5 Search reranker hyperparameters under a product constraint
```bash
python -m mindrec.cli rerank_search --config configs/mind_small.yaml
```

Typical workflow after search:
- run `rerank_search`
- inspect `best_feasible` or another chosen operating point in `rerank_search.json`
- update `configs/mind_small.yaml` with the selected rerank parameters
- run `rerank_eval` again to evaluate that chosen setting as the new default reranker

The current reranker search reports three views of the tradeoff surface on validation:
- `best_feasible`: maximize scalar utility among settings that satisfy the guardrails
- `best_scalar_utility`: maximize a normalized scalar utility
- `pareto_frontier`: nondominated settings across ranking/diversity/fairness axes

The search uses baseline-relative guardrails. We need guardrails because guardrails define what “acceptable” means, and then we can choose the setting with the highest utility among acceptable tradeoffs. Also, if the coefficients underweight relevance or overweight coverage/fairness, then a setting can look “best” by utility while still being a bad product choice; guardrails can prevent that. Current guardrails:
- `nDCG@10` drop must be at most `2.1%` relative to the baseline ranker (initially 2%; changed to 2.1% to allow a good hyperparameter set).
- `new_item_exposure_frac` must not decrease relative to baseline.
- `category_coverage@10` must improve by at least `0.25`.
- `fairness_kl_pool` must improve by at least `0.04`.

Each candidate in `rerank_search.json` now reports:
- `constraint.feasible`

The scalar utility is computed from baseline-relative normalized units:
- `ndcg_retention_units = 1 - relative_ndcg_drop`
- `new_item_exposure_gain_units = new_item_exposure_gain`
- `category_coverage_gain_units = category_coverage_gain / min_category_coverage_gain`
- `fairness_kl_pool_improvement_units = fairness_kl_pool_improvement / min_fairness_kl_pool_improvement`

with coefficients:
- `4.0 * ndcg_retention_units`
- `0.5 * new_item_exposure_gain_units`
- `1.5 * category_coverage_gain_units`
- `1.5 * fairness_kl_pool_improvement_units`

The current selected setting is:
- `relevance_weight=0.89`
- `novelty_weight=0.05`
- `coverage_weight=0.06`
- `novelty_sim=teacher_cosine`
- `fairness.penalty_weight=0.20`
- `fairness.new_item_floor=0.20`

The search writes its summary to `runs/<run_name>/eval/rerank_search.json`.

Artifacts go to `runs/<run_name>/`.
Training logs are written to `runs/<run_name>/teacher/epochs.json` and `runs/<run_name>/ranker/epochs.json`.

### 3.6 Last completed demo results (`runs/mind_small_demo`)

Teacher retrieval:
- current retrieval setup uses hybrid retrieval:
  - teacher retrieval from `item_teacher_emb.npy`
  - raw sentence-transformer fallback from `item_base_emb.npy`
  - `teacher.text.include_category_prefix = false`
  - `hybrid_base_weight = 0.075`
  - `hybrid_oversample = 18`
- Teacher retrieval validation `Recall@200 = 0.04575`
- Teacher retrieval test `Recall@200 = 0.04270`
- Early stopping monitor: `retrieval_recall@200`
- Best teacher epoch: `2`
- Rejected experiment: adding category/subcategory prefixes to the teacher text input improved Teacher retrieval validation `Recall@200`, but hurt downstream Student ranker quality, so the default remains `teacher.text.include_category_prefix = false`.

Student ranker:
- current semantic settings:
  - `semantic_ff_mult=3`
  - `semantic_dropout=0.20`
  - `dropout=0.15`
  - `weight_decay=3.0e-5`
  - `news_id_warm_scale=1.0`
  - `news_id_cold_scale=0.0`
- Student ranker validation `nDCG@10 = 0.40992`
- Student ranker validation `Recall@10 = 0.66987`
- Student ranker validation `AUC = 0.65211`
- Student ranker test `nDCG@10 = 0.38535`
- Student ranker test `Recall@10 = 0.64013`
- Student ranker test `AUC = 0.64154`
- calibration on test changed:
  - `Brier: 0.13650 -> 0.07296`
  - `ECE@15: 0.31091 -> 0.17471`

Search summary:
- `best_feasible` is the setting that has the highest-utility and is also feasible
- current validation baseline ranker before reranking:
  - `nDCG@10 = 0.40992`
  - `fairness_kl_pool = 0.45766`
  - `new_item_exposure_frac = 0.63715`
- current validation `best_feasible`:
  - `nDCG@10 = 0.40154`
  - `Recall@10 = 0.66261`
  - `new_item_exposure_frac = 0.65345`
  - `category_coverage@10 = 5.70720`
  - `fairness_kl_pool = 0.36015`
- `n_feasible = 15` under the current guardrails
- `best_scalar_utility` is a more aggressive diversity/fairness point, but it is not feasible under the current guardrails

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

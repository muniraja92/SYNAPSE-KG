# SYNAPSE-KG v5 — revised experimental protocol (fixed before any v5 run, 5 Nov 2026)

Purpose: address the publishability review of 4 Nov 2026 (synapse-kg-rev/PUBLISHABILITY_REVIEW_2026-11-04.md).
Everything below is fixed before the v5 runs; results are reported whatever their direction.

## Data
- MovieLens-1M (main): ratings >= 4; user (>= 10) and item (>= 5) filters alternated until convergence (k-core); 35% user sample (seed 42).
  (Amendment, 5 Nov, before any neural run: v4's three fixed passes could leave users with fewer than 10 interactions,
  which v4 removed by re-filtering after sampling; filtering to convergence makes that step unnecessary.)
- Amazon Reviews 2023, Video Games (second domain): 5-core benchmark ratings (McAuley Lab), ratings >= 4, same filters,
  20% user sample (seed 42), metadata (title, category path, store) streamed from the official metadata file.
- Stable tie-break: interactions ordered by (timestamp, item id). No dependence on sort kernels.
- Per user: last item = test, second-to-last = validation, third-to-last = fusion item; earlier items = training prefix.
  All models (baselines included) train on the training prefix only.

## Knowledge graph (built from the training prefix and item metadata; all relations are consumed)
- INTERACTED (user <-> item, training prefix), HAS_GENRE / HAS_CATEGORY / HAS_STORE (item <-> attribute),
  TEMPORAL_NEXT (item -> next item in a user's training prefix; directed, weighted by count).
- BRIDGE propagates over this directed graph: forward stream D_out^-1 A, backward stream D_in^-1 A^T.
- Ablation "w/o KG relations": BRIDGE trained on the interaction graph only (BRIDGE-UI), substituted in ORACLE.

## Models (changes)
- DRIP: the positive item is masked out of the history during training.
- MINT: user tower and item tower; momentum copy of the item tower initialised as an exact copy; InfoNCE with in-batch
  negatives plus a queue of 4,096 momentum keys; temperature 0.1; momentum 0.995; the score is the cosine between the user
  tower and the online item tower (the quantity the loss trains).
- NEXIS: P = 4 interests extracted from the training history by self-attentive pooling (ComiRec-SA style), target-aware
  aggregation as before; positive masked out of the history during training.
- New baseline: EASE (lambda selected on validation from {50, 100, 200, 500, 1000, 2000, 5000, 10000}).
- ItemKNN / UserKNN: k from {20, 50, 100, 200, 500} on validation; ORACLE's ItemKNN input uses the selected k.
- (Amendment, 5 Nov, before any neural run: the EASE and kNN grids were extended after the first data preparation selected
  grid-edge values; only deterministic baselines had been run at that point.)

## Training (equal for every neural model)
- Adam; learning rate selected on validation from {5e-4, 1e-3, 2e-3} (seed 0, per dataset), then fixed for all seeds.
- Early stopping: evaluation on validation every 5 epochs, patience 4 evaluations, maximum 200 epochs; best checkpoint restored.
  (Amendment, 5 Nov, during learning-rate selection and before any test evaluation: stopping is allowed only from epoch 50 on.
  A pilot showed that LightGCN with large batches stays on a popularity plateau for 10-40 epochs, which the patience rule
  mistook for convergence. Selection runs that had stopped before epoch 50 were discarded and rerun; runs that stopped at or
  after epoch 50 are unaffected by the change and were kept.)
- Embedding dimension 64; batch size 4,096 (ML-1M) / 512 (Video Games) for MF-BPR, LightGCN and BRIDGE (one graph propagation per
  batch), 1,024 / 256 for DRIP, MINT and NEXIS (about 50-110 updates per epoch on both datasets); other settings as in v4. ML-1M replication samples of v4 are not repeated under v5.

## Fusion
- ORACLE inputs: ItemKNN, Content, Popularity, LightGCN, DRIP, MINT, BRIDGE, NEXIS (scores standardised on fusion pairs).
- ORACLE is fitted on fusion pairs (fusion item + 99 fixed sampled negatives) of 80% of users, early-stopped (patience 10,
  max 100 epochs) on the fusion pairs of the other 20%; validation and test items are never used to fit it.
- Fusion baselines on the same inputs and fusion pairs: mean of standardised inputs; logistic-regression stacking;
  best single input (selected on fusion pairs); ORACLE with MF-BPR as a ninth input; MF-BPR five-seed score ensemble.
- Ablations (ORACLE refitted): without each of the eight inputs; without KG relations (BRIDGE-UI in place of BRIDGE).

## Evaluation
- Primary endpoint: full-ranking NDCG@10 of the test item against all items not in the user's training prefix,
  fusion or validation item (pessimistic ties), averaged over seeds per user.
- Secondary: full-ranking HR@10 and MRR; sampled metrics with 99 fixed negatives (comparability with v4).
- Tests: paired t-test and Wilcoxon on seed-averaged per-user NDCG@10, Holm over comparisons; d_z.
- Uncertainty: two-level bootstrap (users and seeds) 95% interval of the SYNAPSE-KG minus MF-BPR difference.
- Equivalence (pre-specified): TOST with margin +-5% of MF-BPR's full-ranking NDCG@10, alpha 0.05.
- Stratified analyses as in v4 (stratify.py) on the primary endpoint.
- Cost: training time and parameter count per model.

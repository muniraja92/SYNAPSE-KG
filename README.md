# SYNAPSE-KG

Code accompanying the article *SYNAPSE-KG: Meta-Learned Fusion of Target-Aware Attention, Momentum-Contrastive, Graph-Diffusion and Multi-Interest Recommenders* (P. Muniraja and S. M. Satapathy, School of Computer Science and Engineering, Vellore Institute of Technology).

SYNAPSE-KG combines four neural scoring modules (DRIP, MINT, BRIDGE, NEXIS) with collaborative, genre-based content and popularity signals through a gated meta-learner (ORACLE).

## Contents

| Path | Purpose |
|---|---|
| `code/synapse_v3.py` | Full pipeline: data preparation, all baselines, the four modules, ORACLE, ablations. One seed per call. |
| `code/aggregate.py` | Aggregates the seeds: mean ± std tables, paired t-tests and Wilcoxon tests with Holm correction, effect sizes, group analyses, all figures. |
| `requirements.txt` | Python dependencies. |
| `SYNAPSE_KG.ipynb`, `SYNAPSE-KG-Results-Final-*` | **Superseded.** The original notebook and its results. See "Changes from the original notebook" below. Kept for transparency only. |

## Reproducing the results

1. Download MovieLens-1M (`ml-1m.zip`) from GroupLens and unzip it so that `ml-1m/ratings.dat` and `ml-1m/movies.dat` exist.
2. Run the five training seeds and aggregate:

```bash
pip install -r requirements.txt
cd code
python synapse_v3.py prep                       # data preparation and deterministic baselines (cached)
for s in 0 1 2 3 4; do python synapse_v3.py $s; done   # ~40 min per seed on one CPU core; seeds can run in parallel
python aggregate.py                             # writes out/summary.json and figures/
```

Environment variables `ML_ROOT` (default `./ml-1m`) and `OUT` (default `./out`) set the data and output folders.
The data check printed by `prep` must read `users=2082 items=3105 train(original LOO)=202546`.

## Evaluation protocol

* User sample: 35% of MovieLens-1M users with at least 10 positive (rating ≥ 4) interactions, fixed seed 42; items with at least 5 interactions.
* Chronological leave-one-out per user: last item = test, second-to-last = validation, the rest = training prefix. Every model, feature, history and knowledge-graph relation is built from the training prefix only.
* Model selection on validation NDCG@10 only (epochs of every neural model, k of ItemKNN/UserKNN). ORACLE is fitted on validation pairs with early stopping on held-out validation users. The test split is used once.
* Ranking: test item against 99 negatives sampled once (fixed seed) from items the user never interacted with; identical candidates for every model and seed; ties counted against the positive.
* Five training seeds (0–4), mean ± standard deviation. Paired t-test and Wilcoxon signed-rank test on seed-averaged per-user NDCG@10, Holm-corrected; paired effect size d_z.

## Changes from the original notebook

The original notebook (`SYNAPSE_KG.ipynb`) contained evaluation defects that inflated the originally reported numbers. They are corrected in `code/synapse_v3.py`:

1. Content user profiles and DRIP histories were built from each user's full sequence, including the test item (test leakage).
2. ORACLE was fitted on in-sample training pairs; there was no validation split.
3. DRIP and NEXIS were evaluated with raw embedding dot products instead of their target-aware forward passes.
4. LightGCN reused one forward pass per epoch across mini-batches.
5. Each model drew its own random negatives, so comparisons were not paired.
6. Ablations zeroed an input of an already-trained ORACLE instead of retraining it.
7. Knowledge-graph relations were built from full sequences, including test items.

## License

MIT (see `LICENSE`).

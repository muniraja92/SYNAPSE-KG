# SYNAPSE-KG: Evaluating Learned Fusion of Knowledge-Graph-Aware and Neural Recommenders under a Leakage-Free Protocol

Code, protocol, logs, and result files for the article of the same title by Pasupuleti Muniraja and
Shashank Mouli Satapathy (School of Computer Science and Engineering, Vellore Institute of Technology, India),
submitted to *PeerJ Computer Science*.

## Description

SYNAPSE-KG (Semantic-Yielding Neural Attention with Path Synthesis and Encoding for Knowledge Graphs) fuses four neural recommenders with four auxiliary signals through a gated meta-learner, ORACLE
(Output Routing And Component-Level Ensembling):

| Input | Type |
|---|---|
| DRIP (Dynamic Relational Intent Propagation) | target-aware attention over the interaction history |
| MINT (Momentum-based Interest Network Tracking) | momentum-contrastive two-tower model |
| BRIDGE (Bidirectional Relational Information Diffusion with Graph Encoding) | bidirectional propagation over a knowledge graph (interactions, item attributes, temporal transitions) |
| NEXIS (Neural EXpert Interest Synthesis) | multi-interest extraction by self-attentive pooling |
| ItemKNN, Content, Popularity, LightGCN | auxiliary scores |

The study asks whether learned fusion improves on the strongest single model, and whether the learned gate beats
simple fusion rules on the same inputs, under a leakage-free protocol:

- chronological splits;
- the fusion layer fitted on a separate held-out interaction per user;
- equal tuning for all neural models;
- full-catalogue ranking as the primary endpoint (NDCG@10, five training seeds).

It compares SYNAPSE-KG with:

- seven single-model baselines: Popularity, Content, ItemKNN, UserKNN, EASE, MF-BPR, and LightGCN;
- five fusion baselines: mean of inputs, logistic stacking, best single input, ORACLE with MF-BPR as an extra input,
  and a five-seed MF-BPR ensemble.

It also reports ablations, Holm-corrected paired tests, two-level bootstrap intervals, TOST equivalence tests, and
stratified analyses.

**Main result:**

- On both datasets SYNAPSE-KG had the highest mean NDCG@10, but its advantage over the strongest single baseline
  (EASE on MovieLens-1M, UserKNN on Video Games) and over logistic stacking was not statistically significant.
- Removing the knowledge-graph relations made no detectable difference.

## Dataset information

Both datasets are public third-party data and are **not** redistributed here.

| Dataset | Source | Files used | Place in |
|---|---|---|---|
| MovieLens-1M (Harper & Konstan, 2015) | GroupLens Research, https://grouplens.org/datasets/movielens/1m/ | `ratings.dat`, `movies.dat`, `users.dat` | `./ml-1m/` |
| Amazon Reviews 2023, Video Games (Hou et al., 2026) | McAuley Lab, https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 | `benchmark/5core/rating_only/Video_Games.csv`, `raw/meta_categories/meta_Video_Games.jsonl` | `./amazon-vg/` |

Data preprocessing is deterministic and is performed by `synapse_v5.py prep` (Methodology below).
`filter_meta.py` keeps the metadata of the Video Games items that survive filtering, written to `meta_needed.jsonl`.

## Code information

| File | Purpose |
|---|---|
| `synapse_v5.py` | Data preparation, knowledge graph, all models and baselines, learning-rate selection, ORACLE, fusion baselines, ablations. Subcommands: `prep`, `tune`, and seed numbers. |
| `aggregate_v5.py` | Means and standard deviations, paired tests (Holm), two-level bootstrap intervals, TOST equivalence, training statistics. Writes `results.json` and `per_user_ndcg10.npz`. |
| `stratify_v5.py` | Stratified analyses. Writes `stratification.json` and the forest plot `forest.pdf`. |
| `make_tables.py`, `make_figures.py` | All tables and figures of the article, generated from the result files. |
| `make_supplement.py` | The tables of Supplemental Article S1, generated from the result files. |
| `filter_meta.py` | Filters the Video Games item metadata. |
| `PROTOCOL_v5.md` | Evaluation protocol written before the experiments, with dated amendments. |
| `requirements.txt` | Exact package versions. |
| `logs/` | Logs of the learning-rate selection and seed runs. |
| `out_ml1m/`, `out_vg/` | Result files for MovieLens-1M and Video Games (see below). |

Each results folder contains:

- `tune_*.json` and `tuned.json`: learning-rate selection.
- `seed0.pkl` … `seed4.pkl`: per-user ranks of every model and variant, training logs, and gate weights. These are Python
  pickles; load them only from this repository.
- `mf_ensemble_ranks.npy`: per-user test ranks of the MF-BPR five-seed ensemble.
- `results.json`, `per_user_ndcg10.npz`, `stratification.json`: aggregated results.

The same files are provided with the article as Supplemental Data S1 (code and `out_ml1m/`) and Supplemental Data S2
(`out_vg/`).

The `supplementary/` folder holds exact copies of the files submitted with the article: `Supplemental_Data_S1.zip`,
`Supplemental_Data_S2.zip`, and `Supplemental_Article_S1.pdf` (with its LaTeX source; Tables S1-S9 are produced by
`make_supplement.py`).

## Usage instructions

**1. Tables and figures from the stored results (no data download needed, about a minute):**

```bash
pip install -r requirements.txt
python make_tables.py out_ml1m out_vg tables         # Tables 1-7 of the article (LaTeX)
python make_figures.py out_ml1m out_vg figures       # Figures 2 and 3
python make_supplement.py out_ml1m out_vg s_tables.tex   # Tables S1-S9 of Supplemental Article S1
```

Figures 4 and 5 are the stored forest plots `out_ml1m/forest.pdf` and `out_vg/forest.pdf`.

**2. Recompute all statistics from the stored per-seed results.** This step needs the datasets (see Dataset information).
It regenerates the splits and knowledge graphs deterministically, then rebuilds `results.json`,
`per_user_ndcg10.npz`, `stratification.json`, and the forest plots:

```bash
export ML_ROOT=./ml-1m VG_ROOT=./amazon-vg
python filter_meta.py amazon-vg/Video_Games.csv amazon-vg/meta_Video_Games.jsonl amazon-vg/meta_needed.jsonl
DATASET=ml1m OUT=out_ml1m python synapse_v5.py prep
DATASET=vg   OUT=out_vg   python synapse_v5.py prep
python aggregate_v5.py out_ml1m ; python aggregate_v5.py out_vg
python stratify_v5.py out_ml1m ./ml-1m ; python stratify_v5.py out_vg
```

**3. Rerun all experiments.** This takes hours on a CPU:

```bash
for D in ml1m vg; do
  DATASET=$D OUT=out_$D python synapse_v5.py prep        # splits, knowledge graph, deterministic baselines
  DATASET=$D OUT=out_$D python synapse_v5.py tune        # learning-rate selection on validation (seed 0)
  DATASET=$D OUT=out_$D python synapse_v5.py 0 1 2 3 4   # five training seeds
  python aggregate_v5.py out_$D
done
```

Rerunning the seeds also writes the full MF-BPR score matrices `mf_full_seed*.npy`, from which `aggregate_v5.py`
recomputes `mf_ensemble_ranks.npy`.

One seed (all models, fusion baselines, and ablations) took 1.0 to 2.0 hours on MovieLens-1M and about 0.4 hours on
Video Games, with one CPU thread per process. Retrained neural models can differ slightly across platforms because of
floating-point reduction order. The stored seed files are the reference for every reported number.

## Requirements

- Python 3.13
- torch 2.14.1
- numpy 2.1.3
- scipy 1.16.3
- pandas 2.2.3
- scikit-learn 1.6.1
- matplotlib 3.11.2

The same versions are pinned in `requirements.txt`.

The experiments were run on CPU only (Ubuntu 24.04, two virtual cores of an Intel Xeon at 2.1 GHz, 8 GB RAM).
No GPU is needed.

## Methodology

1. **Preprocessing.**
   - Ratings of 4 or higher are kept as positive interactions.
   - User (≥10) and item (≥5) filters are alternated until neither removes anything.
   - A fixed random sample of users is drawn (seed 42): 35% for MovieLens-1M and 20% for Video Games.
   - Interactions are ordered by timestamp, with ties broken by item ID. The last three interactions of each user are
     the fusion, validation, and test items; the earlier ones form the training prefix.
   - 99 sampled negatives per user are drawn once (seed 12345) and used for model selection and for the secondary metrics.
2. **Knowledge graph.** Built from the training prefixes and item metadata only, with these relations:
   - `INTERACTED`;
   - `HAS_GENRE` (MovieLens-1M), or `HAS_CATEGORY` and `HAS_STORE` (Video Games);
   - directed `TEMPORAL_NEXT`.
3. **Training.**
   - Every neural model gets the same learning-rate grid {5e-4, 1e-3, 2e-3}, selected on validation with seed 0.
   - Early stopping: evaluation every 5 epochs, patience 4, at least 50 and at most 200 epochs.
   - Five training seeds (0–4).
4. **Fusion.** ORACLE and the fusion baselines are fitted on the fusion item of each user, never on validation or test data.
5. **Evaluation.**
   - Full-catalogue NDCG@K and HR@K for K = 5, 10, 20, and MRR; sampled metrics are secondary.
   - Paired t-test and Wilcoxon test, Holm-adjusted.
   - Two-level (users × seeds) bootstrap intervals and TOST equivalence (±5%).

See `PROTOCOL_v5.md` and the Materials and Methods section of the article for details.

## Citations

If you use this code, please cite the article:

> Muniraja P, Satapathy SM. SYNAPSE-KG: Evaluating learned fusion of knowledge-graph-aware and neural recommenders
> under a leakage-free protocol. *PeerJ Computer Science* (under review), 2026.

Please also cite the datasets:

- Harper FM, Konstan JA. The MovieLens datasets: history and context. *ACM Transactions on Interactive Intelligent
  Systems* 5(4):19, 2015. https://doi.org/10.1145/2827872
- Hou Y, Li J, Fu X, He Z, Yan A, Chen X, McAuley J. Bridging language and items for retrieval and recommendation:
  benchmarking LLMs as semantic encoders. *Proceedings of ACL 2026 (Long Papers)*, 3251–3265, 2026.
  https://doi.org/10.18653/v1/2026.acl-long.147

## License and contributions

The code is released under the MIT License (see `LICENSE`). The datasets remain under the terms of their providers
(GroupLens Research and the McAuley Lab).

Issues and pull requests are welcome. For questions, contact the corresponding author, Shashank Mouli Satapathy
(shashankmouli.s@vit.ac.in).

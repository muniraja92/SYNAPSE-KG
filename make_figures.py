"""Figures 2-4 of the v5 manuscript.  Usage: python make_figures.py OUT_ML OUT_VG FIGDIR"""
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
ML, VG, FD = sys.argv[1:4]
import os; os.makedirs(FD, exist_ok=True)
R = {'MovieLens-1M': json.load(open(f"{ML}/results.json")), 'Amazon Video Games': json.load(open(f"{VG}/results.json"))}
PU = {'MovieLens-1M': np.load(f"{ML}/per_user_ndcg10.npz"), 'Amazon Video Games': np.load(f"{VG}/per_user_ndcg10.npz")}
plt.rcParams.update({'font.size': 10})
order = ['Popularity', 'Content', 'ItemKNN', 'UserKNN', 'EASE', 'MF-BPR', 'LightGCN', 'MINT', 'DRIP', 'NEXIS', 'BRIDGE-UI', 'BRIDGE', 'SYNAPSE-KG']
col = lambda m: '#c0392b' if m == 'SYNAPSE-KG' else ('#2e86c1' if m in ['DRIP', 'MINT', 'BRIDGE', 'BRIDGE-UI', 'NEXIS'] else '#7f8c8d')
# Figure 2: full-ranking NDCG@10 of all models
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, (nm, r) in zip(axs, R.items()):
    v = [r['main'][m]['full']['NDCG@10'][0] for m in order]; e = [r['main'][m]['full']['NDCG@10'][1] for m in order]
    ax.barh(order, v, xerr=e, color=[col(m) for m in order], edgecolor='black', capsize=2)
    for i, (a, b) in enumerate(zip(v, e)): ax.text(a + b + max(v) * 0.01, i, f'{a:.4f}', va='center', fontsize=8)
    ax.set_title(nm); ax.set_xlim(0, max(v) * 1.22); ax.set_xlabel('NDCG@10, full ranking (mean ± sd over 5 seeds)')
plt.tight_layout(); fig.savefig(f"{FD}/Figure2.pdf"); plt.close(fig)
# Figure 3: SYNAPSE-KG minus single-model baselines, fusion baselines, and ablated variants (relative %, 95% t interval)
groups = [('Single-model baselines', ['EASE', 'UserKNN', 'MF-BPR', 'LightGCN']),
          ('Fusion baselines', ['Mean_of_inputs', 'Logistic_stacking', 'Best_single_input', 'ORACLE_+_MF-BPR_input', 'MF-BPR_seed_ensemble']),
          ('Ablated SYNAPSE-KG', ['wo_KG_relations', 'wo_DRIP', 'wo_MINT', 'wo_BRIDGE', 'wo_NEXIS', 'wo_ItemKNN', 'wo_Content', 'wo_Popularity', 'wo_LightGCN'])]
fig, axs = plt.subplots(1, 2, figsize=(13, 7.5), sharey=True)
for ax, (nm, P) in zip(axs, PU.items()):
    syn = P['SYNAPSE-KG']; y = 0; ticks, labs = [], []
    for g, names in groups:
        y -= 1; ticks.append(y); labs.append(g); y -= 1
        for n in names:
            d = syn - P[n]; base = P[n].mean(); m = 100 * d.mean() / base; h = 100 * stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) / base
            ax.errorbar(m, y, xerr=h, fmt='o', color='#2e4053', ms=4, capsize=2); ticks.append(y)
            labs.append('   ' + n.replace('wo_', 'w/o ').replace('_', ' ').replace('w/o KG relations', 'w/o KG relations')); y -= 1
    ax.axvline(0, color='black', lw=0.8); ax.set_title(nm); ax.grid(axis='x', alpha=.3)
    ax.set_xlabel('SYNAPSE-KG minus comparator, % of comparator NDCG@10 (95% CI)')
axs[0].set_yticks(ticks); axs[0].set_yticklabels(labs, fontsize=8.5)
for t in axs[0].get_yticklabels():
    if not t.get_text().startswith('   '): t.set_fontweight('bold')
plt.tight_layout(); fig.savefig(f"{FD}/Figure3.pdf"); plt.close(fig)
print('figures written')

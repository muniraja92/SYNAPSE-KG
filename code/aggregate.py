import os, pickle, json, glob
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

OUT = os.environ.get("AGG_OUT", "./out"); FIG = os.environ.get("AGG_FIG", "./figures"); os.makedirs(FIG, exist_ok=True)
SUMMARY = os.environ.get("AGG_SUMMARY", OUT + "/summary.json")
KS = [5, 10, 20]
def ranks_from_scores(S): return 1 + (S[:, 1:] >= S[:, :1]).sum(1)
def ndcg10(r): return np.where(r <= 10, 1.0 / np.log2(r + 1), 0.0)
def metrics_from_ranks(r):
    o = {}
    for k in KS:
        o[f'HR@{k}'] = float(np.mean(r <= k)); o[f'NDCG@{k}'] = float(np.mean(np.where(r <= k, 1 / np.log2(r + 1), 0)))
    o['MRR@10'] = float(np.mean(np.where(r <= 10, 1.0 / r, 0))); return o

D = pickle.load(open(f"{OUT}/data.pkl", 'rb')); DET = pickle.load(open(f"{OUT}/deterministic.pkl", 'rb'))
seeds = sorted(int(os.path.basename(f)[4:-4]) for f in glob.glob(f"{OUT}/seed*.pkl"))
R = {s: pickle.load(open(f"{OUT}/seed{s}.pkl", 'rb')) for s in seeds}
print("seeds:", seeds)
det_ranks = {'Popularity': ranks_from_scores(DET['Popularity'][1]), 'ItemKNN': ranks_from_scores(DET['ItemKNN'][1]),
             'UserKNN': ranks_from_scores(DET['UserKNN'][1]), 'Content': ranks_from_scores(DET['Content'][1])}
models = ['Popularity', 'ItemKNN', 'UserKNN', 'Content', 'MF-BPR', 'LightGCN', 'DRIP', 'MINT', 'BRIDGE', 'NEXIS', 'SYNAPSE-KG']
baselines = models[:6]
per_seed = {m: [] for m in models}; per_user = {}
for m in models:
    rs = [det_ranks[m]] * len(seeds) if m in det_ranks else [R[s]['ranks'][m] for s in seeds]
    per_seed[m] = [metrics_from_ranks(r) for r in rs]
    per_user[m] = np.mean([ndcg10(r) for r in rs], 0)
cols = ['HR@5', 'HR@10', 'HR@20', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'MRR@10']
main = {m: {c: (np.mean([d[c] for d in per_seed[m]]), np.std([d[c] for d in per_seed[m]], ddof=1) if len(seeds) > 1 else 0.0) for c in cols} for m in models}
best_base = max(baselines, key=lambda m: main[m]['NDCG@10'][0])
impr = {c: (main['SYNAPSE-KG'][c][0] - main[best_base][c][0]) / main[best_base][c][0] * 100 for c in cols}
# second best per column among all non-SYNAPSE rows? (paper marks best baseline row) -> compute rank among all models
summary = {'seeds': seeds, 'best_baseline': best_base, 'main': {m: {c: list(main[m][c]) for c in cols} for m in models}, 'improvement_vs_best_baseline_pct': impr,
           'knn_k': {'ItemKNN': DET['ItemKNN_k'], 'UserKNN': DET['UserKNN_k']}, 'knn_val': {f"{a}_k{b}": v for (a, b), v in DET['knn_val'].items()}}
# seed-level improvement
summary['seed_level_ndcg10'] = {m: [d['NDCG@10'] for d in per_seed[m]] for m in models}
# statistics
tests = []
for b in baselines:
    x, y = per_user['SYNAPSE-KG'], per_user[b]; d = x - y
    t, p = stats.ttest_rel(x, y); w = stats.wilcoxon(x, y, zero_method='wilcox')
    tests.append(dict(comparison=f'SYNAPSE-KG vs {b}', mean_diff=float(d.mean()), t=float(t), p_t=float(p), W=float(w.statistic), p_w=float(w.pvalue), d_z=float(d.mean() / d.std(ddof=1))))
def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for rnk, i in enumerate(o): run = max(run, min(1, (m - rnk) * ps[i])); adj[i] = run
    return adj
for key in ['p_t', 'p_w']:
    for tt, a in zip(tests, holm([t[key] for t in tests])): tt[key + '_holm'] = float(a)
summary['tests'] = tests
# ablation
abl_names = ['w/o DRIP', 'w/o MINT', 'w/o BRIDGE', 'w/o NEXIS']
abl = {'SYNAPSE-KG (Full)': [metrics_from_ranks(R[s]['ranks']['SYNAPSE-KG']) for s in seeds]}
abl_user = {'SYNAPSE-KG (Full)': per_user['SYNAPSE-KG']}
for a in abl_names:
    abl[a] = [metrics_from_ranks(R[s]['abl_ranks'][a]) for s in seeds]; abl_user[a] = np.mean([ndcg10(R[s]['abl_ranks'][a]) for s in seeds], 0)
abl_tab = {}
full = {c: np.mean([d[c] for d in abl['SYNAPSE-KG (Full)']]) for c in ['NDCG@10', 'HR@10']}
abl_tests = []
for a, v in abl.items():
    row = {c: (np.mean([d[c] for d in v]), np.std([d[c] for d in v], ddof=1) if len(v) > 1 else 0.0) for c in ['NDCG@10', 'HR@10']}
    row['dNDCG'] = (row['NDCG@10'][0] - full['NDCG@10']) / full['NDCG@10'] * 100
    row['dHR'] = (row['HR@10'][0] - full['HR@10']) / full['HR@10'] * 100
    abl_tab[a] = row
    if a != 'SYNAPSE-KG (Full)':
        t, p = stats.ttest_rel(abl_user['SYNAPSE-KG (Full)'], abl_user[a]); abl_tests.append(dict(variant=a, t=float(t), p=float(p)))
for tt, h in zip(abl_tests, holm([t['p'] for t in abl_tests])): tt['p_holm'] = float(h)
summary['ablation'] = {a: {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()} for a, r in abl_tab.items()}
summary['ablation_tests'] = abl_tests
# gates
names = ['ItemKNN', 'Content', 'Popularity', 'DRIP', 'MINT', 'BRIDGE', 'NEXIS', 'LightGCN']
G = np.array([R[s]['gates_test_mean'] for s in seeds]); summary['gates'] = {n: [float(G[:, i].mean()), float(G[:, i].std(ddof=1)) if len(seeds) > 1 else 0.0] for i, n in enumerate(names)}
summary['best_epochs'] = {k: [R[s]['best_epoch'][k] for s in seeds] for k in R[seeds[0]]['best_epoch']}
def ep90(l):
    l = np.asarray(l); tgt = l[0] - 0.9 * (l[0] - l.min()); return int(np.argmax(l <= tgt) + 1)
summary['epochs_to_90pct_loss_drop'] = {k: [ep90(R[s]['losses'][k]) for s in seeds] for k in ['DRIP', 'MINT', 'BRIDGE', 'NEXIS', 'ORACLE']}
# group analyses (seed-averaged per-user NDCG@10, same 99 negatives)
nu = D['n_users']; act = np.array([len(D['train_seq'][u]) for u in range(nu)])
uq = pd.qcut(pd.Series(act), 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
pop = np.bincount(D['train_i'], minlength=D['n_items'])
ipop = pd.Series(pop[D['test_item']]); iq = pd.qcut(ipop.rank(method='first'), 4, labels=['Unpopular', 'Less Popular', 'Popular', 'Very Popular'])
gm = ['MF-BPR', 'LightGCN', 'SYNAPSE-KG']
summary['user_groups'] = {g: {m: float(per_user[m][(uq == g).values].mean()) for m in gm} | {'n': int((uq == g).sum())} for g in uq.cat.categories}
summary['item_groups'] = {g: {m: float(per_user[m][(iq == g).values].mean()) for m in gm} | {'n': int((iq == g).sum())} for g in iq.cat.categories}
# embedding structure: silhouette by primary genre (cosine), all items, per seed
ms = D['movies_subset']; prim = ms.set_index('item')['genres'].str.split('|').str[0].reindex(range(D['n_items'])).values
sil = {k: [] for k in ['BRIDGE', 'LightGCN', 'MF-BPR']}
for s in seeds:
    for k, key in [('BRIDGE', 'emb_BRIDGE'), ('LightGCN', 'emb_LGCN'), ('MF-BPR', 'emb_MF')]:
        sil[k].append(float(silhouette_score(R[s][key], prim, metric='cosine')))
summary['silhouette_primary_genre'] = {k: [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0] for k, v in sil.items()}
summary['kg_counts_train_only'] = D['kg_counts']; summary['n_entities'] = D['n_entities']
summary['dataset'] = {'users': nu, 'items': D['n_items'], 'train': int(len(D['train_u'])), 'val': nu, 'test': nu,
                      'sparsity_train_pct': float((1 - len(D['train_u']) / (nu * D['n_items'])) * 100), 'avg_train_per_user': float(len(D['train_u']) / nu)}
json.dump(summary, open(SUMMARY, 'w'), indent=1, default=float)

# ---------------------------------------------------------------- figures (no embedded "Figure N" titles)
plt.rcParams.update({'font.size': 11})
order = sorted(models, key=lambda m: main[m]['NDCG@10'][0])
col = lambda m: '#c0392b' if m == 'SYNAPSE-KG' else ('#2e86c1' if m in ['DRIP', 'MINT', 'BRIDGE', 'NEXIS'] else '#7f8c8d')
# Fig 1
fig, ax = plt.subplots(figsize=(10, 6))
v = [main[m]['NDCG@10'][0] for m in order]; e = [main[m]['NDCG@10'][1] for m in order]
ax.barh(order, v, xerr=e, color=[col(m) for m in order], edgecolor='black', capsize=3)
for i, (a, b) in enumerate(zip(v, e)): ax.text(a + b + 0.005, i, f'{a:.4f}', va='center', fontsize=9)
ax.set_xlabel(f'NDCG@10 (mean ± std over {len(seeds)} seeds)'); ax.set_xlim(0, max(v) * 1.2); plt.tight_layout(); fig.savefig(f'{FIG}/Figure1_NDCG10_Comparison.pdf'); plt.close(fig)
# Fig 2
fig, ax = plt.subplots(figsize=(10, 6)); sel = ['Popularity', 'ItemKNN', 'UserKNN', 'MF-BPR', 'LightGCN', 'BRIDGE', 'SYNAPSE-KG']; w = 0.12
for j, m in enumerate(sel):
    ax.bar(np.arange(3) + j * w, [main[m][f'HR@{k}'][0] for k in KS], w, yerr=[main[m][f'HR@{k}'][1] for k in KS], label=m, capsize=2, edgecolor='black')
ax.set_xticks(np.arange(3) + w * 3); ax.set_xticklabels([f'HR@{k}' for k in KS]); ax.set_ylabel('Hit Rate'); ax.legend(fontsize=8, ncol=2); plt.tight_layout(); fig.savefig(f'{FIG}/Figure2_HRK_Comparison.pdf'); plt.close(fig)
# Fig 3 / 8 (seed 0 training curves)
L = R[seeds[0]]['losses']; comps = ['DRIP', 'MINT', 'BRIDGE', 'NEXIS', 'ORACLE']
fig, axs = plt.subplots(1, 5, figsize=(20, 4))
for ax, c in zip(axs, comps):
    ax.plot(range(1, len(L[c]) + 1), L[c], marker='o', ms=2); ax.set_title(c); ax.set_xlabel('Epoch'); ax.set_ylabel('Training loss'); ax.grid(alpha=.3)
    be = R[seeds[0]]['best_epoch'][c]; ax.axvline(be, color='red', ls='--', lw=1)
plt.tight_layout(); fig.savefig(f'{FIG}/Figure3_Training_Loss_Curves.pdf'); plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 6))
for c in comps:
    l = np.array(L[c]); ax.plot(range(1, len(l) + 1), (l - l.min()) / (l.max() - l.min() + 1e-12), label=c)
ax.set_xlabel('Epoch'); ax.set_ylabel('Normalized training loss'); ax.legend(); ax.grid(alpha=.3); plt.tight_layout(); fig.savefig(f'{FIG}/Figure8_Convergence_Analysis.pdf'); plt.close(fig)
# Fig 4
fig, ax = plt.subplots(figsize=(10, 6)); an = list(abl_tab.keys())
ax.bar(an, [abl_tab[a]['NDCG@10'][0] for a in an], yerr=[abl_tab[a]['NDCG@10'][1] for a in an], capsize=4, color=['#c0392b'] + ['#2e86c1'] * 4, edgecolor='black')
ax.axhline(abl_tab['SYNAPSE-KG (Full)']['NDCG@10'][0], color='black', ls='--', lw=1)
lo = min(abl_tab[a]['NDCG@10'][0] for a in an); ax.set_ylim(lo * 0.97, max(abl_tab[a]['NDCG@10'][0] for a in an) * 1.02)
ax.set_ylabel(f'NDCG@10 (mean ± std over {len(seeds)} seeds)'); plt.tight_layout(); fig.savefig(f'{FIG}/Figure4_Ablation_Study.pdf'); plt.close(fig)
# Fig 5
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6)); gm_ = G.mean(0); gs_ = G.std(0, ddof=1) if len(seeds) > 1 else np.zeros(8)
a1.bar(names, gm_, yerr=gs_, capsize=3, color=plt.cm.viridis(np.linspace(.2, .8, 8)), edgecolor='black'); a1.tick_params(axis='x', rotation=45); a1.set_ylabel('Mean ORACLE gate weight (test candidates)')
a2.pie(gm_, labels=names, autopct='%1.1f%%', colors=plt.cm.viridis(np.linspace(.2, .8, 8)), startangle=90); plt.tight_layout(); fig.savefig(f'{FIG}/Figure5_Component_Weights.pdf'); plt.close(fig)
# Fig 6 / 7
for fname, grp in [('Figure6_User_Group_Analysis.pdf', summary['user_groups']), ('Figure7_Item_Group_Analysis.pdf', summary['item_groups'])]:
    fig, ax = plt.subplots(figsize=(11, 6)); gs = list(grp.keys())
    for j, m in enumerate(gm): ax.bar(np.arange(len(gs)) + j * 0.25, [grp[g][m] for g in gs], 0.25, label=m, edgecolor='black')
    ax.set_xticks(np.arange(len(gs)) + 0.25); ax.set_xticklabels([f"{g}\n(n={grp[g]['n']})" for g in gs]); ax.set_ylabel(f'NDCG@10 ({len(seeds)}-seed mean, 99 negatives)'); ax.legend(); plt.tight_layout(); fig.savefig(f'{FIG}/{fname}'); plt.close(fig)
# Fig 9 (training-prefix KG)
kc = D['kg_counts']; fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
a1.bar(list(kc.keys()), list(kc.values()), color=plt.cm.Set2(range(5)), edgecolor='black'); a1.tick_params(axis='x', rotation=30); a1.set_ylabel('Triples')
for i, v in enumerate(kc.values()): a1.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
ng = D['n_entities'] - nu - D['n_items']; a2.pie([nu, D['n_items'], ng], labels=[f'Users ({nu:,})', f'Items ({D["n_items"]:,})', f'Genres ({ng})'], autopct='%1.1f%%', startangle=90)
plt.tight_layout(); fig.savefig(f'{FIG}/Figure9_KG_Statistics.pdf'); plt.close(fig)
# Fig 10 (seed 0 t-SNE, 500 items)
rng = np.random.default_rng(42); idx = rng.choice(D['n_items'], 500, replace=False); g = prim[idx]; ug = sorted(set(g))
cm = {x: plt.cm.tab20(i % 20) for i, x in enumerate(ug)}
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
for ax, (k, key) in zip(axs, [('BRIDGE', 'emb_BRIDGE'), ('LightGCN', 'emb_LGCN'), ('MF-BPR', 'emb_MF')]):
    xy = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(R[seeds[0]][key][idx])
    ax.scatter(xy[:, 0], xy[:, 1], c=[cm[x] for x in g], s=15, alpha=.7); ax.set_title(f'{k} (silhouette {summary["silhouette_primary_genre"][k][0]:.3f})')
top = pd.Series(g).value_counts().head(8).index
axs[2].legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cm[x], label=x) for x in top], fontsize=8)
plt.tight_layout(); fig.savefig(f'{FIG}/Figure10_Embedding_tSNE.pdf'); plt.close(fig)
# Fig 11 radar
mets = cols; rm = ['MF-BPR', 'LightGCN', 'UserKNN', 'SYNAPSE-KG']; ang = np.linspace(0, 2 * np.pi, len(mets), endpoint=False).tolist(); ang += ang[:1]
fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, polar=True)
for m in rm:
    vv = [main[m][c][0] for c in mets]; vv += vv[:1]; ax.plot(ang, vv, label=m); ax.fill(ang, vv, alpha=.1)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(mets); ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1)); plt.tight_layout(); fig.savefig(f'{FIG}/Figure11_Radar_Chart.pdf'); plt.close(fig)
# Fig 12 heatmap
hm = ['SYNAPSE-KG', 'MF-BPR', 'UserKNN', 'BRIDGE', 'LightGCN', 'ItemKNN', 'Popularity', 'Content']
M = np.array([[(main[a]['NDCG@10'][0] - main[b]['NDCG@10'][0]) / main[b]['NDCG@10'][0] * 100 for b in hm] for a in hm])
fig, ax = plt.subplots(figsize=(10, 8)); im = ax.imshow(M, cmap='RdYlGn', vmin=-60, vmax=60)
ax.set_xticks(range(len(hm))); ax.set_yticks(range(len(hm))); ax.set_xticklabels(hm, rotation=45, ha='right'); ax.set_yticklabels(hm)
for i in range(len(hm)):
    for j in range(len(hm)): ax.text(j, i, f'{M[i, j]:+.1f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, label='Relative NDCG@10 difference (%) (row vs column)'); plt.tight_layout(); fig.savefig(f'{FIG}/Figure12_Improvement_Heatmap.pdf'); plt.close(fig)
print(json.dumps({k: summary[k] for k in ['best_baseline', 'improvement_vs_best_baseline_pct', 'knn_k']}, indent=1, default=float))

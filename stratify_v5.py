"""Stratified analyses of the SYNAPSE-KG minus strongest-baseline difference in full-ranking NDCG@10 (seed-averaged per user).
Per-stratum mean difference with 95% t interval, paired t and Wilcoxon (Holm over all strata); heterogeneity across the
strata of each attribute (ANOVA F and Kruskal-Wallis, Holm over attributes); equal-weight standardisation with a
stratified bootstrap interval.  Usage: python stratify_v5.py OUT [ML_ROOT]   (ML_ROOT only for MovieLens demographics)"""
import sys, os, json, pickle
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT = sys.argv[1]; ML = sys.argv[2] if len(sys.argv) > 2 else None
D_, _ = pickle.load(open(f"{OUT}/data.pkl", 'rb')); PU = np.load(f"{OUT}/per_user_ndcg10.npz")
BASE = json.load(open(f"{OUT}/results.json"))['best_baseline']          # the strongest single-model baseline
syn, mf = PU['SYNAPSE-KG'], PU[BASE]; d = syn - mf; nu = D_['n_users']
tr, ti = D_['train_seq'], D_['test_item']; pop = np.bincount(D_['train_i'], minlength=D_['n_items'])
attr_rel = list(D_['attrs'].keys())[0]; attrs = D_['attrs'][attr_rel]
act = np.array([len(tr[u]) for u in range(nu)])
def entropy(u):
    c = pd.Series([a for i in tr[u] for a in attrs[i]]).value_counts(normalize=True).values
    return float(-(c * np.log2(c)).sum()) if len(c) else 0.0
ent = np.array([entropy(u) for u in range(nu)]); main = np.array([np.log1p(pop[tr[u]]).mean() for u in range(nu)])
ts = D_['ts']; scale = 86400.0 if max(ts[0]) < 1e11 else 86400000.0
gap = np.array([(ts[u][-1] - ts[u][-4]) / scale for u in range(nu)])         # test time minus last training time, days
Q = ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']
qcut = lambda x: pd.qcut(pd.Series(x).rank(method='first'), 4, labels=Q).astype(str).values
strat = {'User activity': qcut(act), 'Test-item popularity': qcut(pop[ti]), 'User attribute diversity': qcut(ent),
         'User mainstreaminess': qcut(main),
         'Gap to test interaction': pd.cut(gap, [-1, 1, 30, 1e9], labels=['Up to 1 day', '1 to 30 days', 'Over 30 days']).astype(str)}
if ML:
    users = pd.read_csv(f"{ML}/users.dat", sep="::", names=["u", "gender", "age", "occ", "zip"], engine="python").set_index('u')
    U = users.loc[D_['raw_users']]
    strat['User gender'] = U['gender'].map({'M': 'Male', 'F': 'Female'}).values
    strat['User age group'] = U['age'].map({1: 'Under 25', 18: 'Under 25', 25: '25-34', 35: '35-44', 45: '45+', 50: '45+', 56: '45+'}).values
    prim = np.array([attrs[i][0] for i in ti]); top = [g for g, c in pd.Series(prim).value_counts().items() if c >= 100]
    strat['Test-item first-listed genre'] = np.where(np.isin(prim, top), prim, 'Other')
    import re
    yr = np.array([float(re.search(r'\((\d{4})\)\s*$', D_['texts'][i]).group(1)) for i in ti])
    strat['Test-item release year'] = pd.cut(yr, [0, 1979, 1989, 1995, 2100], labels=['Before 1980', '1980-1989', '1990-1995', '1996-2000']).astype(str)
def ci(x):
    n = len(x); m = x.mean(); q = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n); return m, m - q, m + q
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); a = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, min(1, (m - k) * p[i])); a[i] = run
    return a
rng = np.random.default_rng(2026); rows, het = [], []
for name, lab in strat.items():
    lab = np.asarray(lab); levels = [l for l in pd.unique(lab) if l != 'nan']
    order = Q if set(levels) <= set(Q) else sorted(levels, key=lambda l: (l == 'Other', l))
    if name == 'User age group': order = ['Under 25', '25-34', '35-44', '45+']
    if name == 'Gap to test interaction': order = [l for l in ['Up to 1 day', '1 to 30 days', 'Over 30 days'] if l in levels]
    if name == 'Test-item release year': order = ['Before 1980', '1980-1989', '1990-1995', '1996-2000']
    groups = []
    for l in order:
        msk = lab == l; x = d[msk]; groups.append(x); m, lo, hi = ci(x)
        rows.append(dict(stratification=name, stratum=l, n=int(msk.sum()), synapse=float(syn[msk].mean()), mf=float(mf[msk].mean()),
                         diff=float(m), lo=float(lo), hi=float(hi), rel_pct=float(100 * m / mf[msk].mean()),
                         p_t=float(stats.ttest_1samp(x, 0).pvalue), p_w=float(stats.wilcoxon(x).pvalue)))
    F = stats.f_oneway(*groups); K = stats.kruskal(*groups); base = np.mean([mf[lab == l].mean() for l in order])
    ew = np.mean([g.mean() for g in groups]); bt = [np.mean([rng.choice(g, len(g)).mean() for g in groups]) for _ in range(2000)]
    het.append(dict(stratification=name, k=len(groups), p_F=float(F.pvalue), p_KW=float(K.pvalue), ew_rel_pct=float(100 * ew / base),
                    ew_lo_pct=float(100 * np.percentile(bt, 2.5) / base), ew_hi_pct=float(100 * np.percentile(bt, 97.5) / base)))
for key in ['p_t', 'p_w']:
    for r, a in zip(rows, holm([r[key] for r in rows])): r[key + '_holm'] = float(a)
for key in ['p_F', 'p_KW']:
    for h, a in zip(het, holm([h[key] for h in het])): h[key + '_holm'] = float(a)
json.dump(dict(baseline=BASE, strata=rows, heterogeneity=het, n_strata=len(rows)), open(f"{OUT}/stratification.json", 'w'), indent=1)
# forest plot
fig, ax = plt.subplots(figsize=(10, 0.22 * (len(rows) + 2 * len(strat)) + 1.0)); y, ticks, labs, bold, cur = 0, [], [], [], None
for r in rows:
    if r['stratification'] != cur:
        cur = r['stratification']; y -= 1; ticks.append(y); labs.append(cur); bold.append(True); y -= 1
    sig = r['p_t_holm'] < 0.05 and r['p_w_holm'] < 0.05
    ax.errorbar(100 * r['diff'], y, xerr=[[100 * (r['diff'] - r['lo'])], [100 * (r['hi'] - r['diff'])]],
                fmt='s' if sig else 'o', color='#c0392b' if sig else '#2e4053', ms=4, capsize=2, lw=1)
    ticks.append(y); labs.append(f"   {r['stratum']} (n={r['n']})"); bold.append(False); y -= 1
ax.axvline(0, color='black', lw=0.8); ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=7.5); ax.set_ylim(y + 0.5, 0.5)
for t, b in zip(ax.get_yticklabels(), bold):
    if b: t.set_fontweight('bold')
ax.set_xlabel(f'NDCG@10 difference x 100, SYNAPSE-KG minus {BASE} (mean and 95% CI)'); ax.grid(axis='x', alpha=.3)
plt.tight_layout(); fig.savefig(f"{OUT}/forest.pdf"); plt.close(fig)
print(json.dumps([(h['stratification'], round(h['p_F_holm'], 4), round(h['p_KW_holm'], 4), round(h['ew_rel_pct'], 2)) for h in het]))
print([(r['stratification'][:14], r['stratum'], round(r['rel_pct'], 1)) for r in rows if r['p_t_holm'] < .05 and r['p_w_holm'] < .05])

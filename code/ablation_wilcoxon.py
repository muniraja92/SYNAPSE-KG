"""Add the Wilcoxon signed-rank test (and d_z) to the ablation comparisons, mirroring the main comparisons.
Reads the five seed result files of the run and the run's summary.json; writes the new fields into summary.json
under 'ablation_tests' (keys W, p_w, p_w_holm, d_z) after checking that the recomputed paired t-tests match exactly.
Usage: python ablation_wilcoxon.py <dir with seed*.pkl> summary.json"""
import sys, pickle, json, glob, numpy as np
from scipy import stats
d, sp = sys.argv[1], sys.argv[2]; S = json.load(open(sp))
ndcg10 = lambda r: np.where(r <= 10, 1.0 / np.log2(r + 1), 0.0)
R = {s: pickle.load(open(f"{d}/seed{s}.pkl", 'rb')) for s in S['seeds']}
full = np.mean([ndcg10(R[s]['ranks']['SYNAPSE-KG']) for s in S['seeds']], 0)
def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for rnk, i in enumerate(o): run = max(run, min(1, (m - rnk) * ps[i])); adj[i] = run
    return adj
pw = []
for t in S['ablation_tests']:
    v = np.mean([ndcg10(R[s]['abl_ranks'][t['variant']]) for s in S['seeds']], 0)
    tt, p = stats.ttest_rel(full, v); assert abs(p - t['p']) <= 1e-9 * max(1, p) and abs(tt - t['t']) < 1e-6, (t, tt, p)
    w = stats.wilcoxon(full, v, zero_method='wilcox'); diff = full - v
    t.update(W=float(w.statistic), p_w=float(w.pvalue), d_z=float(diff.mean() / diff.std(ddof=1))); pw.append(w.pvalue)
for t, h in zip(S['ablation_tests'], holm(pw)): t['p_w_holm'] = float(h)
json.dump(S, open(sp, 'w'), indent=1)
for t in S['ablation_tests']: print(t['variant'], f"t Holm {t['p_holm']:.3g}  Wilcoxon Holm {t['p_w_holm']:.3g}  d_z {t['d_z']:+.3f}")

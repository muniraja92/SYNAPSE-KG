"""Aggregate the v5 runs of one dataset into results.json (all numbers reported in the manuscript).
Usage: python aggregate_v5.py OUT        (OUT holds data.pkl, deterministic.pkl, tuned.json, seed*.pkl, and mf_full_seed*.npy or mf_ensemble_ranks.npy)"""
import sys, os, glob, json, pickle
import numpy as np
from scipy import stats

OUT = sys.argv[1]; B_BOOT = 2000; MARGIN = 0.05; RNG = np.random.default_rng(2026)
D_, _ = pickle.load(open(f"{OUT}/data.pkl", 'rb')); DET = pickle.load(open(f"{OUT}/deterministic.pkl", 'rb'))
seeds = sorted(int(os.path.basename(f)[4:-4]) for f in glob.glob(f"{OUT}/seed*.pkl"))
R = {s: pickle.load(open(f"{OUT}/seed{s}.pkl", 'rb')) for s in seeds}
tuned = json.load(open(f"{OUT}/tuned.json"))
nu, ni = D_['n_users'], D_['n_items']; KS = [5, 10, 20]
nd = lambda r, k=10: np.where(r <= k, 1.0 / np.log2(r + 1), 0.0)

DETM = ['Popularity', 'ItemKNN', 'UserKNN', 'Content', 'EASE']
NEUR = ['MF-BPR', 'LightGCN', 'DRIP', 'MINT', 'BRIDGE', 'BRIDGE-UI', 'NEXIS']
FUSB = ['Mean of inputs', 'Logistic stacking', 'Best single input', 'ORACLE + MF-BPR input']
INPUTS = ['ItemKNN', 'Content', 'Popularity', 'LightGCN', 'DRIP', 'MINT', 'BRIDGE', 'NEXIS']
ABL = ['w/o KG relations'] + [f'w/o {k}' for k in INPUTS]

def ranks(m, kind='full'):
    key = 'ranks_full' if kind == 'full' else 'ranks_sampled'
    return [DET[key][m]] * len(seeds) if m in DETM else [R[s][key][m] for s in seeds]

# MF-BPR five-seed score ensemble (z-scored per seed, averaged), full ranking. Its per-user test ranks are stored in
# mf_ensemble_ranks.npy so that the published archive does not need the large full score matrices (mf_full_seed*.npy).
def ranks_full(S, target, excl):
    S = S.astype(np.float32, copy=True); st = S[np.arange(len(S)), target].copy(); S[excl] = -np.inf; S[np.arange(len(S)), target] = st
    return (S >= st[:, None]).sum(1)
if all(os.path.exists(f"{OUT}/mf_full_seed{s}.npy") for s in seeds):
    ens = None
    for s in seeds:
        M = np.load(f"{OUT}/mf_full_seed{s}.npy").astype(np.float32); M = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-8)
        ens = M if ens is None else ens + M
    ens_r = ranks_full(ens, D_['test_item'], D_['excl_test']); del ens
    np.save(f"{OUT}/mf_ensemble_ranks.npy", ens_r)
else:
    ens_r = np.load(f"{OUT}/mf_ensemble_ranks.npy")

def metr(rs):
    o = {}
    for k in KS:
        o[f'NDCG@{k}'] = [float(nd(r, k).mean()) for r in rs]; o[f'HR@{k}'] = [float(np.mean(r <= k)) for r in rs]
    o['MRR'] = [float(np.mean(1.0 / r)) for r in rs]
    return {k: [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0] for k, v in o.items()}

models = DETM + NEUR + ['SYNAPSE-KG'] + FUSB
main = {m: {'full': metr(ranks(m)), 'sampled': metr(ranks(m, 'sampled'))} for m in models}
main['MF-BPR seed ensemble'] = {'full': metr([ens_r]), 'sampled': None}
per_user = {m: np.mean([nd(r) for r in ranks(m)], 0) for m in models}; per_user['MF-BPR seed ensemble'] = nd(ens_r)
per_seed = {m: np.array([nd(r).mean() for r in ranks(m)]) for m in models}

def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); a = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, min(1, (m - k) * p[i])); a[i] = run
    return a
def compare(a, others):
    out = []
    for b in others:
        d = per_user[a] - per_user[b]; t = stats.ttest_rel(per_user[a], per_user[b]); w = stats.wilcoxon(per_user[a], per_user[b])
        out.append(dict(comparison=f'{a} vs {b}', base=b, mean_diff=float(d.mean()), rel_pct=float(100 * d.mean() / per_user[b].mean()),
                        t=float(t.statistic), p_t=float(t.pvalue), W=float(w.statistic), p_w=float(w.pvalue), d_z=float(d.mean() / d.std(ddof=1))))
    for key in ['p_t', 'p_w']:
        for o, h in zip(out, holm([o[key] for o in out])): o[key + '_holm'] = float(h)
    return out
baselines = ['Popularity', 'ItemKNN', 'UserKNN', 'Content', 'EASE', 'MF-BPR', 'LightGCN']
tests = compare('SYNAPSE-KG', baselines + FUSB + ['MF-BPR seed ensemble'])
for a in ABL: per_user[a] = np.mean([nd(r) for r in ranks(a)], 0)
abl_tests = compare('SYNAPSE-KG', ABL)
abl = {a: {'full': metr(ranks(a)), 'rel_pct': float(100 * (np.mean([nd(r).mean() for r in ranks(a)]) / main['SYNAPSE-KG']['full']['NDCG@10'][0] - 1))} for a in ABL}

# strongest single baseline by mean full NDCG@10
best_base = max(baselines, key=lambda m: main[m]['full']['NDCG@10'][0])

# two-level bootstrap (users and seeds) for SYNAPSE-KG minus a baseline, relative to the baseline mean
def boot(b):
    A = np.stack([nd(r) for r in ranks('SYNAPSE-KG')]); Bm = np.stack([nd(r) for r in ranks(b)]) if b != 'MF-BPR seed ensemble' else None
    S = len(seeds); out = []
    for _ in range(B_BOOT):
        us = RNG.integers(0, nu, nu); ss = RNG.integers(0, S, S)
        a = A[ss][:, us].mean(); bb = Bm[ss][:, us].mean(); out.append(100 * (a - bb) / bb)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
boot_ci = {b: boot(b) for b in sorted(set(['MF-BPR', best_base, 'EASE', 'UserKNN']))}

# equivalence (TOST) against MF-BPR with a pre-specified margin of +-5% of MF-BPR's mean NDCG@10
def tost(b):
    d = per_user['SYNAPSE-KG'] - per_user[b]; m = MARGIN * per_user[b].mean(); n = len(d); se = d.std(ddof=1) / np.sqrt(n)
    p_low = 1 - stats.t.cdf((d.mean() + m) / se, n - 1); p_up = stats.t.cdf((d.mean() - m) / se, n - 1)
    q = stats.t.ppf(0.95, n - 1)
    return dict(margin_abs=float(m), margin_pct=100 * MARGIN, p_lower=float(p_low), p_upper=float(p_up), p_tost=float(max(p_low, p_up)),
                ci90_pct=[float(100 * (d.mean() - q * se) / per_user[b].mean()), float(100 * (d.mean() + q * se) / per_user[b].mean())])
equiv = {b: tost(b) for b in sorted(set(['MF-BPR', best_base]))}
seed_level = {b: dict(zip(['t', 'p'], map(float, stats.ttest_rel(per_seed['SYNAPSE-KG'], per_seed[b])))) for b in ['MF-BPR', best_base] if b in per_seed}

info = {}
for m in NEUR:
    I = [R[s]['info'][m] for s in seeds]
    info[m] = dict(lr=tuned[m]['lr'], best_epoch=[i['best_epoch'] for i in I], epochs_run=[i['epochs_run'] for i in I],
                   minutes=float(np.mean([i['seconds'] for i in I]) / 60), params=I[0]['params'], tune_val=tuned[m]['val'], tune_epochs=tuned[m]['epochs'])
info['SYNAPSE-KG'] = dict(oracle_best_epoch=[R[s]['info']['SYNAPSE-KG']['best_epoch'] for s in seeds], oracle_minutes=float(np.mean([R[s]['info']['SYNAPSE-KG']['seconds'] for s in seeds]) / 60))
G = np.array([R[s]['gates_test_mean'] for s in seeds])
res = dict(dataset=dict(users=nu, items=ni, train=int(len(D_['train_u'])), pool_users=int(D_['pool_users']),
                        density_pct=float(100 * len(D_['train_u']) / (nu * ni)), avg_train=float(len(D_['train_u']) / nu), kg=D_['kg'],
                        det_selected=DET['sel'], tfidf_terms=int(D_['tfidf_terms'])),
           seeds=seeds, main=main, best_baseline=best_base, tests=tests, ablation=abl, ablation_tests=abl_tests,
           boot_ci_pct=boot_ci, equivalence=equiv, seed_level=seed_level, per_seed_ndcg10={m: per_seed[m].tolist() for m in ['SYNAPSE-KG', 'MF-BPR', best_base]},
           gates={k: [float(G[:, i].mean()), float(G[:, i].std(ddof=1)) if len(seeds) > 1 else 0.0] for i, k in enumerate(INPUTS)},
           stacking_coef={k: float(np.mean([R[s]['stacking_coef'][k] for s in seeds])) for k in INPUTS},
           best_single_input=[R[s]['best_single_input'] for s in seeds], training=info)
json.dump(res, open(f"{OUT}/results.json", 'w'), indent=1)
np.savez_compressed(f"{OUT}/per_user_ndcg10.npz", **{m.replace(' ', '_').replace('/', ''): v for m, v in per_user.items()})
print(json.dumps({'best_baseline': best_base, 'SYN': main['SYNAPSE-KG']['full']['NDCG@10'], 'MF': main['MF-BPR']['full']['NDCG@10'],
                  'tests': [(t['base'], round(t['rel_pct'], 2), round(t['p_t_holm'], 4), round(t['p_w_holm'], 4)) for t in tests],
                  'abl': [(t['base'], round(t['rel_pct'], 2), round(t['p_t_holm'], 4)) for t in abl_tests], 'boot': boot_ci, 'equiv': equiv}, indent=1))

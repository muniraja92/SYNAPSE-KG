"""Supplemental Text S1 of the SYNAPSE-KG v5 manuscript: every table is generated from the stored result files.
Usage: python make_supplement.py OUT_ML OUT_VG OUTFILE"""
import sys, json
ML, VG, OUTF = sys.argv[1:4]
D = {'MovieLens-1M': ML, 'Amazon Video Games': VG}
R = {k: json.load(open(f"{v}/results.json")) for k, v in D.items()}
T = {k: json.load(open(f"{v}/tuned.json")) for k, v in D.items()}
S = {k: json.load(open(f"{v}/stratification.json")) for k, v in D.items()}
NEURAL = ['MF-BPR', 'LightGCN', 'DRIP', 'MINT', 'BRIDGE', 'BRIDGE-UI', 'NEXIS']
INPUTS = ['ItemKNN', 'Content', 'Popularity', 'LightGCN', 'DRIP', 'MINT', 'BRIDGE', 'NEXIS']
METRICS = ['NDCG@5', 'HR@5', 'NDCG@10', 'HR@10', 'NDCG@20', 'HR@20', 'MRR']
SINGLE = ['Popularity', 'Content', 'ItemKNN', 'UserKNN', 'EASE', 'MF-BPR', 'LightGCN', 'MINT', 'DRIP', 'NEXIS', 'BRIDGE-UI', 'BRIDGE', 'SYNAPSE-KG']
FUSION = ['SYNAPSE-KG', 'Mean of inputs', 'Logistic stacking', 'Best single input', 'ORACLE + MF-BPR input', 'MF-BPR seed ensemble']
LRS = ['0.0005', '0.001', '0.002']
f4 = lambda x: f"{x:.4f}"
def sg(x, d=2):
    s = f"{x:+.{d}f}"; return s.replace('-', '$-$')
def fp(p): return '$<$0.001' if p < 0.001 else f"{p:.3f}"
def esc(s): return s.replace('&', '\\&').replace('_', '\\_').replace('%', '\\%')
def table(caption, label, colspec, head, rows, size='\\footnotesize'):
    return ("\\begin{table}[!ht]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n" + size +
            "\n\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{" + colspec + "}\n\\toprule\n" + head +
            " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")
out = []
# S1: learning-rate selection
rows = []
for ds in D:
    rows.append("\\multicolumn{8}{@{}l}{\\textit{" + ds + "}} \\\\")
    for m in NEURAL:
        t = T[ds][m]
        cells = [f4(t['val'][l]) + ('$^{\\dagger}$' if float(l) == t['lr'] else '') for l in LRS]
        ep = [str(t['epochs'][l]) for l in LRS]
        rows.append(f"{m} & " + " & ".join(cells) + " & " + " & ".join(ep) + f" & {t['lr']:g} \\\\")
    rows.append("\\midrule")
out.append(table("Learning-rate selection on seed 0. Validation NDCG@10 against 99 sampled negatives for each learning rate, the epoch of the best validation score, and the selected learning rate ($^{\\dagger}$).",
                 "tab:s_lr", "@{}l ccc ccc c@{}",
                 "\\textbf{Model} & \\multicolumn{3}{c}{\\textbf{Validation NDCG@10}} & \\multicolumn{3}{c}{\\textbf{Best epoch}} & \\textbf{Selected} \\\\\n & 5e-4 & 1e-3 & 2e-3 & 5e-4 & 1e-3 & 2e-3 & ",
                 rows[:-1]))
# S2: deterministic baselines
grids = {'ItemKNN': '\\{20, 50, 100, 200, 500\\}', 'UserKNN': '\\{20, 50, 100, 200, 500\\}', 'EASE': '\\{50, 100, 200, 500, 1{,}000, 2{,}000, 5{,}000, 10{,}000\\}'}
edge = {'ItemKNN': 500, 'UserKNN': 500, 'EASE': 10000}
rows = []
for m in ['ItemKNN', 'UserKNN', 'EASE']:
    vals = [R[ds]['dataset']['det_selected'][m] for ds in D]
    cells = [f"{v:,}".replace(',', '{,}') + ('$^{\\ast}$' if v == edge[m] else '') for v in vals]
    rows.append(f"{m} & {'$k$' if m != 'EASE' else '$\\lambda$'} & ${grids[m]}$ & " + " & ".join(cells) + " \\\\")
out.append(table("Hyperparameters of the deterministic baselines selected on validation. $^{\\ast}$: the selected value lies at the upper edge of the grid.",
                 "tab:s_det", "@{}l c l c c@{}", "\\textbf{Model} & \\textbf{Parameter} & \\textbf{Grid} & \\textbf{MovieLens-1M} & \\textbf{Video Games}", rows))
# S3: per-seed NDCG@10
rows = []
for ds in D:
    ps = R[ds]['per_seed_ndcg10']; bb = R[ds]['best_baseline']
    rows.append("\\multicolumn{6}{@{}l}{\\textit{" + ds + "}} \\\\")
    for m in ['SYNAPSE-KG', 'MF-BPR', bb]:
        rows.append(f"{m} & " + " & ".join(f4(v) for v in ps[m]) + " \\\\")
    sl = R[ds]['seed_level']
    rows.append("\\multicolumn{6}{@{}l}{Paired $t$-test on seed means: " + "; ".join(f"vs {k}, $t = {v['t']:.2f}$, $p = {v['p']:.3f}$" for k, v in sl.items()) + "} \\\\")
    rows.append("\\midrule")
out.append(table("Full-catalogue NDCG@10 of each training seed for SYNAPSE-KG, MF-BPR, and the strongest single-model baseline (deterministic, so constant across seeds).",
                 "tab:s_seeds", "@{}l ccccc@{}", "\\textbf{Model} & \\textbf{Seed 0} & \\textbf{Seed 1} & \\textbf{Seed 2} & \\textbf{Seed 3} & \\textbf{Seed 4}", rows[:-1]))
# S4: training per seed
rows = []
for ds in D:
    tr = R[ds]['training']
    rows.append("\\multicolumn{4}{@{}l}{\\textit{" + ds + "}} \\\\")
    for m in NEURAL:
        rows.append(f"{m} & {', '.join(map(str, tr[m]['best_epoch']))} & {', '.join(map(str, tr[m]['epochs_run']))} & {tr[m]['minutes']:.1f} \\\\")
    rows.append(f"ORACLE & {', '.join(map(str, tr['SYNAPSE-KG']['oracle_best_epoch']))} & -- & {tr['SYNAPSE-KG']['oracle_minutes']:.1f} \\\\")
    rows.append("\\midrule")
out.append(table("Training of the neural models per seed (seeds 0--4): epoch of the restored checkpoint, epochs run before early stopping, and mean training time per seed in minutes. ORACLE is early-stopped on the fusion pairs of held-out users (at most 100 epochs).",
                 "tab:s_train", "@{}l l l r@{}", "\\textbf{Model} & \\textbf{Best epoch} & \\textbf{Epochs run} & \\textbf{Minutes}", rows[:-1]))
# S5-S6: all metrics, full and sampled, single models
for regime, lab in [('full', 'full-catalogue ranking'), ('sampled', '99 sampled negatives')]:
    rows = []
    for ds in D:
        rows.append("\\multicolumn{8}{@{}l}{\\textit{" + ds + "}} \\\\")
        for m in SINGLE + FUSION[1:]:
            v = R[ds]['main'][m][regime]
            rows.append(f"{m} & " + " & ".join(('--' if v is None else f4(v[k][0])) for k in METRICS) + " \\\\")
            if m == 'SYNAPSE-KG': rows.append("\\addlinespace")
        rows.append("\\midrule")
    out.append(table(f"All metrics under {lab} (mean over five seeds) for the single models, SYNAPSE-KG, and the fusion baselines (below the gap). --: not computed (the MF-BPR seed ensemble is evaluated by full ranking only).",
                     f"tab:s_{regime}", "@{}l ccccccc@{}", "\\textbf{Model} & " + " & ".join(f"\\textbf{{{k}}}" for k in METRICS), rows[:-1], size='\\scriptsize'))
# S7: gates, stacking, best single input
rows = []
for m in INPUTS:
    rows.append(f"{m} & " + " & ".join(f"{R[ds]['gates'][m][0]:.3f} ({R[ds]['gates'][m][1]:.3f}) & {sg(R[ds]['stacking_coef'][m], 3)}" for ds in D) + " \\\\")
bs = " & ".join("\\multicolumn{2}{c}{" + ", ".join(R[ds]['best_single_input']) + "}" for ds in D)
rows += ["\\midrule", "Best single input (seeds 0--4) & " + bs + " \\\\"]
out.append(table("Fusion weights. Mean ORACLE gate weight over all user--item pairs scored in full ranking (mean over seeds, standard deviation over seeds in parentheses), mean logistic-stacking coefficient on standardized scores, and the best single input selected on the fusion pairs in each seed.",
                 "tab:s_gates", "@{}l cc cc@{}", "& \\multicolumn{2}{c}{\\textbf{MovieLens-1M}} & \\multicolumn{2}{c}{\\textbf{Video Games}} \\\\\n\\textbf{Input} & \\textbf{Gate} & \\textbf{Stacking} & \\textbf{Gate} & \\textbf{Stacking}", rows))
# S8-S9: per-stratum results
for ds in D:
    s = S[ds]; rows = []; cur = None
    for r in s['strata']:
        if r['stratification'] != cur:
            cur = r['stratification']; rows.append("\\multicolumn{7}{@{}l}{\\textit{" + esc(cur) + "}} \\\\")
        star = '$^{\\ast}$' if r['p_t_holm'] < 0.05 and r['p_w_holm'] < 0.05 else ''
        rows.append(f"\\quad {esc(r['stratum'])} & {r['n']} & {f4(r['synapse'])} & {f4(r['mf'])} & {sg(100 * r['diff'])} [{sg(100 * r['lo'])}, {sg(100 * r['hi'])}]{star} & {fp(r['p_t_holm'])} & {fp(r['p_w_holm'])} \\\\")
    out.append(table(f"Per-stratum results on {ds}: number of users, mean full-catalogue NDCG@10 of SYNAPSE-KG and of {s['baseline']}, difference $\\times 100$ with 95\\% $t$-interval, and Holm-adjusted $p$-values (over all {s['n_strata']} strata) of the paired $t$-test and the Wilcoxon test. $^{{\\ast}}$: both adjusted $p$-values below 0.05.",
                     "tab:s_strata_" + ('ml' if ds.startswith('Movie') else 'vg'), "@{}l r cc c cc@{}",
                     f"\\textbf{{Stratum}} & $n$ & \\textbf{{SYNAPSE-KG}} & \\textbf{{{s['baseline']}}} & \\textbf{{Difference}} & $p_t$ & $p_W$", rows, size='\\scriptsize'))
open(OUTF, 'w').write("\n".join(out))
print('supplement tables written')

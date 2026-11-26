"""Generate the LaTeX tables of the v5 manuscript from results.json / stratification.json of both datasets.
Usage: python make_tables.py OUT_ML OUT_VG TEXDIR"""
import sys, json
ML, VG, TD = sys.argv[1], sys.argv[2], sys.argv[3]
R = {'ml': json.load(open(f"{ML}/results.json")), 'vg': json.load(open(f"{VG}/results.json"))}
NAME = {'ml': 'MovieLens-1M', 'vg': 'Amazon Video Games'}
def sg(x, d=2):
    return ('+' if x >= 0 else '$-$') + f'{abs(x):.{d}f}'
def pv(p): return '$<$0.001' if p < 0.001 else f'{p:.3f}'
def w(name, s): open(f"{TD}/{name}.tex", 'w').write(s)

# dataset table
rows = []
for k in ['ml', 'vg']:
    d = R[k]['dataset']
    rows.append(f"{NAME[k]} & {d['pool_users']:,} & {d['users']:,} & {d['items']:,} & {d['train']:,} & {d['avg_train']:.1f} & {d['density_pct']:.2f}\\% \\\\")
w('t_dataset', r"""\begin{table}[!ht]
\caption{Datasets after filtering and user sampling. Eligible users: users remaining after filtering, from which the sample is drawn. Each user additionally has one fusion, one validation, and one test interaction.}
\label{tab:dataset_statistics}
\footnotesize
\begin{tabular}{@{}l rrrrrr@{}}
\toprule
\textbf{Dataset} & \textbf{Eligible users} & \textbf{Users} & \textbf{Items} & \textbf{Training interactions} & \textbf{Per user} & \textbf{Density} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}""")

# KG table
rel_order = ['INTERACTED', 'HAS_GENRE', 'HAS_CATEGORY', 'HAS_STORE', 'TEMPORAL_NEXT']
desc = {'INTERACTED': 'user -- item (training prefix)', 'HAS_GENRE': 'item -- genre', 'HAS_CATEGORY': 'item -- category label',
        'HAS_STORE': 'item -- store (publisher or brand)', 'TEMPORAL_NEXT': r'item $\rightarrow$ next item (directed)'}
rows = []
for rel in rel_order:
    c = [R[k]['dataset']['kg']['counts_unique'].get(rel) for k in ['ml', 'vg']]
    rows.append(f"\\texttt{{{rel.replace('_', chr(92) + '_')}}} & {desc[rel]} & " + " & ".join(f"{x:,}" if x is not None else '--' for x in c) + r" \\")
ents = [R[k]['dataset']['kg']['n_attr_entities'] for k in ['ml', 'vg']]
rows.append(r"\midrule" + "\n" + "Attribute entities & & " + " & ".join(", ".join(f"{v:,} " + {'HAS_GENRE': 'genres', 'HAS_CATEGORY': 'categories', 'HAS_STORE': 'stores'}[k_] for k_, v in e.items()) for e in ents) + r" \\")
rows.append("All entities & users, items, attributes & " + " & ".join(f"{R[k]['dataset']['kg']['n_nodes']:,}" for k in ['ml', 'vg']) + r" \\")
w('t_kg', r"""\begin{table}[!ht]
\caption{Knowledge-graph relations built from the training prefixes and item metadata (number of distinct edges). \texttt{TEMPORAL\_NEXT} edges are weighted by $\log(1+c)$ for $c$ occurrences.}
\label{tab:kg_statistics}
\footnotesize
\begin{tabular}{@{}l l rr@{}}
\toprule
\textbf{Relation} & \textbf{Links} & \textbf{MovieLens-1M} & \textbf{Video Games} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}""")

# main results (full ranking primary; sampled secondary)
order = ['Popularity', 'ItemKNN', 'UserKNN', 'Content', 'EASE', 'MF-BPR', 'LightGCN', 'DRIP', 'MINT', 'BRIDGE', 'BRIDGE-UI', 'NEXIS', 'SYNAPSE-KG']
cols = ['NDCG@10', 'NDCG@20', 'HR@10', 'HR@20', 'MRR']
blocks = []
for k in ['ml', 'vg']:
    M = R[k]['main']; best = {c: max((M[m]['full'][c][0], m) for m in order)[1] for c in cols}
    second = {c: sorted(((M[m]['full'][c][0], m) for m in order), reverse=True)[1][1] for c in cols}
    blocks.append(r"\midrule" + "\n" + f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{NAME[k]}}}}} \\\\")
    for m in order:
        cells = []
        for c in cols:
            v, s_ = M[m]['full'][c]; t = f"{v:.4f}" + (f"\\,{{\\scriptsize$\\pm$.{round(s_ * 1e4):04d}}}" if s_ > 1e-9 else '')
            cells.append(f"\\textbf{{{t}}}" if best[c] == m else (f"\\underline{{{t}}}" if second[c] == m else t))
        smp = M[m]['sampled']; cells.append(f"{smp['NDCG@10'][0]:.4f}"); cells.append(f"{smp['HR@10'][0]:.4f}")
        if m == 'DRIP': blocks.append(r"\cmidrule{1-8}")
        if m == 'SYNAPSE-KG': blocks.append(r"\cmidrule{1-8}")
        blocks.append(f"{m} & " + " & ".join(cells) + r" \\")
w('t_main', r"""\begin{table}[!ht]
\caption{Test results (mean $\pm$ standard deviation over five training seeds; deterministic methods have zero variance). Full-catalogue ranking is the primary protocol; the last two columns use 99 sampled negatives. Bold: best; underlined: second best (full ranking).}
\label{tab:main_results}
\footnotesize
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{@{}l ccccc cc@{}}
\toprule
 & \multicolumn{5}{c}{\textbf{Full ranking}} & \multicolumn{2}{c}{\textbf{Sampled (99 negatives)}} \\
\cmidrule(lr){2-6}\cmidrule(l){7-8}
\textbf{Method} & \textbf{NDCG@10} & \textbf{NDCG@20} & \textbf{HR@10} & \textbf{HR@20} & \textbf{MRR} & \textbf{NDCG@10} & \textbf{HR@10} \\
""" + "\n".join(blocks) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}""")

# significance + fusion baselines
def test_rows(k, names):
    T = {t['base']: t for t in R[k]['tests']}; out = []
    for b in names:
        t = T[b]; s = 'Yes' if t['p_t_holm'] < .05 and t['p_w_holm'] < .05 else 'No'
        out.append(f"{b} & {sg(t['rel_pct'])}\\% & {sg(t['mean_diff'], 4)} & {pv(t['p_t_holm'])} & {pv(t['p_w_holm'])} & {sg(t['d_z'], 3)} & {s} \\\\")
    return out
base_names = ['Popularity', 'ItemKNN', 'UserKNN', 'Content', 'EASE', 'MF-BPR', 'LightGCN']
fus_names = ['Mean of inputs', 'Logistic stacking', 'Best single input', 'ORACLE + MF-BPR input', 'MF-BPR seed ensemble']
blocks = []
for k in ['ml', 'vg']:
    blocks += [r"\midrule", f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{NAME[k]}: single-model baselines}}}} \\\\"] + test_rows(k, base_names)
    blocks += [f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{NAME[k]}: fusion baselines}}}} \\\\"] + test_rows(k, fus_names)
w('t_tests', r"""\begin{table}[!ht]
\caption{SYNAPSE-KG compared with single-model and fusion baselines on per-user full-ranking NDCG@10 (seed-averaged; $n$ = number of users). $\Delta$: difference of SYNAPSE-KG minus the baseline (relative and absolute); $p$-values Holm-adjusted over the twelve comparisons of each dataset; Sig.: both adjusted $p$-values below 0.05.}
\label{tab:statistical_tests}
\footnotesize
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{@{}l cc cc c c@{}}
\toprule
\textbf{Baseline} & \textbf{$\Delta$ (\%)} & \textbf{$\Delta$} & \textbf{$p_{\text{Holm}}$ ($t$)} & \textbf{$p_{\text{Holm}}$ (Wilcoxon)} & \textbf{$d_z$} & \textbf{Sig.} \\
""" + "\n".join(blocks) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}""")

# ablation
blocks = []
abl = ['w/o KG relations'] + [f'w/o {x}' for x in ['DRIP', 'MINT', 'BRIDGE', 'NEXIS', 'ItemKNN', 'Content', 'Popularity', 'LightGCN']]
for k in ['ml', 'vg']:
    T = {t['base']: t for t in R[k]['ablation_tests']}; A = R[k]['ablation']
    blocks += [r"\midrule", f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{NAME[k]}}} (full model: {R[k]['main']['SYNAPSE-KG']['full']['NDCG@10'][0]:.4f})}} \\\\"]
    for a in abl:
        t = T[a]; s = '$^{*}$' if t['p_t_holm'] < .05 and t['p_w_holm'] < .05 else ''
        lab = a.replace('w/o KG relations', 'w/o KG relations (BRIDGE-UI for BRIDGE)')
        blocks.append(f"{lab} & {A[a]['full']['NDCG@10'][0]:.4f} & {sg(A[a]['rel_pct'])}\\%{s} & {pv(t['p_t_holm'])} & {pv(t['p_w_holm'])} \\\\")
w('t_ablation', r"""\begin{table}[!ht]
\caption{Ablations: ORACLE refitted without one input, or with BRIDGE replaced by BRIDGE-UI (no knowledge-graph relations). Full-ranking NDCG@10 (five-seed mean), relative change, and Holm-adjusted $p$-values (over the nine ablations of each dataset) of paired tests against the full model on per-user NDCG@10; $^{*}$: both significant (no ablation meets this criterion).}
\label{tab:ablation}
\footnotesize
\begin{tabular}{@{}l c c cc@{}}
\toprule
\textbf{Variant} & \textbf{NDCG@10} & \textbf{Change} & \textbf{$p_{\text{Holm}}$ ($t$)} & \textbf{$p_{\text{Holm}}$ (W)} \\
""" + "\n".join(blocks) + r"""
\bottomrule
\end{tabular}
\end{table}""")

# training and cost
blocks = []
for k in ['ml', 'vg']:
    blocks += [r"\midrule", f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{NAME[k]}}}}} \\\\"]
    for m in ['MF-BPR', 'LightGCN', 'DRIP', 'MINT', 'BRIDGE', 'BRIDGE-UI', 'NEXIS']:
        I = R[k]['training'][m]; be = I['best_epoch']
        blocks.append(f"{m} & {I['lr']:g} & {min(be)}--{max(be)} & {sum(I['epochs_run']) / len(I['epochs_run']):.0f} & {I['minutes']:.1f} & {I['params']:,} \\\\")
    blocks.append(f"ORACLE (fit and full scoring) & $10^{{-3}}$ & {min(R[k]['training']['SYNAPSE-KG']['oracle_best_epoch'])}--{max(R[k]['training']['SYNAPSE-KG']['oracle_best_epoch'])} & -- & {R[k]['training']['SYNAPSE-KG']['oracle_minutes']:.1f} & -- \\\\")
w('t_training', r"""\begin{table}[!ht]
\caption{Training: learning rate selected on validation, range of the selected (best) epoch over the five seeds, mean number of epochs run before early stopping, mean training time per seed (minutes, one CPU thread), and number of parameters.}
\label{tab:training}
\footnotesize
\begin{tabular}{@{}l c c c c r@{}}
\toprule
\textbf{Model} & \textbf{Learning rate} & \textbf{Best epoch} & \textbf{Epochs run} & \textbf{Minutes} & \textbf{Parameters} \\
""" + "\n".join(blocks) + r"""
\bottomrule
\end{tabular}
\end{table}""")

# Stratified analyses: heterogeneity across the strata of each attribute
S = {'ml': json.load(open(f"{ML}/stratification.json")), 'vg': json.load(open(f"{VG}/stratification.json"))}
fp = lambda p: '$<$0.001' if p < 0.001 else f"{p:.3f}"
sg = lambda x: f"{x:+.1f}".replace('-', '$-$')
blocks = []
for k, nm in [('ml', 'MovieLens-1M'), ('vg', 'Amazon Video Games')]:
    nsig = sum(1 for r in S[k]['strata'] if r['p_t_holm'] < 0.05 and r['p_w_holm'] < 0.05)
    blocks.append(r"\multicolumn{5}{@{}l}{\textit{" + nm + f" (comparator: {S[k]['baseline']}; {S[k]['n_strata']} strata, {['none','one','two','three','four','five','six','seven','eight','nine'][nsig] if nsig < 10 else nsig} with a significant difference)" + r"}} \\")
    for h in S[k]['heterogeneity']:
        blocks.append(f"{h['stratification']} & {h['k']} & {fp(h['p_F_holm'])} & {fp(h['p_KW_holm'])} & {sg(h['ew_rel_pct'])} [{sg(h['ew_lo_pct'])}, {sg(h['ew_hi_pct'])}] \\\\")
    blocks.append(r"\midrule")
blocks = blocks[:-1]
w('t_strat', r"""\begin{table}[!ht]
\caption{Stratified analyses of the per-user difference in full-ranking NDCG@10 between SYNAPSE-KG and the strongest single-model baseline. $k$: number of strata; $p$-values of the ANOVA $F$-test and the Kruskal--Wallis test for heterogeneity across strata, Holm-adjusted over the attributes of each dataset; last column: relative difference after giving every stratum equal weight, with a 95\% stratified bootstrap interval. A stratum difference is counted as significant when the Holm-adjusted (over all strata) $t$-test and Wilcoxon $p$-values are both below 0.05.}
\label{tab:stratified}
\footnotesize
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{@{}l c c c c@{}}
\toprule
\textbf{Attribute} & \textbf{$k$} & \textbf{$p_{\text{Holm}}$ ($F$)} & \textbf{$p_{\text{Holm}}$ (KW)} & \textbf{Equal-weight difference (\%)} \\
\midrule
""" + "\n".join(blocks) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}""")
print('tables written')

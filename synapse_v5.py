"""
SYNAPSE-KG v5: data preparation, baselines, the four neural scorers (DRIP, MINT, BRIDGE, NEXIS), the ORACLE fusion layer,
fusion baselines and ablations (protocol: PROTOCOL_v5.md).

  DATASET=ml1m|vg  ML_ROOT=...  VG_ROOT=...  OUT=...
  python synapse_v5.py prep            # data, splits, knowledge graph, deterministic baselines
  python synapse_v5.py tune            # learning-rate selection on validation (seed 0) -> OUT/tuned.json
  python synapse_v5.py <seed> [...]    # full run for the given training seeds -> OUT/seed<k>.pkl
Optional: SAMPLE_MODE=stratified SAMPLE_SEED=<s> for replication samples.
"""
import os, sys, time, json, pickle, random
from collections import defaultdict
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression

torch.set_num_threads(1)
DATASET = os.environ.get("DATASET", "ml1m")
ML_ROOT = os.environ.get("ML_ROOT", "./ml-1m")
VG_ROOT = os.environ.get("VG_ROOT", "./amazon-vg")
OUT = os.environ.get("OUT", f"./out_{DATASET}"); os.makedirs(OUT, exist_ok=True)
DATA_SEED = 42
SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", DATA_SEED))
SAMPLE_MODE = os.environ.get("SAMPLE_MODE", "random")
SAMPLE_FRAC = {"ml1m": 0.35, "vg": 0.20}[DATASET]
BS_GRAPH, BS_SEQ = {"ml1m": (4096, 1024), "vg": (512, 256)}[DATASET]   # updates per epoch: graph models ~50 on both datasets; sequence models ~195 (ML-1M) and ~110 (VG)
EVAL_SEED = 12345; N_NEG = 99; KS = [5, 10, 20]
EVAL_EVERY, PATIENCE, MIN_EPOCHS, MAX_EPOCHS = 5, 4, 50, 200
LR_GRID = [5e-4, 1e-3, 2e-3]
HIST_LEN = 20; D = 64
SMOKE = int(os.environ.get("SMOKE_EPOCHS", "0"))   # >0 only for pipeline tests

# ---------------------------------------------------------------- data
def _filter(df, ucol, icol):
    """Alternate user (>= 10) and item (>= 5) filters until neither removes anything (k-core convergence)."""
    while True:
        n0 = len(df)
        uc = df.groupby(ucol).size(); df = df[df[ucol].isin(uc[uc >= 10].index)]
        ic = df.groupby(icol).size(); df = df[df[icol].isin(ic[ic >= 5].index)]
        if len(df) == n0: return df

def _sample_users(df, ucol):
    rng = np.random.default_rng(SAMPLE_SEED); all_users = df[ucol].unique()
    if SAMPLE_MODE == 'stratified':
        cnt = df.groupby(ucol).size().reindex(all_users)
        q = pd.qcut(cnt.rank(method='first'), 4, labels=False).values
        sampled = np.concatenate([rng.choice(all_users[q == k], int(round((q == k).sum() * SAMPLE_FRAC)), replace=False) for k in range(4)])
    else:
        sampled = rng.choice(all_users, int(len(all_users) * SAMPLE_FRAC), replace=False)
    return df[df[ucol].isin(sampled)], len(all_users)

def load_data():
    """Returns per-user chronological item sequences (stable tie-break on item id), item attributes, item texts."""
    if DATASET == 'ml1m':
        r = pd.read_csv(f"{ML_ROOT}/ratings.dat", sep="::", names=["u", "i", "rating", "t"], engine="python")
        mv = pd.read_csv(f"{ML_ROOT}/movies.dat", sep="::", names=["i", "title", "genres"], engine="python", encoding="latin-1")
        df = _filter(r[r['rating'] >= 4][['u', 'i', 't']], 'u', 'i'); df, pool = _sample_users(df, 'u')
        items = sorted(df['i'].unique()); meta = mv.set_index('i').loc[items]
        attrs = {'HAS_GENRE': [g.split('|') for g in meta['genres']]}
        texts = [g.replace('|', ' ') + ' ' + t for g, t in zip(meta['genres'], meta['title'])]
        raw_ids = items
    else:
        r = pd.read_csv(f"{VG_ROOT}/Video_Games.csv").rename(columns={'user_id': 'u', 'parent_asin': 'i', 'timestamp': 't'})
        df = _filter(r[r['rating'] >= 4][['u', 'i', 't']], 'u', 'i'); df, pool = _sample_users(df, 'u')
        items = sorted(df['i'].unique())
        M = {}
        for line in open(f"{VG_ROOT}/meta_needed.jsonl"):
            j = json.loads(line); M[j['parent_asin']] = j
        cats = [[c for c in (M.get(i, {}).get('categories') or []) if c not in ('Video Games', 'Games')] for i in items]
        stores = [[M[i]['store']] if M.get(i, {}).get('store') else [] for i in items]
        attrs = {'HAS_CATEGORY': cats, 'HAS_STORE': stores}
        texts = [' '.join([M.get(i, {}).get('title') or ''] + (M.get(i, {}).get('categories') or []) + [M.get(i, {}).get('store') or '']) for i in items]
        raw_ids = items
    users = sorted(df['u'].unique())
    u2i = {u: k for k, u in enumerate(users)}; m2i = {m: k for k, m in enumerate(items)}
    df = df.assign(user=df['u'].map(u2i), item=df['i'].map(m2i))
    df = df.sort_values(['user', 't', 'item'], kind='stable')      # deterministic: ties broken by item id
    seqs = {u: g['item'].tolist() for u, g in df.groupby('user')}
    ts = {u: g['t'].tolist() for u, g in df.groupby('user')}
    return dict(seqs=seqs, ts=ts, attrs=attrs, texts=texts, n_users=len(users), n_items=len(items),
                raw_users=users, raw_items=raw_ids, pool_users=pool)

def build_splits(seqs, nu, ni):
    tr = {u: s[:-3] for u, s in seqs.items()}
    fus = np.array([seqs[u][-3] for u in range(nu)]); val = np.array([seqs[u][-2] for u in range(nu)]); test = np.array([seqs[u][-1] for u in range(nu)])
    rows = np.concatenate([[u] * len(tr[u]) for u in range(nu)]); cols = np.concatenate([tr[u] for u in range(nu)])
    train_mask = np.zeros((nu, ni), dtype=bool); train_mask[rows, cols] = True
    allm = train_mask.copy(); allm[np.arange(nu), fus] = True; allm[np.arange(nu), val] = True; allm[np.arange(nu), test] = True
    rng = np.random.default_rng(EVAL_SEED)
    def negs():
        o = np.zeros((nu, N_NEG), dtype=np.int64)
        for u in range(nu): o[u] = rng.choice(np.flatnonzero(~allm[u]), N_NEG, replace=False)
        return o
    vn, tn, fn = negs(), negs(), negs()
    excl_test = train_mask.copy(); excl_test[np.arange(nu), fus] = True; excl_test[np.arange(nu), val] = True
    return dict(train_seq=tr, fus_item=fus, val_item=val, test_item=test, train_u=rows, train_i=cols, train_mask=train_mask,
                fus_cand=np.column_stack([fus, fn]), val_cand=np.column_stack([val, vn]), test_cand=np.column_stack([test, tn]),
                excl_test=excl_test)

def build_kg(D_):
    """Knowledge graph from the training prefix and item metadata. Node order: users, items, attribute entities."""
    nu, ni = D_['n_users'], D_['n_items']; ents = {}; triples = defaultdict(list)
    for rel, lists in D_['attrs'].items():
        for i, lst in enumerate(lists):
            for a in lst:
                key = (rel, a)
                if key not in ents: ents[key] = nu + ni + len(ents)
                triples[rel].append((nu + i, ents[key]))
    for u in range(nu):
        for i in D_['train_seq'][u]: triples['INTERACTED'].append((u, nu + i))
    nxt = defaultdict(int)
    for u in range(nu):
        s = D_['train_seq'][u]
        for a, b in zip(s[:-1], s[1:]):
            if a != b: nxt[(nu + a, nu + b)] += 1
    triples['TEMPORAL_NEXT'] = list(nxt.keys())
    n_nodes = nu + ni + len(ents)
    def adj(rels):
        r, c, w = [], [], []
        for rel in rels:
            if rel == 'TEMPORAL_NEXT':
                for (a, b), k in nxt.items(): r.append(a); c.append(b); w.append(np.log1p(k))     # directed
            else:
                for a, b in triples[rel]: r += [a, b]; c += [b, a]; w += [1.0, 1.0]               # undirected
        return csr_matrix((w, (r, c)), shape=(n_nodes, n_nodes))
    def rownorm(A):
        d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1; return sp.diags(1 / d) @ A
    rels_all = [k for k in triples]
    A = adj(rels_all); Aui = adj(['INTERACTED'])
    counts = {k: (len(v) if k != 'TEMPORAL_NEXT' else int(sum(nxt.values()))) for k, v in triples.items()}
    counts_unique = {k: len(v) for k, v in triples.items()}
    ent_types = defaultdict(int)
    for (rel, _), _id in ents.items(): ent_types[rel] += 1
    return dict(fwd=rownorm(A).tocsr(), bwd=rownorm(A.T.tocsr()).tocsr(), ui=rownorm(Aui).tocsr(), n_nodes=n_nodes,
                counts=counts, counts_unique=counts_unique, n_attr_entities=dict(ent_types))

def content_features(D_):
    tf = TfidfVectorizer(max_features=500, stop_words='english').fit_transform(D_['texts'])
    ic = normalize(TruncatedSVD(n_components=64, random_state=DATA_SEED).fit_transform(tf))
    uc = np.zeros((D_['n_users'], 64))
    for u in range(D_['n_users']): uc[u] = ic[D_['train_seq'][u]].mean(0)
    return normalize(uc).astype(np.float32), ic.astype(np.float32), int(tf.shape[1])

# ---------------------------------------------------------------- metrics
def ranks_sampled(S): return 1 + (S[:, 1:] >= S[:, :1]).sum(1)
def ranks_full(S, target, excl):
    S = S.astype(np.float32, copy=True); st = S[np.arange(len(S)), target].copy(); S[excl] = -np.inf
    S[np.arange(len(S)), target] = st
    return (S >= st[:, None]).sum(1)
def ndcg_r(r, k=10): return np.where(r <= k, 1.0 / np.log2(r + 1), 0.0)
def metrics(r):
    o = {}
    for k in KS: o[f'HR@{k}'] = float(np.mean(r <= k)); o[f'NDCG@{k}'] = float(ndcg_r(r, k).mean())
    o['MRR'] = float(np.mean(1.0 / r)); return o

# ---------------------------------------------------------------- sparse op
class SPA:
    def __init__(self, M): self.M = M.tocsr().astype(np.float32); self.MT = self.M.T.tocsr()
class _SpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, op): ctx.op = op; return torch.from_numpy(op.M @ x.detach().numpy())
    @staticmethod
    def backward(ctx, g): return torch.from_numpy(ctx.op.MT @ g.contiguous().numpy()), None
def spmm(op, x): return _SpMM.apply(x, op)

# ---------------------------------------------------------------- non-parametric models
class ItemKNN:
    def __init__(s, D_):
        ui = csr_matrix((np.ones(len(D_['train_u'])), (D_['train_i'], D_['train_u'])), shape=(D_['n_items'], D_['n_users']))
        n = normalize(ui, axis=1); s.sim = (n @ n.T).toarray().astype(np.float32); s.tr = D_['train_seq']
    def score_pairs(s, C, k):
        o = np.zeros(C.shape, dtype=np.float32)
        for u in range(C.shape[0]):
            S = s.sim[C[u]][:, s.tr[u]]; kk = min(k, S.shape[1]); o[u] = np.partition(S, -kk, axis=1)[:, -kk:].sum(1)
        return o
    def score_full(s, k):
        o = np.zeros((len(s.tr), s.sim.shape[0]), dtype=np.float32)
        for u in range(len(s.tr)):
            S = s.sim[:, s.tr[u]]; kk = min(k, S.shape[1]); o[u] = np.partition(S, -kk, axis=1)[:, -kk:].sum(1)
        return o

class UserKNN:
    def __init__(s, D_):
        R = csr_matrix((np.ones(len(D_['train_u'])), (D_['train_u'], D_['train_i'])), shape=(D_['n_users'], D_['n_items']))
        n = normalize(R, axis=1); s.sim = (n @ n.T).toarray().astype(np.float32); np.fill_diagonal(s.sim, -np.inf); s.R = R.toarray().astype(np.float32)
    def score_full(s, k):
        o = np.zeros(s.R.shape, dtype=np.float32)
        for u in range(len(o)):
            idx = np.argpartition(-s.sim[u], k)[:k]; w = s.sim[u, idx]; o[u] = (w @ s.R[idx]) / (np.abs(w).sum() + 1e-8)
        return o

def ease_full(D_, lam):
    X = csr_matrix((np.ones(len(D_['train_u'])), (D_['train_u'], D_['train_i'])), shape=(D_['n_users'], D_['n_items']))
    G = (X.T @ X).toarray().astype(np.float64); G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G); B = -P / np.diag(P); B[np.diag_indices_from(B)] = 0
    return (X @ B).astype(np.float32)

def pick(full, C): return np.take_along_axis(full, C, 1)

# ---------------------------------------------------------------- neural models
def xav(m):
    for x in m.modules():
        if isinstance(x, nn.Embedding): nn.init.xavier_uniform_(x.weight)
        elif isinstance(x, nn.Linear):
            nn.init.xavier_uniform_(x.weight)
            if x.bias is not None: nn.init.zeros_(x.bias)

class MFBPR(nn.Module):
    def __init__(s, nu, ni):
        super().__init__(); s.user_emb = nn.Embedding(nu, D); s.item_emb = nn.Embedding(ni, D); xav(s)
    def loss(s, u, p, n):
        U, P, N = s.user_emb(u), s.item_emb(p), s.item_emb(n)
        return -F.logsigmoid((U * P).sum(1) - (U * N).sum(1)).mean() + 1e-3 * (U.pow(2).sum() + P.pow(2).sum() + N.pow(2).sum()) / len(u)
    def embs(s): return s.user_emb.weight.detach().numpy(), s.item_emb.weight.detach().numpy()

class LightGCN(nn.Module):
    def __init__(s, nu, ni, A, L=3):
        super().__init__(); s.nu, s.L, s.A = nu, L, A; s.user_emb = nn.Embedding(nu, D); s.item_emb = nn.Embedding(ni, D); xav(s)
    def forward(s):
        e = torch.cat([s.user_emb.weight, s.item_emb.weight]); out = [e]
        for _ in range(s.L): e = spmm(s.A, e); out.append(e)
        e = torch.stack(out, 1).mean(1); return e[:s.nu], e[s.nu:]
    def loss(s, u, p, n):
        Ue, Ie = s(); U, P, N = Ue[u], Ie[p], Ie[n]
        return -F.logsigmoid((U * P).sum(1) - (U * N).sum(1)).mean() + 1e-3 * (s.user_emb.weight[u].pow(2).sum() + s.item_emb.weight[p].pow(2).sum() + s.item_emb.weight[n].pow(2).sum()) / len(u)
    def embs(s):
        with torch.no_grad(): U, I = s()
        return U.numpy(), I.numpy()

class DRIP(nn.Module):
    def __init__(s, nu, ni, heads=4):
        super().__init__(); s.user_emb = nn.Embedding(nu, D); s.item_emb = nn.Embedding(ni, D)
        s.attention = nn.MultiheadAttention(D, heads, batch_first=True, dropout=0.1); s.query_proj = nn.Linear(D, D)
        s.intent_mlp = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Dropout(0.2), nn.Linear(D, D)); xav(s)
    def _kv(s, H):
        mha = s.attention; d = D; h = mha.num_heads; Wq, Wk, Wv = mha.in_proj_weight.split(d); bq, bk, bv = mha.in_proj_bias.split(d)
        he = s.item_emb(H); B, L = H.shape
        return (he @ Wk.T + bk).view(B, L, h, d // h), (he @ Wv.T + bv).view(B, L, h, d // h)
    def _score(s, u, i, K, V, M):
        """Multi-head attention with one query (the candidate), identical to nn.MultiheadAttention; then the intent MLP."""
        mha = s.attention; d = D; h = mha.num_heads; Wq = mha.in_proj_weight[:d]; bq = mha.in_proj_bias[:d]
        ue, ie = s.user_emb(u), s.item_emb(i); Q = (s.query_proj(ie) @ Wq.T + bq).view(-1, h, d // h) / (d // h) ** 0.5
        att = (Q.unsqueeze(1) * K).sum(-1).masked_fill(~M.unsqueeze(-1), float('-inf')).softmax(1)     # B x L x h
        att = F.dropout(att, mha.dropout, s.training)
        a = mha.out_proj((att.unsqueeze(-1) * V).sum(1).reshape(-1, d))
        return (s.intent_mlp(torch.cat([ue + a, ie], -1)) * ie).sum(-1)
    def forward(s, u, i, H, M):
        K, V = s._kv(H); return s._score(u, i, K, V, M)
    def bpr(s, u, p, n, H, M):
        K, V = s._kv(H); return s._score(u, p, K, V, M), s._score(u, n, K, V, M)     # history projected once
    @torch.no_grad()
    def full(s, users, H, M):
        """Scores of all items for the given users (eval mode); same arithmetic as forward(), vectorised over items."""
        mha = s.attention; d = D; h = mha.num_heads; hd = d // h
        Wq, Wk, Wv = mha.in_proj_weight.split(d); bq, bk, bv = mha.in_proj_bias.split(d)
        IE = s.item_emb.weight; Q = (s.query_proj(IE) @ Wq.T + bq).view(-1, h, hd) / hd ** 0.5      # ni x h x hd
        out = torch.zeros(len(users), IE.shape[0])
        for k, u in enumerate(users):
            he = s.item_emb(H[u]); K_ = (he @ Wk.T + bk).view(-1, h, hd); V = (he @ Wv.T + bv).view(-1, h, hd)
            att = torch.einsum('nhd,lhd->nhl', Q, K_).masked_fill(~M[u].view(1, 1, -1), float('-inf')).softmax(-1)
            a = mha.out_proj(torch.einsum('nhl,lhd->nhd', att, V).reshape(-1, d))
            z = s.intent_mlp(torch.cat([s.user_emb.weight[u].expand_as(a) + a, IE], -1)); out[k] = (z * IE).sum(-1)
        return out

class NEXIS(nn.Module):
    """P interests extracted from the history by self-attentive pooling, aggregated with target-aware attention."""
    def __init__(s, nu, ni, P=4):
        super().__init__(); s.P = P; s.user_emb = nn.Embedding(nu, D); s.item_emb = nn.Embedding(ni, D)
        s.W1 = nn.Linear(D, D, bias=False); s.W2 = nn.Linear(D, P, bias=False)
        s.att = nn.Sequential(nn.Linear(2 * D, D), nn.Tanh(), nn.Linear(D, 1)); xav(s)
    def interests(s, H, M):
        h = s.item_emb(H)                                            # B x L x d
        a = s.W2(torch.tanh(s.W1(h))).masked_fill(~M.unsqueeze(-1), -1e9)   # B x L x P
        a = F.softmax(a, 1); return torch.einsum('blp,bld->bpd', a, h)      # B x P x d
    def _score(s, u, i, c):
        ie = s.item_emb(i); w = F.softmax(s.att(torch.cat([c, ie.unsqueeze(1).expand(-1, s.P, -1)], -1)).squeeze(-1), -1)
        return ((s.user_emb(u) + (c * w.unsqueeze(-1)).sum(1)) * ie).sum(-1)
    def forward(s, u, i, H, M): return s._score(u, i, s.interests(H, M))
    def bpr(s, u, p, n, H, M):
        c = s.interests(H, M); return s._score(u, p, c), s._score(u, n, c)                # interests extracted once
    @torch.no_grad()
    def full(s, users, H, M):
        IE = s.item_emb.weight; L1, L2 = s.att[0], s.att[2]; Wc, Wi = L1.weight[:, :D], L1.weight[:, D:]; IEp = IE @ Wi.T + L1.bias
        out = torch.zeros(len(users), IE.shape[0])
        for k, u in enumerate(users):
            c = s.interests(H[u:u + 1], M[u:u + 1])[0]                                   # P x d
            w = L2(torch.tanh(IEp.unsqueeze(1) + (c @ Wc.T).unsqueeze(0))).squeeze(-1).softmax(-1)   # ni x P
            out[k] = ((s.user_emb.weight[u] + w @ c) * IE).sum(-1)
        return out

class MINT(nn.Module):
    """Symmetric momentum contrast between a user tower and an item tower (MoCo-style key encoders and queues)."""
    def __init__(s, nu, ni, m=0.995, t=0.1, K=4096):
        super().__init__(); s.m, s.t, s.K = m, t, K
        s.user_emb = nn.Embedding(nu, D); s.item_emb = nn.Embedding(ni, D)
        s.utow = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D)); s.itow = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        xav(s)
        s.user_emb_k = nn.Embedding(nu, D); s.item_emb_k = nn.Embedding(ni, D)
        s.utow_k = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D)); s.itow_k = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        for o, k in s._pairs(): k.load_state_dict(o.state_dict())                    # keys start as exact copies
        for _, k in s._pairs():
            for p in k.parameters(): p.requires_grad = False
        s.register_buffer('qi', F.normalize(torch.randn(K, D), dim=1)); s.register_buffer('qu', F.normalize(torch.randn(K, D), dim=1))
        s.ptr = 0
    def _pairs(s): return [(s.user_emb, s.user_emb_k), (s.item_emb, s.item_emb_k), (s.utow, s.utow_k), (s.itow, s.itow_k)]
    @torch.no_grad()
    def _mom(s):
        for o, k in s._pairs():
            for po, pk in zip(o.parameters(), k.parameters()): pk.data.mul_(s.m).add_(po.data * (1 - s.m))
    def loss(s, u, p):
        qu = F.normalize(s.utow(s.user_emb(u)), dim=1); qi = F.normalize(s.itow(s.item_emb(p)), dim=1)
        with torch.no_grad():
            s._mom(); ki = F.normalize(s.itow_k(s.item_emb_k(p)), dim=1); ku = F.normalize(s.utow_k(s.user_emb_k(u)), dim=1)
        lab = torch.arange(len(u))
        Qi, Qu = s.qi.clone(), s.qu.clone()
        l1 = F.cross_entropy(torch.cat([qu @ ki.T, qu @ Qi.T], 1) / s.t, lab)
        l2 = F.cross_entropy(torch.cat([qi @ ku.T, qi @ Qu.T], 1) / s.t, lab)
        with torch.no_grad():
            n = len(u); idx = (torch.arange(n) + s.ptr) % s.K; s.qi[idx] = ki; s.qu[idx] = ku; s.ptr = int((s.ptr + n) % s.K)
        return (l1 + l2) / 2
    def embs(s):
        with torch.no_grad():
            return F.normalize(s.utow(s.user_emb.weight), dim=1).numpy(), F.normalize(s.itow(s.item_emb.weight), dim=1).numpy()

class BRIDGE(nn.Module):
    """Two residual propagation streams over a (directed) graph: forward D_out^-1 A, backward D_in^-1 A^T."""
    def __init__(s, nu, ni, n_nodes, fwd, bwd, steps=3):
        super().__init__(); s.nu, s.ni = nu, ni; s.fwd_op, s.bwd_op = fwd, bwd
        s.emb = nn.Embedding(n_nodes, D)
        s.fwd = nn.ModuleList([nn.Linear(D, D) for _ in range(steps)]); s.bwd = nn.ModuleList([nn.Linear(D, D) for _ in range(steps)])
        s.fusion = nn.Sequential(nn.Linear(2 * D, D), nn.LayerNorm(D), nn.ReLU(), nn.Linear(D, D)); s.alpha = nn.Parameter(torch.tensor(0.0))
        xav(s)
    def prop(s, e, A, T):
        h = e; outs = [h]
        for t in T: h = F.relu(h + t(spmm(A, h))); outs.append(h)
        return torch.stack(outs, 0).mean(0)
    def forward(s):
        e = s.emb.weight; a = torch.sigmoid(s.alpha)
        f = s.fusion(torch.cat([a * s.prop(e, s.fwd_op, s.fwd), (1 - a) * s.prop(e, s.bwd_op, s.bwd)], -1))
        return f[:s.nu], f[s.nu:s.nu + s.ni]
    def loss(s, u, p, n):
        Ue, Ie = s(); U, P, N = Ue[u], Ie[p], Ie[n]
        return -F.logsigmoid((U * P).sum(1) - (U * N).sum(1)).mean() + 1e-3 * (s.emb.weight[u].pow(2).sum() + s.emb.weight[s.nu + p].pow(2).sum() + s.emb.weight[s.nu + n].pow(2).sum()) / len(u)
    def embs(s):
        with torch.no_grad(): U, I = s()
        return U.numpy(), I.numpy()

class ORACLE(nn.Module):
    def __init__(s, C, hd=64, t0=0.5):
        super().__init__(); s.C = C; s.norms = nn.ModuleList([nn.BatchNorm1d(1) for _ in range(C)])
        s.encoder = nn.Sequential(nn.Linear(C, hd), nn.LayerNorm(hd), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hd, hd), nn.LayerNorm(hd), nn.ReLU(), nn.Dropout(0.2))
        s.gate = nn.Sequential(nn.Linear(hd, hd), nn.ReLU(), nn.Linear(hd, C)); s.temperature = nn.Parameter(torch.tensor(t0)); s.output = nn.Linear(hd, 1)
    def forward(s, x):
        n = torch.cat([s.norms[c](x[:, c:c + 1]) for c in range(s.C)], 1); h = s.encoder(n)
        g = F.softmax(s.gate(h) / s.temperature.abs().clamp(min=0.1), -1)
        return (n * g).sum(-1) + s.output(h).squeeze(-1), g

# ---------------------------------------------------------------- training helpers
def neg_sample(users, mask, ni):
    neg = np.random.randint(0, ni, len(users)); bad = mask[users, neg]
    while bad.any(): neg[bad] = np.random.randint(0, ni, bad.sum()); bad = mask[users, neg]
    return neg

def train_es(model, lr, bs, step_fn, val_fn, n_train, tag, wd=0.0):
    """Early stopping: validate every EVAL_EVERY epochs, stop after PATIENCE evaluations without improvement."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    best, state, best_ep, losses, bad, t0 = -1, None, 0, [], 0, time.time()
    max_ep = SMOKE or MAX_EPOCHS
    for ep in range(max_ep):
        model.train(); idx = np.random.permutation(n_train); tot = nb = 0
        for s0 in range(0, n_train, bs):
            b = idx[s0:s0 + bs]
            if len(b) < 2: continue
            opt.zero_grad(); l = step_fn(b); l.backward(); opt.step(); tot += l.item(); nb += 1
        losses.append(tot / max(nb, 1))
        if (ep + 1) % EVAL_EVERY == 0 or ep + 1 == max_ep:
            model.eval(); v = val_fn()
            if v > best: best, best_ep, bad, state = v, ep + 1, 0, {k: t.detach().clone() for k, t in model.state_dict().items()}
            else: bad += 1
            if bad >= PATIENCE and ep + 1 >= min(MIN_EPOCHS, max_ep): break
    model.load_state_dict(state); model.eval()
    info = dict(best_val=best, best_epoch=best_ep, epochs_run=len(losses), losses=losses, seconds=time.time() - t0,
                params=int(sum(p.numel() for p in model.parameters())))
    print(f"    {tag} lr={lr}: val NDCG@10={best:.4f} at epoch {best_ep} (ran {len(losses)}; {info['seconds']:.0f}s)", flush=True)
    return info

def pair_scores(fn, C, chunk=20000):
    n, m = C.shape; users = np.repeat(np.arange(n), m); items = C.reshape(-1); out = np.zeros(n * m, dtype=np.float32)
    for s0 in range(0, n * m, chunk): out[s0:s0 + chunk] = fn(users[s0:s0 + chunk], items[s0:s0 + chunk])
    return out.reshape(n, m)

def full_scores(fn, nu, ni, chunk=200000):
    out = np.zeros((nu, ni), dtype=np.float32); per = max(1, chunk // ni)
    for u0 in range(0, nu, per):
        us = np.arange(u0, min(nu, u0 + per)); uu = np.repeat(us, ni); ii = np.tile(np.arange(ni), len(us))
        out[us] = fn(uu, ii).reshape(len(us), ni)
    return out

val_nd = lambda S: float(ndcg_r(ranks_sampled(S)).mean())

# ---------------------------------------------------------------- component training (returns a scorer)
def train_component(name, D_, KG, lr, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    nu, ni = D_['n_users'], D_['n_items']; tu, ti, mask = D_['train_u'], D_['train_i'], D_['train_mask']; ntr = len(tu)
    VC = D_['val_cand']; H, HM = torch.LongTensor(D_['hist']), torch.BoolTensor(D_['hist_mask'])
    T = lambda x: torch.LongTensor(x)
    if name in ('MF-BPR', 'LightGCN', 'BRIDGE', 'BRIDGE-UI', 'MINT'):
        if name == 'MF-BPR': m = MFBPR(nu, ni); step = lambda b: m.loss(T(tu[b]), T(ti[b]), T(neg_sample(tu[b], mask, ni))); bs = BS_GRAPH
        elif name == 'LightGCN': m = LightGCN(nu, ni, KG['lgcn']); step = lambda b: m.loss(T(tu[b]), T(ti[b]), T(neg_sample(tu[b], mask, ni))); bs = BS_GRAPH
        elif name == 'BRIDGE': m = BRIDGE(nu, ni, KG['n_nodes'], KG['fwd_op'], KG['bwd_op']); step = lambda b: m.loss(T(tu[b]), T(ti[b]), T(neg_sample(tu[b], mask, ni))); bs = BS_GRAPH
        elif name == 'BRIDGE-UI': m = BRIDGE(nu, ni, nu + ni, KG['ui_op'], KG['ui_op']); step = lambda b: m.loss(T(tu[b]), T(ti[b]), T(neg_sample(tu[b], mask, ni))); bs = BS_GRAPH
        else: m = MINT(nu, ni); step = lambda b: m.loss(T(tu[b]), T(ti[b])); bs = BS_SEQ
        info = train_es(m, lr, bs, step, lambda: val_nd(pick_emb(*m.embs(), VC)), ntr, name)
        U, I = m.embs(); return info, (lambda C: pick_emb(U, I, C)), (lambda: (U @ I.T).astype(np.float32)), (I if name in ('MF-BPR', 'LightGCN', 'BRIDGE') else None)
    if name == 'DRIP': m = DRIP(nu, ni)
    else: m = NEXIS(nu, ni)
    def fn(u, i):
        with torch.no_grad(): u = T(u); return m(u, T(i), H[u], HM[u]).numpy()
    def step(b):
        u = T(tu[b]); p = T(ti[b]); n = T(neg_sample(tu[b], mask, ni)); Hb = H[u]; Mb = HM[u] & (Hb != p.unsqueeze(1))   # positive masked
        Mb = torch.where(Mb.any(1, keepdim=True), Mb, HM[u])
        ps, ns = m.bpr(u, p, n, Hb, Mb); return -F.logsigmoid(ps - ns).mean()
    info = train_es(m, lr, BS_SEQ, step, lambda: val_nd(pair_scores(fn, VC)), ntr, name, wd=1e-5 if name == 'DRIP' else 0.0)
    return info, (lambda C: pair_scores(fn, C)), (lambda: m.full(np.arange(nu), H, HM).numpy()), None

def pick_emb(U, I, C): return (U[:, None, :] * I[C]).sum(-1)

NEURAL = ['MF-BPR', 'LightGCN', 'DRIP', 'MINT', 'BRIDGE', 'BRIDGE-UI', 'NEXIS']

# ---------------------------------------------------------------- ORACLE fitting and fusion baselines
def fit_oracle(Xf, yf, uf, seed, max_ep=100, patience=10):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed); uu = np.unique(uf); rng.shuffle(uu); ho = np.isin(uf, uu[:int(0.2 * len(uu))])
    mu, sd = Xf[~ho].mean(0), Xf[~ho].std(0) + 1e-8
    Xt, yt = torch.FloatTensor((Xf[~ho] - mu) / sd), torch.FloatTensor(yf[~ho]); Xh, yh = torch.FloatTensor((Xf[ho] - mu) / sd), torch.FloatTensor(yf[ho])
    model = ORACLE(Xf.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=1e-3); crit = nn.BCEWithLogitsLoss()
    best, state, best_ep, bad = 1e9, None, 0, 0
    for ep in range(max_ep):
        model.train(); idx = torch.randperm(len(Xt))
        for s0 in range(0, len(Xt), 1024):
            b = idx[s0:s0 + 1024]
            if len(b) < 2: continue
            opt.zero_grad(); l = crit(model(Xt[b])[0], yt[b]); l.backward(); opt.step()
        model.eval()
        with torch.no_grad(): hl = crit(model(Xh)[0], yh).item()
        if hl < best - 1e-6: best, best_ep, bad, state = hl, ep + 1, 0, {k: t.clone() for k, t in model.state_dict().items()}
        else: bad += 1
        if bad >= patience: break
    model.load_state_dict(state); model.eval(); return model, mu, sd, best_ep

def oracle_full(model, mu, sd, Ffull, chunk_users=64):
    nu, ni, C = Ffull.shape; out = np.zeros((nu, ni), dtype=np.float32); gsum = np.zeros(C); n = 0
    with torch.no_grad():
        for u0 in range(0, nu, chunk_users):
            x = torch.FloatTensor((Ffull[u0:u0 + chunk_users].reshape(-1, C) - mu) / sd); p, g = model(x)
            out[u0:u0 + chunk_users] = p.numpy().reshape(-1, ni); gsum += g.numpy().sum(0); n += len(x)
    return out, gsum / n

def oracle_pairs(model, mu, sd, Fp):
    nu, m, C = Fp.shape
    with torch.no_grad(): p, g = model(torch.FloatTensor((Fp.reshape(-1, C) - mu) / sd))
    return p.numpy().reshape(nu, m), g.numpy().reshape(nu, m, C)

# ---------------------------------------------------------------- one seed
INPUTS = ['ItemKNN', 'Content', 'Popularity', 'LightGCN', 'DRIP', 'MINT', 'BRIDGE', 'NEXIS']

def run_seed(seed, D_, KG, DET, tuned):
    nu, ni = D_['n_users'], D_['n_items']; FC, VC, TC, ex, tgt = D_['fus_cand'], D_['val_cand'], D_['test_cand'], D_['excl_test'], D_['test_item']
    res = {'seed': seed, 'info': {}, 'ranks_full': {}, 'ranks_sampled': {}}
    fus, test_p, full = {}, {}, {}
    for k in ['ItemKNN', 'Content', 'Popularity']: fus[k], full[k] = DET[k]['fus'], DET[k]['full']
    for name in NEURAL:
        info, pairs, fullf, emb = train_component(name, D_, KG, tuned[name], seed)
        res['info'][name] = info; fus[name] = pairs(FC); F_ = fullf(); full[name] = F_
        res['ranks_full'][name] = ranks_full(F_, tgt, ex); res['ranks_sampled'][name] = ranks_sampled(pairs(TC))
        if emb is not None: res.setdefault('item_emb', {})[name] = emb
    np.save(f"{OUT}/mf_full_seed{seed}.npy", full['MF-BPR'].astype(np.float16))
    y = np.tile(np.r_[1.0, np.zeros(N_NEG)], nu); uf = np.repeat(np.arange(nu), N_NEG + 1)
    def fused(names, tag):
        Xf = np.stack([fus[k] for k in names], -1); Ff = np.stack([full[k] for k in names], -1)
        model, mu, sd, ep = fit_oracle(Xf.reshape(-1, len(names)), y, uf, seed)
        S, g = oracle_full(model, mu, sd, Ff); Tp = np.stack([pick(full[k], TC) for k in names], -1)
        St, _ = oracle_pairs(model, mu, sd, Tp)
        res['ranks_full'][tag] = ranks_full(S, tgt, ex); res['ranks_sampled'][tag] = ranks_sampled(St)
        res['info'][tag] = dict(best_epoch=ep); return g
    t0 = time.time(); g = fused(INPUTS, 'SYNAPSE-KG'); res['gates_test_mean'] = g; res['info']['SYNAPSE-KG']['seconds'] = time.time() - t0
    fused(INPUTS + ['MF-BPR'], 'ORACLE + MF-BPR input')
    fused([k if k != 'BRIDGE' else 'BRIDGE-UI' for k in INPUTS], 'w/o KG relations')
    for rm in INPUTS: fused([k for k in INPUTS if k != rm], f'w/o {rm}')
    # fusion baselines on the same inputs and fusion pairs (scores accumulated input by input to limit memory)
    Xf = np.stack([fus[k] for k in INPUTS], -1).reshape(-1, len(INPUTS)); mu, sd = Xf.mean(0), Xf.std(0) + 1e-8
    Tp = np.stack([pick(full[k], TC) for k in INPUTS], -1)
    lr_ = LogisticRegression(max_iter=2000, C=1.0).fit((Xf - mu) / sd, y)
    for tag, wts in [('Mean of inputs', np.full(len(INPUTS), 1.0 / len(INPUTS))), ('Logistic stacking', lr_.coef_[0])]:
        S = np.zeros((nu, ni), dtype=np.float32)
        for j, k in enumerate(INPUTS): S += (wts[j] / sd[j]) * (full[k] - mu[j])
        res['ranks_full'][tag] = ranks_full(S, tgt, ex); res['ranks_sampled'][tag] = ranks_sampled(((Tp - mu) / sd) @ wts); del S
    res['stacking_coef'] = dict(zip(INPUTS, lr_.coef_[0].tolist()))
    best_in = max(INPUTS, key=lambda k: val_nd(fus[k])); res['best_single_input'] = best_in
    res['ranks_full']['Best single input'] = res['ranks_full'].get(best_in, ranks_full(full[best_in], tgt, ex))
    res['ranks_sampled']['Best single input'] = res['ranks_sampled'].get(best_in, ranks_sampled(pick(full[best_in], TC)))
    return res

# ---------------------------------------------------------------- main
def prep():
    t = time.time(); L = load_data(); nu, ni = L['n_users'], L['n_items']
    D_ = build_splits(L['seqs'], nu, ni); D_.update(n_users=nu, n_items=ni, attrs=L['attrs'], texts=L['texts'], raw_users=L['raw_users'],
                                                  raw_items=L['raw_items'], pool_users=L['pool_users'], ts=L['ts'])
    H = np.zeros((nu, HIST_LEN), dtype=np.int64); HM = np.zeros((nu, HIST_LEN), dtype=bool)
    for u in range(nu): h = D_['train_seq'][u][-HIST_LEN:]; H[u, :len(h)] = h; HM[u, :len(h)] = True
    D_['hist'], D_['hist_mask'] = H, HM
    KGd = build_kg(D_); D_['kg'] = {k: KGd[k] for k in ['counts', 'counts_unique', 'n_attr_entities', 'n_nodes']}
    D_['uc'], D_['ic'], D_['tfidf_terms'] = content_features(D_)
    pickle.dump((D_, KGd), open(f"{OUT}/data.pkl", 'wb'))
    print(f"[data] {DATASET}: users={nu} items={ni} train={len(D_['train_u'])} pool_users={L['pool_users']} kg={D_['kg']} ({time.time()-t:.0f}s)", flush=True)
    # deterministic models: hyperparameters selected on validation
    t = time.time(); DET = {'sel': {}, 'val': {}}; tgt, ex = D_['test_item'], D_['excl_test']
    ik = ItemKNN(D_); uk = UserKNN(D_)
    for nm, obj, grid in [('ItemKNN', ik, [20, 50, 100, 200, 500]), ('UserKNN', uk, [20, 50, 100, 200, 500]), ('EASE', None, [50, 100, 200, 500, 1000, 2000, 5000, 10000])]:
        best = None
        for k in grid:
            Sfull = ease_full(D_, k) if nm == 'EASE' else (obj.score_full(k) if nm == 'UserKNN' else None)
            Sv = ik.score_pairs(D_['val_cand'], k) if nm == 'ItemKNN' else pick(Sfull, D_['val_cand'])
            v = val_nd(Sv); DET['val'][f'{nm}_{k}'] = v
            if best is None or v > best[1]: best = (k, v)
        DET['sel'][nm] = best[0]
        Sfull = ik.score_full(best[0]) if nm == 'ItemKNN' else (uk.score_full(best[0]) if nm == 'UserKNN' else ease_full(D_, best[0]))
        DET[nm] = {'full': Sfull, 'fus': pick(Sfull, D_['fus_cand'])}
        print(f"  {nm}: {best[0]} (val {best[1]:.4f})", flush=True)
    pop = np.bincount(D_['train_i'], minlength=ni).astype(np.float32); pop = (pop - pop.min()) / (pop.max() - pop.min() + 1e-8)
    DET['Popularity'] = {'full': np.tile(pop, (nu, 1))}; DET['Popularity']['fus'] = pick(DET['Popularity']['full'], D_['fus_cand'])
    C = (D_['uc'] @ D_['ic'].T).astype(np.float32); DET['Content'] = {'full': C, 'fus': pick(C, D_['fus_cand'])}
    DET['ranks_full'] = {k: ranks_full(DET[k]['full'], tgt, ex) for k in ['Popularity', 'ItemKNN', 'UserKNN', 'Content', 'EASE']}
    DET['ranks_sampled'] = {k: ranks_sampled(pick(DET[k]['full'], D_['test_cand'])) for k in ['Popularity', 'ItemKNN', 'UserKNN', 'Content', 'EASE']}
    pickle.dump(DET, open(f"{OUT}/deterministic.pkl", 'wb'), protocol=4); print(f"[det] {time.time()-t:.0f}s", flush=True)

def load_all():
    D_, KGd = pickle.load(open(f"{OUT}/data.pkl", 'rb')); DET = pickle.load(open(f"{OUT}/deterministic.pkl", 'rb'))
    nu, ni = D_['n_users'], D_['n_items']
    rows, cols = D_['train_u'], D_['train_i'] + nu
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nu + ni, nu + ni)); adj = adj + adj.T
    dinv = sp.diags(np.power(np.asarray(adj.sum(1)).ravel() + 1e-8, -0.5))
    KG = dict(lgcn=SPA(dinv @ adj @ dinv), fwd_op=SPA(KGd['fwd']), bwd_op=SPA(KGd['bwd']), ui_op=SPA(KGd['ui'][:nu + ni, :nu + ni]), n_nodes=KGd['n_nodes'])
    return D_, KG, DET

if __name__ == "__main__":
    arg = sys.argv[1:]
    if arg and arg[0] == 'prep': prep(); sys.exit()
    D_, KG, DET = load_all()
    if arg and arg[0] == 'tune':
        for nm in (arg[1:] or NEURAL):
            tf = f"{OUT}/tuned_{nm}.json"
            if os.path.exists(tf): continue
            r = {}
            for lr in LR_GRID:
                pf = f"{OUT}/tune_{nm}_{lr}.json"
                if os.path.exists(pf): r[lr] = json.load(open(pf)); continue
                i0 = train_component(nm, D_, KG, lr, 0)[0]; i0 = {k: v for k, v in i0.items() if k != 'losses'}
                json.dump(i0, open(pf, 'w')); r[lr] = i0
            T_ = {'lr': max(LR_GRID, key=lambda lr: r[lr]['best_val']), 'val': {str(lr): r[lr]['best_val'] for lr in LR_GRID},
                  'epochs': {str(lr): r[lr]['best_epoch'] for lr in LR_GRID}, 'seconds': {str(lr): r[lr]['seconds'] for lr in LR_GRID}}
            json.dump(T_, open(tf, 'w'), indent=1); print(f"[tune] {nm}: {T_}", flush=True)
        sys.exit()
    if not os.path.exists(f"{OUT}/tuned.json"):
        json.dump({nm: json.load(open(f"{OUT}/tuned_{nm}.json")) for nm in NEURAL}, open(f"{OUT}/tuned.json", 'w'), indent=1)
    tuned = {k: v['lr'] for k, v in json.load(open(f"{OUT}/tuned.json")).items()}
    for s in [int(x) for x in arg]:
        f = f"{OUT}/seed{s}.pkl"
        if os.path.exists(f): continue
        t = time.time(); print(f"=== seed {s}", flush=True); r = run_seed(s, D_, KG, DET, tuned)
        pickle.dump(r, open(f + '.tmp', 'wb'), protocol=4); os.replace(f + '.tmp', f)
        nd = lambda m: ndcg_r(r['ranks_full'][m]).mean()
        print(f"=== seed {s} done in {time.time()-t:.0f}s | full NDCG@10 SYNAPSE={nd('SYNAPSE-KG'):.4f} MF={nd('MF-BPR'):.4f}", flush=True)

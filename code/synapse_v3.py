"""
SYNAPSE-KG v3 re-run with a corrected evaluation protocol.

Faithful to the model definitions in W15_SYNAPSE_KG.ipynb (cell 1, the version whose
outputs are reported in the manuscript), with the following protocol corrections:
  (C1) chronological leave-one-out with a VALIDATION item (second-to-last) and a TEST item (last);
       every model, feature and history is built from the training prefix only;
  (C2) content user profiles and DRIP histories exclude validation/test items
       (the original used the full sequence, i.e. included the test item);
  (C3) ORACLE is fitted on validation pairs (not on in-sample training pairs) with
       early stopping on a held-out 20% of validation users;
  (C4) the number of epochs of every neural model and k of ItemKNN/UserKNN are selected on
       validation NDCG@10; the test split is used exactly once per model and seed;
  (C5) DRIP and NEXIS are scored with their full forward pass (target-aware attention /
       interest attention), as specified by the paper's equations; the original code
       scored them with raw embedding dot-products;
  (C6) LightGCN recomputes propagation for every mini-batch (the original reused one
       forward pass per epoch with retain_graph=True);
  (C7) all models are ranked against the SAME fixed 99 sampled negatives per user
       (negatives exclude every item the user interacted with); ties are counted
       pessimistically;
  (C8) ablations retrain ORACLE without the removed component (the original zeroed the
       component's input to an already-trained ORACLE);
  (C9) 5 training seeds; the data sample is fixed (seed 42) as in the paper.
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

torch.set_num_threads(1)
ML_ROOT = os.environ.get("ML_ROOT", "./ml-1m")   # folder with ratings.dat and movies.dat
OUT = os.environ.get("OUT", "./out")
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cpu")

DATA_SEED = 42          # identical user sample to the paper
EVAL_SEED = 12345       # fixed negatives for validation/test
N_NEG = 99
KS = [5, 10, 20]
EVAL_EVERY = 5
SMOKE_EPOCHS = int(os.environ.get("SMOKE_EPOCHS", "0"))   # >0 only for a quick pipeline test; never for reported runs

# ---------------------------------------------------------------- data (identical preprocessing)
def load_and_preprocess():
    ratings = pd.read_csv(f"{ML_ROOT}/ratings.dat", sep="::", names=["userId", "movieId", "rating", "timestamp"], engine="python", encoding="latin-1")
    movies = pd.read_csv(f"{ML_ROOT}/movies.dat", sep="::", names=["movieId", "title", "genres"], engine="python", encoding="latin-1")
    df = ratings[ratings['rating'] >= 4][['userId', 'movieId', 'timestamp', 'rating']].copy()
    for _ in range(3):
        uc = df.groupby('userId').size(); df = df[df['userId'].isin(uc[uc >= 10].index)]
        ic = df.groupby('movieId').size(); df = df[df['movieId'].isin(ic[ic >= 5].index)]
    rng = np.random.default_rng(DATA_SEED)
    all_users = df['userId'].unique()
    sampled = rng.choice(all_users, int(len(all_users) * 0.35), replace=False)
    df = df[df['userId'].isin(sampled)]
    uc = df.groupby('userId').size(); df = df[df['userId'].isin(uc[uc >= 10].index)]
    users = sorted(df['userId'].unique()); items = sorted(df['movieId'].unique())
    u2i = {u: i for i, u in enumerate(users)}; m2i = {m: i for i, m in enumerate(items)}
    df['user'] = df['userId'].map(u2i); df['item'] = df['movieId'].map(m2i)
    df = df.sort_values(['user', 'timestamp']).reset_index(drop=True)
    seqs = {}
    for u, g in df.groupby('user'):
        seqs[u] = g.sort_values('timestamp')['item'].tolist()
    movies_subset = movies[movies['movieId'].isin(m2i)].copy(); movies_subset['item'] = movies_subset['movieId'].map(m2i)
    return seqs, movies_subset, len(users), len(items)

def build_splits(seqs, n_users, n_items):
    train_seq = {u: s[:-2] for u, s in seqs.items()}
    val_item = np.array([seqs[u][-2] for u in range(n_users)])
    test_item = np.array([seqs[u][-1] for u in range(n_users)])
    rows = np.concatenate([[u] * len(train_seq[u]) for u in range(n_users)])
    cols = np.concatenate([train_seq[u] for u in range(n_users)])
    train_mask = np.zeros((n_users, n_items), dtype=bool); train_mask[rows, cols] = True
    all_mask = train_mask.copy(); all_mask[np.arange(n_users), val_item] = True; all_mask[np.arange(n_users), test_item] = True
    rng = np.random.default_rng(EVAL_SEED)
    def negs():
        out = np.zeros((n_users, N_NEG), dtype=np.int64)
        for u in range(n_users):
            pool = np.flatnonzero(~all_mask[u]); out[u] = rng.choice(pool, N_NEG, replace=False)
        return out
    val_neg = negs(); test_neg = negs()
    return dict(train_seq=train_seq, val_item=val_item, test_item=test_item, train_u=rows, train_i=cols,
                train_mask=train_mask, val_cand=np.column_stack([val_item, val_neg]),
                test_cand=np.column_stack([test_item, test_neg]))

def build_kg_counts(train_seq, movies_subset, n_users, n_items):
    """Same KG construction as the notebook, but from the training prefix only (no val/test items)."""
    rc = defaultdict(int)
    rc['INTERACTED'] = sum(len(s) for s in train_seq.values())
    genres = set(); [genres.update(g.split('|')) for g in movies_subset['genres']]
    rc['HAS_GENRE'] = sum(len(g.split('|')) for g in movies_subset['genres'])
    co = 0
    for u, its in train_seq.items():
        for i in range(len(its)):
            for j in range(i + 1, min(i + 6, len(its))):
                if co < 25000: co += 1
    rc['CO_WATCHED'] = co
    item_genres = {int(r['item']): set(r['genres'].split('|')) for _, r in movies_subset.iterrows()}
    items_list = list(item_genres.keys()); rng = np.random.default_rng(DATA_SEED); rng.shuffle(items_list)
    gs = 0
    for i, a in enumerate(items_list[:500]):
        for b in items_list[i + 1:i + 50]:
            if len(item_genres[a] & item_genres[b]) >= 2 and gs < 10000: gs += 1
    rc['GENRE_SIMILAR'] = gs
    rc['TEMPORAL_NEXT'] = sum(max(0, len(s) - 1) for s in train_seq.values())
    return dict(rc), n_users + n_items + len(genres)

# ---------------------------------------------------------------- metrics
def ranks_from_scores(S):
    """S: [n_users, 100], column 0 = positive. Pessimistic tie handling."""
    pos = S[:, :1]
    return 1 + (S[:, 1:] >= pos).sum(1)

def metrics_from_ranks(r):
    out = {}
    for k in KS:
        out[f'HR@{k}'] = float(np.mean(r <= k))
        out[f'NDCG@{k}'] = float(np.mean(np.where(r <= k, 1.0 / np.log2(r + 1), 0.0)))
    out['MRR@10'] = float(np.mean(np.where(r <= 10, 1.0 / r, 0.0)))
    return out

def ndcg10(r): return np.where(r <= 10, 1.0 / np.log2(r + 1), 0.0)

# ---------------------------------------------------------------- non-parametric components
class ItemKNN:
    def __init__(self, train_u, train_i, n_users, n_items):
        ui = csr_matrix((np.ones(len(train_u)), (train_i, train_u)), shape=(n_items, n_users))
        n = normalize(ui, axis=1); self.sim = (n @ n.T).toarray().astype(np.float32)
        self.hist = defaultdict(list)
        for u, i in zip(train_u, train_i): self.hist[u].append(i)
        self.hist = {u: np.array(v) for u, v in self.hist.items()}
    def score(self, C, k):
        out = np.zeros(C.shape, dtype=np.float32)
        for u in range(C.shape[0]):
            H = self.hist[u]; S = self.sim[C[u]][:, H]
            kk = min(k, S.shape[1])
            out[u] = np.partition(S, -kk, axis=1)[:, -kk:].sum(1)
        return out

class UserKNN:
    def __init__(self, train_u, train_i, n_users, n_items):
        self.R = csr_matrix((np.ones(len(train_u)), (train_u, train_i)), shape=(n_users, n_items))
        n = normalize(self.R, axis=1); self.sim = (n @ n.T).toarray().astype(np.float32)
        self.Rd = self.R.toarray().astype(np.float32)
    def score(self, C, k):
        out = np.zeros(C.shape, dtype=np.float32)
        for u in range(C.shape[0]):
            s = self.sim[u]; idx = np.argsort(s)[-k - 1:-1]; w = s[idx]
            out[u] = (w[:, None] * self.Rd[idx][:, C[u]]).sum(0) / (np.abs(w).sum() + 1e-8)
        return out

def content_features(train_seq, movies_subset, n_users, n_items):
    texts = []
    for i in range(n_items):
        row = movies_subset[movies_subset['item'] == i]
        texts.append(row.iloc[0]['genres'].replace('|', ' ') + ' ' + row.iloc[0]['title'] if len(row) else 'unknown')
    tfidf = TfidfVectorizer(max_features=500, stop_words='english').fit_transform(texts)
    ic = normalize(TruncatedSVD(n_components=64, random_state=DATA_SEED).fit_transform(tfidf))
    uc = np.zeros((n_users, 64))
    for u in range(n_users):
        if train_seq[u]: uc[u] = ic[train_seq[u]].mean(0)     # TRAINING PREFIX ONLY (C2)
    return normalize(uc).astype(np.float32), ic.astype(np.float32)

# ---------------------------------------------------------------- neural models (definitions from the notebook)
class MFBPR(nn.Module):
    def __init__(s, nu, ni, d=64):
        super().__init__(); s.user_emb = nn.Embedding(nu, d); s.item_emb = nn.Embedding(ni, d)
        nn.init.xavier_uniform_(s.user_emb.weight); nn.init.xavier_uniform_(s.item_emb.weight)
    def loss(s, u, p, n):
        U, P, N = s.user_emb(u), s.item_emb(p), s.item_emb(n)
        l = -F.logsigmoid((U * P).sum(1) - (U * N).sum(1)).mean()
        return l + 0.001 * (U.norm(2).pow(2) + P.norm(2).pow(2) + N.norm(2).pow(2)) / len(u)
    def embs(s): return s.user_emb.weight.detach().numpy(), s.item_emb.weight.detach().numpy()

class LightGCN(nn.Module):
    def __init__(s, nu, ni, d=64, L=3):
        super().__init__(); s.nu = nu; s.L = L
        s.user_emb = nn.Embedding(nu, d); s.item_emb = nn.Embedding(ni, d)
        nn.init.xavier_uniform_(s.user_emb.weight); nn.init.xavier_uniform_(s.item_emb.weight)
    def forward(s, A):
        e = torch.cat([s.user_emb.weight, s.item_emb.weight]); out = [e]
        for _ in range(s.L): e = spmm(A, e); out.append(e)
        e = torch.stack(out, 1).mean(1); return e[:s.nu], e[s.nu:]
    def loss(s, A, u, p, n):
        Ue, Ie = s(A); U, P, N = Ue[u], Ie[p], Ie[n]
        l = -F.logsigmoid((U * P).sum(1) - (U * N).sum(1)).mean()
        return l + 0.001 * (s.user_emb.weight[u].norm(2).pow(2) + s.item_emb.weight[p].norm(2).pow(2) + s.item_emb.weight[n].norm(2).pow(2)) / len(u)
    def embs(s, A):
        with torch.no_grad(): U, I = s(A)
        return U.numpy(), I.numpy()

class DRIP(nn.Module):
    def __init__(s, nu, ni, d=64, heads=4):
        super().__init__()
        s.user_emb = nn.Embedding(nu, d); s.item_emb = nn.Embedding(ni, d)
        s.attention = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.1)
        s.query_proj = nn.Linear(d, d)
        s.intent_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(0.2), nn.Linear(d, d))
        for m in s.modules():
            if isinstance(m, nn.Embedding): nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(s, u, i, H, M):
        ue, ie = s.user_emb(u), s.item_emb(i); he = s.item_emb(H)
        q = s.query_proj(ie).unsqueeze(1)
        a, _ = s.attention(q, he, he, key_padding_mask=~M)
        z = s.intent_mlp(torch.cat([ue + a.squeeze(1), ie], -1))
        return (z * ie).sum(-1)

class MINT(nn.Module):
    def __init__(s, nu, ni, d=64, m=0.995, t=0.1):
        super().__init__(); s.m, s.t = m, t
        s.user_emb = nn.Embedding(nu, d); s.item_emb = nn.Embedding(ni, d)
        s.user_emb_k = nn.Embedding(nu, d); s.item_emb_k = nn.Embedding(ni, d)
        s.projector = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        s.user_emb_k.weight.data.copy_(s.user_emb.weight.data); s.item_emb_k.weight.data.copy_(s.item_emb.weight.data)
        for p in list(s.user_emb_k.parameters()) + list(s.item_emb_k.parameters()): p.requires_grad = False
        nn.init.xavier_uniform_(s.user_emb.weight); nn.init.xavier_uniform_(s.item_emb.weight)
    @torch.no_grad()
    def _mom(s):
        s.user_emb_k.weight.data.mul_(s.m).add_(s.user_emb.weight.data * (1 - s.m))
        s.item_emb_k.weight.data.mul_(s.m).add_(s.item_emb.weight.data * (1 - s.m))
    def loss(s, u, p, n):
        q = F.normalize(s.projector(s.user_emb(u) + s.item_emb(p)), dim=-1)
        with torch.no_grad():
            s._mom()
            kp = F.normalize(s.projector(s.user_emb_k(u) + s.item_emb_k(p)), dim=-1)
            kn = F.normalize(s.projector(s.user_emb_k(u) + s.item_emb_k(n)), dim=-1)
        logits = torch.cat([(q * kp).sum(-1, keepdim=True), (q * kn).sum(-1, keepdim=True)], 1) / s.t
        return F.cross_entropy(logits, torch.zeros(len(u), dtype=torch.long))
    def embs(s): return s.user_emb.weight.detach().numpy(), s.item_emb.weight.detach().numpy()

class BRIDGE(nn.Module):
    def __init__(s, nu, ni, d=64, steps=3):
        super().__init__(); s.nu = nu
        s.user_emb = nn.Embedding(nu, d); s.item_emb = nn.Embedding(ni, d)
        s.fwd = nn.ModuleList([nn.Linear(d, d) for _ in range(steps)]); s.bwd = nn.ModuleList([nn.Linear(d, d) for _ in range(steps)])
        s.fusion = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, d))
        s.alpha = nn.Parameter(torch.tensor(0.5))
        nn.init.xavier_uniform_(s.user_emb.weight); nn.init.xavier_uniform_(s.item_emb.weight)
    def prop(s, e, A, T):
        h = e; outs = [h]
        for t in T: h = F.relu(h + t(spmm(A, h))); outs.append(h)
        return torch.stack(outs, 0).mean(0)
    def forward(s, A, At):
        e = torch.cat([s.user_emb.weight, s.item_emb.weight]); a = torch.sigmoid(s.alpha)
        f = s.fusion(torch.cat([a * s.prop(e, A, s.fwd), (1 - a) * s.prop(e, At, s.bwd)], -1))
        return f[:s.nu], f[s.nu:]
    def embs(s, A, At):
        with torch.no_grad(): U, I = s(A, At)
        return U.numpy(), I.numpy()

class NEXIS(nn.Module):
    def __init__(s, nu, ni, d=64, P=4):
        super().__init__(); s.P = P
        s.user_emb = nn.Embedding(nu, d); s.item_emb = nn.Embedding(ni, d)
        s.ext = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)) for _ in range(P)])
        s.att = nn.Sequential(nn.Linear(2 * d, d), nn.Tanh(), nn.Linear(d, 1))
        for m in s.modules():
            if isinstance(m, nn.Embedding): nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(s, u, i):
        ue, ie = s.user_emb(u), s.item_emb(i)
        c = torch.stack([f(ue) for f in s.ext], 1)
        w = F.softmax(s.att(torch.cat([c, ie.unsqueeze(1).expand(-1, s.P, -1)], -1)).squeeze(-1), -1)
        return ((c * w.unsqueeze(-1)).sum(1) * ie).sum(-1)

class ORACLE(nn.Module):
    def __init__(s, C, hd=64, t0=0.5):
        super().__init__(); s.C = C
        s.norms = nn.ModuleList([nn.BatchNorm1d(1) for _ in range(C)])
        s.encoder = nn.Sequential(nn.Linear(C, hd), nn.LayerNorm(hd), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(hd, hd), nn.LayerNorm(hd), nn.ReLU(), nn.Dropout(0.2))
        s.gate = nn.Sequential(nn.Linear(hd, hd), nn.ReLU(), nn.Linear(hd, C))
        s.temperature = nn.Parameter(torch.tensor(t0)); s.output = nn.Linear(hd, 1)
    def forward(s, x):
        n = torch.cat([s.norms[c](x[:, c:c + 1]) for c in range(s.C)], 1)
        h = s.encoder(n); g = F.softmax(s.gate(h) / s.temperature.abs().clamp(min=0.1), -1)
        return (n * g).sum(-1) + s.output(h).squeeze(-1), g

# ---------------------------------------------------------------- helpers
def neg_sample(users, mask, n_items):
    neg = np.random.randint(0, n_items, len(users)); bad = mask[users, neg]
    while bad.any():
        neg[bad] = np.random.randint(0, n_items, bad.sum()); bad = mask[users, neg]
    return neg

class SPA:
    """Sparse operator backed by scipy CSR (same math as torch.sparse.mm; ~4x faster on 1 CPU)."""
    def __init__(self, M): self.M = M.tocsr().astype(np.float32); self.MT = self.M.T.tocsr()
class _SpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, op):
        ctx.op = op; return torch.from_numpy(op.M @ x.detach().numpy())
    @staticmethod
    def backward(ctx, g): return torch.from_numpy(ctx.op.MT @ g.contiguous().numpy()), None
def spmm(op, x): return _SpMM.apply(x, op)
def sp_tensor(M): return SPA(M)

def emb_scores(U, I, C): return (U[:, None, :] * I[C]).sum(-1)

def batched_pair_scores(fn, C, chunk=20000):
    n, m = C.shape; users = np.repeat(np.arange(n), m); items = C.reshape(-1); out = np.zeros(n * m, dtype=np.float32)
    for s in range(0, n * m, chunk):
        out[s:s + chunk] = fn(users[s:s + chunk], items[s:s + chunk])
    return out.reshape(n, m)

def train_loop(model, params_lr, epochs, bs, step_fn, val_fn, n_train, tag):
    if SMOKE_EPOCHS: epochs = min(epochs, SMOKE_EPOCHS)
    opt = torch.optim.Adam(*params_lr) if isinstance(params_lr, tuple) else params_lr
    best, best_state, best_ep, losses = -1, None, 0, []
    for ep in range(epochs):
        model.train(); idx = np.random.permutation(n_train); tot = 0; nb = 0
        for s in range(0, n_train, bs):
            b = idx[s:s + bs]; opt.zero_grad(); l = step_fn(b); l.backward(); opt.step(); tot += l.item(); nb += 1
        losses.append(tot / nb)
        if (ep + 1) % EVAL_EVERY == 0 or ep + 1 == epochs:
            model.eval(); v = val_fn()
            if v > best: best, best_ep, best_state = v, ep + 1, {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    print(f"    {tag}: best val NDCG@10={best:.4f} at epoch {best_ep}/{epochs}", flush=True)
    return losses, best_ep, best

def fit_oracle(Xv, yv, users_v, seed, epochs=100):
    """Fit ORACLE on validation pairs; early stopping on 20% held-out validation users."""
    rng = np.random.default_rng(seed); uu = np.unique(users_v); rng.shuffle(uu)
    ho = set(uu[:int(0.2 * len(uu))]); m_ho = np.array([u in ho for u in users_v])
    mu, sd = Xv[~m_ho].mean(0), Xv[~m_ho].std(0) + 1e-8
    Xt = torch.FloatTensor((Xv[~m_ho] - mu) / sd); yt = torch.FloatTensor(yv[~m_ho])
    Xh = torch.FloatTensor((Xv[m_ho] - mu) / sd); yh = torch.FloatTensor(yv[m_ho])
    model = ORACLE(Xv.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=1e-3); crit = nn.BCEWithLogitsLoss()
    best, state, best_ep, losses = 1e9, None, 0, []
    for ep in range(epochs):
        model.train(); idx = torch.randperm(len(Xt)); tot = 0; nb = 0
        for s in range(0, len(Xt), 256):
            b = idx[s:s + 256]
            if len(b) < 2: continue
            opt.zero_grad(); p, _ = model(Xt[b]); l = crit(p, yt[b]); l.backward(); opt.step(); tot += l.item(); nb += 1
        losses.append(tot / nb)
        model.eval()
        with torch.no_grad(): hl = crit(model(Xh)[0], yh).item()
        if hl < best: best, best_ep, state = hl, ep + 1, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state); model.eval()
    return model, mu, sd, losses, best_ep

def oracle_score(model, mu, sd, feats):
    n, m, C = feats.shape
    with torch.no_grad():
        p, g = model(torch.FloatTensor((feats.reshape(-1, C) - mu) / sd))
    return p.numpy().reshape(n, m), g.numpy().reshape(n, m, C)

# ---------------------------------------------------------------- one seed
def run_seed(seed, D, feats_static, knn_k):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    nu, ni = D['n_users'], D['n_items']; tu, ti = D['train_u'], D['train_i']; mask = D['train_mask']
    VC, TC = D['val_cand'], D['test_cand']; ntr = len(tu)
    val_nd = lambda S: float(ndcg10(ranks_from_scores(S)).mean())
    A = D['A']; At = D['At']; H, HM = D['hist'], D['hist_mask']
    res = {'losses': {}, 'best_epoch': {}, 'val_best': {}}
    comp_val, comp_test = {}, {}

    # MF-BPR
    t0 = time.time(); mf = MFBPR(nu, ni)
    def mf_step(b):
        u = tu[b]; return mf.loss(torch.LongTensor(u), torch.LongTensor(ti[b]), torch.LongTensor(neg_sample(u, mask, ni)))
    res['losses']['MF-BPR'], res['best_epoch']['MF-BPR'], res['val_best']['MF-BPR'] = train_loop(
        mf, (mf.parameters(), 1e-3), 40, 1024, mf_step, lambda: val_nd(emb_scores(*mf.embs(), VC)), ntr, 'MF-BPR')
    U, I = mf.embs(); comp_val['MF-BPR'], comp_test['MF-BPR'] = emb_scores(U, I, VC), emb_scores(U, I, TC); res['emb_MF'] = I
    # LightGCN (per-batch propagation, C6)
    lg = LightGCN(nu, ni)
    def lg_step(b):
        u = tu[b]; return lg.loss(A, torch.LongTensor(u), torch.LongTensor(ti[b]), torch.LongTensor(neg_sample(u, mask, ni)))
    res['losses']['LightGCN'], res['best_epoch']['LightGCN'], res['val_best']['LightGCN'] = train_loop(
        lg, (lg.parameters(), 1e-3), 30, 1024, lg_step, lambda: val_nd(emb_scores(*lg.embs(A), VC)), ntr, 'LightGCN')
    U, I = lg.embs(A); comp_val['LightGCN'], comp_test['LightGCN'] = emb_scores(U, I, VC), emb_scores(U, I, TC); res['emb_LGCN'] = I
    # DRIP (training-prefix histories, full forward scoring; C2, C5)
    dr = DRIP(nu, ni); Ht, HMt = torch.LongTensor(H), torch.BoolTensor(HM)
    def dr_fn(u, i):
        with torch.no_grad():
            u = torch.LongTensor(u); return dr(u, torch.LongTensor(i), Ht[u], HMt[u]).numpy()
    def dr_step(b):
        u = tu[b]; ut = torch.LongTensor(u); n = torch.LongTensor(neg_sample(u, mask, ni))
        ps = dr(ut, torch.LongTensor(ti[b]), Ht[ut], HMt[ut]); ns = dr(ut, n, Ht[ut], HMt[ut])
        return -F.logsigmoid(ps - ns).mean()
    res['losses']['DRIP'], res['best_epoch']['DRIP'], res['val_best']['DRIP'] = train_loop(
        dr, torch.optim.Adam(dr.parameters(), lr=5e-4, weight_decay=1e-5), 50, 512, dr_step,
        lambda: val_nd(batched_pair_scores(dr_fn, VC)), ntr, 'DRIP')
    comp_val['DRIP'], comp_test['DRIP'] = batched_pair_scores(dr_fn, VC), batched_pair_scores(dr_fn, TC)
    # MINT
    mi = MINT(nu, ni)
    def mi_step(b):
        u = tu[b]; return mi.loss(torch.LongTensor(u), torch.LongTensor(ti[b]), torch.LongTensor(neg_sample(u, mask, ni)))
    res['losses']['MINT'], res['best_epoch']['MINT'], res['val_best']['MINT'] = train_loop(
        mi, (mi.parameters(), 5e-4), 40, 512, mi_step, lambda: val_nd(emb_scores(*mi.embs(), VC)), ntr, 'MINT')
    U, I = mi.embs(); comp_val['MINT'], comp_test['MINT'] = emb_scores(U, I, VC), emb_scores(U, I, TC)
    # BRIDGE
    br = BRIDGE(nu, ni)
    def br_step(b):
        u = tu[b]; p = ti[b]; n = neg_sample(u, mask, ni); Ue, Ie = br(A, At)
        Uu, P, N = Ue[u], Ie[p], Ie[n]
        l = -F.logsigmoid((Uu * P).sum(1) - (Uu * N).sum(1)).mean()
        return l + 0.001 * (br.user_emb.weight[u].norm(2).pow(2) + br.item_emb.weight[p].norm(2).pow(2) + br.item_emb.weight[n].norm(2).pow(2)) / len(u)
    res['losses']['BRIDGE'], res['best_epoch']['BRIDGE'], res['val_best']['BRIDGE'] = train_loop(
        br, (br.parameters(), 1e-3), 40, 1024, br_step, lambda: val_nd(emb_scores(*br.embs(A, At), VC)), ntr, 'BRIDGE')
    U, I = br.embs(A, At); comp_val['BRIDGE'], comp_test['BRIDGE'] = emb_scores(U, I, VC), emb_scores(U, I, TC); res['emb_BRIDGE'] = I
    # NEXIS (full forward scoring; C5)
    nx = NEXIS(nu, ni)
    def nx_fn(u, i):
        with torch.no_grad(): return nx(torch.LongTensor(u), torch.LongTensor(i)).numpy()
    def nx_step(b):
        u = torch.LongTensor(tu[b]); n = torch.LongTensor(neg_sample(tu[b], mask, ni))
        return -F.logsigmoid(nx(u, torch.LongTensor(ti[b])) - nx(u, n)).mean()
    res['losses']['NEXIS'], res['best_epoch']['NEXIS'], res['val_best']['NEXIS'] = train_loop(
        nx, (nx.parameters(), 5e-4), 40, 512, nx_step, lambda: val_nd(batched_pair_scores(nx_fn, VC)), ntr, 'NEXIS')
    comp_val['NEXIS'], comp_test['NEXIS'] = batched_pair_scores(nx_fn, VC), batched_pair_scores(nx_fn, TC)
    print(f"    neural components trained in {time.time()-t0:.0f}s", flush=True)

    # ORACLE: same 8 inputs, same order as the notebook
    names = ['ItemKNN', 'Content', 'Popularity', 'DRIP', 'MINT', 'BRIDGE', 'NEXIS', 'LightGCN']
    for k in ['ItemKNN', 'Content', 'Popularity']:
        comp_val[k], comp_test[k] = feats_static[k]
    Fv = np.stack([comp_val[k] for k in names], -1); Ft = np.stack([comp_test[k] for k in names], -1)
    # validation pairs: val item + first 10 fixed validation negatives (1:10 as in the notebook)
    Xv = Fv[:, :11, :].reshape(-1, 8); yv = np.tile(np.r_[1.0, np.zeros(10)], nu); uv = np.repeat(np.arange(nu), 11)
    orc, mu, sd, olosses, oep = fit_oracle(Xv, yv, uv, seed)
    S, G = oracle_score(orc, mu, sd, Ft)
    res['losses']['ORACLE'] = olosses; res['best_epoch']['ORACLE'] = oep
    res['gates_test_mean'] = G.reshape(-1, 8).mean(0)
    ranks = {m: ranks_from_scores(comp_test[m]) for m in ['MF-BPR', 'LightGCN', 'DRIP', 'MINT', 'BRIDGE', 'NEXIS']}
    ranks['SYNAPSE-KG'] = ranks_from_scores(S)
    # Ablation: retrain ORACLE without the component (C8)
    abl = {}
    for rm in ['DRIP', 'MINT', 'BRIDGE', 'NEXIS']:
        keep = [j for j, k in enumerate(names) if k != rm]
        o2, mu2, sd2, _, _ = fit_oracle(Xv[:, keep], yv, uv, seed)
        S2, _ = oracle_score(o2, mu2, sd2, Ft[:, :, keep]); abl[f'w/o {rm}'] = ranks_from_scores(S2)
    res['ranks'] = ranks; res['abl_ranks'] = abl
    # Embedding diagnostics computed here so the stored seed result stays compact (identical maths
    # to the former aggregate step): genre silhouette on all items, and the 500 rows used for t-SNE.
    from sklearn.metrics import silhouette_score
    ms = D['movies_subset']; prim = ms.set_index('item')['genres'].str.split('|').str[0].reindex(range(ni)).values
    tsne_idx = np.random.default_rng(42).choice(ni, 500, replace=False)
    res['sil'] = {}; res['tsne_emb'] = {}
    for k, key in [('BRIDGE', 'emb_BRIDGE'), ('LightGCN', 'emb_LGCN'), ('MF-BPR', 'emb_MF')]:
        res['sil'][k] = float(silhouette_score(res[key], prim, metric='cosine'))
        res['tsne_emb'][k] = res[key][tsne_idx].astype(np.float32)
        del res[key]
    return res

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    PREP_ONLY = len(sys.argv) > 1 and sys.argv[1] == "prep"
    seeds = [] if PREP_ONLY else ([int(x) for x in sys.argv[1:]] or [0, 1, 2, 3, 4])
    cache = f"{OUT}/data.pkl"
    if os.path.exists(cache):
        D = pickle.load(open(cache, 'rb'))
    else:
        t = time.time(); seqs, movies_subset, nu, ni = load_and_preprocess()
        full_train = sum(len(s) - 1 for s in seqs.values())
        print(f"[check] users={nu} items={ni} train(original LOO)={full_train} test={nu}", flush=True)
        D = build_splits(seqs, nu, ni); D.update(n_users=nu, n_items=ni)
        D['kg_counts'], D['n_entities'] = build_kg_counts(D['train_seq'], movies_subset, nu, ni)
        D['kg_counts_original_protocol'] = None
        uc, ic = content_features(D['train_seq'], movies_subset, nu, ni); D['uc'], D['ic'] = uc, ic
        D['movies_subset'] = movies_subset
        # histories: last 20 training items (C2)
        H = np.zeros((nu, 20), dtype=np.int64); HM = np.zeros((nu, 20), dtype=bool)
        for u in range(nu):
            h = D['train_seq'][u][-20:]; H[u, :len(h)] = h; HM[u, :len(h)] = True
        D['hist'], D['hist_mask'] = H, HM
        pickle.dump(D, open(cache, 'wb')); print(f"[data] prepared in {time.time()-t:.0f}s", flush=True)
    nu, ni = D['n_users'], D['n_items']
    rows = D['train_u']; cols = D['train_i'] + nu
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nu + ni, nu + ni)); adj = adj + adj.T
    dinv = sp.diags(np.power(np.asarray(adj.sum(1)).flatten() + 1e-8, -0.5)); An = dinv @ adj @ dinv
    D['A'] = sp_tensor(An); D['At'] = sp_tensor(An.T.tocsr())     # identical to notebook; A is symmetric
    D['A_symmetric_maxabsdiff'] = float(abs(An - An.T).max())

    # deterministic components (computed once): KNN k selected on validation (C4)
    det = f"{OUT}/deterministic.pkl"
    if os.path.exists(det):
        DET = pickle.load(open(det, 'rb'))
    else:
        t = time.time(); DET = {'knn_val': {}}
        ik = ItemKNN(D['train_u'], D['train_i'], nu, ni); uk = UserKNN(D['train_u'], D['train_i'], nu, ni)
        for name, obj in [('ItemKNN', ik), ('UserKNN', uk)]:
            best = None
            for k in [20, 50, 100, 200]:
                v = float(ndcg10(ranks_from_scores(obj.score(D['val_cand'], k))).mean()); DET['knn_val'][(name, k)] = v
                if best is None or v > best[1]: best = (k, v)
            DET[name + '_k'] = best[0]
            DET[name] = (obj.score(D['val_cand'], best[0]), obj.score(D['test_cand'], best[0]))
            print(f"  {name}: k={best[0]} (val NDCG@10={best[1]:.4f})", flush=True)
        # the ORACLE input 'ItemKNN' uses the notebook's k=100 so the fused model is unchanged
        DET['ItemKNN_k100'] = (ik.score(D['val_cand'], 100), ik.score(D['test_cand'], 100))
        pop = np.bincount(D['train_i'], minlength=ni).astype(np.float32); popn = (pop - pop.min()) / (pop.max() - pop.min() + 1e-8)
        DET['Popularity'] = (popn[D['val_cand']], popn[D['test_cand']])
        DET['Content'] = (emb_scores(D['uc'], D['ic'], D['val_cand']), emb_scores(D['uc'], D['ic'], D['test_cand']))
        pickle.dump(DET, open(det, 'wb')); print(f"[det] {time.time()-t:.0f}s", flush=True)
    feats_static = {'ItemKNN': DET['ItemKNN_k100'], 'Content': DET['Content'], 'Popularity': DET['Popularity']}
    for s in seeds:
        f = f"{OUT}/seed{s}.pkl"
        if os.path.exists(f): continue
        t = time.time(); print(f"=== seed {s}", flush=True)
        r = run_seed(s, D, feats_static, None)
        pickle.dump(r, open(f + '.tmp', 'wb')); os.replace(f + '.tmp', f)
        print(f"=== seed {s} done in {time.time()-t:.0f}s | SYNAPSE NDCG@10={ndcg10(r['ranks']['SYNAPSE-KG']).mean():.4f} MF={ndcg10(r['ranks']['MF-BPR']).mean():.4f}", flush=True)

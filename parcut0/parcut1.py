#!/usr/bin/env python3
# --------------------------------------------------------------------
#  Hierarchical Reasoning Model for Parallel Cut Discovery (V1.1)
# --------------------------------------------------------------------
#  Adapted from the HRM Sudoku solver to discover parallel cuts in
#  directly-follows graphs, inspired by the Inductive Miner algorithm.
#  The model learns to group activities that share identical dependency
#  footprints.
#
#  Core adaptations:
#    • Synthetic data generator for perfect parallel cut graphs.
#    • Dataset yields flattened adjacency matrices and group labels.
#    • Model architecture reasons over activities, not grid cells.
#    • Loss and accuracy metrics tailored for group prediction.
# --------------------------------------------------------------------
#  (c) 2025 – public-domain demo
#
#  Original paper reference: "Hierarchical Reasoning Model"
#                            arXiv:2506.21734
# --------------------------------------------------------------------
import argparse, math, os, random, time
from pathlib import Path
from typing import Tuple

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# --------------------------- utils ----------------------------------
torch.set_float32_matmul_precision("high")  # ↑ GPU matmul throughput

def set_seed(s: int):
    """Set seed for reproducibility."""
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# --------------------------------------------------------------------
# 1.  Constants & utility
# --------------------------------------------------------------------
MAX_ACTS     = 50                       # hard upper-bound
MAX_GROUPS   = 5                        # generate ∈[2,5] ⊗-groups
PAD_VALUE    = MAX_GROUPS               # “unused” label for absent groups

def sorted_groups(labels: torch.Tensor) -> torch.Tensor:
    """Relabel groups so that 0 ≺ 1 ≺ 2… is determined by the
       *lowest* activity-index that belongs to each group."""
    # Find unique, sorted group labels present in the tensor
    uniq = labels.unique(sorted=True)
    # Create a mapping from old labels to new, ordered labels (0, 1, 2, ...)
    order = {g.item(): i for i, g in enumerate(uniq)}
    # Apply the mapping to the original labels tensor
    return torch.tensor([order[int(g)] for g in labels], dtype=torch.long)

# --------------------------------------------------------------------
# 2.  Synthetic DF-graph generator
# --------------------------------------------------------------------
def gen_parallel_df(num_groups: int = None,
                    max_acts: int = MAX_ACTS) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (A, g) where
      A – (n,n) 0/1 adjacency of the directly-follows graph
      g – length-n group indices 0,1,…,k-1 (deterministic)
    """
    if num_groups is None:
        num_groups = random.randint(2, MAX_GROUPS)

    # Split |A| activities into k non-empty groups (size ≥1)
    n = random.randint(num_groups, max_acts)
    # Draw k-1 cut points and sort them to make contiguous segments
    cuts = sorted(random.sample(range(1, n), num_groups - 1))
    sizes = [b - a for a, b in zip([0] + cuts, cuts + [n])]
    groups = [list(range(sum(sizes[:i]), sum(sizes[:i+1]))) for i in range(num_groups)]

    A = torch.zeros((n, n), dtype=torch.float32)

    # Fully connect every pair of *distinct* groups in both directions
    for g1 in range(num_groups):
        for g2 in range(g1 + 1, num_groups):
            for i in groups[g1]:
                for j in groups[g2]:
                    A[i, j] = 1.
                    A[j, i] = 1.

    g = torch.cat([torch.full((len(grp),), k, dtype=torch.long)
                   for k, grp in enumerate(groups)])

    return A, g

# --------------------------------------------------------------------
# 3.  Dataset
# --------------------------------------------------------------------
class ParallelCutDataset(torch.utils.data.Dataset):
    """
    Each sample:
        x  – flattened adjacency (n²,), left-padded to MAX_ACTS² with -1
        y  – group labels   (MAX_ACTS,)  values 0-4 or PAD_VALUE
        m  – mask for valid activities  (MAX_ACTS,) 1/0
    """
    def __init__(self, num_samples: int):
        self.samples = [gen_parallel_df() for _ in tqdm(range(num_samples), desc="Generating graphs")]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        A, g = self.samples[idx]
        n    = A.size(0)

        # ---------- inputs ----------
        adj_flat   = A.flatten()
        pad_len    = MAX_ACTS*MAX_ACTS - adj_flat.numel()
        x          = F.pad(adj_flat, (0, pad_len), value=-1)        # sentinel

        # ---------- targets ----------
        g_sorted   = sorted_groups(g)
        y          = F.pad(g_sorted,
                           (0, MAX_ACTS - n),
                           value=PAD_VALUE)
        m          = torch.zeros(MAX_ACTS, dtype=torch.long)
        m[:n]      = 1
        return x.long(), y.long(), m

# ------------------ normalisation / init helpers --------------------
class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm (bias-free, ε=1e-8)."""
    def __init__(self, d, eps=1e-8):
        super().__init__(); self.scale = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * x / rms

def lecun_init_(mod: nn.Module):
    """LeCun-normal init for Linear/Embedding; bias=None by design."""
    if isinstance(mod, nn.Linear):
        fan_in = mod.weight.size(1)
        nn.init.trunc_normal_(mod.weight, std=1.0 / math.sqrt(fan_in), a=-2, b=2)
    elif isinstance(mod, nn.Embedding):
        fan_in = mod.weight.size(0)
        nn.init.trunc_normal_(mod.weight, std=1.0 / math.sqrt(fan_in), a=-2, b=2)


# ------------------ Transformer block (fast variant) ----------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=256, dropout=0.1,
                 use_rms=True):
        super().__init__()
        self.norm1 = RMSNorm(d_model) if use_rms else nn.LayerNorm(d_model)
        self.norm2 = RMSNorm(d_model) if use_rms else nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0,
                                          batch_first=True, bias=False)
        # SwiGLU FFN: 2×d_ff then GLU halves back to d_ff
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2*d_ff, bias=False),
            nn.GLU(dim=-1),          # (x•sigmoid) gate, faster than GELU
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                      need_weights=False)[0]
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x

# ------------------------ HRM encoder stack -------------------------
class EncoderStack(nn.Module):
    def __init__(self, layers=2, **kw):
        super().__init__(); self.seq = nn.ModuleList([TransformerBlock(**kw) for _ in range(layers)])
    def forward(self, x):
        for blk in self.seq: x = blk(x)
        return x

# --------------------------------------------------------------------
# 4.  Model adaptations (Corrected)
# --------------------------------------------------------------------
class HRMParallel(nn.Module):
    """
    • Sequence-length = MAX_ACTS² (flattened adjacency).
    • Vocabulary = {-1, 0, 1}  (pad, no-arc, arc) → 3 tokens
    • Predict a group label (0-4 / PAD) for every *activity*,
      so we need a second mapping from cell-tokens to activities.
      The simplest way: summarise each row with mean-pooling.
    """
    def __init__(self, d_model=128, n_heads=4, d_ff=256,
                 depth_L=2, depth_H=2, N=2, T=2, dropout=0.1,
                 use_rms=True):
        super().__init__()
        self.d_model = d_model
        # --- FIX: Store N and T as instance attributes ---
        self.N, self.T = N, T
        self.embed   = nn.Embedding(3, d_model)          # -1,0,1 → 0,1,2
        self.pos     = nn.Embedding(MAX_ACTS*MAX_ACTS, d_model)

        self.L = EncoderStack(layers=depth_L, d_model=d_model,
                              n_heads=n_heads, d_ff=d_ff,
                              dropout=dropout, use_rms=use_rms)
        self.H = EncoderStack(layers=depth_H, d_model=d_model,
                              n_heads=n_heads, d_ff=d_ff,
                              dropout=dropout, use_rms=use_rms)

        self.head = nn.Linear(d_model, MAX_GROUPS + 1, bias=False)  # +PAD

        self.apply(lecun_init_)

    def forward(self, z, x):
        tok = (x + 1).clamp(0, 2)                        # (-1→0, 0→1, 1→2)
        e   = self.embed(tok) + self.pos.weight.unsqueeze(0)

        B = e.size(0)
        zH, zL = (torch.zeros(B, MAX_ACTS*MAX_ACTS, self.d_model,
                              device=e.device),
                  torch.zeros(B, MAX_ACTS*MAX_ACTS, self.d_model,
                              device=e.device)) if z is None else z

        # --- FIX: Use self.N and self.T ---
        with torch.no_grad():
            for i in range(self.N * self.T - 1):
                zL = self.L(zL + zH + e)
                if (i+1) % self.T == 0: zH = self.H(zH + zL)
        zL  = self.L(zL + zH + e)        # final grad step
        zH  = self.H(zH + zL)

        # -------- pool rows → activities (B,MAX_ACTS,d_model) --------
        acts = zH.view(B, MAX_ACTS, MAX_ACTS, self.d_model).mean(2)

        return (zH, zL), self.head(acts)                 # logits per activity

# --------------------------------------------------------------------
# 5.  Loss
# --------------------------------------------------------------------
def ce_loss_parallel(logits, tgt, m):
    """Cross-entropy loss that ignores padding."""
    return F.cross_entropy(logits.view(-1, MAX_GROUPS + 1),
                           tgt.view(-1),
                           ignore_index=PAD_VALUE,
                           reduction='mean')

# ----------------- accuracy helpers ---------------------------------
@torch.no_grad()
def parallel_cut_accuracies(logits, tgt, m):
    """Calculates graph-level and activity-level accuracy."""
    p = logits.argmax(-1)
    
    # Correctly predicted activities (only consider masked/valid activities)
    correct_acts = ((p == tgt) * m).sum().item()
    total_acts = m.sum().item()

    # A graph is solved if all its activities are correctly classified
    graph_correct = ((p == tgt) * m).view(p.size(0), -1).sum(-1) == m.view(p.size(0), -1).sum(-1)
    graph_acc = graph_correct.float().mean().item()

    return graph_acc, correct_acts, total_acts

# ---------------------------- train loop ----------------------------
def run_epoch(model, loader, device, loss_fn,
              opt=None, scaler=None, sched=None, clip=None):
    is_train = opt is not None
    model.train(is_train)
    agg = {'loss':0,'graph':0,'act_c':0,'act_n':0,'batches':0}
    pbar = tqdm(loader, desc=("Train" if is_train else "Eval"), leave=False)

    for x,y,m in pbar:
        x,y,m = [t.to(device,non_blocking=True) for t in (x,y,m)]
        with torch.set_grad_enabled(is_train):
            _, out = model(None, x) # Start with zero state for each graph
            loss = loss_fn(out, y, m)

        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if sched: sched.step()

        g_acc, a_corr, a_total = parallel_cut_accuracies(out, y, m)
        agg['loss']+=loss.item(); agg['graph']+=g_acc
        agg['act_c']+=a_corr;   agg['act_n']+=a_total; agg['batches']+=1
        if agg['act_n'] > 0:
            pbar.set_postfix(loss=f"{agg['loss']/agg['batches']:.3f}",
                             acc=f"{(agg['act_c']/agg['act_n'])*100:.1f}%")

    l = agg['loss']/agg['batches']
    gr = agg['graph']/agg['batches']
    act = agg['act_c']/agg['act_n'] if agg['act_n'] > 0 else 0
    return l, gr, act

# ------------------------------ main -------------------------------
def main(a):
    set_seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() and not a.cpu else 'cpu'

    # Model, Loss, and Data
    loss_fn = ce_loss_parallel
    model = HRMParallel(d_model=a.d_model, n_heads=a.n_heads, d_ff=a.d_ff,
                        depth_L=a.depth_L, depth_H=a.depth_H, N=a.N, T=a.T,
                        dropout=a.dropout, use_rms=not a.no_rms)
    if a.compile: model = torch.compile(model, mode="reduce-overhead")
    model.to(device); print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_ds = ParallelCutDataset(a.num_train)
    test_ds  = ParallelCutDataset(a.num_test)
    dl_tr = torch.utils.data.DataLoader(train_ds, batch_size=a.batch, shuffle=True,
                                        num_workers=a.workers, pin_memory=True, drop_last=True)
    dl_te = torch.utils.data.DataLoader(test_ds, batch_size=a.batch, shuffle=False,
                                        num_workers=a.workers, pin_memory=True)

    # Optimizer, Scheduler, and Scaler
    opt = optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9,0.95), weight_decay=a.wd)
    steps = math.ceil(len(dl_tr))*a.epochs
    sched = OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1, anneal_strategy='cos')
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda' and not a.no_amp))

    print(f"\nTraining on {device.upper()} ...")
    for ep in range(a.epochs):
        t0=time.time()
        tr = run_epoch(model, dl_tr, device, loss_fn, opt, scaler, sched, a.clip)
        te = run_epoch(model, dl_te, device, loss_fn)

        print(f"Epoch {ep:02d} | "
              f"Train L {tr[0]:.4f}, Graph {tr[1]*100:5.1f}%, Act {tr[2]*100:5.2f}% | "
              f"Test L {te[0]:.4f}, Graph {te[1]*100:5.1f}%, Act {te[2]*100:5.2f}% | "
              f"LR {sched.get_last_lr()[0]:.2e} | {(time.time()-t0):.1f}s")

        if te[1] > 0.999: print("Early-stop: graph acc ≥99.9%"); break

# ----------------------------- CLI ----------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser("HRM for Parallel Cut Discovery")
    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch",  type=int, default=64)
    p.add_argument("--num-train", type=int, default=5000)
    p.add_argument("--num-test",  type=int, default=1000)
    # optimisation
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    # model
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--depth-L", type=int, default=2)
    p.add_argument("--depth-H", type=int, default=2)
    p.add_argument("--N", type=int, default=2, help="Hierarchical outer loops")
    p.add_argument("--T", type=int, default=2, help="Hierarchical inner loops")
    p.add_argument("--dropout", type=float, default=0.1)
    # toggles
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no-rms", action="store_true", help="use LayerNorm instead")
    p.add_argument("--no-amp", action="store_true", help="disable automatic mixed precision")
    # system
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    
    # Tiny demo / smoke-test
    print("--- Running smoke-test ---")
    set_seed(0)
    ds  = ParallelCutDataset(1)
    x,y,m = ds[0]
    print(f"Sample adjacency shape: {x.shape}, groups: {y[:m.sum()].tolist()}")
    model = HRMParallel(N=args.N, T=args.T) # Pass N,T to model
    _, out = model(None, x.unsqueeze(0))
    print(f"Prediction shape: {out.shape}")
    print("--- Smoke-test complete ---\n")

    main(args)

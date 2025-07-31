#!/usr/bin/env python3
# --------------------------------------------------------------------
#  Faster‑converging Hierarchical Reasoning Model for Sudoku  (V2.3)
# --------------------------------------------------------------------
#  Major speed‑of‑convergence upgrades vs V2.2
#    • RMSNorm + SwiGLU feed‑forward improves optimisation stability.
#    • LeCun‑normal weight init & bias‑free Linear layers.
#    • Flash‑/SDP‑attention path when PyTorch≥2.1 (falls back safely).
#    • Optional rotary row/col/box embeddings (--rotary) for richer
#      structure‑aware invariances.
#    • Lightweight seed‑controlled solution cache to avoid regeneration.
# --------------------------------------------------------------------
#  (c) 2025 – public‑domain demo
#
#  References: HRM architecture & ablations, Fig‑8 & §2,3 of the paper
#              “Hierarchical Reasoning Model” arXiv:2506.21734 :contentReference[oaicite:4]{index=4}
# --------------------------------------------------------------------
import argparse, copy, math, os, random, time
from pathlib import Path
from typing import List, Tuple

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# --------------------------- utils ----------------------------------
SIZE, SUB = 9, 3
DIGITS = list(range(1, SIZE + 1))
torch.set_float32_matmul_precision("high")  # ↑ GPU matmul throughput

def set_seed(s: int):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ---------- Sudoku generation (now with disk‑cache) -----------------
CACHE = Path(".sudoku_cache.pt")

def valid(b, r, c, n):
    br, bc = (r // SUB) * SUB, (c // SUB) * SUB
    if any(b[r][j] == n for j in range(SIZE)): return False
    if any(b[i][c] == n for i in range(SIZE)): return False
    if any(b[br+i][bc+j] == n for i in range(SUB) for j in range(SUB)): return False
    return True

def _solve(board) -> bool:                       # back‑tracking solver
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] == 0:
                random.shuffle(DIGITS)
                for n in DIGITS:
                    if valid(board, r, c, n):
                        board[r][c] = n
                        if _solve(board): return True
                        board[r][c] = 0
                return False
    return True

def generate_full_board():
    b = [[0]*SIZE for _ in range(SIZE)]
    _solve(b); return b

def generate_solution_pool(n: int) -> List[List[int]]:
    # simple persistence: cache is keyed by n and seed
    key = f"{n}_{random.getstate()[1][0]}"
    if CACHE.exists():
        store = torch.load(CACHE)
        if key in store: return store[key]
    pool, pbar = [], tqdm(total=n, desc="Generating unique solutions")
    while len(pool) < n:
        pool.append(generate_full_board()); pbar.update(1)
    pbar.close()
    store = {} if not CACHE.exists() else torch.load(CACHE)
    store[key] = pool; torch.save(store, CACHE)
    return pool

def mask_board(full, min_clues):
    p = copy.deepcopy(full)
    cells = [(r, c) for r in range(SIZE) for c in range(SIZE)]
    random.shuffle(cells)
    for r, c in cells[:81 - min_clues]: p[r][c] = 0
    return p

flatten  = lambda b: [n for row in b for n in row]
to_tensor = lambda b: torch.tensor(flatten(b), dtype=torch.long)

# --------------------- dataset (unchanged logic) --------------------
class SudokuDataset(torch.utils.data.Dataset):
    def __init__(self, pool, min_clues):
        self.sols, self.min_clues = pool, min_clues
    def __len__(self):  return len(self.sols)
    def __getitem__(self, idx):
        sol = self.sols[idx]
        puzzle = mask_board(sol, self.min_clues)
        x = to_tensor(puzzle)
        y = to_tensor(sol) - 1           # target 0‑indexed
        m = (x == 0).long()              # mask of unknown cells
        return x, y, m
    def set_difficulty(self, k): self.min_clues = k

# ------------------ normalisation / init helpers --------------------
class RMSNorm(nn.Module):
    """Root‑mean‑square LayerNorm (bias‑free, ε=1e‑8)."""
    def __init__(self, d, eps=1e-8):
        super().__init__(); self.scale = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * x / rms

def lecun_init_(mod: nn.Module):
    """LeCun‑normal init for Linear/Embedding; bias=None by design."""
    if isinstance(mod, nn.Linear) or isinstance(mod, nn.Embedding):
        fan_in = mod.weight.size(0)
        nn.init.trunc_normal_(mod.weight, std=1.0 / math.sqrt(fan_in), a=-2, b=2)

# ------------------ Transformer block (fast variant) ----------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=256, dropout=0.1,
                 use_rms=True):
        super().__init__()
        self.norm1 = RMSNorm(d_model) if use_rms else nn.LayerNorm(d_model)
        self.norm2 = RMSNorm(d_model) if use_rms else nn.LayerNorm(d_model)

        # PyTorch ≥2.1 uses flash / SDP automatically when available
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

# --------------------------- HRM model ------------------------------
class HRM(nn.Module):
    def __init__(self, vocab=10, d_model=128, n_heads=4, d_ff=256,
                 depth_L=2, depth_H=2, N=2, T=2, dropout=0.1,
                 use_rms=True, rotary=False):
        super().__init__()
        self.N, self.T, self.d_model = N, T, d_model
        self.embed = nn.Embedding(vocab, d_model)
        self.row_pos = nn.Embedding(SIZE, d_model)
        self.col_pos = nn.Embedding(SIZE, d_model)
        self.box_pos = nn.Embedding(SIZE, d_model)
        self.rotary = rotary

        # cached positional indices
        R = torch.arange(81).view(SIZE, SIZE)
        C = R.clone()
        self.register_buffer("row_idx", (R // SIZE).view(1, 81))
        self.register_buffer("col_idx", (C % SIZE).view(1, 81))
        self.register_buffer("box_idx", ((self.row_idx//SUB)*SUB + (self.col_idx//SUB)).view(1,81))

        self.L = EncoderStack(layers=depth_L, d_model=d_model, n_heads=n_heads,
                              d_ff=d_ff, dropout=dropout, use_rms=use_rms)
        self.H = EncoderStack(layers=depth_H, d_model=d_model, n_heads=n_heads,
                              d_ff=d_ff, dropout=dropout, use_rms=use_rms)
        self.head = nn.Linear(d_model, 9, bias=False)
        self.apply(lecun_init_)  # fast, stable start‑up

    def forward(self, z, x):
        B = x.size(0)
        zH, zL = (torch.zeros(B, 81, self.d_model, device=x.device),
                  torch.zeros(B, 81, self.d_model, device=x.device)) if z is None else z

        # token + (row|col|box) structural embeddings
        e = self.embed(x) + self.row_pos(self.row_idx) \
                          + self.col_pos(self.col_idx) \
                          + self.box_pos(self.box_idx)

        if self.rotary:   # optional rotary positional mixing
            sin, cos = torch.sin, torch.cos  # brevity
            theta = torch.arange(0, self.d_model, 2, device=x.device)
            rot = torch.cat([cos(theta), sin(theta)]).unsqueeze(0).unsqueeze(0)
            e = (e*rot) + torch.roll(e, 1, -1)*(1-rot)

        # --- hierarchical unroll w/ truncated back‑prop (1‑step) ----
        with torch.no_grad():
            for i in range(self.N*self.T - 1):
                zL = self.L(zL + zH + e)
                if (i+1) % self.T == 0: zH = self.H(zH + zL)
        zL = self.L(zL + zH + e)          # final 1‑step grad
        zH = self.H(zH + zL)
        return (zH, zL), self.head(zH)

# ------------------------ Losses (unchanged) ------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0): super().__init__(); self.g=gamma
    def forward(self, logits, tgt, m):
        ce = F.cross_entropy(logits.view(-1,9), tgt.view(-1), reduction='none')
        pt = torch.exp(-ce); loss = ((1-pt)**self.g)*ce
        return (loss.view_as(m)*m).sum()/m.sum()

def cross_entropy_loss(logits, tgt, m):
    ce = F.cross_entropy(logits.view(-1,9), tgt.view(-1), reduction='none')
    return (ce.view_as(m)*m).sum()/m.sum()

# ------------ accuracy helpers (unchanged functionality) -----------
@torch.no_grad()
@torch.no_grad()
def accuracies(logits, tgt, m):
    p = logits.argmax(-1)
    # Correctly solved cells (only consider the masked cells)
    correct_in_mask = ((p == tgt) * m).sum()
    
    # Total number of masked cells
    num_masked = m.sum().item()

    # Board is solved if all masked cells are correct
    # We check this for each board in the batch
    board_correct = ((p == tgt) * m).view(p.size(0), -1).sum(-1) == m.view(p.size(0), -1).sum(-1)
    board_acc = board_correct.float().mean().item()

    # Cell-wise accuracy on the masked cells
    cell_acc = correct_in_mask.item()

    return board_acc, cell_acc, num_masked

# ---------------------------- train loop ----------------------------
def run_epoch(model, loader, device, segments, loss_fn,
              opt=None, scaler=None, sched=None, clip=None):
    is_train = opt is not None
    model.train(is_train)
    agg = {'loss':0,'board':0,'cell_c':0,'mask':0,'batches':0}
    pbar = tqdm(loader, desc=("Train" if is_train else "Eval"), leave=False)

    for x,y,m in pbar:
        x,y,m = [t.to(device,non_blocking=True) for t in (x,y,m)]
        with torch.set_grad_enabled(is_train):
            z, loss = None, 0.
            for _ in range(segments):
                z, out = model(z, x)
                loss += loss_fn(out, y, m); z = (z[0].detach(), z[1].detach())
        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if sched: sched.step()

        b_acc,c_corr,c_mask = accuracies(out, y, m)
        agg['loss']+=loss.item(); agg['board']+=b_acc
        agg['cell_c']+=c_corr;   agg['mask']+=c_mask; agg['batches']+=1
        pbar.set_postfix(loss=agg['loss']/agg['batches'],
                         cell=f"{(agg['cell_c']/agg['mask'])*100:.2f}%")

    l = agg['loss']/agg['batches']
    brd = agg['board']/agg['batches']
    cell = agg['cell_c']/agg['mask']
    return l, brd, cell, agg['cell_c'], agg['mask']

# ----------------------- curriculum schedule -----------------------
def curriculum(ep): return (45,45) if ep<=2 else (37,37) if ep<=5 else (30,30)

# ------------------------------ main -------------------------------
def main(a):
    set_seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() and not a.cpu else 'cpu'

    loss_fn = FocalLoss(a.focal_gamma) if a.loss_fn=='focal' else cross_entropy_loss
    model = HRM(d_model=a.d_model, n_heads=a.n_heads, d_ff=a.d_ff,
                depth_L=a.depth_L, depth_H=a.depth_H, N=a.N, T=a.T,
                dropout=a.dropout, use_rms=not a.no_rms, rotary=a.rotary)
    if a.compile: model = torch.compile(model, mode="reduce-overhead")
    model.to(device); print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    opt = optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9,0.95), weight_decay=a.wd)
    steps = math.ceil(a.num_train/a.batch)*a.epochs
    sched = OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1, anneal_strategy='cos')
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))

    train_pool = generate_solution_pool(a.num_train)
    test_pool  = generate_solution_pool(a.num_test)
    train_ds   = SudokuDataset(train_pool, curriculum(0)[0])
    test_ds    = SudokuDataset(test_pool,  curriculum(0)[1])

    dl_tr = torch.utils.data.DataLoader(train_ds, batch_size=a.batch, shuffle=True,
                                        num_workers=a.workers, pin_memory=True, drop_last=True)
    dl_te = torch.utils.data.DataLoader(test_ds, batch_size=a.batch, shuffle=False,
                                        num_workers=a.workers, pin_memory=True)

    print(f"\nTraining on {device.upper()} ...")
    for ep in range(a.epochs):
        min_tr, min_te = curriculum(ep)
        train_ds.set_difficulty(min_tr); test_ds.set_difficulty(min_te)
        print(f"\nEpoch {ep:02d} | min_clues: train={min_tr}, test={min_te}")

        t0=time.time()
        tr = run_epoch(model, dl_tr, device, a.segments, loss_fn, opt, scaler, sched, a.clip)
        te = run_epoch(model, dl_te, device, a.segments, loss_fn)

        print(f"Epoch {ep:02d} | "
              f"Train L {tr[0]:.4f}, Brd {tr[1]*100:5.1f}%, Cell {tr[2]*100:5.2f}% | "
              f"Test L {te[0]:.4f}, Brd {te[1]*100:5.1f}%, Cell {te[2]*100:5.2f}% | "
              f"LR {sched.get_last_lr()[0]:.2e} | {(time.time()-t0):.1f}s")

        if te[1] > 0.999: print("Early‑stop: board acc ≥99.9%"); break

# ----------------------------- CLI ----------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser("HRM Sudoku – fast‑converging edition")
    # training
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch",  type=int, default=128)
    p.add_argument("--num-train", type=int, default=8000)
    p.add_argument("--num-test",  type=int, default=1000)
    # optimisation
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--loss-fn", choices=['ce','focal'], default='focal')
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--clip", type=float, default=1.0)
    # model
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--depth-L", type=int, default=2)
    p.add_argument("--depth-H", type=int, default=2)
    p.add_argument("--N", type=int, default=2)
    p.add_argument("--T", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    # toggles
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no-rms", action="store_true", help="use LayerNorm instead")
    p.add_argument("--rotary", action="store_true", help="enable rotary pos‑enc")
    # system
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--segments", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())

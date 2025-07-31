#!/usr/bin/env python3
# -------------------------------------------------------------------
# Faster‑converging Hierarchical Reasoning Model for Sudoku (V2.1)
# -------------------------------------------------------------------
#   (c) 2025 – public‑domain demo
#
# Engineering Optimizations:
#   1. Pre-generates solutions to avoid slow re-generation during training.
#   2. Uses `torch.compile()` for JIT-compilation speedup.
#   3. Optimized DataLoader with parallel workers and pinned memory.
#
# Machine Learning Model Improvements:
#   1. Structural Positional Embeddings for row, column, and box identities.
#   2. Dropout Regularization and Pre-Layer Normalization for stable training.
#
# New in this version:
#   - Added detailed cell-level accuracy tracking (correct cells / total blanks)
#     for both training and evaluation to provide a more granular view of
#     learning progress.
# -------------------------------------------------------------------
import random, math, time, copy, argparse, os
from typing import List, Tuple
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# ---------------- Sudoku utilities (unchanged) ----------------
SIZE, SUB = 9, 3
DIGITS = list(range(1, SIZE + 1))

def valid(b, r, c, n):
    box_r, box_c = (r // SUB) * SUB, (c // SUB) * SUB
    if any(b[r][j] == n for j in range(SIZE)): return False
    if any(b[i][c] == n for i in range(SIZE)): return False
    if any(b[box_r + i][box_c + j] == n for i in range(SUB) for j in range(SUB)): return False
    return True

def solve(board) -> bool:
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] == 0:
                random.shuffle(DIGITS)
                for n in DIGITS:
                    if valid(board, r, c, n):
                        board[r][c] = n
                        if solve(board): return True
                        board[r][c] = 0
                return False
    return True

def generate_full_board():
    b = [[0] * SIZE for _ in range(SIZE)]
    solve(b)
    return b

def generate_solution_pool(num_puzzles: int) -> List[List[int]]:
    print(f"Generating {num_puzzles} unique Sudoku solutions... (this may take a moment)")
    pool = []
    with tqdm(total=num_puzzles, desc="Generating Solutions") as pbar:
        while len(pool) < num_puzzles:
            pool.append(generate_full_board())
            pbar.update(1)
    return pool

def mask_board(full_board, min_clues):
    p = copy.deepcopy(full_board)
    cells = [(r, c) for r in range(SIZE) for c in range(SIZE)]
    random.shuffle(cells)
    num_to_remove = 81 - min_clues
    for i in range(num_to_remove):
        r, c = cells[i]
        p[r][c] = 0
    return p

def flatten(b): return [n for row in b for n in row]
def to_tensor(b): return torch.tensor(flatten(b), dtype=torch.long)

# --------------- Optimized Dataset with curriculum ---------------------
class SudokuDataset(torch.utils.data.Dataset):
    def __init__(self, solution_pool: List[List[int]], min_clues: int):
        self.solutions = solution_pool
        self.min_clues = min_clues

    def __len__(self):
        return len(self.solutions)

    def __getitem__(self, idx):
        solution_board = self.solutions[idx]
        puzzle_board = mask_board(solution_board, self.min_clues)
        puzzle_tensor = to_tensor(puzzle_board)
        solution_tensor = to_tensor(solution_board) - 1
        mask_tensor = (puzzle_tensor == 0).long()
        return puzzle_tensor, solution_tensor, mask_tensor
    
    def set_difficulty(self, min_clues: int):
        self.min_clues = min_clues

# --- REVISED HRM Blocks (with structural bias & regularization) ---
class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normed_x = self.norm1(x)
        h, _ = self.attn(normed_x, normed_x, normed_x, need_weights=False)
        x = x + self.dropout(h)
        normed_x = self.norm2(x)
        ff_out = self.ff(normed_x)
        x = x + self.dropout(ff_out)
        return x

class EncoderStack(nn.Module):
    def __init__(self, layers=2, **kw):
        super().__init__()
        self.seq = nn.ModuleList([TransformerBlock(**kw) for _ in range(layers)])
    def forward(self, x):
        for blk in self.seq: x = blk(x)
        return x

class HRM(nn.Module):
    def __init__(self, vocab=10, d_model=128, n_heads=4, d_ff=256, depth_L=2, depth_H=2, N=2, T=2, dropout=0.1):
        super().__init__()
        self.N, self.T, self.d_model = N, T, d_model
        self.embed = nn.Embedding(vocab, d_model)
        self.row_pos_embed = nn.Embedding(SIZE, d_model)
        self.col_pos_embed = nn.Embedding(SIZE, d_model)
        self.box_pos_embed = nn.Embedding(SIZE, d_model)
        
        row_indices = torch.arange(81).view(SIZE, SIZE) // SIZE
        self.register_buffer("row_indices", row_indices.view(1, 81))
        col_indices = torch.arange(81).view(SIZE, SIZE) % SIZE
        self.register_buffer("col_indices", col_indices.view(1, 81))
        box_indices = (self.row_indices // SUB) * SUB + (self.col_indices // SUB)
        self.register_buffer("box_indices", box_indices.view(1, 81))

        self.L = EncoderStack(layers=depth_L, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.H = EncoderStack(layers=depth_H, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.head = nn.Linear(d_model, 9)

    def forward(self, z, x):
        B = x.size(0)
        if z is None:
            zH = zL = torch.zeros(B, 81, self.head.in_features, device=x.device)
        else:
            zH, zL = z
            
        digit_emb = self.embed(x)
        row_emb = self.row_pos_embed(self.row_indices)
        col_emb = self.col_pos_embed(self.col_indices)
        box_emb = self.box_pos_embed(self.box_indices)
        xemb = digit_emb + row_emb + col_emb + box_emb
        
        with torch.no_grad():
            for i in range(self.N * self.T - 1):
                zL = self.L(zL + zH + xemb)
                if (i + 1) % self.T == 0: zH = self.H(zH + zL)
        
        zL = self.L(zL + zH + xemb)
        zH = self.H(zH + zL)
        logits = self.head(zH)
        return (zH, zL), logits

# -------- Loss & Accuracy Functions (with cell-level detail) --------
def ce_loss(logits, target, mask):
    ce = nn.functional.cross_entropy(logits.view(-1, 9), target.view(-1), reduction='none').view_as(mask)
    return (ce * mask).sum() / mask.sum()

def calculate_accuracies(logits, target, mask) -> Tuple[float, int, int]:
    """Calculates both board-level and cell-level accuracies."""
    with torch.no_grad():
        pred = logits.argmax(-1)
        
        # Board accuracy: all 81 cells must be correct
        board_solved = (pred == target).view(pred.size(0), -1).all(dim=1)
        board_acc = board_solved.float().mean().item()

        # Cell accuracy: correct predictions only on the blank cells
        correct_cells = (pred == target) * mask
        num_correct = correct_cells.sum().item()
        num_masked = mask.sum().item()
        
    return board_acc, num_correct, num_masked

# -------------- Refactored Train / eval loops -----------------------------
def train_one_epoch(model, loader, opt, scaler, sched, device, segments, max_gn):
    model.train()
    total_loss, total_board_acc = 0., 0.
    total_correct_cells, total_masked_cells, num_batches = 0, 0, 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for x, y, m in pbar:
        x, y, m = [t.to(device, non_blocking=True) for t in (x, y, m)]
        
        z, loss = None, 0.
        for _ in range(segments):
            z, logits = model(z, x)
            loss += ce_loss(logits, y, m)
            z = (z[0].detach(), z[1].detach())

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_gn)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        if sched: sched.step()
        
        board_acc, num_correct, num_masked = calculate_accuracies(logits, y, m)
        total_loss += loss.item()
        total_board_acc += board_acc
        total_correct_cells += num_correct
        total_masked_cells += num_masked
        num_batches += 1
        
        pbar.set_postfix(
            loss=total_loss/num_batches,
            cell_acc=f"{(total_correct_cells/total_masked_cells)*100:.2f}%"
        )
    
    epoch_loss = total_loss / num_batches
    epoch_board_acc = total_board_acc / num_batches
    epoch_cell_acc = total_correct_cells / total_masked_cells if total_masked_cells > 0 else 0
    return epoch_loss, epoch_board_acc, epoch_cell_acc, total_correct_cells, total_masked_cells

def evaluate(model, loader, device, segments):
    model.eval()
    total_loss, total_board_acc = 0., 0.
    total_correct_cells, total_masked_cells, num_batches = 0, 0, 0
    
    with torch.no_grad():
        for x, y, m in tqdm(loader, desc="Evaluating", leave=False):
            x, y, m = [t.to(device, non_blocking=True) for t in (x, y, m)]
            
            z, loss = None, 0.
            for _ in range(segments):
                z, logits = model(z, x)
                loss += ce_loss(logits, y, m)
                z = (z[0].detach(), z[1].detach())
            
            board_acc, num_correct, num_masked = calculate_accuracies(logits, y, m)
            total_loss += loss.item()
            total_board_acc += board_acc
            total_correct_cells += num_correct
            total_masked_cells += num_masked
            num_batches += 1

    epoch_loss = total_loss / num_batches
    epoch_board_acc = total_board_acc / num_batches
    epoch_cell_acc = total_correct_cells / total_masked_cells if total_masked_cells > 0 else 0
    return epoch_loss, epoch_board_acc, epoch_cell_acc, total_correct_cells, total_masked_cells

# ------------------------ main -------------------------------
def curriculum(epoch):
    if epoch <= 2: return 45, 45
    if epoch <= 5: return 37, 37
    return 30, 30

def main(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    
    model = HRM(
        d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
        depth_L=args.depth_L, depth_H=args.depth_H,
        N=args.N, T=args.T, dropout=args.dropout
    )

    if args.compile:
        print("Compiling the model with torch.compile()... (first batch will be slow)")
        model = torch.compile(model, mode="reduce-overhead")
    model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {param_count:,} trainable parameters.")

    opt = optim.AdamW(model.parameters(), lr=args.peak_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(args.num_train / args.batch)
    sched = OneCycleLR(opt, max_lr=args.peak_lr,
                     total_steps=steps_per_epoch * args.epochs,
                     pct_start=0.1, anneal_strategy='cos')
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    train_solutions = generate_solution_pool(args.num_train)
    test_solutions = generate_solution_pool(args.num_test)

    min_clues_train, min_clues_test = curriculum(0)
    train_dataset = SudokuDataset(train_solutions, min_clues_train)
    test_dataset = SudokuDataset(test_solutions, min_clues_test)

    dl_tr = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_te = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False,
                                        num_workers=args.num_workers, pin_memory=True)
    
    print(f"\nStarting training on {device.upper()}...")
    for ep in range(args.epochs):
        t0 = time.time()
        
        min_clues_train, min_clues_test = curriculum(ep)
        train_dataset.set_difficulty(min_clues_train)
        test_dataset.set_difficulty(min_clues_test)
        print(f"\nEpoch {ep:02d} | Difficulty (min_clues): Train={min_clues_train}, Test={min_clues_test}")

        tr_loss, tr_brd, tr_cell, tr_corr, tr_mask = train_one_epoch(
            model, dl_tr, opt, scaler, sched, device, args.segments, args.clip)
        
        te_loss, te_brd, te_cell, te_corr, te_mask = evaluate(
            model, dl_te, device, args.segments)
        
        print(f"Epoch {ep:02d} | Train Loss: {tr_loss:.4f}, Board Acc: {tr_brd*100:5.1f}%, Cell Acc: {tr_cell*100:5.2f}% ({tr_corr:,}/{tr_mask:,}) | "
              f"Test Loss: {te_loss:.4f}, Board Acc: {te_brd*100:5.1f}%, Cell Acc: {te_cell*100:5.2f}% ({te_corr:,}/{te_mask:,}) | "
              f"LR: {sched.get_last_lr()[0]:.1e} | "
              f"Time: {time.time()-t0:.2f}s")
        
        if te_brd > 0.999:
            print("Test board accuracy reached >99.9%. Stopping early.")
            break

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Efficient training for a Hierarchical Reasoning Model on Sudoku.")
    p.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    p.add_argument("--batch", type=int, default=128, help="Batch size.")
    p.add_argument("--num-train", type=int, default=8000, help="Number of puzzles in the training set.")
    p.add_argument("--num-test", type=int, default=1000, help="Number of puzzles in the test set.")
    p.add_argument("--peak-lr", type=float, default=1e-3, help="Peak learning rate for OneCycleLR.")
    p.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    
    p.add_argument("--d-model", type=int, default=128, help="Model dimension.")
    p.add_argument("--n-heads", type=int, default=4, help="Number of attention heads.")
    p.add_argument("--d-ff", type=int, default=256, help="Dimension of the feed-forward layer.")
    p.add_argument("--depth-L", type=int, default=2, help="Number of layers in the L encoder.")
    p.add_argument("--depth-H", type=int, default=2, help="Number of layers in the H encoder.")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for regularization.")
    
    p.add_argument("--N", type=int, default=2, help="Model parameter N.")
    p.add_argument("--T", type=int, default=2, help="Model parameter T.")
    p.add_argument("--segments", type=int, default=3, help="Number of refinement segments per step.")
    
    p.add_argument("--clip", type=float, default=1.0, help="Gradient clipping value.")
    p.add_argument("--cpu", action="store_true", help="Force use of CPU even if CUDA is available.")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile() for JIT speedup.")
    p.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1), help="Number of worker processes for data loading.")
    main(p.parse_args())

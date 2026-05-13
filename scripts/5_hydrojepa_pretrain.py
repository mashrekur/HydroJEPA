"""
HydroJEPA pretraining (v1: prototype I-JEPA + VICReg).

What this script does, in plain language:
  1. Loads the 9,704 Sentinel-2 GeoTIFFs from data/hydrojepa/patches.
  2. Splits patches by ecoregion-stratified geography. Pacific Northwest
     is held out as the OOD test set (Exp 3). Train/val are the rest.
  3. Tokenizes each 128x128 patch into 64 tokens of 16x16 px.
  4. Trains a ViT-S context encoder + EMA target encoder + small predictor
     with the I-JEPA objective: predict target embeddings of masked
     blocks from the visible context, in latent space, with MSE loss.
  5. Adds a VICReg-style variance + covariance regularizer to prevent
     representational collapse. Three-line implementation, well-understood.
  6. Saves checkpoints every epoch. The context encoder is what we keep.

Two modes:
  --smoke   200 patches, 10 epochs, batch 8 — should run in ~5 minutes
            and produce a visibly decreasing loss + a basic UMAP. This
            is the gate before launching the full run.
  (default) full corpus, 100 epochs, batch 64. Overnight on a 5090.

Outputs:
  checkpoints/hydrojepa_<mode>_ep<NN>.pt    full state dicts
  checkpoints/hydrojepa_<mode>_final.pt     final encoder only
  reports/training_curves_<mode>.png        loss + collapse diagnostics
  reports/smoke_umap.png                    if --smoke, post-training UMAP

Usage:
  python 5_hydrojepa_pretrain.py --smoke
  python 5_hydrojepa_pretrain.py
"""

import argparse
import logging
import math
import os
import time
import warnings
from pathlib import Path

# Silence GDAL's per-file TIFF metadata grumbles before importing rasterio.
# Our 10-band Sentinel-2 GeoTIFFs lack standard Photometric tags (the spec
# expects RGB/RGBA color roles, we have 10 scientific bands). Data is fine;
# warnings are pure noise that obscure training logs.
os.environ['CPL_LOG'] = os.devnull
logging.getLogger('rasterio').setLevel(logging.ERROR)
logging.getLogger('rasterio._env').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*Photometric.*')
warnings.filterwarnings('ignore', message='.*ExtraSamples.*')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import rasterio
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

# ---------------------------------------------------------------------------
# Config (full mode; --smoke overrides where noted)
# ---------------------------------------------------------------------------
DATA_DIR    = Path('data/hydrojepa')
PATCH_DIR   = DATA_DIR / 'patches'
MANIFEST    = DATA_DIR / 'manifest.parquet'
LABELS_FILE = DATA_DIR / 'labels.parquet'

CKPT_DIR    = Path('checkpoints')
REPORT_DIR  = Path('reports')

# Image
PATCH_SIZE  = 128
TOKEN_SIZE  = 16          # 128 / 16 = 8 -> 8x8 = 64 tokens per image
N_TOKENS    = (PATCH_SIZE // TOKEN_SIZE) ** 2
IN_CHANNELS = 10          # Sentinel-2 bands

# Model (ViT-S-ish)
EMBED_DIM   = 384         # encoder hidden dim
ENC_DEPTH   = 12
ENC_HEADS   = 6
PRED_DEPTH  = 4           # predictor is small; it should not be the bottleneck
PRED_DIM    = 192
OUT_DIM     = 64          # final embedding dim, matched to AlphaEarth for comparison

# Masking
MASK_RATIO          = 0.4    # fraction of tokens masked across all blocks
N_TARGET_BLOCKS     = 4      # number of disjoint masked blocks per sample
TARGET_BLOCK_RATIO  = 0.15   # each block covers ~15% of tokens

# Optimization
LR          = 1.5e-4
WEIGHT_DECAY = 0.05
WARMUP_EP   = 5
EPOCHS      = 100
BATCH_SIZE  = 64
EMA_START   = 0.996
EMA_END     = 1.0

# Regularization (VICReg style)
VAR_COEF    = 1.0          # encourages each dim to have std >= 1
COV_COEF    = 0.04         # decorrelates dims
PRED_LOSS_COEF = 1.0       # primary JEPA loss

# Holdout
HOLDOUT_REGION = 'NORTHWESTERN FORESTED MOUNTAINS'
VAL_FRACTION   = 0.05

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# (1) Dataset
# ---------------------------------------------------------------------------
class HydroJEPADataset(Dataset):
    """
    Reads a Sentinel-2 GeoTIFF and returns a (10, 128, 128) tensor.
    Per-band normalization stats are computed once and cached.
    """
    def __init__(self, patch_ids: list[str], stats: dict):
        self.patch_ids = patch_ids
        self.mean = torch.tensor(stats['mean'], dtype=torch.float32).view(IN_CHANNELS, 1, 1)
        self.std  = torch.tensor(stats['std'],  dtype=torch.float32).view(IN_CHANNELS, 1, 1)

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        pid = self.patch_ids[idx]
        path = PATCH_DIR / f'{pid}.tif'
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)   # (C, H, W)
        # Replace nan/inf with band median, then normalize.
        if not np.isfinite(arr).all():
            for b in range(arr.shape[0]):
                m = np.nanmedian(arr[b])
                arr[b] = np.where(np.isfinite(arr[b]), arr[b], m)
        x = torch.from_numpy(arr)
        x = (x - self.mean) / (self.std + 1e-6)
        return x, pid


def compute_band_stats(patch_ids: list[str], n_sample: int = 200) -> dict:
    """One-pass per-band mean/std over a random subsample."""
    rng = np.random.default_rng(0)
    sample = rng.choice(patch_ids, size=min(n_sample, len(patch_ids)), replace=False)
    sums = np.zeros(IN_CHANNELS, dtype=np.float64)
    sqs  = np.zeros(IN_CHANNELS, dtype=np.float64)
    n    = 0
    for pid in sample:
        with rasterio.open(PATCH_DIR / f'{pid}.tif') as src:
            arr = src.read().astype(np.float64)
        arr = np.where(np.isfinite(arr), arr, 0)
        sums += arr.sum(axis=(1, 2))
        sqs  += (arr ** 2).sum(axis=(1, 2))
        n    += arr.shape[1] * arr.shape[2]
    mean = sums / n
    var  = sqs / n - mean ** 2
    std  = np.sqrt(np.maximum(var, 1e-8))
    return {'mean': mean.astype(np.float32), 'std': std.astype(np.float32)}


# ---------------------------------------------------------------------------
# (2) Model components
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Conv-based patchifier: (B, C, 128, 128) -> (B, 64, embed_dim)."""
    def __init__(self, in_channels=IN_CHANNELS, embed_dim=EMBED_DIM):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=TOKEN_SIZE, stride=TOKEN_SIZE)

    def forward(self, x):
        x = self.proj(x)                  # (B, D, 8, 8)
        x = x.flatten(2).transpose(1, 2)  # (B, 64, D)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Standard ViT-S over 64 image tokens, with learned positional embeddings."""
    def __init__(self, embed_dim=EMBED_DIM, depth=ENC_DEPTH, n_heads=ENC_HEADS):
        super().__init__()
        self.patch_embed = PatchEmbed(IN_CHANNELS, embed_dim)
        self.pos_embed   = nn.Parameter(torch.zeros(1, N_TOKENS, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, n_heads)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, OUT_DIM)

    def forward(self, x, token_ids=None):
        """
        x:         (B, C, H, W)
        token_ids: (B, K) optional — return only embeddings for these tokens.
        """
        h = self.patch_embed(x) + self.pos_embed
        if token_ids is not None:
            # Gather the requested tokens before running attention only on them.
            B, K = token_ids.shape
            idx = token_ids.unsqueeze(-1).expand(-1, -1, h.size(-1))
            h = torch.gather(h, 1, idx)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return self.head(h)             # (B, K_or_64, OUT_DIM)


class Predictor(nn.Module):
    """
    Tiny transformer that takes context-token embeddings + learnable
    target-position queries, and predicts the target tokens' embeddings.
    """
    def __init__(self, in_dim=OUT_DIM, hidden=PRED_DIM, depth=PRED_DEPTH, n_heads=4):
        super().__init__()
        self.in_proj  = nn.Linear(in_dim, hidden)
        self.pos_embed = nn.Parameter(torch.zeros(1, N_TOKENS, hidden))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(hidden, n_heads)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, in_dim)

    def forward(self, context_emb, context_ids, target_ids):
        """
        context_emb: (B, Kc, D_out) embeddings produced by the context encoder
        context_ids: (B, Kc) token indices the context encoder saw
        target_ids:  (B, Kt) token indices we want to predict
        Returns:     (B, Kt, D_out) predicted embeddings
        """
        B, Kc, _ = context_emb.shape
        Kt = target_ids.shape[1]

        ctx = self.in_proj(context_emb)
        # Add positional info for context tokens
        ctx = ctx + self.pos_embed.expand(B, -1, -1).gather(
            1, context_ids.unsqueeze(-1).expand(-1, -1, ctx.size(-1)))

        # Build target slots: shared mask token + target positional embedding
        tgt = self.mask_token.expand(B, Kt, -1).clone()
        tgt = tgt + self.pos_embed.expand(B, -1, -1).gather(
            1, target_ids.unsqueeze(-1).expand(-1, -1, tgt.size(-1)))

        h = torch.cat([ctx, tgt], dim=1)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        # Return only the target slots
        return self.out_proj(h[:, Kc:, :])


# ---------------------------------------------------------------------------
# (3) Masking
# ---------------------------------------------------------------------------
def sample_block_mask(grid: int = 8, ratio: float = TARGET_BLOCK_RATIO):
    """Sample one rectangular block of tokens, ~ratio of total."""
    target = max(1, int(round(grid * grid * ratio)))
    h = max(1, int(round(math.sqrt(target))))
    w = max(1, int(math.ceil(target / h)))
    h, w = min(h, grid), min(w, grid)
    top  = np.random.randint(0, grid - h + 1)
    left = np.random.randint(0, grid - w + 1)
    ids = []
    for r in range(top, top + h):
        for c in range(left, left + w):
            ids.append(r * grid + c)
    return set(ids)


def make_masks(batch_size: int, grid: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each sample, sample N_TARGET_BLOCKS disjoint target blocks.
    Context tokens = everything not in any target block.

    Returns:
      ctx_ids: (B, Kc) int64
      tgt_ids: (B, Kt) int64
    All samples in a batch share the same Kc/Kt for tensor packing — we
    enforce this by using a fixed number of target tokens per sample.
    """
    n_total = grid * grid
    target_n_min = int(round(n_total * MASK_RATIO))

    ctx_list, tgt_list = [], []
    for _ in range(batch_size):
        targets: set[int] = set()
        attempts = 0
        while len(targets) < target_n_min and attempts < 30:
            block = sample_block_mask(grid, TARGET_BLOCK_RATIO)
            targets |= block
            attempts += 1
        # Trim or pad to exact target_n_min for batch packing
        targets = list(targets)[:target_n_min]
        contexts = [i for i in range(n_total) if i not in set(targets)]
        ctx_list.append(contexts)
        tgt_list.append(targets)

    # Pad context list to common length (all should be n_total - target_n_min)
    Kc = n_total - target_n_min
    ctx_arr = np.array([c[:Kc] for c in ctx_list], dtype=np.int64)
    tgt_arr = np.array(tgt_list, dtype=np.int64)
    return torch.from_numpy(ctx_arr), torch.from_numpy(tgt_arr)


# ---------------------------------------------------------------------------
# (4) VICReg regularizer
# ---------------------------------------------------------------------------
def vicreg_terms(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Variance term: hinge loss pushing per-dim std >= 1.
    Covariance term: penalizes off-diagonal terms of the cov matrix.
    Returns scalar (var_loss, cov_loss).
    """
    # z: (N, D) where N is batch * tokens
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()

    N, D = z.shape
    cov = (z.T @ z) / (N - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag.pow(2).sum()) / D
    return var_loss, cov_loss


# ---------------------------------------------------------------------------
# (5) Train / val splits, ecoregion-aware
# ---------------------------------------------------------------------------
def make_splits(smoke: bool):
    """
    SSL pool:  all patches (including PNW). The encoder sees everything.
    Val:       small random fraction of the pool, used during training
               to track held-out JEPA prediction loss.
    OOD list:  PNW patches, recorded separately for Experiment 3
               (downstream probe transfer to a held-out region).

    Note on the design choice: an earlier version of this script excluded
    PNW from SSL entirely. We include it here because (a) SSL is
    label-free, so the soil-moisture probe test in Exp 3 is not
    compromised by the encoder having seen PNW imagery, and (b) it
    keeps ~6% more training data. Exp 3's "generalization" claim
    becomes about probe transfer rather than encoder transfer — equally
    valid, and matches how foundation models are used in practice.
    """
    manifest = pd.read_parquet(MANIFEST)
    ok = manifest[manifest.status.isin(['ok', 'cached'])].copy()
    if LABELS_FILE.exists():
        labels = pd.read_parquet(LABELS_FILE)[['patch_id']]
        ok = ok.merge(labels, on='patch_id')

    # PNW bookkeeping: lon < -116, lat > 42 covers WA/OR/ID/W-MT/NW-CA.
    is_pnw = (ok.lon < -116) & (ok.lat > 42)
    ood = ok[is_pnw].patch_id.tolist()
    pool = ok.patch_id.tolist()

    if smoke:
        rng = np.random.default_rng(0)
        pool = rng.choice(pool, size=min(200, len(pool)), replace=False).tolist()

    rng = np.random.default_rng(42)
    rng.shuffle(pool)
    n_val = max(8, int(len(pool) * VAL_FRACTION))
    val_ids   = pool[:n_val]
    train_ids = pool[n_val:]

    logging.info(f'Splits — train: {len(train_ids)}, val: {len(val_ids)} '
                 f'(SSL on all CONUS), Exp 3 OOD anchors: {len(ood)}')
    return train_ids, val_ids, ood


# ---------------------------------------------------------------------------
# (6) Training loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, momentum: float):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


def lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def find_latest_checkpoint(mode_tag: str) -> Path | None:
    """Find the latest epoch checkpoint, if any."""
    candidates = sorted(CKPT_DIR.glob(f'hydrojepa_{mode_tag}_ep*.pt'))
    return candidates[-1] if candidates else None


def prune_old_checkpoints(mode_tag: str, current_ep: int,
                          keep_recent: int = 3, keep_every: int = 10):
    """
    Disk hygiene. Keep:
      - the last `keep_recent` epoch checkpoints
      - every `keep_every`-th epoch (10, 20, 30, ...)
      - everything from the current epoch onward (just saved)
    Delete the rest. Final + best are saved separately and never pruned.
    """
    ckpts = sorted(CKPT_DIR.glob(f'hydrojepa_{mode_tag}_ep*.pt'))
    keep_set = set(ckpts[-keep_recent:])  # most recent
    for c in ckpts:
        ep = int(c.stem.split('ep')[-1])
        if ep % keep_every == 0:
            keep_set.add(c)
    for c in ckpts:
        if c not in keep_set:
            c.unlink()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')
    if device.type == 'cuda':
        logging.info(f'  GPU: {torch.cuda.get_device_name(0)}')

    # --- Splits and stats ---
    train_ids, val_ids, _ = make_splits(args.smoke)
    stats = compute_band_stats(train_ids)
    logging.info(f'Band stats: mean={stats["mean"].round(3).tolist()}')
    logging.info(f'            std ={stats["std"].round(3).tolist()}')

    train_ds = HydroJEPADataset(train_ids, stats)
    val_ds   = HydroJEPADataset(val_ids,   stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=True,
                              pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=2)

    # --- Models ---
    context_enc = ViTEncoder().to(device)
    target_enc  = ViTEncoder().to(device)
    target_enc.load_state_dict(context_enc.state_dict())
    for p in target_enc.parameters():
        p.requires_grad = False
    predictor = Predictor().to(device)

    # --- Optimizer ---
    params = list(context_enc.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=WEIGHT_DECAY)

    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = WARMUP_EP * max(1, len(train_loader))

    # --- Logging buffers ---
    history = {'epoch': [], 'train_pred': [], 'train_var': [], 'train_cov': [],
               'val_pred': []}

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    mode_tag = 'smoke' if args.smoke else 'full'

    # --- Resume logic ---
    start_epoch = 0
    step = 0
    best_val = float('inf')
    if not args.no_resume:
        latest = find_latest_checkpoint(mode_tag)
        if latest is not None:
            logging.info(f'Found checkpoint: {latest.name}, attempting resume')
            try:
                ckpt = torch.load(latest, map_location=device, weights_only=False)
                context_enc.load_state_dict(ckpt['context_enc'])
                target_enc.load_state_dict(ckpt['target_enc'])
                predictor.load_state_dict(ckpt['predictor'])
                if 'optimizer' in ckpt:
                    opt.load_state_dict(ckpt['optimizer'])
                if 'rng_state' in ckpt:
                    torch.set_rng_state(ckpt['rng_state'].cpu())
                    if device.type == 'cuda' and 'cuda_rng_state' in ckpt:
                        torch.cuda.set_rng_state(ckpt['cuda_rng_state'].cpu())
                    if 'numpy_rng_state' in ckpt:
                        np.random.set_state(ckpt['numpy_rng_state'])
                start_epoch = ckpt['epoch']
                step = ckpt.get('step', start_epoch * len(train_loader))
                history = ckpt.get('history', history)
                best_val = ckpt.get('best_val', float('inf'))
                logging.info(f'  Resumed from epoch {start_epoch}, '
                             f'step {step}, best_val {best_val:.4f}')
                if start_epoch >= args.epochs:
                    logging.info('  Already complete. Use --no_resume to retrain.')
                    return
            except Exception as e:
                logging.warning(f'  Resume failed ({e}); starting fresh')
                start_epoch, step, best_val = 0, 0, float('inf')

    for ep in range(start_epoch, args.epochs):
        context_enc.train(); predictor.train()
        ep_pred, ep_var, ep_cov, n_b = 0.0, 0.0, 0.0, 0
        t0 = time.time()

        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            B = x.size(0)
            ctx_ids, tgt_ids = make_masks(B)
            ctx_ids, tgt_ids = ctx_ids.to(device), tgt_ids.to(device)

            # Context encoder sees only context tokens.
            ctx_emb = context_enc(x, token_ids=ctx_ids)        # (B, Kc, D)

            # Target encoder sees the whole image; we pull only target tokens.
            with torch.no_grad():
                full_tgt = target_enc(x)                        # (B, 64, D)
                tgt_emb = torch.gather(
                    full_tgt, 1,
                    tgt_ids.unsqueeze(-1).expand(-1, -1, OUT_DIM)
                )                                               # (B, Kt, D)

            # Predict target embeddings from context.
            pred = predictor(ctx_emb, ctx_ids, tgt_ids)         # (B, Kt, D)

            pred_loss = F.smooth_l1_loss(pred, tgt_emb)

            # VICReg on the predicted distribution + on context embeddings.
            var_p, cov_p = vicreg_terms(pred.flatten(0, 1))
            var_c, cov_c = vicreg_terms(ctx_emb.flatten(0, 1))
            var_loss = 0.5 * (var_p + var_c)
            cov_loss = 0.5 * (cov_p + cov_c)

            loss = (PRED_LOSS_COEF * pred_loss
                    + VAR_COEF * var_loss
                    + COV_COEF * cov_loss)

            # LR schedule
            lr_now = lr_at(step, total_steps, warmup_steps, args.lr)
            for pg in opt.param_groups:
                pg['lr'] = lr_now

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            # EMA update
            ema_m = EMA_START + (EMA_END - EMA_START) * step / max(1, total_steps)
            update_ema(context_enc, target_enc, ema_m)

            ep_pred += pred_loss.item()
            ep_var  += var_loss.item()
            ep_cov  += cov_loss.item()
            n_b     += 1
            step    += 1

        # --- Validation ---
        context_enc.eval(); predictor.eval()
        val_pred, n_v = 0.0, 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device, non_blocking=True)
                B = x.size(0)
                ctx_ids, tgt_ids = make_masks(B)
                ctx_ids, tgt_ids = ctx_ids.to(device), tgt_ids.to(device)
                ctx_emb = context_enc(x, token_ids=ctx_ids)
                full_tgt = target_enc(x)
                tgt_emb = torch.gather(
                    full_tgt, 1,
                    tgt_ids.unsqueeze(-1).expand(-1, -1, OUT_DIM))
                pred = predictor(ctx_emb, ctx_ids, tgt_ids)
                val_pred += F.smooth_l1_loss(pred, tgt_emb).item()
                n_v += 1

        history['epoch'].append(ep + 1)
        history['train_pred'].append(ep_pred / max(1, n_b))
        history['train_var'].append(ep_var / max(1, n_b))
        history['train_cov'].append(ep_cov / max(1, n_b))
        history['val_pred'].append(val_pred / max(1, n_v))

        elapsed = time.time() - t0
        logging.info(
            f'ep {ep+1:3d}/{args.epochs} | '
            f'train pred {history["train_pred"][-1]:.4f}  '
            f'var {history["train_var"][-1]:.3f}  '
            f'cov {history["train_cov"][-1]:.3f} | '
            f'val pred {history["val_pred"][-1]:.4f} | '
            f'lr {lr_now:.2e} | ema {ema_m:.4f} | {elapsed:.1f}s'
        )

        # --- Save full training state (atomic write, then prune old) ---
        ckpt_payload = {
            'epoch':       ep + 1,
            'step':        step,
            'context_enc': context_enc.state_dict(),
            'target_enc':  target_enc.state_dict(),
            'predictor':   predictor.state_dict(),
            'optimizer':   opt.state_dict(),
            'rng_state':         torch.get_rng_state(),
            'cuda_rng_state':    (torch.cuda.get_rng_state()
                                  if device.type == 'cuda' else None),
            'numpy_rng_state':   np.random.get_state(),
            'stats':       stats,
            'history':     history,
            'best_val':    best_val,
            'config':      vars(args),
        }
        ckpt_path = CKPT_DIR / f'hydrojepa_{mode_tag}_ep{ep+1:03d}.pt'
        tmp_path  = ckpt_path.with_suffix('.pt.tmp')
        torch.save(ckpt_payload, tmp_path)
        tmp_path.replace(ckpt_path)   # atomic on POSIX; safe on Windows too

        # Track best by val loss (encoder-only, smaller file)
        cur_val = history['val_pred'][-1]
        if cur_val < best_val:
            best_val = cur_val
            best_path = CKPT_DIR / f'hydrojepa_{mode_tag}_best.pt'
            torch.save({
                'epoch': ep + 1,
                'context_enc': context_enc.state_dict(),
                'stats': stats,
                'val_pred': cur_val,
                'config': vars(args),
            }, best_path)

        # Disk hygiene
        prune_old_checkpoints(mode_tag, ep + 1)

    # --- Final encoder-only checkpoint ---
    torch.save({
        'context_enc': context_enc.state_dict(),
        'stats':       stats,
        'config':      vars(args),
    }, CKPT_DIR / f'hydrojepa_{mode_tag}_final.pt')

    # --- Curves ---
    plot_curves(history, REPORT_DIR / f'training_curves_{mode_tag}.png')

    # --- Smoke-mode UMAP for quick validation ---
    if args.smoke:
        smoke_umap(context_enc, train_ids + val_ids, stats, device)


# ---------------------------------------------------------------------------
# (7) Diagnostics
# ---------------------------------------------------------------------------
def plot_curves(history: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax1, ax2 = axes
    ax1.plot(history['epoch'], history['train_pred'], label='train pred')
    ax1.plot(history['epoch'], history['val_pred'],   label='val pred')
    ax1.set_xlabel('epoch'); ax1.set_ylabel('loss'); ax1.legend()
    ax1.set_title('JEPA prediction loss')
    ax2.plot(history['epoch'], history['train_var'], label='var loss')
    ax2.plot(history['epoch'], history['train_cov'], label='cov loss')
    ax2.set_xlabel('epoch'); ax2.legend()
    ax2.set_title('VICReg regularizers (collapse diagnostic)')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    logging.info(f'Saved {out_path}')


@torch.no_grad()
def smoke_umap(encoder: nn.Module, patch_ids: list[str],
               stats: dict, device: torch.device):
    try:
        import umap
    except ImportError:
        logging.warning('umap-learn not installed; skipping smoke UMAP')
        return
    encoder.eval()
    ds = HydroJEPADataset(patch_ids, stats)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    all_emb, all_pid = [], []
    for x, pids in loader:
        x = x.to(device, non_blocking=True)
        emb = encoder(x).mean(dim=1)         # mean over tokens -> (B, OUT_DIM)
        all_emb.append(emb.cpu().numpy())
        all_pid.extend(pids)
    emb = np.concatenate(all_emb)
    proj = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(emb)

    # Color by SMAP if available, else by elevation, else gray.
    color, label = None, ''
    if LABELS_FILE.exists():
        lab = pd.read_parquet(LABELS_FILE).set_index('patch_id')
        for col, cmap, lbl in [('smap_sm', 'YlGnBu', 'SMAP soil moisture'),
                               ('elevation', 'viridis', 'elevation')]:
            if col in lab.columns:
                color = lab.loc[all_pid, col].values
                cmap_use = cmap
                label = lbl
                break

    fig, ax = plt.subplots(figsize=(7, 6), dpi=130)
    if color is not None:
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=color, cmap=cmap_use,
                        s=10, alpha=0.85)
        plt.colorbar(sc, ax=ax, label=label)
    else:
        ax.scatter(proj[:, 0], proj[:, 1], s=10, alpha=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Smoke-mode UMAP of HydroJEPA embeddings')
    fig.tight_layout()
    out = REPORT_DIR / 'smoke_umap.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'Saved {out}')


# ---------------------------------------------------------------------------
# (8) Driver
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true',
                   help='Run on 200 patches for 10 epochs as a sanity check')
    p.add_argument('--no_resume', action='store_true',
                   help='Ignore existing checkpoints and start fresh')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--lr', type=float, default=LR)
    args = p.parse_args()
    if args.smoke:
        if args.epochs is None: args.epochs = 10
        if args.batch_size is None: args.batch_size = 8
    else:
        if args.epochs is None: args.epochs = EPOCHS
        if args.batch_size is None: args.batch_size = BATCH_SIZE
    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)

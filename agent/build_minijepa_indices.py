"""
agent/build_minijepa_indices.py
Build per-Mini-JEPA FAISS indices over the patch corpus.

Mirrors the pattern from 5_1_minijepa_pretrain.py and 6_1_minijepa_evaluation.py:

  1. Load 5_hydrojepa_pretrain.py as a module
  2. For each modality, patch s5.PATCH_DIR / s5.MANIFEST / s5.IN_CHANNELS
     so the dataset reads the right files and the encoder builds with the
     right channel count
  3. For modis_lst and topo_soil, apply the inf-aware dataset hardening
     so -inf nodata pixels don't NaN out the encoder
  4. Filter the per-modality manifest to status in (ok, cached) and
     intersect with labels.parquet so we only encode patches that have
     both the GeoTIFF on disk AND a row of labels — this matches what
     scripts 11/12/13 do
  5. Mean-pool the 64 ViT tokens to a single 64-d vector, write FAISS

Output (under agent/minijepa_index/):
  s2_optical.{npy,faiss}      mean-pooled embeddings + L2 index
  s2_optical_keys.parquet     (patch_id, lon, lat) row mapping
  ...same for s1_sar, s2_phenology, modis_lst, topo_soil
  alphaearth.{npy,faiss}      pulled directly from labels.parquet AE columns
  alphaearth_keys.parquet
  <modality>.done             sentinel; reruns skip done modalities

Resumable per modality. Per-model output is small (~3 MB each), total < 50 MB.

Run:
    python agent/build_minijepa_indices.py
    python agent/build_minijepa_indices.py --only s1_sar modis_lst   # subset
    python agent/build_minijepa_indices.py --force                   # rebuild all
"""

import argparse
import importlib.util
import logging
import os
import sys
import warnings
from pathlib import Path

# Silence rasterio noise BEFORE importing anything that touches GDAL
os.environ['CPL_LOG'] = os.devnull
warnings.filterwarnings('ignore', message='.*Photometric.*')
warnings.filterwarnings('ignore', message='.*ExtraSamples.*')
logging.getLogger('rasterio').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import torch
import faiss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HJ_ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR    = HJ_ROOT / 'data' / 'hydrojepa'
LABELS_FILE = DATA_DIR / 'labels.parquet'
CKPT_DIR    = HJ_ROOT / 'checkpoints'
INDEX_DIR   = HJ_ROOT / 'agent' / 'minijepa_index'

# Modality registry — keep in sync with 5_1_minijepa_pretrain.py and
# 6_1_minijepa_evaluation.py. Each entry:
#   ckpt_name        — basename in checkpoints/
#   patches_subdir   — patches dir under data/hydrojepa/
#   manifest_name    — manifest parquet under data/hydrojepa/
#   in_channels      — input channel count for this modality
#   needs_inf_hardening — whether to apply inf-aware dataset hardening
MODALITIES = {
    's2_optical': {
        'ckpt_name':           'hydrojepa_full_best.pt',
        'patches_subdir':      'patches',
        'manifest_name':       'manifest.parquet',
        'in_channels':         10,
        'needs_inf_hardening': False,
    },
    's1_sar': {
        'ckpt_name':           'hydrojepa_s1_sar_best.pt',
        'patches_subdir':      'patches_s1_sar',
        'manifest_name':       'manifest_s1_sar.parquet',
        'in_channels':         2,
        'needs_inf_hardening': False,
    },
    's2_phenology': {
        'ckpt_name':           'hydrojepa_s2_phenology_best.pt',
        'patches_subdir':      'patches_s2_phenology',
        'manifest_name':       'manifest_s2_phenology.parquet',
        'in_channels':         40,
        'needs_inf_hardening': False,
    },
    'modis_lst': {
        'ckpt_name':           'hydrojepa_modis_lst_best.pt',
        'patches_subdir':      'patches_modis_lst',
        'manifest_name':       'manifest_modis_lst.parquet',
        'in_channels':         2,
        'needs_inf_hardening': True,
    },
    'topo_soil': {
        'ckpt_name':           'hydrojepa_topo_soil_best.pt',
        'patches_subdir':      'patches_topo_soil',
        'manifest_name':       'manifest_topo_soil.parquet',
        'in_channels':         6,
        'needs_inf_hardening': True,
    },
}

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------------
# Pretrain module loader (one process-wide instance, repeatedly re-patched)
# ---------------------------------------------------------------------------
def load_pretrain_module():
    """Import script 5 as a fresh module so we can mutate its globals."""
    spec = importlib.util.spec_from_file_location(
        'pretrain', str(HJ_ROOT / '5_hydrojepa_pretrain.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['pretrain'] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# -inf-aware dataset hardening — verbatim port of the logic from
# 5_1_minijepa_pretrain.py and 6_1_minijepa_evaluation.py
# ---------------------------------------------------------------------------
def harden_dataset_for_nodata(s5):
    """Replace s5.HydroJEPADataset with a variant that handles -inf nodata.

    modis_lst (~0.8% of patches) and topo_soil (~26% of patches) have
    nodata encoded as -inf. The original dataset uses nanmedian which
    returns -inf when an entire band is bad, propagating garbage through
    the encoder. This replacement converts non-finite pixels to NaN, then
    fills with per-band median (falling back to 0 if the entire band is NaN).
    """
    import torch
    import rasterio
    from torch.utils.data import Dataset

    PATCH_DIR_REF = lambda: s5.PATCH_DIR
    IN_CH_REF     = lambda: s5.IN_CHANNELS

    class HydroJEPADatasetHardened(Dataset):
        def __init__(self, patch_ids, stats):
            self.patch_ids = patch_ids
            in_ch = IN_CH_REF()
            self.mean = torch.tensor(stats['mean'], dtype=torch.float32).view(in_ch, 1, 1)
            self.std  = torch.tensor(stats['std'],  dtype=torch.float32).view(in_ch, 1, 1)

        def __len__(self):
            return len(self.patch_ids)

        def __getitem__(self, idx):
            pid = self.patch_ids[idx]
            path = PATCH_DIR_REF() / f'{pid}.tif'
            with rasterio.open(path) as src:
                arr = src.read().astype(np.float32)
            arr = np.where(np.isfinite(arr), arr, np.nan)
            if np.isnan(arr).any():
                for b in range(arr.shape[0]):
                    band = arr[b]
                    if np.isnan(band).all():
                        fill = 0.0
                    else:
                        fill = np.nanmedian(band)
                        if not np.isfinite(fill):
                            fill = 0.0
                    arr[b] = np.where(np.isnan(arr[b]), fill, arr[b])
            x = torch.from_numpy(arr)
            x = (x - self.mean) / (self.std + 1e-6)
            return x, pid

    s5.HydroJEPADataset = HydroJEPADatasetHardened


# ---------------------------------------------------------------------------
# Per-modality patch_id resolution
# ---------------------------------------------------------------------------
def resolve_patch_ids(modality: str, manifest_path: Path,
                      labels_df: pd.DataFrame) -> list[str]:
    """
    Filter to patch_ids that:
      (a) the manifest marks as ok or cached  (file exists on disk)
      (b) appear in labels.parquet            (have label rows)
    Same filter scripts 11/12/13 use.
    """
    manifest = pd.read_parquet(manifest_path)
    if 'status' not in manifest.columns:
        # Older manifest schema — assume all rows are ok
        ok = manifest
        logging.info(f'  {modality}: manifest has no status column, taking all rows')
    else:
        ok = manifest[manifest['status'].isin(['ok', 'cached'])]
        logging.info(f'  {modality}: manifest has {len(manifest)} rows, '
                     f'{len(ok)} ok/cached')

    label_ids = set(labels_df['patch_id'].astype(str))
    keep = ok[ok['patch_id'].astype(str).isin(label_ids)]
    patch_ids = keep['patch_id'].astype(str).tolist()

    logging.info(f'  {modality}: {len(patch_ids)} patches survive '
                 f'(manifest ok ∩ labels.parquet)')
    return patch_ids


# ---------------------------------------------------------------------------
# Per-modality embedding encode
# ---------------------------------------------------------------------------
def encode_modality(modality: str, cfg: dict, labels_df: pd.DataFrame,
                    s5, device: torch.device, batch_size: int = 64
                    ) -> tuple[np.ndarray, list[str]]:
    """Patch s5 globals, build encoder, run inference, return (embs, patch_ids)."""

    ckpt_path = CKPT_DIR / cfg['ckpt_name']
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')

    patches_dir   = DATA_DIR / cfg['patches_subdir']
    manifest_path = DATA_DIR / cfg['manifest_name']
    if not patches_dir.exists():
        raise FileNotFoundError(f'Missing patches dir: {patches_dir}')
    if not manifest_path.exists():
        raise FileNotFoundError(f'Missing manifest: {manifest_path}')

    # Patch s5 module-level globals — the dataset and encoder both read these
    # at runtime (see 5_hydrojepa_pretrain.py:67 PATCH_DIR, line 132 path
    # construction, and ViTEncoder reading IN_CHANNELS at __init__ time).
    s5.PATCH_DIR   = patches_dir
    s5.MANIFEST    = manifest_path
    s5.IN_CHANNELS = cfg['in_channels']

    if cfg['needs_inf_hardening']:
        harden_dataset_for_nodata(s5)
        logging.info(f'  {modality}: applied -inf-aware dataset hardening')

    # Resolve which patches to encode
    patch_ids = resolve_patch_ids(modality, manifest_path, labels_df)
    if not patch_ids:
        raise RuntimeError(f'{modality}: no patches survived filtering')

    # Sanity-check the encoder builds with the right channel count
    test_enc = s5.ViTEncoder()
    actual_in = test_enc.patch_embed.proj.in_channels
    if actual_in != cfg['in_channels']:
        raise RuntimeError(
            f'{modality}: ViTEncoder built with in_channels={actual_in} '
            f'but expected {cfg["in_channels"]} — IN_CHANNELS patch failed.')
    del test_enc

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder = s5.ViTEncoder().to(device)
    encoder.load_state_dict(ckpt['context_enc'])
    encoder.eval()

    if 'stats' not in ckpt:
        raise RuntimeError(
            f'{modality}: checkpoint missing band stats. '
            f'Re-pretrain or pull stats from a side channel.')
    stats = ckpt['stats']

    ds = s5.HydroJEPADataset(patch_ids, stats)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    embs = []
    seen = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            tokens = encoder(x.to(device))           # (B, T, D)
            mean_pooled = tokens.mean(dim=1)         # (B, D)
            embs.append(mean_pooled.cpu().numpy().astype(np.float32))
            seen += x.shape[0]
            if (i + 1) % 20 == 0:
                logging.info(f'    encoded {seen}/{len(ds)}')

    arr = np.concatenate(embs, axis=0)
    logging.info(f'  {modality}: encoded shape={arr.shape}')
    return arr, patch_ids


# ---------------------------------------------------------------------------
# AlphaEarth: pull straight from labels.parquet
# ---------------------------------------------------------------------------
def encode_alphaearth(labels: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    ae_cols = [f'A{i:02d}' for i in range(64)]
    missing = [c for c in ae_cols if c not in labels.columns]
    if missing:
        raise RuntimeError(f'labels.parquet missing AE columns: {missing[:3]}...')
    sub = labels.dropna(subset=ae_cols).copy()
    arr = sub[ae_cols].values.astype(np.float32)
    return arr, sub['patch_id'].astype(str).tolist()


# ---------------------------------------------------------------------------
# FAISS write
# ---------------------------------------------------------------------------
def write_index(name: str, embs: np.ndarray, patch_ids: list[str],
                labels: pd.DataFrame):
    """Write embeddings + FAISS index + sentinel + (patch_id, lon, lat) keys."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / f'{name}.npy', embs)

    # Build keys parquet by joining patch_ids back to lon/lat in labels
    lookup = labels.set_index(labels['patch_id'].astype(str))[['lon', 'lat']]
    rows = [(pid, float(lookup.loc[pid, 'lon']), float(lookup.loc[pid, 'lat']))
            for pid in patch_ids]
    keys = pd.DataFrame(rows, columns=['patch_id', 'lon', 'lat'])
    keys.to_parquet(INDEX_DIR / f'{name}_keys.parquet')

    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, str(INDEX_DIR / f'{name}.faiss'))
    (INDEX_DIR / f'{name}.done').touch()
    logging.info(f'  wrote {name}.{{npy,faiss}}, {name}_keys.parquet '
                 f'({embs.shape[0]} × {embs.shape[1]})')


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--only', nargs='+', default=None,
                   help='Subset of targets to (re)build. Valid: '
                        + ' '.join(list(MODALITIES) + ['alphaearth']))
    p.add_argument('--force', action='store_true',
                   help='Re-encode even if .done sentinel exists')
    p.add_argument('--batch_size', type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    targets = list(MODALITIES.keys()) + ['alphaearth']
    if args.only:
        targets = [t for t in targets if t in args.only]
        if not targets:
            sys.exit(f'No matching targets in --only={args.only}')

    labels = pd.read_parquet(LABELS_FILE)
    logging.info(f'Loaded labels.parquet ({len(labels)} rows)')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')

    # Load script 5 once, re-patch its globals per modality
    s5 = load_pretrain_module()
    # Snapshot the original dataset class so each modality starts clean —
    # the inf-hardening swap is sticky otherwise and would also affect a
    # subsequent non-hardened modality
    original_dataset_cls = s5.HydroJEPADataset

    for tgt in targets:
        sentinel = INDEX_DIR / f'{tgt}.done'
        if sentinel.exists() and not args.force:
            logging.info(f'[{tgt}] already built, skipping (use --force to redo)')
            continue

        logging.info(f'[{tgt}] encoding ...')
        if tgt == 'alphaearth':
            embs, patch_ids = encode_alphaearth(labels)
        else:
            # Reset dataset class in case a previous modality applied hardening
            s5.HydroJEPADataset = original_dataset_cls
            embs, patch_ids = encode_modality(
                tgt, MODALITIES[tgt], labels, s5, device,
                batch_size=args.batch_size)

        write_index(tgt, embs, patch_ids, labels)

    logging.info('All requested indices built.')


if __name__ == '__main__':
    main()

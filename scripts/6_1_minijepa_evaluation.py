"""
HydroJEPA Mini-JEPA evaluation (6_1).

Runs scripts 11, 12, 13 once per Mini-JEPA modality, with paths patched
so each model writes its dimension dictionary, geometry profile, and
AE complementarity into its own clean per-model report directory:

  reports/minijepas/
    s2_optical/
      interpretability/
      manifold_geometry/
      complementarity/
    modis_lst/
      interpretability/
      manifold_geometry/
      complementarity/
    topo_soil/        ...
    s1_sar/           ...
    s2_phenology/     ...

The existing reports at reports/{interpretability,manifold_geometry,
complementarity}/ (S2-Optical, already produced by the original script
11/12/13 runs) are NOT touched. We mirror them into reports/minijepas/
s2_optical/ so the agent has uniform access to all five Mini-JEPAs.

Why this matters: the agent router needs to load each Mini-JEPA's
metadata uniformly. One directory per model, three subdirectories per
model, every model identical. Future scripts (agent_planner) iterate
reports/minijepas/*/ rather than special-casing the optical baseline.

Run examples:
  # Run all three evaluations on one modality:
  python 6_1_minijepa_evaluation.py --modality modis_lst

  # Run on every Mini-JEPA the script can find (skips missing checkpoints):
  python 6_1_minijepa_evaluation.py --modality all

  # Skip a specific evaluation step (useful during iteration):
  python 6_1_minijepa_evaluation.py --modality s1_sar --skip 13

  # Mirror the existing S2-Optical reports into the minijepas/ tree:
  python 6_1_minijepa_evaluation.py --mirror_s2_optical

Resumable: scripts 11/12/13 cache embeddings and intermediate outputs
inside their own report directories, so re-runs of a modality skip
expensive computation if the cache is present.
"""

import argparse
import importlib.util
import logging
import shutil
import sys
import warnings
from pathlib import Path

# Silence rasterio noise BEFORE importing anything that touches GDAL
import os
os.environ['CPL_LOG'] = os.devnull
warnings.filterwarnings('ignore', message='.*Photometric.*')
warnings.filterwarnings('ignore', message='.*ExtraSamples.*')
logging.getLogger('rasterio').setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------------
# Modality registry — keep in sync with 5_1_minijepa_pretrain.py
# ---------------------------------------------------------------------------
MODALITIES = {
    'modis_lst':    {'in_channels': 2,  'needs_inf_hardening': True},
    'topo_soil':    {'in_channels': 6,  'needs_inf_hardening': True},
    's1_sar':       {'in_channels': 2,  'needs_inf_hardening': False},
    's2_phenology': {'in_channels': 40, 'needs_inf_hardening': False},
}

CKPT_DIR = Path('checkpoints')
DATA_DIR = Path('data/hydrojepa')
MINIJEPAS_DIR = Path('reports/minijepas')

# Existing S2-Optical reports (from the original script 11/12/13 runs)
S2_OPTICAL_REPORTS = {
    'interpretability':   Path('reports/interpretability'),
    'manifold_geometry':  Path('reports/manifold_geometry'),
    'complementarity':    Path('reports/complementarity'),
}


# ---------------------------------------------------------------------------
# Module loading — scripts 11/12/13 have leading-digit filenames
# ---------------------------------------------------------------------------
def load_module(filename: str, alias: str):
    here = Path(__file__).resolve().parent
    src = here / filename
    if not src.exists():
        raise FileNotFoundError(f'{filename} not found at {src}')
    spec = importlib.util.spec_from_file_location(alias, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# -inf-aware dataset hardening — same logic as 5_1_minijepa_pretrain.py
# Applied to script 5's HydroJEPADataset / compute_band_stats so the
# embeddings produced by script 11 don't NaN on -inf-containing patches.
# ---------------------------------------------------------------------------
def harden_dataset_for_nodata(s5):
    import numpy as np
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
    logging.info('  applied -inf-aware dataset hardening')


# ---------------------------------------------------------------------------
# Patching scripts 11/12/13 to point at the right modality
# ---------------------------------------------------------------------------
def patch_for_modality(modality: str, s5, model_dir: Path):
    """Set the module-level paths in script 5 so its dataset reads the
    Mini-JEPA's modality-specific patches and channel count.

    Returns a dict of (Cfg overrides) to apply to scripts 11/12/13.
    """
    cfg = MODALITIES[modality]

    # Script 5 module: dataset + encoder construction read these at runtime
    s5.PATCH_DIR   = DATA_DIR / f'patches_{modality}'
    s5.MANIFEST    = DATA_DIR / f'manifest_{modality}.parquet'
    s5.IN_CHANNELS = cfg['in_channels']

    if not s5.PATCH_DIR.exists():
        raise FileNotFoundError(
            f'[{modality}] patches not found at {s5.PATCH_DIR}. '
            f'Run 1_1_hydrojepa_download.py --modality {modality} first.')
    if not s5.MANIFEST.exists():
        raise FileNotFoundError(
            f'[{modality}] manifest not found at {s5.MANIFEST}.')

    if cfg['needs_inf_hardening']:
        harden_dataset_for_nodata(s5)

    # Script 11/12/13 Cfg overrides: per-modality directories
    interp_dir = model_dir / 'interpretability'
    geom_dir   = model_dir / 'manifold_geometry'
    comp_dir   = model_dir / 'complementarity'
    interp_dir.mkdir(parents=True, exist_ok=True)
    geom_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    return {
        'HJ_CKPT':           CKPT_DIR / f'hydrojepa_{modality}_best.pt',
        'MANIFEST':          s5.MANIFEST,
        'INTERP_REPORT_DIR': interp_dir,
        'INTERP_EMB_CACHE':  interp_dir / 'hydrojepa_embeddings.npy',
        'GEOM_REPORT_DIR':   geom_dir,
        'COMP_REPORT_DIR':   comp_dir,
    }


def apply_to_script11(s11, paths: dict, s5):
    """Patch script 11's Cfg AND its private copy of the pretrain module.

    Script 11 loads its own copy of 5_hydrojepa_pretrain.py at import time
    (search for 'spec_from_file_location' near the top of script 11). That
    private copy is a SEPARATE module object from the one 6_1 patched.
    Without redirecting s11.pretrain, script 11 calls ViTEncoder() with
    the unpatched default IN_CHANNELS=10 and state_dict load fails with
    a shape mismatch on patch_embed.proj.weight.
    """
    s11.Cfg.MANIFEST    = paths['MANIFEST']
    s11.Cfg.HJ_CKPT     = paths['HJ_CKPT']
    s11.Cfg.REPORT_DIR  = paths['INTERP_REPORT_DIR']
    s11.Cfg.EMB_CACHE   = paths['INTERP_EMB_CACHE']
    if hasattr(s11.Cfg, 'INTERP_DIR'):
        s11.Cfg.INTERP_DIR = paths['INTERP_REPORT_DIR']
    # Redirect script 11's pretrain reference to the SAME module object
    # we already patched in 6_1. Now ViTEncoder() inside script 11 sees
    # the patched IN_CHANNELS / PATCH_DIR / MANIFEST.
    s11.pretrain = s5


def apply_to_script12(s12, paths: dict):
    s12.Cfg.EMB_CACHE  = paths['INTERP_EMB_CACHE']
    s12.Cfg.REPORT_DIR = paths['GEOM_REPORT_DIR']
    if hasattr(s12.Cfg, 'INTERP_DIR'):
        s12.Cfg.INTERP_DIR = paths['INTERP_REPORT_DIR']
    # Replace load_data() so it reads the modality's manifest instead of the
    # hardcoded S2-Optical 'data/hydrojepa/manifest.parquet'. Otherwise, when
    # a Mini-JEPA's manifest has even one failed patch, the row count check
    # fails (labels has 9704 rows, this modality's embeddings have 9703).
    _replace_load_data(s12, paths['MANIFEST'])


def apply_to_script13(s13, paths: dict):
    s13.Cfg.EMB_CACHE  = paths['INTERP_EMB_CACHE']
    s13.Cfg.REPORT_DIR = paths['COMP_REPORT_DIR']
    if hasattr(s13.Cfg, 'INTERP_DIR'):
        s13.Cfg.INTERP_DIR = paths['INTERP_REPORT_DIR']
    _replace_load_data(s13, paths['MANIFEST'])


def _replace_load_data(mod, modality_manifest_path):
    """Swap mod.load_data with one that uses the modality-specific manifest.

    Script 12's original load_data reads embeddings, then loads labels.parquet
    (shared across modalities), then if row counts don't match, falls back to
    filtering labels by the manifest at Cfg.DATA_DIR/'manifest.parquet'. That
    fallback uses the S2-Optical manifest, which says all 9,704 patches are
    ok — but a Mini-JEPA whose download had even one 503 failure has 9,703
    embeddings, and the filter doesn't drop the right row.

    We replace the whole function with one that uses the modality's manifest.
    Script 13 has the same issue and gets the same fix.
    """
    import numpy as np
    import pandas as pd

    if mod.__name__ == 's12':
        # script 12: returns (E, labels)
        def load_data_patched():
            mod.Cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
            if not mod.Cfg.EMB_CACHE.exists():
                raise FileNotFoundError(
                    f'{mod.Cfg.EMB_CACHE} missing. Run script 11 first.')
            E = np.load(mod.Cfg.EMB_CACHE)
            labels = pd.read_parquet(mod.Cfg.LABELS_FILE).reset_index(drop=True)
            if len(labels) != E.shape[0]:
                manifest = pd.read_parquet(modality_manifest_path)
                ok_ids = manifest[manifest.status.isin(['ok', 'cached'])].patch_id
                labels = labels[labels.patch_id.isin(ok_ids)].reset_index(drop=True)
            if len(labels) != E.shape[0]:
                raise RuntimeError(
                    f'labels rows {len(labels)} vs embeddings {E.shape[0]} '
                    f'after filtering by {modality_manifest_path}')
            import logging
            logging.info(f'Loaded {E.shape[0]:,} x {E.shape[1]} embeddings '
                         f'(modality manifest: {modality_manifest_path.name})')
            return E, labels
        mod.load_data = load_data_patched

    elif mod.__name__ == 's13':
        # script 13: returns df with AE dims, HJ dims (as H00..H63), and env vars
        def load_data_patched():
            import logging
            mod.Cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
            if not mod.Cfg.EMB_CACHE.exists():
                raise FileNotFoundError(
                    f'{mod.Cfg.EMB_CACHE} missing — run script 11 first')
            emb_hj = np.load(mod.Cfg.EMB_CACHE)
            labels = pd.read_parquet(mod.Cfg.LABELS_FILE)
            manifest = pd.read_parquet(modality_manifest_path)
            ok_ids = manifest[manifest.status.isin(['ok', 'cached'])].patch_id
            df = labels[labels.patch_id.isin(ok_ids)].reset_index(drop=True)
            if len(df) != emb_hj.shape[0]:
                raise RuntimeError(
                    f'labels {len(df)} vs HJ embeddings {emb_hj.shape[0]} '
                    f'after filtering by {modality_manifest_path}')
            # Mirror script 13's column coercion exactly
            for c in mod.Cfg.AE_DIMS:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
            for j, col in enumerate(mod.Cfg.HJ_DIMS):
                df[col] = emb_hj[:, j]
            for c in mod.Cfg.ENV_VARS:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
            logging.info(f'Joint table: {len(df)} rows, {len(mod.Cfg.AE_DIMS)} AE + '
                         f'{len(mod.Cfg.HJ_DIMS)} HJ dims, '
                         f'{len([v for v in mod.Cfg.ENV_VARS if v in df.columns])} env vars '
                         f'(modality manifest: {modality_manifest_path.name})')
            return df
        mod.load_data = load_data_patched


# ---------------------------------------------------------------------------
# Per-modality evaluation runner
# ---------------------------------------------------------------------------
def evaluate_one_modality(modality: str, args, s5, s11, s12, s13):
    logging.info('=' * 70)
    logging.info(f'evaluating Mini-JEPA: {modality}')
    logging.info('=' * 70)

    ckpt_path = CKPT_DIR / f'hydrojepa_{modality}_best.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f'[{modality}] checkpoint missing: {ckpt_path}. '
            f'Train it first with 5_1_minijepa_pretrain.py --modality {modality}.')

    model_dir = MINIJEPAS_DIR / modality
    paths = patch_for_modality(modality, s5, model_dir)
    apply_to_script11(s11, paths, s5)
    apply_to_script12(s12, paths)
    apply_to_script13(s13, paths)

    logging.info(f'  checkpoint:  {paths["HJ_CKPT"]}')
    logging.info(f'  patches:     {s5.PATCH_DIR}')
    logging.info(f'  reports ->   {model_dir}/')

    # Sanity-check the encoder was rebuilt with the right channel count.
    # This catches the same monkeypatch failure mode as 5_1.
    test_enc = s5.ViTEncoder()
    actual_in = test_enc.patch_embed.proj.in_channels
    expected_in = MODALITIES[modality]['in_channels']
    if actual_in != expected_in:
        raise RuntimeError(
            f'[{modality}] channel patch failed: ViTEncoder built with '
            f'in_channels={actual_in} but modality expects {expected_in}.')
    del test_enc
    logging.info(f'  encoder ok:  in_channels={actual_in}')

    # ── Step 11: interpretability + dimension dictionary ─────────────────
    # Scripts 11/12/13 each call their own parse_args() inside main(),
    # so we can't pass an args namespace. We shim sys.argv with the flags
    # we want, run main(), then restore.
    if 11 not in args.skip:
        logging.info(f'\n[{modality}] running script 11 (interpretability)...')
        argv_flags = []
        if args.skip_rf:        argv_flags.append('--skip_rf')
        if args.skip_block:     argv_flags.append('--skip_block')
        if args.recompute_emb:  argv_flags.append('--recompute_emb')
        _call_with_argv(s11, argv_flags)
    else:
        logging.info(f'[{modality}] skipping script 11')

    # ── Step 12: manifold geometry ───────────────────────────────────────
    if 12 not in args.skip:
        logging.info(f'\n[{modality}] running script 12 (manifold geometry)...')
        argv_flags = []
        if args.skip_local:      argv_flags.append('--skip_local')
        if args.skip_multiscale: argv_flags.append('--skip_multiscale')
        _call_with_argv(s12, argv_flags)
    else:
        logging.info(f'[{modality}] skipping script 12')

    # ── Step 13: AE complementarity ──────────────────────────────────────
    if 13 not in args.skip:
        logging.info(f'\n[{modality}] running script 13 (AE complementarity)...')
        _call_with_argv(s13, [])
    else:
        logging.info(f'[{modality}] skipping script 13')

    logging.info(f'\n[{modality}] evaluation complete. See {model_dir}/')


def _call_with_argv(mod, flags: list):
    """Run mod.main() with sys.argv shimmed to the given flags.

    Scripts 11/12/13 call parse_args() inside main(), reading sys.argv
    directly. We replace sys.argv with what we want, call main(), then
    restore. Wrapped in try/finally to guarantee restoration even if
    the script raises.
    """
    saved_argv = sys.argv
    try:
        sys.argv = [mod.__name__] + flags
        mod.main()
    finally:
        sys.argv = saved_argv


# ---------------------------------------------------------------------------
# S2-Optical mirroring
# ---------------------------------------------------------------------------
def mirror_s2_optical():
    """Copy the existing S2-Optical reports (already produced by the
    original script 11/12/13 runs against hydrojepa_full_best.pt) into
    reports/minijepas/s2_optical/ so the agent can iterate uniformly.

    Uses copy rather than symlink to be portable across WSL/Windows.
    """
    target = MINIJEPAS_DIR / 's2_optical'
    target.mkdir(parents=True, exist_ok=True)

    for sub, src in S2_OPTICAL_REPORTS.items():
        dst = target / sub
        if not src.exists():
            logging.warning(f'S2-Optical {sub} source missing: {src}')
            continue
        if dst.exists():
            logging.info(f's2_optical/{sub} already mirrored, skipping')
            continue
        shutil.copytree(src, dst)
        logging.info(f'mirrored {src} -> {dst}')


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n', 1)[0])
    p.add_argument('--modality',
                   choices=sorted(MODALITIES) + ['all'],
                   help='Which Mini-JEPA to evaluate. "all" runs every modality '
                        'whose checkpoint exists.')
    p.add_argument('--mirror_s2_optical', action='store_true',
                   help='Mirror the existing S2-Optical reports into '
                        'reports/minijepas/s2_optical/ and exit.')
    p.add_argument('--skip', type=int, nargs='*', default=[],
                   choices=[11, 12, 13],
                   help='Skip specific evaluation steps (e.g. --skip 13).')
    # Pass-through flags for script 11
    p.add_argument('--skip_rf',        action='store_true')
    p.add_argument('--skip_block',     action='store_true')
    p.add_argument('--recompute_emb',  action='store_true')
    # Pass-through flags for script 12
    p.add_argument('--skip_local',      action='store_true')
    p.add_argument('--skip_multiscale', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    if args.mirror_s2_optical:
        mirror_s2_optical()
        if not args.modality:
            return  # mirror-only invocation

    if not args.modality:
        raise SystemExit('--modality is required (or use --mirror_s2_optical alone).')

    s5  = load_module('5_hydrojepa_pretrain.py',         'pretrain')
    s11 = load_module('11_hydrojepa_interpretability.py', 's11')
    s12 = load_module('12_hydrojepa_manifold_geometry.py', 's12')
    s13 = load_module('13_hydrojepa_ae_complementarity.py', 's13')

    if args.modality == 'all':
        targets = []
        for m in MODALITIES:
            if (CKPT_DIR / f'hydrojepa_{m}_best.pt').exists():
                targets.append(m)
            else:
                logging.warning(f'skipping {m}: no checkpoint at '
                                f'{CKPT_DIR}/hydrojepa_{m}_best.pt')
        if not targets:
            raise SystemExit('No Mini-JEPA checkpoints found.')
        logging.info(f'evaluating Mini-JEPAs: {targets}')
    else:
        targets = [args.modality]

    results = []
    for m in targets:
        try:
            evaluate_one_modality(m, args, s5, s11, s12, s13)
            results.append((m, 'ok', None))
        except Exception as e:
            # Print the full exception details — script 11/12/13 errors can
            # be multi-line (state_dict mismatches list every key) and we
            # need the full text to diagnose.
            import traceback
            full_tb = traceback.format_exc()
            logging.error(f'[{m}] FAILED:\n{full_tb}')
            err_short = f'{type(e).__name__}: {str(e).split(chr(10), 1)[0][:200]}'
            results.append((m, 'failed', err_short))

    logging.info('')
    logging.info('=' * 70)
    logging.info('EVALUATION SUMMARY')
    logging.info('=' * 70)
    for m, status, err in results:
        line = f'  {m:<14} {status}'
        if err:
            line += f'  | {err}'
        logging.info(line)
    n_ok = sum(1 for _, s, _ in results if s == 'ok')
    logging.info(f'  total: {n_ok}/{len(results)} succeeded')


if __name__ == '__main__':
    main()

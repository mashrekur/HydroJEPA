"""
HydroJEPA — Physical Interpretability of Embedding Dimensions
==============================================================

Mirrors the methodology of Rahman (2026) "Physically Interpretable AlphaEarth
Foundation Model Embeddings" (Paper 1). Adapted for HydroJEPA's smaller scale
and 9,704-patch CONUS corpus.

Three convergent measures of dimension-variable relationship:
  1. Spearman rank correlation (linear, monotonic)
  2. Random Forest permutation importance (nonlinear)
  3. Spatial block-CV R² (robustness to spatial autocorrelation)

Then builds the dimension dictionary by reconciling Spearman and RF rankings,
matching the format of the AE paper for direct comparison.

Inputs:
  data/hydrojepa/labels.parquet        — 64-d HydroJEPA + 64-d AE + env vars
  checkpoints/hydrojepa_full_best.pt   — frozen encoder

Outputs (under reports/interpretability/):
  hydrojepa_spearman_matrix.csv         — 64 dims x N_env matrix (signed rho)
  hydrojepa_pval_matrix.csv             — same shape, p-values
  hydrojepa_rf_importance.csv           — permutation importance per (dim, var)
  hydrojepa_rf_r2.csv                   — RF cross-val R² per env variable
  hydrojepa_rf_r2_blocked.csv           — same with 2-deg block CV
  hydrojepa_dimension_dictionary.csv    — semantic label per dimension
  hydrojepa_corr_heatmap.png            — Figure 1 in paper 1 style
  hydrojepa_dimension_dict_table.png    — sortable dictionary visualization

Run:
  python 11_hydrojepa_interpretability.py
  python 11_hydrojepa_interpretability.py --skip_rf       # use cached RF
  python 11_hydrojepa_interpretability.py --recompute_emb # recompute HJ embeddings
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

os.environ['CPL_LOG'] = os.devnull
logging.getLogger('rasterio').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*Photometric.*')
warnings.filterwarnings('ignore', message='.*ExtraSamples.*')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, KFold

from tqdm import tqdm

# Reuse encoder from script 5
import importlib.util
spec = importlib.util.spec_from_file_location('pretrain', '5_hydrojepa_pretrain.py')
pretrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pretrain)


# ────────────────────────────────────────────────────────────────────────────
# Config — matches Paper 1 style + variable category palette
# ────────────────────────────────────────────────────────────────────────────
class Cfg:
    DATA_DIR    = Path('data/hydrojepa')
    LABELS_FILE = DATA_DIR / 'labels.parquet'
    MANIFEST    = DATA_DIR / 'manifest.parquet'
    CKPT_DIR    = Path('checkpoints')
    REPORT_DIR  = Path('reports/interpretability')
    EMB_CACHE   = REPORT_DIR / 'hydrojepa_embeddings.npy'

    HJ_CKPT     = CKPT_DIR / 'hydrojepa_full_best.pt'

    HJ_DIMS     = [f'H{i:02d}' for i in range(64)]    # HydroJEPA dim labels

    # Hydrology-relevant subset of variables we have in labels.parquet.
    # Paper 1 had 26 vars on the planetary AE grid; we have what we have.
    ENV_VARS = [
        'smap_sm', 'elevation', 'prism_ppt_mm', 'prism_tmean_c',
        'aridity_proxy', 'koppen', 'nlcd_class',
    ]
    ENV_LABELS = {
        'smap_sm':       'Soil Moisture (SMAP)',
        'elevation':     'Elevation (m)',
        'prism_ppt_mm':  'Precipitation (mm/yr)',
        'prism_tmean_c': 'Temperature Mean (°C)',
        'aridity_proxy': 'Aridity (P/PET proxy)',
        'koppen':        'Köppen Class',
        'nlcd_class':    'NLCD Land Cover',
    }
    ENV_CATEGORY = {
        'smap_sm':       'Hydrology',
        'elevation':     'Terrain',
        'prism_ppt_mm':  'Climate',
        'prism_tmean_c': 'Temperature',
        'aridity_proxy': 'Hydrology',
        'koppen':        'Climate',
        'nlcd_class':    'Vegetation',
    }
    CATEGORY_COLORS = {
        'Terrain':     '#8C564B',
        'Soil':        '#C49C6B',
        'Vegetation':  '#2CA02C',
        'Temperature': '#D62728',
        'Climate':     '#1F77B4',
        'Hydrology':   '#17BECF',
        'Urban':       '#7F7F7F',
        'Radiation':   '#BCBD22',
    }

    # RF parameters — scaled for our smaller corpus
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH    = 12
    RF_MIN_SAMPLES_LEAF = 20
    RF_N_JOBS       = 4
    PERM_N_REPEATS  = 10
    N_FOLDS         = 5
    SPATIAL_BLOCK_DEG = 2.0
    MIN_VALID       = 200
    SEED            = 42


def banner(msg):
    logging.info('')
    logging.info('=' * 70)
    logging.info(f'  {msg}')
    logging.info('=' * 70)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# ────────────────────────────────────────────────────────────────────────────
# 1. Build the joint table: HydroJEPA embeddings + env vars
# ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_hydrojepa_embeddings(patch_ids: list[str], device) -> np.ndarray:
    """Run frozen encoder over patches, return mean-pooled (n, 64)."""
    ckpt = torch.load(Cfg.HJ_CKPT, map_location=device, weights_only=False)
    encoder = pretrain.ViTEncoder().to(device)
    encoder.load_state_dict(ckpt['context_enc'])
    encoder.eval()
    ds = pretrain.HydroJEPADataset(patch_ids, ckpt['stats'])
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    embs = []
    for x, _ in tqdm(loader, desc='  encoding'):
        e = encoder(x.to(device, non_blocking=True)).mean(dim=1)
        embs.append(e.cpu().numpy())
    arr = np.concatenate(embs, axis=0)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def build_dataset(recompute_emb: bool = False) -> pd.DataFrame:
    """Returns df with columns: patch_id, lon, lat, H00..H63, env vars."""
    Cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(Cfg.MANIFEST)
    ok = manifest[manifest.status.isin(['ok', 'cached'])]
    labels = pd.read_parquet(Cfg.LABELS_FILE)
    df = labels.merge(ok[['patch_id']], on='patch_id')

    # Coerce env columns
    for c in Cfg.ENV_VARS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')

    # HydroJEPA embeddings (cache to disk so re-runs are fast)
    if Cfg.EMB_CACHE.exists() and not recompute_emb:
        logging.info(f'Loading cached HydroJEPA embeddings: {Cfg.EMB_CACHE}')
        emb = np.load(Cfg.EMB_CACHE)
    else:
        logging.info('Computing HydroJEPA embeddings (one-time)...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        emb = compute_hydrojepa_embeddings(df.patch_id.tolist(), device)
        np.save(Cfg.EMB_CACHE, emb)
        logging.info(f'Cached embeddings: {Cfg.EMB_CACHE}')

    if emb.shape[0] != len(df):
        raise RuntimeError(f'Embedding rows {emb.shape[0]} != df rows {len(df)}')

    for j, col in enumerate(Cfg.HJ_DIMS):
        df[col] = emb[:, j]

    return df


# ────────────────────────────────────────────────────────────────────────────
# 2. Spearman correlation, mirroring Paper 1
# ────────────────────────────────────────────────────────────────────────────
def compute_spearman(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    banner(f'Spearman correlation: {len(Cfg.HJ_DIMS)} dims × {len(Cfg.ENV_VARS)} vars')
    corr = np.full((len(Cfg.HJ_DIMS), len(Cfg.ENV_VARS)), np.nan)
    pval = np.full_like(corr, np.nan)

    for j, ev in enumerate(tqdm(Cfg.ENV_VARS, desc='  vars')):
        if ev not in df.columns:
            continue
        ev_v = df[ev].values
        ev_m = np.isfinite(ev_v)
        if ev_m.sum() < Cfg.MIN_VALID:
            continue
        for i, ad in enumerate(Cfg.HJ_DIMS):
            ad_v = df[ad].values
            both = ev_m & np.isfinite(ad_v)
            if both.sum() < Cfg.MIN_VALID:
                continue
            r, p = spearmanr(ad_v[both], ev_v[both])
            corr[i, j] = r
            pval[i, j] = p

    cdf = pd.DataFrame(corr, index=Cfg.HJ_DIMS, columns=Cfg.ENV_VARS)
    pdf = pd.DataFrame(pval, index=Cfg.HJ_DIMS, columns=Cfg.ENV_VARS)
    n_05 = (np.abs(corr[np.isfinite(corr)]) > 0.5).sum()
    n_07 = (np.abs(corr[np.isfinite(corr)]) > 0.7).sum()
    logging.info(f'  |ρ| > 0.5: {n_05},  |ρ| > 0.7: {n_07}')

    flat = []
    for i, d in enumerate(Cfg.HJ_DIMS):
        for j, v in enumerate(Cfg.ENV_VARS):
            if np.isfinite(corr[i, j]):
                flat.append((d, v, corr[i, j]))
    flat.sort(key=lambda x: abs(x[2]), reverse=True)
    logging.info('  Top 10 (|ρ|):')
    for d, v, r in flat[:10]:
        logging.info(f'    {d}  ×  {Cfg.ENV_LABELS[v]:25s}  ρ = {r:+.4f}')
    return cdf, pdf


# ────────────────────────────────────────────────────────────────────────────
# 3. Random Forest permutation importance + cross-val R²
# ────────────────────────────────────────────────────────────────────────────
def compute_rf_importance(df: pd.DataFrame, rng) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    banner(f'Random Forest permutation importance')
    imp = pd.DataFrame(np.nan, index=Cfg.HJ_DIMS, columns=Cfg.ENV_VARS)
    rf_r2 = []
    rf_details = []

    for ev in tqdm(Cfg.ENV_VARS, desc='  vars'):
        if ev not in df.columns:
            continue
        cols = Cfg.HJ_DIMS + [ev]
        sub = df[cols].dropna()
        if len(sub) < Cfg.MIN_VALID:
            rf_r2.append({'variable': ev, 'r2_cv': np.nan,
                          'r2_cv_std': np.nan, 'n_samples': len(sub)})
            continue
        X = sub[Cfg.HJ_DIMS].values.astype('float32')
        y = sub[ev].values.astype('float32')

        rf = RandomForestRegressor(
            n_estimators=Cfg.RF_N_ESTIMATORS,
            max_depth=Cfg.RF_MAX_DEPTH,
            min_samples_leaf=Cfg.RF_MIN_SAMPLES_LEAF,
            n_jobs=Cfg.RF_N_JOBS,
            random_state=Cfg.SEED,
        )
        cv = cross_val_score(rf, X, y, cv=Cfg.N_FOLDS,
                             scoring='r2', n_jobs=Cfg.RF_N_JOBS)
        rf_r2.append({'variable': ev, 'r2_cv': float(cv.mean()),
                      'r2_cv_std': float(cv.std()), 'n_samples': len(X)})
        rf.fit(X, y)
        perm = permutation_importance(
            rf, X, y, n_repeats=Cfg.PERM_N_REPEATS,
            random_state=Cfg.SEED, n_jobs=Cfg.RF_N_JOBS)
        for i, ad in enumerate(Cfg.HJ_DIMS):
            imp.loc[ad, ev] = perm.importances_mean[i]
        top3 = np.argsort(perm.importances_mean)[::-1][:3]
        rf_details.append({
            'variable': ev,
            'top1': Cfg.HJ_DIMS[top3[0]], 'top1_imp': float(perm.importances_mean[top3[0]]),
            'top2': Cfg.HJ_DIMS[top3[1]], 'top2_imp': float(perm.importances_mean[top3[1]]),
            'top3': Cfg.HJ_DIMS[top3[2]], 'top3_imp': float(perm.importances_mean[top3[2]]),
        })
        logging.info(f'  {Cfg.ENV_LABELS[ev]:25s}: R²={cv.mean():.3f}±{cv.std():.3f}  '
                     f'top: {Cfg.HJ_DIMS[top3[0]]}, {Cfg.HJ_DIMS[top3[1]]}, {Cfg.HJ_DIMS[top3[2]]}')

    return imp, pd.DataFrame(rf_r2), pd.DataFrame(rf_details)


# ────────────────────────────────────────────────────────────────────────────
# 4. Spatial block cross-validation (2-deg blocks, mirrors Paper 1)
# ────────────────────────────────────────────────────────────────────────────
def compute_spatial_cv(df: pd.DataFrame, rng) -> pd.DataFrame:
    banner(f'Spatial block CV ({Cfg.SPATIAL_BLOCK_DEG}° blocks)')
    df = df.copy()
    df['block_lon'] = (df.lon // Cfg.SPATIAL_BLOCK_DEG).astype(int)
    df['block_lat'] = (df.lat // Cfg.SPATIAL_BLOCK_DEG).astype(int)
    df['block_id']  = df['block_lon'] * 1000 + df['block_lat']
    block_ids = df['block_id'].unique()
    logging.info(f'  {len(block_ids)} blocks total')

    rows = []
    for ev in tqdm(Cfg.ENV_VARS, desc='  vars'):
        if ev not in df.columns:
            continue
        sub = df[Cfg.HJ_DIMS + [ev, 'block_id']].dropna()
        if len(sub) < Cfg.MIN_VALID:
            continue
        block_groups = list(sub.groupby('block_id'))
        rng.shuffle(block_groups)

        # 5-fold block CV
        fold_r2 = []
        fold_size = len(block_groups) // Cfg.N_FOLDS
        for k in range(Cfg.N_FOLDS):
            test_blocks = block_groups[k * fold_size:(k + 1) * fold_size]
            test_idx  = pd.concat([g for _, g in test_blocks]).index
            train_idx = sub.index.difference(test_idx)
            if len(test_idx) < 50 or len(train_idx) < 200:
                continue
            X_tr = sub.loc[train_idx, Cfg.HJ_DIMS].values
            y_tr = sub.loc[train_idx, ev].values
            X_te = sub.loc[test_idx,  Cfg.HJ_DIMS].values
            y_te = sub.loc[test_idx,  ev].values
            rf = RandomForestRegressor(
                n_estimators=Cfg.RF_N_ESTIMATORS,
                max_depth=Cfg.RF_MAX_DEPTH,
                min_samples_leaf=Cfg.RF_MIN_SAMPLES_LEAF,
                n_jobs=Cfg.RF_N_JOBS,
                random_state=Cfg.SEED,
            )
            rf.fit(X_tr, y_tr)
            yhat = rf.predict(X_te)
            ss_res = float(((y_te - yhat) ** 2).sum())
            ss_tot = float(((y_te - y_te.mean()) ** 2).sum() + 1e-12)
            fold_r2.append(1.0 - ss_res / ss_tot)
        if fold_r2:
            rows.append({
                'variable': ev,
                'r2_block_mean': float(np.mean(fold_r2)),
                'r2_block_std':  float(np.std(fold_r2)),
                'n_folds': len(fold_r2),
            })
            logging.info(f'  {Cfg.ENV_LABELS[ev]:25s}: '
                         f'block R²={np.mean(fold_r2):.3f}±{np.std(fold_r2):.3f}')
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# 5. Build the dimension dictionary (Paper 1 style)
# ────────────────────────────────────────────────────────────────────────────
def build_dictionary(corr: pd.DataFrame, imp: pd.DataFrame) -> pd.DataFrame:
    banner('HydroJEPA dimension dictionary')
    rows = []
    for d in Cfg.HJ_DIMS:
        sp_row = corr.loc[d].dropna()
        rf_row = imp.loc[d].dropna().astype(float)
        e = {'dimension': d}
        if len(sp_row):
            sp_sorted = sp_row.abs().sort_values(ascending=False)
            top = sp_sorted.index[0]
            e['sp_primary']  = top
            e['sp_rho']      = float(sp_row[top])
            e['sp_abs_max']  = float(abs(sp_row[top]))
            e['sp_category'] = Cfg.ENV_CATEGORY.get(top, '?')
            for r in range(min(3, len(sp_sorted))):
                v = sp_sorted.index[r]
                e[f'sp_top{r+1}_var'] = v
                e[f'sp_top{r+1}_rho'] = float(sp_row[v])
        else:
            e.update(sp_primary='N/A', sp_rho=np.nan, sp_abs_max=0,
                     sp_category='?')
        if len(rf_row):
            top = rf_row.idxmax()
            e['rf_primary']  = top
            e['rf_imp']      = float(rf_row[top])
            e['rf_category'] = Cfg.ENV_CATEGORY.get(top, '?')
        else:
            e.update(rf_primary='N/A', rf_imp=np.nan, rf_category='?')
        e['agree'] = e.get('sp_primary') == e.get('rf_primary')
        rows.append(e)
    dd = pd.DataFrame(rows).sort_values('sp_abs_max', ascending=False)
    n_agree = int(dd['agree'].sum())
    logging.info(f'  Spearman/RF agreement: {n_agree}/{len(Cfg.HJ_DIMS)} '
                 f'({100*n_agree/len(Cfg.HJ_DIMS):.0f}%)')
    for _, r in dd.head(15).iterrows():
        sp_lbl = Cfg.ENV_LABELS.get(r['sp_primary'], r['sp_primary'])
        rf_lbl = Cfg.ENV_LABELS.get(r['rf_primary'], r['rf_primary'])
        tag = '✓' if r['agree'] else '✗'
        logging.info(f'    {r["dimension"]}: SP→{sp_lbl:20s} '
                     f'(ρ={r["sp_rho"]:+.3f})  RF→{rf_lbl:20s} [{tag}]')
    return dd


# ────────────────────────────────────────────────────────────────────────────
# 6. Figures (match Paper 1 visual style)
# ────────────────────────────────────────────────────────────────────────────
def plt_setup():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.linewidth': 0.8, 'axes.labelsize': 12,
        'axes.titlesize': 13, 'axes.titleweight': 'bold',
        'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
    })


def fig_corr_heatmap(corr: pd.DataFrame, save_path: Path):
    """64 × N_env signed-rho heatmap, dims sorted by |max ρ|."""
    plt_setup()
    order = corr.abs().max(axis=1).sort_values(ascending=False).index
    M = corr.loc[order]

    fig, ax = plt.subplots(figsize=(7, 13), dpi=150)
    cmap = plt.cm.RdBu_r
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    im = ax.imshow(M.values, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(np.arange(len(M.columns)))
    ax.set_xticklabels([Cfg.ENV_LABELS.get(v, v) for v in M.columns],
                       rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(len(M.index)))
    ax.set_yticklabels(M.index, fontsize=7)
    ax.set_xlabel('Environmental variable')
    ax.set_ylabel('HydroJEPA dimension (sorted by |max ρ|)')
    ax.set_title('HydroJEPA dimensions vs environmental variables\n(Spearman ρ)')
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label('Spearman ρ')

    # Annotate strong cells (|ρ|>0.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isfinite(v) and abs(v) > 0.5:
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                        fontsize=6, color='white' if abs(v) > 0.65 else 'black')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'  saved {save_path.name}')


def fig_dimension_dict(dd: pd.DataFrame, save_path: Path):
    """Stylized table of top-30 dimensions colored by category."""
    plt_setup()
    top = dd.head(30).copy()
    top['sp_label'] = top['sp_primary'].map(Cfg.ENV_LABELS).fillna(top['sp_primary'])
    top['rf_label'] = top['rf_primary'].map(Cfg.ENV_LABELS).fillna(top['rf_primary'])

    fig, ax = plt.subplots(figsize=(11, 12), dpi=150)
    ax.axis('off')

    # Column headers
    cols = ['Dim', 'Spearman primary', '|ρ|', 'RF primary', 'RF imp', 'Agreement']
    col_x = [0.05, 0.18, 0.42, 0.55, 0.78, 0.90]
    for x, c in zip(col_x, cols):
        ax.text(x, 0.98, c, fontsize=11, fontweight='bold',
                transform=ax.transAxes)
    ax.axhline(0.965, color='black', lw=0.8, transform=ax.transAxes)

    for i, (_, r) in enumerate(top.iterrows()):
        y = 0.94 - i * 0.030
        sp_color = Cfg.CATEGORY_COLORS.get(r['sp_category'], '#999')
        rf_color = Cfg.CATEGORY_COLORS.get(r['rf_category'], '#999')
        ax.text(col_x[0], y, r['dimension'], fontsize=10,
                fontweight='bold', transform=ax.transAxes)
        ax.text(col_x[1], y, r['sp_label'], fontsize=10,
                color=sp_color, transform=ax.transAxes)
        ax.text(col_x[2], y, f'{r["sp_abs_max"]:.3f}', fontsize=10,
                transform=ax.transAxes)
        ax.text(col_x[3], y, r['rf_label'], fontsize=10,
                color=rf_color, transform=ax.transAxes)
        ax.text(col_x[4], y, f'{r["rf_imp"]:.4f}',
                fontsize=10, transform=ax.transAxes)
        mark = '✓' if r['agree'] else '✗'
        mark_color = '#2CA02C' if r['agree'] else '#D62728'
        ax.text(col_x[5], y, mark, fontsize=12,
                color=mark_color, fontweight='bold', transform=ax.transAxes)

    # Category legend
    handles = [Patch(color=Cfg.CATEGORY_COLORS[c], label=c)
               for c in ['Hydrology', 'Climate', 'Temperature', 'Terrain',
                         'Vegetation', 'Soil']]
    ax.legend(handles=handles, loc='lower center', ncol=6, frameon=False,
              bbox_to_anchor=(0.5, -0.02))
    ax.set_title('HydroJEPA Dimension Dictionary (top 30 by Spearman |ρ|)',
                 fontsize=14, fontweight='bold', pad=10)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'  saved {save_path.name}')


def fig_rf_r2_summary(rf_r2: pd.DataFrame, blk_r2: pd.DataFrame | None,
                      save_path: Path):
    """Bar chart: per-variable RF R² (random) vs spatial-block."""
    plt_setup()
    if rf_r2.empty:
        return
    rf_r2 = rf_r2.set_index('variable')
    if blk_r2 is not None and not blk_r2.empty:
        blk_r2 = blk_r2.set_index('variable')
        merged = rf_r2.join(blk_r2[['r2_block_mean', 'r2_block_std']],
                            how='left')
    else:
        merged = rf_r2.copy()
        merged['r2_block_mean'] = np.nan
        merged['r2_block_std']  = np.nan

    merged = merged.sort_values('r2_cv', ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    y = np.arange(len(merged))
    ax.barh(y - 0.18, merged['r2_cv'], height=0.34,
            xerr=merged['r2_cv_std'], color='#4C72B0',
            label='Random k-fold CV')
    ax.barh(y + 0.18, merged['r2_block_mean'], height=0.34,
            xerr=merged['r2_block_std'], color='#DD8452',
            label=f'{Cfg.SPATIAL_BLOCK_DEG}° spatial block CV')
    ax.set_yticks(y)
    ax.set_yticklabels([Cfg.ENV_LABELS.get(v, v) for v in merged.index],
                       fontsize=10)
    ax.set_xlabel('R² (HydroJEPA → variable)')
    ax.set_title('Per-variable predictability (RF, 5-fold CV)')
    ax.set_xlim(-0.05, 1.0)
    ax.axvline(0, color='black', lw=0.5)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'  saved {save_path.name}')


# ────────────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--skip_rf', action='store_true',
                   help='Skip RF, use cached importance matrix if present')
    p.add_argument('--skip_block', action='store_true',
                   help='Skip spatial-block CV')
    p.add_argument('--recompute_emb', action='store_true',
                   help='Recompute HydroJEPA embeddings even if cached')
    return p.parse_args()


def main():
    args = parse_args()
    Cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(Cfg.SEED)

    df = build_dataset(recompute_emb=args.recompute_emb)
    logging.info(f'Joint table: {len(df):,} rows, '
                 f'{len(Cfg.HJ_DIMS)} HJ dims, {len(Cfg.ENV_VARS)} env vars')

    # 1. Spearman
    corr_df, pval_df = compute_spearman(df)
    corr_df.to_csv(Cfg.REPORT_DIR / 'hydrojepa_spearman_matrix.csv')
    pval_df.to_csv(Cfg.REPORT_DIR / 'hydrojepa_pval_matrix.csv')

    # 2. RF
    imp_path = Cfg.REPORT_DIR / 'hydrojepa_rf_importance.csv'
    r2_path  = Cfg.REPORT_DIR / 'hydrojepa_rf_r2.csv'
    if args.skip_rf and imp_path.exists() and r2_path.exists():
        logging.info('Skipping RF (using cached files)')
        imp_df = pd.read_csv(imp_path, index_col=0)
        rf_r2_df = pd.read_csv(r2_path)
    else:
        imp_df, rf_r2_df, rf_details = compute_rf_importance(df, rng)
        imp_df.to_csv(imp_path)
        rf_r2_df.to_csv(r2_path, index=False)
        rf_details.to_csv(Cfg.REPORT_DIR / 'hydrojepa_rf_details.csv', index=False)

    # 3. Spatial block CV
    if not args.skip_block:
        blk_path = Cfg.REPORT_DIR / 'hydrojepa_rf_r2_blocked.csv'
        blk_df = compute_spatial_cv(df, rng)
        blk_df.to_csv(blk_path, index=False)
    else:
        blk_df = None

    # 4. Dictionary
    dd = build_dictionary(corr_df, imp_df)
    dd.to_csv(Cfg.REPORT_DIR / 'hydrojepa_dimension_dictionary.csv', index=False)

    # 5. Figures
    fig_corr_heatmap(corr_df, Cfg.REPORT_DIR / 'hydrojepa_corr_heatmap.png')
    fig_dimension_dict(dd, Cfg.REPORT_DIR / 'hydrojepa_dimension_dict_table.png')
    fig_rf_r2_summary(rf_r2_df, blk_df,
                      Cfg.REPORT_DIR / 'hydrojepa_rf_r2_summary.png')

    # Manifest
    summary = {
        'n_patches':      int(len(df)),
        'n_dimensions':   len(Cfg.HJ_DIMS),
        'n_variables':    len([v for v in Cfg.ENV_VARS if v in df.columns]),
        'n_strong_corr':  int((np.abs(corr_df.values) > 0.5).sum()),
        'n_very_strong':  int((np.abs(corr_df.values) > 0.7).sum()),
        'spearman_rf_agreement': int(dd['agree'].sum()),
        'top_dimension':  dd.iloc[0].to_dict(),
    }
    with open(Cfg.REPORT_DIR / 'hydrojepa_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    banner('Done')
    logging.info(f'Outputs in {Cfg.REPORT_DIR}/')


if __name__ == '__main__':
    main()

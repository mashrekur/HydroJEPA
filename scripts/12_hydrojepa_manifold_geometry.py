"""
HydroJEPA — Manifold Geometry Characterization
================================================

Mirrors the methodology of Rahman et al (2026) "Characterizing AlphaEarth
Embedding Geometry" (Paper 2). Adapted for HydroJEPA's 9,704-patch CONUS
corpus.

Three complementary geometric measurements:
  1. GLOBAL — eigendecomposition of cov matrix; participation ratio
  2. LOCAL — Levina-Bickel intrinsic dim, local PCA at probe locations,
             tangent space angles, local-global PC1 alignment
  3. MULTI-SCALE — local PCA at 4 neighborhood sizes (k = 20, 50, 100, 500)

All figures match Paper 2's visual style (4-panel layouts, colormaps, legends)
to enable direct cross-paper comparison.

Inputs:
  reports/interpretability/hydrojepa_embeddings.npy   (cached by script 11)
  data/hydrojepa/labels.parquet                       (patch coords + env vars)

Outputs (under reports/manifold_geometry/):
  hydrojepa_global_covariance.csv
  hydrojepa_local_pca.csv
  hydrojepa_multiscale.csv
  fig_global_covariance.png        — eigenspectrum + participation ratio
  fig_intrinsic_dimensionality.png — Levina-Bickel ID across CONUS + by elev
  fig_local_geometry.png           — local PR, tangent angles, alignment
  fig_multiscale_alignment.png     — alignment vs neighborhood size
  fig_dominant_dimension_map.png   — which dim dominates locally, mapped

Run:
  python 12_hydrojepa_manifold_geometry.py
  python 12_hydrojepa_manifold_geometry.py --skip_local   # quick re-render
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

os.environ['CPL_LOG'] = os.devnull
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from tqdm import tqdm


# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────
class Cfg:
    DATA_DIR    = Path('data/hydrojepa')
    LABELS_FILE = DATA_DIR / 'labels.parquet'
    INTERP_DIR  = Path('reports/interpretability')
    EMB_CACHE   = INTERP_DIR / 'hydrojepa_embeddings.npy'
    REPORT_DIR  = Path('reports/manifold_geometry')

    HJ_DIMS = [f'H{i:02d}' for i in range(64)]
    N_DIMS  = 64

    # Probes — scaled for our 9,704-patch corpus
    N_PROBES        = 2_000
    K_LOCAL         = 100
    K_INTRINSIC     = 20
    SCALE_LIST      = [20, 50, 100, 500]
    SEED            = 42

    CONUS_EXTENT = [-125.0, -66.5, 24.5, 49.5]


def banner(msg):
    logging.info('')
    logging.info('=' * 70)
    logging.info(f'  {msg}')
    logging.info('=' * 70)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def plt_setup():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.linewidth': 0.8, 'axes.labelsize': 12,
        'axes.titlesize': 13, 'axes.titleweight': 'bold',
        'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
    })


# ────────────────────────────────────────────────────────────────────────────
# Load embeddings + co-located metadata
# ────────────────────────────────────────────────────────────────────────────
def load_data() -> tuple[np.ndarray, pd.DataFrame]:
    Cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not Cfg.EMB_CACHE.exists():
        raise FileNotFoundError(
            f'{Cfg.EMB_CACHE} missing. Run script 11 first to generate.')
    E = np.load(Cfg.EMB_CACHE)

    labels = pd.read_parquet(Cfg.LABELS_FILE).reset_index(drop=True)
    if len(labels) != E.shape[0]:
        # Filter labels to match cached embeddings (they were built from
        # the same eval set in script 11)
        from pathlib import Path as P
        manifest = pd.read_parquet(Cfg.DATA_DIR / 'manifest.parquet')
        ok_ids = manifest[manifest.status.isin(['ok', 'cached'])].patch_id
        labels = labels[labels.patch_id.isin(ok_ids)].reset_index(drop=True)
    if len(labels) != E.shape[0]:
        raise RuntimeError(f'labels rows {len(labels)} vs embeddings {E.shape[0]}')

    logging.info(f'Loaded {E.shape[0]:,} × {E.shape[1]} embeddings')
    return E, labels


# ────────────────────────────────────────────────────────────────────────────
# 1. Global covariance + participation ratio
# ────────────────────────────────────────────────────────────────────────────
def global_covariance(E: np.ndarray) -> dict:
    banner('Global covariance & participation ratio')
    cov = np.cov(E, rowvar=False)
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]                  # descending
    eigvecs = eigvecs[:, ::-1]
    eigvals = np.maximum(eigvals, 0)
    var_ratio = eigvals / eigvals.sum()
    cum_var = np.cumsum(var_ratio)

    pos = eigvals[eigvals > 1e-12]
    pr = float((pos.sum() ** 2) / (pos ** 2).sum())

    n80 = int(np.searchsorted(cum_var, 0.80) + 1)
    n90 = int(np.searchsorted(cum_var, 0.90) + 1)
    n95 = int(np.searchsorted(cum_var, 0.95) + 1)
    logging.info(f'  PR = {pr:.2f}   n80={n80}, n90={n90}, n95={n95} (out of 64)')
    logging.info(f'  PC1 explains {var_ratio[0]*100:.1f}%, '
                 f'PC2 {var_ratio[1]*100:.1f}%, PC3 {var_ratio[2]*100:.1f}%')

    return {
        'eigvals': eigvals, 'eigvecs': eigvecs, 'var_ratio': var_ratio,
        'cum_var': cum_var, 'pr': pr, 'n80': n80, 'n90': n90, 'n95': n95,
    }


# ────────────────────────────────────────────────────────────────────────────
# 2. Levina-Bickel intrinsic dimensionality
# ────────────────────────────────────────────────────────────────────────────
def intrinsic_dim_lb(E: np.ndarray, k: int) -> np.ndarray:
    """Per-point Levina-Bickel MLE intrinsic dimensionality."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(E)
    d, _ = nn.kneighbors(E)
    d = d[:, 1:]                              # drop self
    rk = d[:, -1:]
    log_ratios = np.log(np.maximum(rk, 1e-12) / np.maximum(d, 1e-12))
    inv_id = log_ratios[:, :-1].mean(axis=1)
    inv_id = np.where(inv_id > 1e-6, inv_id, np.nan)
    return 1.0 / inv_id


def compute_intrinsic(E: np.ndarray) -> dict:
    banner(f'Intrinsic dimensionality (Levina-Bickel, k={Cfg.K_INTRINSIC})')
    id_per_point = intrinsic_dim_lb(E, Cfg.K_INTRINSIC)
    valid = np.isfinite(id_per_point)
    logging.info(f'  mean ID = {np.nanmean(id_per_point):.2f} '
                 f'(median {np.nanmedian(id_per_point):.2f})')
    return {'id_per_point': id_per_point, 'mean_id': float(np.nanmean(id_per_point)),
            'median_id': float(np.nanmedian(id_per_point)),
            'valid_frac': float(valid.mean())}


# ────────────────────────────────────────────────────────────────────────────
# 3. Local PCA at probe locations
# ────────────────────────────────────────────────────────────────────────────
def local_pca(E: np.ndarray, labels: pd.DataFrame,
              global_evecs: np.ndarray) -> pd.DataFrame:
    banner(f'Local PCA at {Cfg.N_PROBES} probes (k={Cfg.K_LOCAL})')
    rng = np.random.default_rng(Cfg.SEED)

    # Stratify probes by elevation if available (mirrors Paper 2)
    if 'elevation' in labels.columns:
        elev = pd.to_numeric(labels['elevation'], errors='coerce').to_numpy()
        bins = [-100, 100, 500, 1000, 2000, 5000]
        groups = pd.cut(elev, bins=bins, labels=False)
        per_group = Cfg.N_PROBES // len(bins[:-1])
        idx = []
        for g in range(len(bins) - 1):
            mask = (groups == g) & np.isfinite(elev)
            if mask.sum() == 0:
                continue
            ids = np.where(mask)[0]
            n = min(per_group, len(ids))
            idx.extend(rng.choice(ids, n, replace=False))
        probe_idx = np.array(idx)
    else:
        probe_idx = rng.choice(len(E), Cfg.N_PROBES, replace=False)

    nn = NearestNeighbors(n_neighbors=Cfg.K_LOCAL + 1).fit(E)
    global_pc1 = global_evecs[:, 0]
    global_pc2 = global_evecs[:, 1]

    rows = []
    for pi, pidx in enumerate(tqdm(probe_idx, desc='  probes')):
        _, nbr = nn.kneighbors(E[pidx:pidx + 1])
        nbr_idx = nbr[0, 1:]
        nbhd = E[nbr_idx]
        pca = PCA(n_components=min(20, Cfg.K_LOCAL - 1)).fit(nbhd)

        eigs = pca.explained_variance_
        eigs = eigs[eigs > 1e-12]
        local_pr = (eigs.sum() ** 2) / (eigs ** 2).sum() if len(eigs) else np.nan
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n80_local = int(np.searchsorted(cumvar, 0.80) + 1)

        local_pc1 = pca.components_[0]
        local_pc2 = pca.components_[1] if pca.components_.shape[0] > 1 else np.zeros(64)
        align_pc1 = float(abs(np.dot(local_pc1, global_pc1)))
        align_pc2 = float(abs(np.dot(local_pc2, global_pc2)))

        # Top contributing dim to local PC1
        dom_idx = int(np.argmax(np.abs(local_pc1)))

        # Distance to nearest probe (for spatial reference)
        rows.append({
            'probe_idx': int(pidx),
            'lon': float(labels.lon.iloc[pidx]),
            'lat': float(labels.lat.iloc[pidx]),
            'local_pr': float(local_pr),
            'local_n80': n80_local,
            'pc1_var': float(pca.explained_variance_ratio_[0]),
            'align_pc1': align_pc1,
            'align_pc2': align_pc2,
            'dom_dim': Cfg.HJ_DIMS[dom_idx],
            'dom_dim_idx': dom_idx,
            'dom_weight': float(local_pc1[dom_idx]),
            'local_pc1': local_pc1.tolist(),
        })
    df = pd.DataFrame(rows)

    # Tangent angles between consecutive probe PC1 directions
    pcs = np.array([r['local_pc1'] for r in rows])
    angles = []
    for i in range(len(pcs) - 1):
        c = abs(np.dot(pcs[i], pcs[i + 1]))
        c = np.clip(c, -1, 1)
        angles.append(np.degrees(np.arccos(c)))
    df['tangent_angle_to_next'] = angles + [np.nan]

    rand_baseline = np.sqrt(2.0 / (np.pi * Cfg.N_DIMS))
    logging.info(f'  mean local PR = {df.local_pr.mean():.2f}')
    logging.info(f'  mean PC1 alignment = {df.align_pc1.mean():.3f} '
                 f'(random baseline {rand_baseline:.3f})')
    logging.info(f'  median tangent angle = {df.tangent_angle_to_next.median():.0f}°')
    logging.info(f'  fraction tangent > 60° = '
                 f'{(df.tangent_angle_to_next > 60).mean():.0%}')

    df.drop(columns=['local_pc1'], inplace=True)
    return df


# ────────────────────────────────────────────────────────────────────────────
# 4. Multi-scale analysis (smaller scope than Paper 2 — 4 scales)
# ────────────────────────────────────────────────────────────────────────────
def multiscale_local_pca(E: np.ndarray, global_evecs: np.ndarray) -> pd.DataFrame:
    banner(f'Multi-scale local PCA at {Cfg.SCALE_LIST}')
    rng = np.random.default_rng(Cfg.SEED + 1)
    n_probes_per_scale = min(500, len(E) // 4)
    probe_idx = rng.choice(len(E), n_probes_per_scale, replace=False)
    global_pc1 = global_evecs[:, 0]

    rows = []
    for k in Cfg.SCALE_LIST:
        if k + 1 > len(E):
            continue
        nn = NearestNeighbors(n_neighbors=k + 1).fit(E)
        for pidx in probe_idx:
            _, nbr = nn.kneighbors(E[pidx:pidx + 1])
            nbhd = E[nbr[0, 1:]]
            try:
                pca = PCA(n_components=min(10, k - 1)).fit(nbhd)
            except Exception:
                continue
            local_pc1 = pca.components_[0]
            align = float(abs(np.dot(local_pc1, global_pc1)))
            eigs = pca.explained_variance_
            eigs = eigs[eigs > 1e-12]
            local_pr = (eigs.sum() ** 2) / (eigs ** 2).sum()
            rows.append({'scale_k': k, 'probe_idx': int(pidx),
                         'align_pc1': align, 'local_pr': float(local_pr),
                         'pc1_var': float(pca.explained_variance_ratio_[0])})
        logging.info(f'  k={k}: mean align = '
                     f'{np.mean([r["align_pc1"] for r in rows if r["scale_k"]==k]):.3f}, '
                     f'mean PR = '
                     f'{np.mean([r["local_pr"] for r in rows if r["scale_k"]==k]):.2f}')
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# 5. Figures (Paper 2 visual style)
# ────────────────────────────────────────────────────────────────────────────
def fig_global(g: dict, save_path: Path):
    plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    a, b = axes

    # (a) Cumulative variance
    a.plot(np.arange(1, 65), g['cum_var'] * 100, color='#1F77B4', lw=2)
    a.axhline(80, color='gray', lw=0.5, ls='--')
    a.axhline(90, color='gray', lw=0.5, ls='--')
    a.axvline(g['n80'], color='red', lw=0.8, ls=':', label=f'80%: {g["n80"]} PCs')
    a.axvline(g['n90'], color='red', lw=0.8, ls=':', label=f'90%: {g["n90"]} PCs')
    a.set_xlabel('Number of principal components')
    a.set_ylabel('Cumulative variance explained (%)')
    a.set_title(f'(a) Effective dimensionality (PR = {g["pr"]:.1f})')
    a.set_xlim(1, 64); a.set_ylim(0, 100)
    a.legend()

    # (b) PC1 + PC2 loadings on individual HJ dims
    b.bar(np.arange(64) - 0.18, g['eigvecs'][:, 0], width=0.36,
          color='#1F77B4', alpha=0.8, label=f'PC1 ({g["var_ratio"][0]*100:.1f}%)')
    b.bar(np.arange(64) + 0.18, g['eigvecs'][:, 1], width=0.36,
          color='#FF7F0E', alpha=0.8, label=f'PC2 ({g["var_ratio"][1]*100:.1f}%)')
    b.axhline(0, color='black', lw=0.5)
    b.set_xlabel('HydroJEPA dimension index')
    b.set_ylabel('Eigenvector weight')
    b.set_title('(b) Top-2 principal axes')
    b.set_xlim(-1, 64)
    b.legend()

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'  saved {save_path.name}')


def fig_intrinsic(id_data: dict, labels: pd.DataFrame, save_path: Path,
                  global_pr: float):
    plt_setup()
    id_per = id_data['id_per_point']
    valid = np.isfinite(id_per)
    fig = plt.figure(figsize=(15, 5.2), dpi=150)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1.2, 0.9])

    # (a) ID histogram
    a = fig.add_subplot(gs[0])
    a.hist(id_per[valid], bins=40, color='#5B8DB8', edgecolor='white')
    a.axvline(np.nanmean(id_per), color='red', lw=1.5, ls='--',
              label=f'mean = {np.nanmean(id_per):.2f}')
    a.axvline(global_pr, color='black', lw=1.0, ls=':',
              label=f'global PR = {global_pr:.1f}')
    a.set_xlabel('Local intrinsic dimensionality')
    a.set_ylabel('count')
    a.set_title('(a) ID distribution')
    a.legend()

    # (b) ID across CONUS
    b = fig.add_subplot(gs[1])
    sc = b.scatter(labels.lon[valid], labels.lat[valid],
                   c=id_per[valid], s=4, cmap='YlOrRd', alpha=0.85,
                   vmin=np.nanpercentile(id_per, 5),
                   vmax=np.nanpercentile(id_per, 95))
    b.set_xlim(Cfg.CONUS_EXTENT[0], Cfg.CONUS_EXTENT[1])
    b.set_ylim(Cfg.CONUS_EXTENT[2], Cfg.CONUS_EXTENT[3])
    b.set_xticks([]); b.set_yticks([])
    b.set_aspect('equal', adjustable='box')
    b.set_title('(b) Local ID across CONUS')
    plt.colorbar(sc, ax=b, fraction=0.04, pad=0.02, label='Local ID')

    # (c) Boxplot by elevation if available
    c = fig.add_subplot(gs[2])
    if 'elevation' in labels.columns:
        elev = pd.to_numeric(labels['elevation'], errors='coerce').to_numpy()
        bins = [-100, 100, 500, 1000, 2000, 5000]
        names = ['<100', '100-500', '500-1k', '1k-2k', '>2k']
        groups = pd.cut(elev, bins=bins, labels=names)
        data = [id_per[(groups == n) & valid] for n in names]
        c.boxplot(data, labels=names, showfliers=False)
        c.set_xlabel('Elevation (m)')
        c.set_ylabel('Local ID')
        c.set_title('(c) ID by elevation band')
    fig.suptitle('HydroJEPA intrinsic dimensionality',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'  saved {save_path.name}')


def fig_local(local_df: pd.DataFrame, save_path: Path, n_dims: int = 64):
    plt_setup()
    rand_baseline = np.sqrt(2.0 / (np.pi * n_dims))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    a, b, c, d = axes.flatten()

    # (a) Local PR distribution
    a.hist(local_df.local_pr.dropna(), bins=40, color='#5b8db8', edgecolor='white')
    a.axvline(local_df.local_pr.mean(), color='red', lw=1.5, ls='--',
              label=f'mean = {local_df.local_pr.mean():.2f}')
    a.set_xlabel('Local participation ratio')
    a.set_ylabel('count')
    a.set_title('(a) Local effective dimensionality')
    a.legend()

    # (b) Tangent angle distribution
    ta = local_df.tangent_angle_to_next.dropna()
    b.hist(ta, bins=40, color='#c97a3a', edgecolor='white')
    b.axvline(60, color='red', lw=0.8, ls=':')
    b.axvline(ta.median(), color='black', lw=1.0, ls='--',
              label=f'median = {ta.median():.0f}°')
    b.set_xlabel('Tangent angle between adjacent probes (°)')
    b.set_title(f'(b) Tangent rotation '
                f'({(ta > 60).mean()*100:.0f}% > 60°)')
    b.legend()

    # (c) Alignment with global PC1, PC2
    c.hist(local_df.align_pc1.dropna(), bins=30, alpha=0.6,
           color='#1F77B4', label='|cos(local PC1, global PC1)|',
           edgecolor='white')
    c.hist(local_df.align_pc2.dropna(), bins=30, alpha=0.6,
           color='#FF7F0E', label='|cos(local PC2, global PC2)|',
           edgecolor='white')
    c.axvline(rand_baseline, color='red', lw=1.0, ls=':',
              label=f'random baseline = {rand_baseline:.3f}')
    c.set_xlabel('|cosine|')
    c.set_title('(c) Local-global PC alignment')
    c.legend()

    # (d) Spatial map of alignment
    sc = d.scatter(local_df.lon, local_df.lat,
                   c=local_df.align_pc1, s=12, cmap='RdYlGn',
                   vmin=0, vmax=1, alpha=0.85)
    d.set_xlim(Cfg.CONUS_EXTENT[0], Cfg.CONUS_EXTENT[1])
    d.set_ylim(Cfg.CONUS_EXTENT[2], Cfg.CONUS_EXTENT[3])
    d.set_xticks([]); d.set_yticks([])
    d.set_aspect('equal', adjustable='box')
    d.set_title('(d) PC1 alignment across CONUS')
    plt.colorbar(sc, ax=d, fraction=0.04, pad=0.02, label='|align|')

    fig.suptitle('HydroJEPA local geometry', fontsize=14,
                 fontweight='bold', y=1.00)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'  saved {save_path.name}')


def fig_multiscale(ms_df: pd.DataFrame, save_path: Path, n_dims: int = 64):
    plt_setup()
    if ms_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)
    a, b = axes
    rand_baseline = np.sqrt(2.0 / (np.pi * n_dims))

    by_scale = ms_df.groupby('scale_k')
    scales  = sorted(ms_df['scale_k'].unique())
    align_means = [by_scale.get_group(k)['align_pc1'].mean() for k in scales]
    align_stds  = [by_scale.get_group(k)['align_pc1'].std()  for k in scales]
    pr_means    = [by_scale.get_group(k)['local_pr'].mean()  for k in scales]

    a.errorbar(scales, align_means, yerr=align_stds,
               marker='o', color='#1F77B4', capsize=4, lw=2)
    a.axhline(rand_baseline, color='red', lw=0.8, ls=':',
              label=f'random baseline ({rand_baseline:.3f})')
    a.set_xscale('log')
    a.set_xlabel('Neighborhood size k')
    a.set_ylabel('Mean |cos(local PC1, global PC1)|')
    a.set_title('(a) Alignment vs scale')
    a.legend()

    b.plot(scales, pr_means, marker='s', color='#FF7F0E', lw=2)
    b.set_xscale('log')
    b.set_xlabel('Neighborhood size k')
    b.set_ylabel('Mean local participation ratio')
    b.set_title('(b) Effective dim vs scale')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'  saved {save_path.name}')


def fig_dom_dim_map(local_df: pd.DataFrame, save_path: Path):
    """Spatial map of which HJ dim dominates locally."""
    plt_setup()
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    top_dims = local_df['dom_dim'].value_counts().head(8).index.tolist()
    cmap = plt.cm.tab10
    for i, dim in enumerate(top_dims):
        sub = local_df[local_df.dom_dim == dim]
        ax.scatter(sub.lon, sub.lat, color=cmap(i % 10),
                   label=f'{dim} (n={len(sub)})', s=8, alpha=0.75)
    other = local_df[~local_df.dom_dim.isin(top_dims)]
    if len(other):
        ax.scatter(other.lon, other.lat, color='#ccc',
                   label=f'other (n={len(other)})', s=6, alpha=0.4)
    ax.set_xlim(Cfg.CONUS_EXTENT[0], Cfg.CONUS_EXTENT[1])
    ax.set_ylim(Cfg.CONUS_EXTENT[2], Cfg.CONUS_EXTENT[3])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Dominant local HydroJEPA dimension across CONUS')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9,
              frameon=False)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'  saved {save_path.name}')


# ────────────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--skip_local',      action='store_true')
    p.add_argument('--skip_multiscale', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    Cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    E, labels = load_data()

    # 1. Global covariance
    g = global_covariance(E)
    pd.DataFrame({
        'eigenvalue': g['eigvals'],
        'var_ratio': g['var_ratio'],
        'cum_var': g['cum_var'],
    }).to_csv(Cfg.REPORT_DIR / 'hydrojepa_global_covariance.csv', index_label='pc')
    np.save(Cfg.REPORT_DIR / 'hydrojepa_global_eigvecs.npy', g['eigvecs'])
    fig_global(g, Cfg.REPORT_DIR / 'fig_global_covariance.png')

    # 2. Intrinsic dimensionality
    id_data = compute_intrinsic(E)
    np.save(Cfg.REPORT_DIR / 'hydrojepa_id_per_point.npy', id_data['id_per_point'])
    fig_intrinsic(id_data, labels,
                  Cfg.REPORT_DIR / 'fig_intrinsic_dimensionality.png',
                  global_pr=g['pr'])

    # 3. Local PCA
    if not args.skip_local:
        local_df = local_pca(E, labels, g['eigvecs'])
        local_df.to_csv(Cfg.REPORT_DIR / 'hydrojepa_local_pca.csv', index=False)
        fig_local(local_df, Cfg.REPORT_DIR / 'fig_local_geometry.png',
                  n_dims=Cfg.N_DIMS)
        fig_dom_dim_map(local_df,
                        Cfg.REPORT_DIR / 'fig_dominant_dimension_map.png')
    else:
        local_df = None

    # 4. Multi-scale
    if not args.skip_multiscale:
        ms_df = multiscale_local_pca(E, g['eigvecs'])
        ms_df.to_csv(Cfg.REPORT_DIR / 'hydrojepa_multiscale.csv', index=False)
        fig_multiscale(ms_df,
                       Cfg.REPORT_DIR / 'fig_multiscale_alignment.png',
                       n_dims=Cfg.N_DIMS)

    # Summary JSON
    summary = {
        'n_patches': int(len(E)),
        'global_pr': g['pr'],
        'pcs_for_80_var': g['n80'],
        'pcs_for_90_var': g['n90'],
        'mean_intrinsic_dim': id_data['mean_id'],
        'pr_to_id_ratio': id_data['mean_id'] / g['pr'],
    }
    if local_df is not None:
        summary.update({
            'mean_local_pr': float(local_df.local_pr.mean()),
            'mean_pc1_alignment': float(local_df.align_pc1.mean()),
            'random_baseline_alignment': float(np.sqrt(2.0/(np.pi*Cfg.N_DIMS))),
            'pct_tangent_above_60': float(
                (local_df.tangent_angle_to_next > 60).mean()),
        })
    with open(Cfg.REPORT_DIR / 'hydrojepa_geometry_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    banner('Done')
    logging.info(f'Outputs in {Cfg.REPORT_DIR}/')


if __name__ == '__main__':
    main()

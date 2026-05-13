"""
HydroJEPA vs AlphaEarth — Complementarity Analysis
====================================================

Tests whether HydroJEPA and AlphaEarth encode environmental signal in
complementary ways. The answer determines whether the dual-embedding
agentic system makes sense or whether the two are redundant.

Three views of the same question:

  1. PER-VARIABLE PRIMARY DIMENSION COMPARISON
     For each environmental variable, identify the strongest-correlated
     dimension in each model. Are they encoding the same thing? Compare
     |Spearman ρ| side by side.

  2. INFORMATION SHARING via CCA
     Canonical correlation analysis between the 64-d AE and 64-d HJ
     embeddings. How much variance is shared? How many directions are
     common vs unique to each model?

  3. JOINT-VS-SEPARATE PREDICTIVE GAIN
     For each environmental variable, fit RF using:
        a) AE only (64 dims)
        b) HJ only (64 dims)
        c) Concatenated (128 dims)
     Quantify the (c)-vs-max(a,b) gain. Variables with positive gain are
     ones where the two models capture COMPLEMENTARY signal.

Inputs:
  reports/interpretability/hydrojepa_embeddings.npy    (cached HJ embeddings)
  data/hydrojepa/labels.parquet                        (AE A00..A63 + env vars)

Outputs (under reports/complementarity/):
  primary_dim_comparison.csv               — per-var: AE top dim vs HJ top dim
  cca_loadings.csv                         — canonical correlations + components
  joint_predictive_gain.csv                — RF R²: AE, HJ, joint, gain
  fig_primary_dim_comparison.png           — bar chart of |ρ| AE vs HJ
  fig_cca_correlations.png                 — canonical correlation spectrum
  fig_predictive_gain.png                  — gain bar chart
  fig_complementarity_summary.png          — single-figure summary

Run:
  python 13_hydrojepa_ae_complementarity.py
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
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from tqdm import tqdm


class Cfg:
    DATA_DIR    = Path('data/hydrojepa')
    LABELS_FILE = DATA_DIR / 'labels.parquet'
    INTERP_DIR  = Path('reports/interpretability')
    EMB_CACHE   = INTERP_DIR / 'hydrojepa_embeddings.npy'
    REPORT_DIR  = Path('reports/complementarity')

    HJ_DIMS = [f'H{i:02d}' for i in range(64)]
    AE_DIMS = [f'A{i:02d}' for i in range(64)]

    ENV_VARS = ['smap_sm', 'elevation', 'prism_ppt_mm', 'prism_tmean_c',
                'aridity_proxy', 'koppen', 'nlcd_class']
    ENV_LABELS = {
        'smap_sm':       'Soil Moisture (SMAP)',
        'elevation':     'Elevation (m)',
        'prism_ppt_mm':  'Precipitation (mm/yr)',
        'prism_tmean_c': 'Temperature Mean (°C)',
        'aridity_proxy': 'Aridity (P/PET)',
        'koppen':        'Köppen Class',
        'nlcd_class':    'NLCD Land Cover',
    }

    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH    = 12
    N_FOLDS         = 5
    RF_N_JOBS       = 4
    SEED            = 42


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
# Load data
# ────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    Cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not Cfg.EMB_CACHE.exists():
        raise FileNotFoundError(f'{Cfg.EMB_CACHE} missing — run script 11 first')
    emb_hj = np.load(Cfg.EMB_CACHE)

    labels = pd.read_parquet(Cfg.LABELS_FILE)
    manifest = pd.read_parquet(Cfg.DATA_DIR / 'manifest.parquet')
    ok_ids = manifest[manifest.status.isin(['ok', 'cached'])].patch_id
    df = labels[labels.patch_id.isin(ok_ids)].reset_index(drop=True)
    if len(df) != emb_hj.shape[0]:
        raise RuntimeError(f'labels {len(df)} vs HJ embeddings {emb_hj.shape[0]}')

    # Coerce AE columns to plain numeric (same fix as elsewhere)
    for c in Cfg.AE_DIMS:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
    for j, col in enumerate(Cfg.HJ_DIMS):
        df[col] = emb_hj[:, j]
    for c in Cfg.ENV_VARS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')

    logging.info(f'Joint table: {len(df)} rows, {len(Cfg.AE_DIMS)} AE + '
                 f'{len(Cfg.HJ_DIMS)} HJ dims, '
                 f'{len([v for v in Cfg.ENV_VARS if v in df.columns])} env vars')
    return df


# ────────────────────────────────────────────────────────────────────────────
# 1. Per-variable primary dimension comparison
# ────────────────────────────────────────────────────────────────────────────
def primary_dim_comparison(df: pd.DataFrame) -> pd.DataFrame:
    banner('Per-variable primary dimension: AE vs HydroJEPA')
    rows = []
    for ev in Cfg.ENV_VARS:
        if ev not in df.columns:
            continue
        ev_v = df[ev].values
        ev_m = np.isfinite(ev_v)
        if ev_m.sum() < 200:
            continue
        # Best AE dim
        best_ae = (None, 0.0)
        for d in Cfg.AE_DIMS:
            v = df[d].values
            both = ev_m & np.isfinite(v)
            if both.sum() < 200:
                continue
            r, _ = spearmanr(v[both], ev_v[both])
            if abs(r) > abs(best_ae[1]):
                best_ae = (d, r)
        # Best HJ dim
        best_hj = (None, 0.0)
        for d in Cfg.HJ_DIMS:
            v = df[d].values
            both = ev_m & np.isfinite(v)
            if both.sum() < 200:
                continue
            r, _ = spearmanr(v[both], ev_v[both])
            if abs(r) > abs(best_hj[1]):
                best_hj = (d, r)
        rows.append({
            'variable':       ev,
            'label':          Cfg.ENV_LABELS.get(ev, ev),
            'ae_top_dim':     best_ae[0],
            'ae_top_rho':     best_ae[1],
            'ae_top_abs_rho': abs(best_ae[1]),
            'hj_top_dim':     best_hj[0],
            'hj_top_rho':     best_hj[1],
            'hj_top_abs_rho': abs(best_hj[1]),
            'gap':            abs(best_ae[1]) - abs(best_hj[1]),
        })
        logging.info(f'  {Cfg.ENV_LABELS[ev]:25s}: '
                     f'AE {best_ae[0]} ρ={best_ae[1]:+.3f}   '
                     f'HJ {best_hj[0]} ρ={best_hj[1]:+.3f}   '
                     f'Δ|ρ|={abs(best_ae[1]) - abs(best_hj[1]):+.3f}')
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# 2. CCA: how much information is shared?
# ────────────────────────────────────────────────────────────────────────────
def cca_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    banner('Canonical correlation: AE ↔ HydroJEPA')
    sub = df[Cfg.AE_DIMS + Cfg.HJ_DIMS].dropna()
    X = sub[Cfg.AE_DIMS].values
    Y = sub[Cfg.HJ_DIMS].values
    # Standardize
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)

    n_components = min(20, X.shape[1], Y.shape[1])
    cca = CCA(n_components=n_components, max_iter=500).fit(X, Y)
    Xc, Yc = cca.transform(X, Y)
    canon_corr = np.array([np.corrcoef(Xc[:, i], Yc[:, i])[0, 1]
                           for i in range(n_components)])

    # How many "shared directions" exceed e.g. 0.5 corr?
    n_shared_05 = int((canon_corr > 0.5).sum())
    n_shared_07 = int((canon_corr > 0.7).sum())
    n_shared_09 = int((canon_corr > 0.9).sum())

    logging.info(f'  Top canonical correlations: '
                 f'{[f"{c:.3f}" for c in canon_corr[:5]]}')
    logging.info(f'  > 0.5: {n_shared_05}    > 0.7: {n_shared_07}    > 0.9: {n_shared_09}')

    df_out = pd.DataFrame({
        'cc_idx':       np.arange(n_components),
        'corr':         canon_corr,
    })
    summary = {
        'n_components_05': n_shared_05,
        'n_components_07': n_shared_07,
        'n_components_09': n_shared_09,
        'first_3_corrs': canon_corr[:3].tolist(),
    }
    return df_out, summary


# ────────────────────────────────────────────────────────────────────────────
# 3. Joint-vs-separate predictive gain
# ────────────────────────────────────────────────────────────────────────────
def predictive_gain(df: pd.DataFrame) -> pd.DataFrame:
    banner('Joint vs separate predictive R²')
    rows = []
    for ev in tqdm(Cfg.ENV_VARS, desc='  vars'):
        if ev not in df.columns:
            continue
        sub = df[Cfg.AE_DIMS + Cfg.HJ_DIMS + [ev]].dropna()
        if len(sub) < 500:
            continue
        y = sub[ev].values.astype('float32')

        def cv_r2(X):
            rf = RandomForestRegressor(
                n_estimators=Cfg.RF_N_ESTIMATORS,
                max_depth=Cfg.RF_MAX_DEPTH,
                n_jobs=Cfg.RF_N_JOBS, random_state=Cfg.SEED)
            return cross_val_score(rf, X.astype('float32'), y,
                                   cv=Cfg.N_FOLDS, scoring='r2',
                                   n_jobs=Cfg.RF_N_JOBS)

        ae_scores = cv_r2(sub[Cfg.AE_DIMS].values)
        hj_scores = cv_r2(sub[Cfg.HJ_DIMS].values)
        joint_scores = cv_r2(sub[Cfg.AE_DIMS + Cfg.HJ_DIMS].values)

        gain_over_max = float(joint_scores.mean()
                              - max(ae_scores.mean(), hj_scores.mean()))

        rows.append({
            'variable':     ev,
            'label':        Cfg.ENV_LABELS.get(ev, ev),
            'r2_ae_mean':   float(ae_scores.mean()),
            'r2_ae_std':    float(ae_scores.std()),
            'r2_hj_mean':   float(hj_scores.mean()),
            'r2_hj_std':    float(hj_scores.std()),
            'r2_joint_mean': float(joint_scores.mean()),
            'r2_joint_std':  float(joint_scores.std()),
            'gain_over_max': gain_over_max,
            'n_samples':     int(len(sub)),
        })
        logging.info(f'  {Cfg.ENV_LABELS[ev]:25s}: '
                     f'AE={ae_scores.mean():.3f}  HJ={hj_scores.mean():.3f}  '
                     f'joint={joint_scores.mean():.3f}  gain={gain_over_max:+.3f}')
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# Figures
# ────────────────────────────────────────────────────────────────────────────
def fig_primary_dim(df: pd.DataFrame, save_path: Path):
    plt_setup()
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    df = df.sort_values('ae_top_abs_rho', ascending=True)
    y = np.arange(len(df))
    ax.barh(y - 0.18, df['ae_top_abs_rho'], height=0.34,
            color='#5B8DB8', label='AlphaEarth')
    ax.barh(y + 0.18, df['hj_top_abs_rho'], height=0.34,
            color='#3A7D44', label='HydroJEPA')
    ax.set_yticks(y)
    ax.set_yticklabels(df['label'])
    ax.set_xlabel('|Spearman ρ| of best dimension')
    ax.set_title('Primary dimension correlation: AE vs HydroJEPA')
    ax.set_xlim(0, 1)
    # Annotate dim names
    for i, r in enumerate(df.itertuples()):
        ax.text(r.ae_top_abs_rho + 0.01, i - 0.18, r.ae_top_dim,
                fontsize=8, va='center', color='#5B8DB8')
        ax.text(r.hj_top_abs_rho + 0.01, i + 0.18, r.hj_top_dim,
                fontsize=8, va='center', color='#3A7D44')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'  saved {save_path.name}')


def fig_cca(cca_df: pd.DataFrame, save_path: Path):
    plt_setup()
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.bar(cca_df.cc_idx + 1, cca_df.corr, color='#7B4DB7', alpha=0.85)
    ax.axhline(0.5, color='gray', lw=0.8, ls='--', label='0.5')
    ax.axhline(0.7, color='red',  lw=0.8, ls='--', label='0.7')
    ax.set_xlabel('Canonical component index')
    ax.set_ylabel('Canonical correlation')
    ax.set_title('CCA: AlphaEarth ↔ HydroJEPA shared directions')
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'  saved {save_path.name}')


def fig_predictive_gain(df: pd.DataFrame, save_path: Path):
    plt_setup()
    df = df.sort_values('r2_ae_mean', ascending=True)
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    y = np.arange(len(df))
    ax.barh(y - 0.27, df['r2_ae_mean'], height=0.25,
            xerr=df['r2_ae_std'], color='#5B8DB8',
            label='AE only', capsize=3)
    ax.barh(y, df['r2_hj_mean'], height=0.25,
            xerr=df['r2_hj_std'], color='#3A7D44',
            label='HydroJEPA only', capsize=3)
    ax.barh(y + 0.27, df['r2_joint_mean'], height=0.25,
            xerr=df['r2_joint_std'], color='#7B4DB7',
            label='Joint (concat 128 dims)', capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df['label'])
    ax.set_xlabel('5-fold RF R²')
    ax.set_title('Joint AE+HydroJEPA prediction vs separate models')
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlim(-0.05, 1.0)

    # Annotate gain
    for i, r in enumerate(df.itertuples()):
        sign = '+' if r.gain_over_max >= 0 else ''
        ax.text(r.r2_joint_mean + 0.01, i + 0.27,
                f'gain {sign}{r.gain_over_max:.3f}',
                fontsize=8, va='center', color='#7B4DB7')

    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'  saved {save_path.name}')


def fig_summary(prim_df, cca_df, gain_df, cca_summary, save_path):
    plt_setup()
    fig = plt.figure(figsize=(15, 5.5), dpi=150)
    gs = gridspec.GridSpec(1, 3, figure=fig)

    # (a) primary dim |ρ| AE vs HJ scatter
    a = fig.add_subplot(gs[0])
    a.scatter(prim_df.ae_top_abs_rho, prim_df.hj_top_abs_rho,
              s=50, color='#7B4DB7', alpha=0.8)
    for _, r in prim_df.iterrows():
        a.text(r.ae_top_abs_rho + 0.01, r.hj_top_abs_rho - 0.02,
               r.label.split(' (')[0], fontsize=8)
    a.plot([0, 1], [0, 1], color='gray', ls='--', lw=0.8)
    a.set_xlabel('|ρ|  best AE dim')
    a.set_ylabel('|ρ|  best HJ dim')
    a.set_title('(a) Primary dim |ρ|')
    a.set_xlim(0, 1); a.set_ylim(0, 1)

    # (b) CCA spectrum
    b = fig.add_subplot(gs[1])
    b.bar(cca_df.cc_idx + 1, cca_df.corr, color='#7B4DB7', alpha=0.85)
    b.axhline(0.5, color='gray', lw=0.8, ls='--')
    b.axhline(0.7, color='red',  lw=0.8, ls='--')
    b.set_xlabel('Canonical component')
    b.set_ylabel('Canonical correlation')
    b.set_title(f'(b) CCA: {cca_summary["n_components_07"]} dirs > 0.7')
    b.set_ylim(0, 1)

    # (c) gain distribution
    c = fig.add_subplot(gs[2])
    sorted_gain = gain_df.sort_values('gain_over_max', ascending=True)
    bars = c.barh(np.arange(len(sorted_gain)), sorted_gain.gain_over_max,
                  color=['#D62728' if g < 0 else '#2CA02C'
                         for g in sorted_gain.gain_over_max])
    c.set_yticks(np.arange(len(sorted_gain)))
    c.set_yticklabels(sorted_gain.label)
    c.set_xlabel('Joint R² − max(AE, HJ)')
    c.set_title('(c) Predictive gain')
    c.axvline(0, color='black', lw=0.5)

    fig.suptitle('AE ↔ HydroJEPA complementarity', fontsize=14,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'  saved {save_path.name}')


# ────────────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--skip_predictive', action='store_true',
                   help='Skip the slow joint RF gain experiment')
    return p.parse_args()


def main():
    args = parse_args()
    Cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    # 1. Primary dim comparison
    prim_df = primary_dim_comparison(df)
    prim_df.to_csv(Cfg.REPORT_DIR / 'primary_dim_comparison.csv', index=False)

    # 2. CCA
    cca_df, cca_summary = cca_analysis(df)
    cca_df.to_csv(Cfg.REPORT_DIR / 'cca_loadings.csv', index=False)

    # 3. Predictive gain
    if not args.skip_predictive:
        gain_df = predictive_gain(df)
        gain_df.to_csv(Cfg.REPORT_DIR / 'joint_predictive_gain.csv', index=False)
    else:
        gain_df = pd.DataFrame()

    # Figures
    fig_primary_dim(prim_df,
                    Cfg.REPORT_DIR / 'fig_primary_dim_comparison.png')
    fig_cca(cca_df, Cfg.REPORT_DIR / 'fig_cca_correlations.png')
    if not gain_df.empty:
        fig_predictive_gain(gain_df,
                            Cfg.REPORT_DIR / 'fig_predictive_gain.png')
    if not gain_df.empty:
        import matplotlib.gridspec as gridspec
        # Only run summary fig if all three pieces are present
        fig_summary(prim_df, cca_df, gain_df, cca_summary,
                    Cfg.REPORT_DIR / 'fig_complementarity_summary.png')

    summary = {
        'cca_summary': cca_summary,
        'best_var_for_hj':
            (prim_df.sort_values('hj_top_abs_rho', ascending=False)
                    .iloc[0].to_dict() if len(prim_df) else None),
        'biggest_joint_gain':
            (gain_df.sort_values('gain_over_max', ascending=False)
                    .iloc[0].to_dict() if not gain_df.empty else None),
    }
    with open(Cfg.REPORT_DIR / 'complementarity_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    banner('Done')
    logging.info(f'Outputs in {Cfg.REPORT_DIR}/')


if __name__ == '__main__':
    main()

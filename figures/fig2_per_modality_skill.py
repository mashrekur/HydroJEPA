"""
figures/fig2_per_modality_skill.py

Figure 2 — Each Mini-JEPA encodes different physics

Top row (headline):
  Five small CONUS maps. For each Mini-JEPA, we show predicted-vs-actual
  fit on that Mini-JEPA's strongest predictive variable (per STRONGEST_VAR
  in _style.py). The map color encodes the absolute residual: lighter where
  the model agrees with ground truth, darker where it disagrees. This makes
  "specialization" geographic instead of abstract — MODIS-Thermal is faithful
  on temperature across CONUS; Topo-Soil is faithful on elevation; S1-SAR's
  fit on its best variable is visibly weaker, etc.

  Each map is annotated with the standalone CV R² in a corner box.

Bottom row (supporting):
  The existing 7×5 RF-R² heatmap, restyled for paper.

Run:
  python fig2_per_modality_skill.py --dry-run
  python fig2_per_modality_skill.py --no-maps
  python fig2_per_modality_skill.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (
    apply_rc, save_figure, project_root, output_dir, dry_run_report,
    MODALITY_ORDER, MODALITY_LABEL, MODALITY_COLOR, STRONGEST_VAR,
    ENVVAR_LABEL, ENVVAR_ORDER, SKILL_CMAP, modality_cmap,
    add_conus_basemap, annotation_box,
)

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Path resolver — tries multiple candidate locations per file
# ---------------------------------------------------------------------------
def first_existing(*candidates: Path) -> Path | None:
    """Return the first candidate that exists on disk; None if none do."""
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_paths():
    """Build a {label: Path} dict where each Path is the *first* existing
    candidate among likely locations on disk. Stores None for missing.
    """
    root = project_root()

    # Possible locations for the embeddings cache and R² CSV per modality:
    # 1. Per-Mini-JEPA tree:  reports/minijepas/<modality>/interpretability/
    # 2. Flat baseline tree:  reports/interpretability/  (S2-Optical only,
    #                         from the original single-model run)
    # 3. Mirror at root:      <modality>__hydrojepa_*.csv
    # 4. eval_results_for_review/ mirror
    def emb_candidates(m):
        return (
            root / 'reports' / 'minijepas' / m / 'interpretability' / 'hydrojepa_embeddings.npy',
            # The flat tree's file is unprefixed; it's the S2-Optical baseline
            *( (root / 'reports' / 'interpretability' / 'hydrojepa_embeddings.npy',)
                if m == 's2_optical' else () ),
        )

    def r2_candidates(m):
        return (
            root / 'reports' / 'minijepas' / m / 'interpretability' / 'hydrojepa_rf_r2.csv',
            root / 'eval_results_for_review' / f'{m}__hydrojepa_rf_r2.csv',
            root / f'{m}__hydrojepa_rf_r2.csv',
            *( (root / 'reports' / 'interpretability' / 'hydrojepa_rf_r2.csv',)
                if m == 's2_optical' else () ),
        )

    def manifest_candidates(m):
        # S2-Optical's original manifest has no modality suffix
        if m == 's2_optical':
            return (
                root / 'data' / 'hydrojepa' / 'manifest.parquet',
                root / 'data' / 'hydrojepa' / 'manifest_s2_optical.parquet',
            )
        return (root / 'data' / 'hydrojepa' / f'manifest_{m}.parquet',)

    paths = {
        'labels': first_existing(root / 'data' / 'hydrojepa' / 'labels.parquet'),
    }

    for m in MODALITY_ORDER:
        paths[f'{m}_emb']      = first_existing(*emb_candidates(m))
        paths[f'{m}_r2']       = first_existing(*r2_candidates(m))
        paths[f'{m}_manifest'] = first_existing(*manifest_candidates(m))

    return paths


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_modality(modality: str, paths: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (embeddings, dataframe-with-lon-lat-and-envvars) aligned by row.

    Tries the per-modality manifest first; falls back to labels.parquet
    keyed on patch_id if the manifest doesn't exist.
    """
    emb_path = paths[f'{modality}_emb']
    if emb_path is None or not emb_path.exists():
        raise FileNotFoundError(f'Embeddings missing for {modality}')
    E = np.load(emb_path)

    manifest_path = paths[f'{modality}_manifest']
    labels_path   = paths['labels']

    if manifest_path is not None and manifest_path.exists():
        manifest = pd.read_parquet(manifest_path)
    elif labels_path is not None:
        manifest = pd.read_parquet(labels_path)
    else:
        raise FileNotFoundError(
            f'No manifest or labels file found for {modality}; '
            f'cannot recover patch lon/lat'
        )

    labels = pd.read_parquet(labels_path) if labels_path else manifest

    # Align by patch_id if available; else assume row-order
    if 'patch_id' in manifest.columns and 'patch_id' in labels.columns:
        # Bring lon/lat from manifest, env vars from labels
        keep_cols = ['patch_id']
        for c in ('lon', 'lat'):
            if c in manifest.columns:
                keep_cols.append(c)
        df = manifest[keep_cols].merge(
            labels.drop(columns=[c for c in ('lon', 'lat')
                                  if c in labels.columns and c in manifest.columns]),
            on='patch_id', how='left',
        )
    else:
        df = manifest.copy()
        for c in labels.columns:
            if c not in df.columns:
                df[c] = labels[c].values[:len(df)]

    if 'lon' not in df.columns or 'lat' not in df.columns:
        raise ValueError(
            f'{modality}: lon/lat not present after merge. '
            f'manifest cols={list(manifest.columns)[:8]}; '
            f'labels cols={list(labels.columns)[:8]}'
        )

    if len(df) != len(E):
        n = min(len(df), len(E))
        df = df.iloc[:n].reset_index(drop=True)
        E = E[:n]

    return E, df


# ---------------------------------------------------------------------------
# Cross-validated RF prediction (mirrors script 11's settings)
# ---------------------------------------------------------------------------
def cv_predictions(E: np.ndarray, y: np.ndarray, *,
                   n_folds: int = 5, classifier: bool = False,
                   seed: int = 42) -> np.ndarray:
    """Out-of-fold RF predictions over all samples.

    Returns a length-N array; entries where y is NaN remain NaN.
    """
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import KFold

    valid = ~pd.isna(y)
    valid = np.asarray(valid)
    Xv = E[valid]
    yv = y[valid]

    pred_full = np.full(len(y), np.nan, dtype=float)
    if len(yv) < 50:
        return pred_full

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pred_v = np.empty(len(yv), dtype=float)

    for tr, te in kf.split(Xv):
        if classifier:
            mdl = RandomForestClassifier(n_estimators=200, max_depth=20,
                                          n_jobs=-1, random_state=seed)
            mdl.fit(Xv[tr], yv[tr])
            pred_v[te] = mdl.predict(Xv[te]).astype(float)
        else:
            mdl = RandomForestRegressor(n_estimators=200, max_depth=20,
                                         n_jobs=-1, random_state=seed)
            mdl.fit(Xv[tr], yv[tr].astype(float))
            pred_v[te] = mdl.predict(Xv[te])

    pred_full[valid] = pred_v
    return pred_full


def cached_predictions(modality: str, var: str, E: np.ndarray, y: np.ndarray):
    """Cache CV predictions to disk; recompute only if missing."""
    cache_dir = output_dir() / '_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f'pred_{modality}__{var}.npy'
    if cache.exists():
        try:
            arr = np.load(cache)
            if len(arr) == len(y):
                return arr
        except Exception:
            pass
    is_clf = var in ('koppen', 'nlcd_class')
    print(f'  computing CV predictions: {modality} → {var} ({"clf" if is_clf else "reg"})')
    pred = cv_predictions(E, y, classifier=is_clf)
    np.save(cache, pred)
    return pred


def load_r2_table(paths: dict) -> pd.DataFrame:
    """Stack all per-modality RF R² CSVs into one long table."""
    rows = []
    for m in MODALITY_ORDER:
        f = paths.get(f'{m}_r2')
        if f is None or not f.exists():
            continue
        df = pd.read_csv(f)
        df['modality'] = m
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=['variable', 'r2_cv', 'r2_cv_std',
                                      'n_samples', 'modality'])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Hex-binned local R²
# ---------------------------------------------------------------------------
# Local R² per hex aggregates the out-of-fold RF predictions over a
# spatial neighborhood and asks "in this region, how well does the
# Mini-JEPA's embedding predict the within-region variance of its
# target variable?"
#
# This is a stricter test than the headline CV R²: a model that predicts
# precipitation by exploiting continental-scale climate gradients
# (e.g., latitude) scores high globally but its local R² collapses
# because, within any one hex, latitude is roughly constant. A model
# that genuinely encodes the variable scores high both globally AND
# locally — which is the visually distinguishable claim.

def _hex_centers(extent, n_x: int = 40):
    """Generate flat-topped hex centers covering an extent (lonW, lonE, latS, latN).

    Returns (centers Nx2, dx, dy) where dx is hex horizontal spacing
    and dy is hex vertical spacing. The hex apothem geometry is set so
    flat-topped hexes tile cleanly.
    """
    lonW, lonE, latS, latN = extent
    dx = (lonE - lonW) / n_x                     # column spacing
    dy = dx * np.sqrt(3) / 2                     # row spacing for flat-top hex
    cols = np.arange(lonW, lonE + dx, dx)
    rows = np.arange(latS, latN + dy, dy)
    centers = []
    for i, lat in enumerate(rows):
        offset = (dx / 2) if (i % 2 == 1) else 0.0
        for lon in cols:
            centers.append((lon + offset, lat))
    return np.array(centers), dx, dy


def _assign_to_hexes(lons, lats, centers, dx, dy):
    """For each (lon, lat), return the index of the nearest hex center.

    Approximate: uses Euclidean distance on lon/lat (good enough at
    CONUS scale for a viz binning). Vectorized via cKDTree.
    """
    from scipy.spatial import cKDTree
    # Scale lat by aspect to make distance roughly isotropic in display
    aspect = dx / dy
    pts = np.column_stack([lons, lats * aspect])
    centers_scaled = np.column_stack([centers[:, 0], centers[:, 1] * aspect])
    tree = cKDTree(centers_scaled)
    _, idx = tree.query(pts, k=1)
    return idx


def hex_local_r2(lons, lats, y, pred,
                  *, n_x: int = 40, min_n: int = 15,
                  extent=(-125.0, -66.5, 24.5, 49.5)):
    """Compute per-hex local R² from out-of-fold predictions.

    Returns: (hex_centers, hex_r2, hex_n) — three same-length arrays.
      hex_centers : (M, 2) lon, lat of hex centers
      hex_r2      : (M,)   local R² per hex; NaN where n < min_n
      hex_n       : (M,)   number of patches in each hex
    """
    centers, dx, dy = _hex_centers(extent, n_x=n_x)
    valid = ~(np.isnan(y) | np.isnan(pred))
    if valid.sum() < 50:
        return centers, np.full(len(centers), np.nan), np.zeros(len(centers), int)

    lons_v = np.asarray(lons)[valid]
    lats_v = np.asarray(lats)[valid]
    y_v    = np.asarray(y)[valid]
    p_v    = np.asarray(pred)[valid]

    hex_idx = _assign_to_hexes(lons_v, lats_v, centers, dx, dy)
    M = len(centers)
    r2  = np.full(M, np.nan)
    nh  = np.zeros(M, dtype=int)

    for h in range(M):
        mask = hex_idx == h
        n = int(mask.sum())
        nh[h] = n
        if n < min_n:
            continue
        yh = y_v[mask]
        ph = p_v[mask]
        ss_tot = float(np.sum((yh - yh.mean()) ** 2))
        if ss_tot < 1e-12:
            continue                  # constant target in this hex
        ss_res = float(np.sum((yh - ph) ** 2))
        r2[h] = 1.0 - ss_res / ss_tot

    return centers, r2, nh


# ---------------------------------------------------------------------------
# Top row: five CONUS maps of local R²
# ---------------------------------------------------------------------------
def draw_local_r2_map(ax, df_xy: pd.DataFrame, y: np.ndarray, pred: np.ndarray,
                      modality: str, r2_global: float, var_label: str):
    """One CONUS panel: hex-binned local R² in the modality's color.

    Reading: darker hex = model captures more within-region variance.
    Hatched hex = model worse than predicting the regional mean (R² < 0).
    Empty (basemap-colored) hex = too few patches to score (n < 15).
    """
    add_conus_basemap(ax, land=True, states=True)

    centers, hex_r2, hex_n = hex_local_r2(
        df_xy['lon'].values, df_xy['lat'].values, y, pred,
    )

    # Median local R² across scored hexes — the "honest" companion to
    # the headline CV R². Reported in the corner box below.
    scored = ~np.isnan(hex_r2)
    median_local = float(np.nanmedian(hex_r2[scored])) if scored.any() else np.nan

    if not scored.any():
        ax.text(0.5, 0.5, 'insufficient data',
                ha='center', va='center', transform=ax.transAxes,
                color='#888581')
        return

    cmap = modality_cmap(modality)

    # Two passes: failure hexes (R² < 0) get a subdued hatch overlay,
    # successful hexes (R² ∈ [0, 1]) get a full color gradient.
    success_mask = scored & (hex_r2 >= 0)
    fail_mask    = scored & (hex_r2 < 0)

    # Hex marker size: tuned so flat-topped hexes roughly tile at this
    # CONUS aspect. Use a regular hexagon (matplotlib marker 'h').
    hex_size = 78  # marker s param; tweaked visually for a 40-hex grid

    if success_mask.any():
        ax.scatter(
            centers[success_mask, 0], centers[success_mask, 1],
            c=np.clip(hex_r2[success_mask], 0, 1),
            cmap=cmap, vmin=0, vmax=1,
            marker='H', s=hex_size,
            edgecolors='white', linewidths=0.3,
            alpha=0.92, zorder=2,
        )
    if fail_mask.any():
        ax.scatter(
            centers[fail_mask, 0], centers[fail_mask, 1],
            c='#DDDDDD', marker='H', s=hex_size,
            edgecolors='#999999', linewidths=0.3,
            hatch='///', alpha=0.85, zorder=2,
        )

    ax.set_title(MODALITY_LABEL[modality], fontsize=11, fontweight='bold',
                  color=MODALITY_COLOR[modality], pad=4)

    # Two-line corner annotation: global vs median local R²
    g_str = f'{r2_global:.2f}' if not np.isnan(r2_global) else '?'
    l_str = f'{median_local:.2f}' if not np.isnan(median_local) else '?'
    annotation_box(
        ax,
        f'{var_label}\n'
        f'R² (global)  = {g_str}\n'
        f'R² (median local) = {l_str}',
        loc='lower left', fontsize=8.0, weight='bold',
    )


# ---------------------------------------------------------------------------
# Bottom row: restyled R² heatmap
# ---------------------------------------------------------------------------
def draw_r2_heatmap(ax, r2_long: pd.DataFrame):
    if r2_long.empty:
        ax.text(0.5, 0.5, 'no R² CSVs found',
                ha='center', va='center', transform=ax.transAxes,
                color='#888581', fontsize=11)
        ax.axis('off')
        return

    pivot = (r2_long.pivot_table(index='variable', columns='modality',
                                  values='r2_cv', aggfunc='mean')
                    .reindex(index=ENVVAR_ORDER, columns=MODALITY_ORDER))

    data = pivot.values
    ax.imshow(data, cmap=SKILL_CMAP, vmin=0, vmax=1, aspect='auto')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            color = 'white' if v > 0.55 else '#222222'
            weight = 'bold' if v >= 0.90 else 'normal'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight=weight)

    ax.set_xticks(range(len(MODALITY_ORDER)))
    ax.set_xticklabels([MODALITY_LABEL[m] for m in MODALITY_ORDER],
                        rotation=0, ha='center', fontsize=8.5)
    ax.set_yticks(range(len(ENVVAR_ORDER)))
    ax.set_yticklabels([ENVVAR_LABEL[v] for v in ENVVAR_ORDER], fontsize=8.5)

    for tick, m in zip(ax.get_xticklabels(), MODALITY_ORDER):
        tick.set_color(MODALITY_COLOR[m])
        tick.set_fontweight('bold')

    ax.set_title('Predictive skill across the fleet  (RF cross-validated R²)',
                  fontsize=9.5, fontweight='bold', pad=4)

    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(left=False, bottom=False)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true',
                    help='Print resolved data file paths and exit')
    p.add_argument('--no-maps', action='store_true',
                    help='Skip top-row CONUS maps (heatmap only)')
    p.add_argument('--two-rows', action='store_true',
                    help='Lay out the 5 maps in 3+2 rows (each map larger) '
                         'instead of a single row of 5')
    args = p.parse_args()

    paths = resolve_paths()

    if args.dry_run:
        # Dry-run wants to show "what would I read?" — convert None→placeholder
        display_paths = {k: (v if v is not None
                              else Path(f'<not found for {k}>'))
                          for k, v in paths.items()}
        dry_run_report('Figure 2 — per-modality skill', display_paths)
        return

    apply_rc()

    # ---- Top row ----------------------------------------------------------
    panel_data = {}
    if not args.no_maps:
        for m in MODALITY_ORDER:
            var = STRONGEST_VAR[m]
            try:
                E, df = load_modality(m, paths)
            except (FileNotFoundError, ValueError) as e:
                print(f'  [skip] {m}: {e}')
                continue

            if var not in df.columns:
                print(f'  [skip] {m}: variable {var} not in dataframe '
                       f'(have: {[c for c in df.columns][:8]}...)')
                continue

            y = df[var].astype(float).values
            pred = cached_predictions(m, var, E, y)

            panel_data[m] = dict(df=df[['lon', 'lat']].copy(),
                                  y=y, pred=pred, var=var)

    # ---- R² table ---------------------------------------------------------
    r2_long = load_r2_table(paths)

    # ---- Lookup each modality's best-variable R² for annotations ----------
    r2_by_modality = {}
    if not r2_long.empty and 'modality' in r2_long.columns:
        for m in MODALITY_ORDER:
            sub = r2_long[(r2_long['modality'] == m) &
                           (r2_long['variable'] == STRONGEST_VAR[m])]
            r2_by_modality[m] = float(sub['r2_cv'].iloc[0]) if len(sub) else np.nan
    else:
        for m in MODALITY_ORDER:
            r2_by_modality[m] = np.nan

    # ---- Layout -----------------------------------------------------------
    # 2×3 equal-cell grid. 5 maps + heatmap as the 6th cell; every cell
    # is the same size — peers, not a hierarchy. Title centered above.
    if args.two_rows:
        # Legacy alternative layout, kept for back-compat. Matches the
        # earlier 3+2 maps + bottom heatmap arrangement.
        fig = plt.figure(figsize=(13.5, 11.0))
        outer = gridspec.GridSpec(
            3, 1, height_ratios=[2.4, 2.4, 1.0],
            hspace=0.18,
            left=0.03, right=0.99, top=0.91, bottom=0.05,
        )
        gs_top = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[0], wspace=0.08)
        gs_mid = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=outer[1], wspace=0.08)
        ax_hm_spec = outer[2]
        map_axes_specs = [
            gs_top[0, 0], gs_top[0, 1], gs_top[0, 2],
            gs_mid[0, 1], gs_mid[0, 3],
        ]
    else:
        fig = plt.figure(figsize=(15.0, 9.5))
        outer = gridspec.GridSpec(
            2, 3,
            hspace=0.18, wspace=0.06,
            left=0.03, right=0.99, top=0.89, bottom=0.04,
        )
        # Maps fill cells 0-4; heatmap takes cell 5 (bottom-right).
        map_axes_specs = [
            outer[0, 0], outer[0, 1], outer[0, 2],
            outer[1, 0], outer[1, 1],
        ]
        ax_hm_spec = outer[1, 2]

    # ---- Maps ------------------------------------------------------------
    if not args.no_maps:
        for j, m in enumerate(MODALITY_ORDER):
            ax = fig.add_subplot(map_axes_specs[j])
            if m not in panel_data:
                ax.set_title(MODALITY_LABEL[m], fontsize=11,
                              color=MODALITY_COLOR[m])
                ax.text(0.5, 0.5, 'embeddings\nnot found',
                        ha='center', va='center', transform=ax.transAxes,
                        color='#888581')
                ax.axis('off')
                continue
            d = panel_data[m]
            draw_local_r2_map(
                ax, d['df'], d['y'], d['pred'],
                modality=m,
                r2_global=r2_by_modality.get(m, np.nan),
                var_label=ENVVAR_LABEL[d['var']],
            )
    else:
        for j, m in enumerate(MODALITY_ORDER):
            ax = fig.add_subplot(map_axes_specs[j])
            ax.set_title(MODALITY_LABEL[m], color=MODALITY_COLOR[m])
            ax.text(0.5, 0.5, '(skipped: --no-maps)',
                    ha='center', va='center', transform=ax.transAxes,
                    color='#888581', fontsize=9)
            ax.axis('off')

    # ---- Title block (centered, tight to figure) -------------------------
    fig.text(0.5, 0.965,
              'Each Mini-JEPA captures within-region variance of its '
              'strongest variable',
              fontsize=14, fontweight='bold', ha='center', va='top')
    fig.text(0.5, 0.927,
              'darker hex = higher local R² (model captures regional '
              'variance)   •   hatched hex = model fails (R² < 0)   •   '
              'empty hex = too few patches (n < 15)',
              fontsize=8.5, ha='center', va='top',
              color='#555555', style='italic')

    # ---- Heatmap ---------------------------------------------------------
    ax_hm = fig.add_subplot(ax_hm_spec)
    draw_r2_heatmap(ax_hm, r2_long)

    saved = save_figure(fig, 'fig2_per_modality_skill')
    plt.close(fig)
    print('\nSaved:')
    for p_ in saved:
        print(f'  {p_}')


if __name__ == '__main__':
    main()

"""
fig4_manifold_portrait.py

Figure 3b — Manifold portraits per Mini-JEPA. Mirrors the four-piece
geometric characterization from the AlphaEarth Geometry paper (Rahman et
al 2026, Paper 2) applied to each of the five Mini-JEPAs.

Why this exists: a 3D PCA scatter compresses 64 dimensions down to 3 and
loses exactly the information that distinguishes "this model uses 2 dims"
from "this model uses 50 dims". The heterogeneity story lives in the
SPECTRUM (how variance is distributed across all 64 PCs) and in the
SPATIAL variability (how local geometry varies across CONUS), not in any
single 3D view.

Layout — 5 rows (one per Mini-JEPA) × 3 columns:

  Col 1: Variance spectrum
    Cumulative variance from PC1 to PC64. Reference lines at 80%/90%
    with annotated PC counts. The shape of the curve IS the geometric
    character of the global manifold — MODIS-Thermal climbs slowly
    (variance spread across many directions, high global PR ≈ 20),
    while a low-PR modality would climb fast.

  Col 2: Local effective-dimensionality histogram
    Histogram of `local_n80` across the 2,000 probe locations — the
    number of dimensions needed for 80% local variance in each probe's
    neighborhood. Tight narrow histogram = locally uniform manifold;
    wide spread = manifold heterogeneity is locally variable.
    Annotated with mean local_n80 and global PR for reference.

  Col 3: CONUS map of dominant local dimension
    Each probe colored by its single most-important 64-D dimension
    locally (one of H00..H63). Different colors in different regions
    = spatial heterogeneity (different physical processes activate
    different dimensions). One color everywhere = spatially uniform.

Inputs (per modality, with flat-mirror fallbacks for ad-hoc layouts):
  reports/minijepas/<m>/manifold_geometry/hydrojepa_global_covariance.csv
  reports/minijepas/<m>/manifold_geometry/hydrojepa_local_pca.csv
  reports/minijepas/<m>/manifold_geometry/hydrojepa_geometry_summary.json
  (with S2-Optical flat fallback at reports/manifold_geometry/<file>)
  Flat-mirror fallback: ./<m>__hydrojepa_<name>.csv at project root.

Run:
  python fig4_manifold_portrait.py --dry-run
  python fig4_manifold_portrait.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (
    apply_rc, save_figure, project_root, dry_run_report,
    MODALITY_ORDER, MODALITY_LABEL, MODALITY_COLOR,
    add_conus_basemap,
)


warnings.filterwarnings('ignore')


# Number of dominant dimensions to highlight on the CONUS map per modality.
# The rest collapse to a single muted "other" color so the legend stays
# legible and the spatial pattern reads cleanly.
TOP_K_DOM_DIMS = 5


def first_existing(*candidates: Path) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def resolve_paths():
    root = project_root()
    paths = {}
    for m in MODALITY_ORDER:
        paths[f'{m}_global_cov'] = first_existing(
            root / 'reports' / 'minijepas' / m / 'manifold_geometry'
                 / 'hydrojepa_global_covariance.csv',
            *((root / 'reports' / 'manifold_geometry'
                    / 'hydrojepa_global_covariance.csv',)
               if m == 's2_optical' else ()),
            root / f'{m}__hydrojepa_global_covariance.csv',
        )
        paths[f'{m}_local_pca'] = first_existing(
            root / 'reports' / 'minijepas' / m / 'manifold_geometry'
                 / 'hydrojepa_local_pca.csv',
            *((root / 'reports' / 'manifold_geometry'
                    / 'hydrojepa_local_pca.csv',)
               if m == 's2_optical' else ()),
            root / f'{m}__hydrojepa_local_pca.csv',
        )
        paths[f'{m}_geom_summary'] = first_existing(
            root / 'reports' / 'minijepas' / m / 'manifold_geometry'
                 / 'hydrojepa_geometry_summary.json',
            *((root / 'reports' / 'manifold_geometry'
                    / 'hydrojepa_geometry_summary.json',)
               if m == 's2_optical' else ()),
            root / f'{m}__hydrojepa_geometry_summary.json',
        )
    return paths


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_modality(modality: str, paths: dict):
    gc_path = paths.get(f'{modality}_global_cov')
    lp_path = paths.get(f'{modality}_local_pca')
    js_path = paths.get(f'{modality}_geom_summary')

    if not (gc_path and gc_path.exists()):
        raise FileNotFoundError(f'{modality}: global_covariance.csv missing')
    if not (lp_path and lp_path.exists()):
        raise FileNotFoundError(f'{modality}: local_pca.csv missing')
    if not (js_path and js_path.exists()):
        raise FileNotFoundError(f'{modality}: geometry_summary.json missing')

    global_cov = pd.read_csv(gc_path)
    local_pca  = pd.read_csv(lp_path)
    geom       = json.loads(Path(js_path).read_text())

    return dict(global_cov=global_cov, local_pca=local_pca, geom=geom)


# ---------------------------------------------------------------------------
# Column 1: variance spectrum
# ---------------------------------------------------------------------------
def draw_variance_spectrum(ax, modality: str, data: dict):
    """Cumulative variance vs PC index, 1..64. Reference lines at 80% / 90%.

    The shape of this curve is the global geometric character. A curve
    that snaps near-vertical to 100% by PC2 means a ~1-D manifold. A
    curve that climbs slowly past PC20 means variance is spread across
    many directions.
    """
    gc = data['global_cov'].sort_values('pc')
    n_pc = len(gc)
    pcs = np.arange(1, n_pc + 1)
    cum_var = gc['cum_var'].values * 100

    color = MODALITY_COLOR[modality]

    # Main curve
    ax.plot(pcs, cum_var, color=color, linewidth=2.2, zorder=3)
    # Filled area under the curve in a tinted modality color
    ax.fill_between(pcs, 0, cum_var, color=_tint(color, 0.18),
                     alpha=0.55, zorder=2)

    # 80% and 90% reference lines (horizontal)
    ax.axhline(80, color='#888888', linestyle='--', linewidth=0.8,
                alpha=0.8, zorder=1)
    ax.axhline(90, color='#888888', linestyle='--', linewidth=0.8,
                alpha=0.8, zorder=1)

    # Annotated PC counts at the threshold crossings — placed AT the
    # bottom of the plot near the dotted vertical lines to avoid colliding
    # with the curve itself or with each other. The labels read along the
    # x-axis where the threshold crossings actually live.
    n80 = int(data['geom'].get('pcs_for_80_var', _crossing(cum_var, 80)))
    n90 = int(data['geom'].get('pcs_for_90_var', _crossing(cum_var, 90)))
    ax.axvline(n80, color=color, linestyle=':', linewidth=0.9, alpha=0.8, zorder=1)
    ax.axvline(n90, color=color, linestyle=':', linewidth=0.9, alpha=0.8, zorder=1)

    # Bottom-anchored annotations so they don't fight the curve. Both placed
    # below the curve in a stacked vertical layout near the n80/n90 lines.
    ax.text(n80, 12, f' 80% @ {n80}',
             fontsize=11, color='#333333', fontweight='bold',
             va='center', ha='left',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                       edgecolor='none', alpha=0.92))
    ax.text(n90, 28, f' 90% @ {n90}',
             fontsize=11, color='#333333', fontweight='bold',
             va='center', ha='left',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                       edgecolor='none', alpha=0.92))

    # Global PR callout (top-left corner)
    g_pr = data['geom'].get('global_pr', float('nan'))
    ax.text(0.04, 0.94,
             f'Global PR = {g_pr:.1f}',
             transform=ax.transAxes,
             fontsize=13, color='#222222', fontweight='bold',
             va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                       edgecolor=color, linewidth=1.0, alpha=0.94))

    ax.set_xlim(0.5, n_pc + 0.5)
    ax.set_ylim(0, 102)
    ax.set_xlabel('Principal component index', fontsize=12)
    ax.set_ylabel('Cumulative variance (%)', fontsize=12)
    ax.tick_params(labelsize=11)

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')

    ax.grid(True, which='major', linestyle='-', linewidth=0.45,
            color='#E5E5E5', alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def _crossing(curve: np.ndarray, threshold: float) -> int:
    """Index (1-based) of the first PC where cumulative variance ≥ threshold."""
    over = np.where(curve >= threshold)[0]
    return int(over[0] + 1) if len(over) else len(curve)


# ---------------------------------------------------------------------------
# Column 2: local effective-dimensionality histogram
# ---------------------------------------------------------------------------
def draw_local_id_histogram(ax, modality: str, data: dict):
    """Histogram of local_n80 across probe locations.

    local_n80 is "how many PCs does each neighborhood need to capture 80%
    of its local variance" — the local analogue of the global n80 in
    column 1. The DISTRIBUTION of this quantity tells you whether the
    manifold is locally uniform (tight tall histogram) or wildly
    heterogeneous (wide flat histogram).
    """
    lp = data['local_pca']
    if 'local_n80' not in lp.columns:
        # Fall back to local_pr if local_n80 isn't available
        values = lp['local_pr'].dropna().values
        x_label = 'Local participation ratio'
        unit = 'PR'
    else:
        values = lp['local_n80'].dropna().values
        x_label = 'Local n₈₀  (PCs for 80% local variance)'
        unit = 'PCs'

    color = MODALITY_COLOR[modality]

    if len(values) == 0:
        ax.text(0.5, 0.5, 'no local PCA data',
                ha='center', va='center', transform=ax.transAxes,
                color='#888581')
        ax.axis('off')
        return

    # Choose bins from the actual data range, padded slightly
    vmin = float(np.floor(values.min()))
    vmax = float(np.ceil(values.max()))
    if vmax - vmin < 1:
        vmax = vmin + 1
    # Integer bins if local_n80 (always integer); fine bins otherwise
    if 'local_n80' in lp.columns:
        bins = np.arange(vmin, vmax + 2) - 0.5   # integer-centered bins
    else:
        bins = 30

    ax.hist(values, bins=bins, color=color, alpha=0.75,
             edgecolor='white', linewidth=0.6, zorder=3)

    # Force a comparable x-axis range across all five modalities so
    # readers can directly compare the spreads. Without this, a
    # degenerate-spread modality (MODIS-Thermal: all probes at n80=2)
    # gets a tiny x-range that makes its histogram look as wide as
    # everyone else's at first glance. We use the union of all observed
    # n80 values plus a small pad.
    SHARED_XLIM_LOCAL_N80 = (0.5, 10.5)   # covers observed range 1..9 + small pad
    if 'local_n80' in lp.columns:
        ax.set_xlim(*SHARED_XLIM_LOCAL_N80)

    # Mean local value (red dashed)
    mean_v = float(np.mean(values))
    ax.axvline(mean_v, color='#C0392B', linestyle='--', linewidth=2.0,
                zorder=4, label=f'mean = {mean_v:.1f} {unit}')

    # Global PR reference (dotted), placed only if it fits in-axes
    g_pr = data['geom'].get('global_pr', float('nan'))
    if np.isfinite(g_pr) and g_pr <= vmax * 1.05:
        ax.axvline(g_pr, color='#222222', linestyle=':', linewidth=1.6,
                    zorder=4, label=f'global PR = {g_pr:.1f}')

    # Spread annotation: this is the heterogeneity-of-heterogeneity number
    std_v = float(np.std(values))
    spread_text = f'spread (std) = {std_v:.1f}'
    # Highlight the geometrically interesting cases — flag the extreme
    # uniformity of MODIS-Thermal explicitly because the visual (one tall
    # bar) doesn't immediately read as "every probe needs exactly the
    # same dimensionality"
    if std_v < 0.05 and 'local_n80' in lp.columns:
        spread_text += '\n(locally uniform)'
    elif std_v > 1.2 and 'local_n80' in lp.columns:
        spread_text += '\n(locally varied)'
    ax.text(0.96, 0.94,
             spread_text,
             transform=ax.transAxes,
             fontsize=12, color='#222222', fontweight='bold',
             va='top', ha='right', linespacing=1.2,
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                       edgecolor=color, linewidth=1.0, alpha=0.94))

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Probe count', fontsize=12)
    ax.tick_params(labelsize=11)

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')

    ax.grid(True, which='major', linestyle='-', linewidth=0.45,
            color='#E5E5E5', alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    # Compact legend in the upper-left
    ax.legend(loc='upper left', fontsize=10.5, frameon=True,
               framealpha=0.92, edgecolor='#AAAAAA',
               handletextpad=0.5, labelspacing=0.4, borderpad=0.5)


# ---------------------------------------------------------------------------
# Column 3: CONUS map of dominant local dimension
# ---------------------------------------------------------------------------
def draw_dominant_dim_map(ax, modality: str, data: dict):
    """Probes colored by which 64-D dimension dominates locally.

    The legend shows the top-K dominant dims (by global count); everything
    else collapses into a muted "other" category. Spatial pattern reading:
    multiple distinct color regions = the manifold uses different
    dimensions in different parts of CONUS (spatially heterogeneous).
    One color filling most of the map = spatially uniform.
    """
    lp = data['local_pca']
    if not {'lon', 'lat', 'dom_dim'}.issubset(lp.columns):
        ax.text(0.5, 0.5, 'no dom_dim data',
                ha='center', va='center', transform=ax.transAxes,
                color='#888581')
        ax.axis('off')
        return

    add_conus_basemap(ax, land=True, states=True)

    # Top-K dominant dimensions by overall probe count
    dom_counts = lp['dom_dim'].value_counts()
    top_dims = dom_counts.head(TOP_K_DOM_DIMS).index.tolist()
    n_total_dims = len(dom_counts)

    # Categorical palette chosen for distinctness; first slot picks up the
    # modality color so the row identity is reinforced
    modality_color = MODALITY_COLOR[modality]
    categorical_palette = [
        modality_color,
        '#E89A3C',   # warm orange
        '#5B9BD5',   # mid blue
        '#7FB069',   # sage green
        '#B85C9F',   # magenta
        '#3C7872',   # teal
        '#C26E60',   # rust
    ]
    palette = {dim: categorical_palette[i % len(categorical_palette)]
                for i, dim in enumerate(top_dims)}
    other_color = '#CFCBC2'

    # Plot "other" first so top dims render above
    other_mask = ~lp['dom_dim'].isin(top_dims)
    if other_mask.any():
        n_other = int(other_mask.sum())
        ax.scatter(lp.loc[other_mask, 'lon'], lp.loc[other_mask, 'lat'],
                    c=other_color, s=8, alpha=0.40, edgecolors='none',
                    zorder=2, label=f'other  (n={n_other})')

    # Top-K dimensions in palette colors — larger and more saturated so
    # the spatial pattern reads clearly above the "other" backdrop
    for dim in top_dims:
        mask = lp['dom_dim'] == dim
        ax.scatter(lp.loc[mask, 'lon'], lp.loc[mask, 'lat'],
                    c=palette[dim], s=28, alpha=0.95, edgecolors='white',
                    linewidths=0.5, zorder=3,
                    label=f'{dim}  (n={int(mask.sum())})')

    # Spatial-diversity callout: how many distinct dominant dims appear?
    ax.text(0.02, 0.97,
             f'{n_total_dims} dims active\n'
             f'top-{TOP_K_DOM_DIMS} cover '
             f'{dom_counts.head(TOP_K_DOM_DIMS).sum() / len(lp) * 100:.0f}%',
             transform=ax.transAxes,
             fontsize=11.5, color='#222222', fontweight='bold',
             va='top', ha='left', linespacing=1.3,
             bbox=dict(boxstyle='round,pad=0.40', facecolor='white',
                       edgecolor=modality_color, linewidth=1.0, alpha=0.94))

    # Legend on the right edge of the map
    ax.legend(loc='center left', bbox_to_anchor=(1.005, 0.5),
               fontsize=10.5, frameon=False,
               handletextpad=0.5, labelspacing=0.55, borderpad=0.0)


# ---------------------------------------------------------------------------
# Row-header (modality name, on the very left)
# ---------------------------------------------------------------------------
def draw_row_header(ax, modality: str, data: dict):
    """Stamp the modality name and three headline geometry numbers as a
    compact row-label panel.
    """
    ax.axis('off')
    color = MODALITY_COLOR[modality]
    label = MODALITY_LABEL[modality]

    ax.text(0.5, 0.78, label,
             fontsize=17, fontweight='bold', color=color,
             transform=ax.transAxes,
             ha='center', va='center')

    g_pr  = data['geom'].get('global_pr', float('nan'))
    int_d = data['geom'].get('mean_intrinsic_dim', float('nan'))
    l_pr  = data['geom'].get('mean_local_pr', float('nan'))

    metrics_text = (
        f'Global PR\n{g_pr:.1f}\n'
        f'\n'
        f'Intrinsic dim\n{int_d:.1f}\n'
        f'\n'
        f'Mean local PR\n{l_pr:.1f}'
    )
    ax.text(0.5, 0.32, metrics_text,
             fontsize=12, color='#333333',
             transform=ax.transAxes,
             ha='center', va='center', linespacing=1.4)


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
def _tint(hex_color: str, weight: float) -> tuple:
    h = hex_color.lstrip('#')
    r, g, b = tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return (1 - weight * (1 - r),
            1 - weight * (1 - g),
            1 - weight * (1 - b))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true',
                    help='Print resolved data file paths and exit')
    args = p.parse_args()

    paths = resolve_paths()

    if args.dry_run:
        display = {k: (v if v is not None
                        else Path(f'<not found for {k}>'))
                    for k, v in paths.items()}
        dry_run_report('Figure 3b — manifold portraits', display)
        return

    apply_rc()

    per_modality = {}
    for m in MODALITY_ORDER:
        try:
            data = load_modality(m, paths)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f'  [skip] {m}: {e}')
            continue
        per_modality[m] = data
        g_pr  = data['geom'].get('global_pr', float('nan'))
        int_d = data['geom'].get('mean_intrinsic_dim', float('nan'))
        print(f'  [ok]   {m}: global PR={g_pr:.1f}, intrinsic dim={int_d:.1f}')

    if not per_modality:
        raise RuntimeError(
            'No modality manifold-geometry data found. '
            'Run scripts 12 (or 6_1) per-modality first.')

    # ──────────────────────────────────────────────────────────────────────
    # Layout: 5 modality rows × 4 columns
    # (row-header, variance spectrum, local-ID histogram, CONUS dom-dim map)
    # No top banner — captions go in the paper text.
    # ──────────────────────────────────────────────────────────────────────
    n_modalities = len(MODALITY_ORDER)
    row_h = 3.6   # taller rows so maps and panels read comfortably
    fig_h = row_h * n_modalities
    fig = plt.figure(figsize=(20.0, fig_h))

    # The 4-column row template: narrow header + variance spectrum +
    # local-n80 histogram + WIDE CONUS map. The map gets the largest
    # share of width since it carries the most visual information per
    # square inch and benefits most from extra room.
    gs_rows = gridspec.GridSpec(
        n_modalities, 4,
        width_ratios=[0.42, 1.25, 1.25, 2.7],
        hspace=0.32, wspace=0.20,
        left=0.020, right=0.985, top=0.985, bottom=0.020,
    )

    for r, m in enumerate(MODALITY_ORDER):
        ax_id      = fig.add_subplot(gs_rows[r, 0])
        ax_spec    = fig.add_subplot(gs_rows[r, 1])
        ax_lochist = fig.add_subplot(gs_rows[r, 2])
        ax_map     = fig.add_subplot(gs_rows[r, 3])

        if m not in per_modality:
            ax_id.axis('off')
            ax_id.text(0.5, 0.5, MODALITY_LABEL[m],
                        fontsize=16, fontweight='bold',
                        color=MODALITY_COLOR[m],
                        transform=ax_id.transAxes,
                        ha='center', va='center')
            for ax in (ax_spec, ax_lochist, ax_map):
                ax.text(0.5, 0.5, '(geometry data not found)',
                         ha='center', va='center', transform=ax.transAxes,
                         color='#888888', fontsize=11, style='italic')
                ax.axis('off')
            continue

        data = per_modality[m]
        draw_row_header(ax_id, m, data)
        draw_variance_spectrum(ax_spec, m, data)
        draw_local_id_histogram(ax_lochist, m, data)
        draw_dominant_dim_map(ax_map, m, data)

    saved = save_figure(fig, 'fig4_manifold_portrait')
    plt.close(fig)
    print('\nSaved:')
    for p_ in saved:
        print(f'  {p_}')


if __name__ == '__main__':
    main()

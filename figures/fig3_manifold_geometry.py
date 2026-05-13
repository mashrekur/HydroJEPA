"""
fig3_manifold_geometry.py

Figure 3 — Manifold geometry across the Mini-JEPA fleet.

Visual content (no figure-level title; intended for the figure caption):

  Cells 0-4: five CONUS maps. Each map shows ~2,000 probe locations
             colored by *local participation ratio*, a continuous measure
             of how many embedding directions vary jointly within each
             probe's k-nearest neighborhood. Color is the modality color;
             intensity scales with local PR on a shared range across all
             five maps (vmin=1.5, vmax=9.0) so the maps are directly
             comparable. Light = locally simple manifold (few directions
             matter). Dark = locally complex (many directions co-vary).

  Cell 5:    geometric signature scatter — five modality points in
             (global participation ratio, mean local PR) space. Same
             architecture, same training recipe; manifolds spread along
             a tradeoff axis from "globally rich, locally simple" to
             "globally compact, locally complex." MODIS-Thermal and
             S1-SAR sit at opposite corners.

Inputs (per modality):
  reports/minijepas/<m>/manifold_geometry/hydrojepa_local_pca.csv
  reports/minijepas/<m>/manifold_geometry/hydrojepa_geometry_summary.json

Run:
  python fig3_manifold_geometry.py --dry-run
  python fig3_manifold_geometry.py
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
    add_conus_basemap, annotation_box,
)


def first_existing(*candidates: Path) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None


def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _tint_with_white(hex_color: str, weight: float) -> tuple[float, float, float]:
    """Mix a hex color with white. weight=0 → white, weight=1 → original."""
    r, g, b = _hex_to_rgb(hex_color)
    return (1 - weight * (1 - r),
            1 - weight * (1 - g),
            1 - weight * (1 - b))


def fig3_modality_cmap(modality: str, n: int = 256):
    """Per-modality cmap from a light tint of the color → saturated color.

    This is a Figure-3 specific cmap (not the one in _style.py) because
    Figure 3 uses a per-modality percentile stretch and needs even the
    lowest-PR probes to show as visibly tinted. Starting from white would
    make light modality colors (purple, blue, green) read as pure white
    at the low end. Starting from a 25%-saturated tint instead lifts the
    floor so the color identifies the modality even where intensity is low.
    """
    from matplotlib.colors import LinearSegmentedColormap
    base_hex = MODALITY_COLOR[modality]
    tint     = _tint_with_white(base_hex, 0.40)
    base_rgb = _hex_to_rgb(base_hex)
    return LinearSegmentedColormap.from_list(
        f'fig3_{modality}_seq', [tint, base_rgb], N=n,
    )


warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Color scale strategy
# ---------------------------------------------------------------------------
# Each map uses its OWN [p5, p95] of local_pr as the color range, so within-
# map regional structure shows clearly. The absolute scale comparison
# happens via two channels instead of color:
#   1. The three numbers in each map's corner annotation (Global PR,
#      Intrinsic dim, Local PR)
#   2. The geometric signature scatter (cell 5), which places all five
#      Mini-JEPAs on shared axes
# This keeps within-map contrast strong (the geography reads) and pushes
# cross-modality comparison to the right channels for it.

PR_PCTILE_LOW  = 5    # percentile for vmin per modality
PR_PCTILE_HIGH = 95   # percentile for vmax per modality


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def resolve_paths():
    root = project_root()
    paths = {}
    for m in MODALITY_ORDER:
        paths[f'{m}_local_pca'] = first_existing(
            root / 'reports' / 'minijepas' / m / 'manifold_geometry'
                 / 'hydrojepa_local_pca.csv',
            *((root / 'reports' / 'manifold_geometry' / 'hydrojepa_local_pca.csv',)
               if m == 's2_optical' else ()),
        )
        paths[f'{m}_geom_summary'] = first_existing(
            root / 'reports' / 'minijepas' / m / 'manifold_geometry'
                 / 'hydrojepa_geometry_summary.json',
            *((root / 'reports' / 'manifold_geometry' / 'hydrojepa_geometry_summary.json',)
               if m == 's2_optical' else ()),
        )
    return paths


# ---------------------------------------------------------------------------
# Per-modality data loading
# ---------------------------------------------------------------------------
def load_modality_geometry(modality: str, paths: dict):
    """Return (probes_df, geom_summary_dict)."""
    pca_path  = paths[f'{modality}_local_pca']
    json_path = paths[f'{modality}_geom_summary']

    if pca_path is None or not pca_path.exists():
        raise FileNotFoundError(f'{modality}: local_pca.csv missing')
    if json_path is None or not json_path.exists():
        raise FileNotFoundError(f'{modality}: geometry_summary.json missing')

    pca  = pd.read_csv(pca_path)
    geom = json.loads(Path(json_path).read_text())
    return pca, geom


# ---------------------------------------------------------------------------
# Map drawing
# ---------------------------------------------------------------------------
def draw_complexity_map(ax, probes: pd.DataFrame, modality: str,
                         geom_summary: dict):
    """One CONUS panel: probes colored by local PR on a shared scale.

    Modality-colored sequential cmap. Low local_pr → light tint of the
    modality color. High local_pr → fully saturated modality color.
    """
    add_conus_basemap(ax, land=True, states=True)

    cmap = fig3_modality_cmap(modality)

    # Per-modality percentile stretch so the within-map contrast is
    # always strong. Cross-modality comparison happens via the corner
    # annotations and the scatter, not via color.
    pr_all = probes['local_pr'].values
    vmin = float(np.quantile(pr_all, PR_PCTILE_LOW  / 100.0))
    vmax = float(np.quantile(pr_all, PR_PCTILE_HIGH / 100.0))
    # Guard against degenerate range
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0

    # Plot low-complexity first so high-complexity hot spots sit on top
    order = np.argsort(pr_all)
    lons  = probes['lon'].values[order]
    lats  = probes['lat'].values[order]
    pr    = pr_all[order]

    ax.scatter(
        lons, lats,
        c=pr, cmap=cmap, vmin=vmin, vmax=vmax,
        s=14, alpha=0.92, edgecolors='none', zorder=2,
    )

    ax.set_title(MODALITY_LABEL[modality], fontsize=11, fontweight='bold',
                  color=MODALITY_COLOR[modality], pad=4)

    g_pr  = geom_summary.get('global_pr', float('nan'))
    int_d = geom_summary.get('mean_intrinsic_dim', float('nan'))
    l_pr  = geom_summary.get('mean_local_pr', float('nan'))
    annotation_box(
        ax,
        f'Global PR     = {g_pr:.1f}\n'
        f'Intrinsic dim = {int_d:.1f}\n'
        f'Local PR      = {l_pr:.1f}',
        loc='lower left', fontsize=7.8, weight='bold',
    )


# ---------------------------------------------------------------------------
# Geometric signature scatter (cell 5)
# ---------------------------------------------------------------------------
def draw_geometric_signature(ax, summaries: dict[str, dict]):
    """Five modality points in (global PR, mean local PR) space.

    Size encodes intrinsic dim, color encodes modality.
    """
    rows = []
    for m in MODALITY_ORDER:
        s = summaries.get(m)
        if s is None:
            continue
        rows.append(dict(
            modality=m,
            global_pr=s.get('global_pr', np.nan),
            local_pr=s.get('mean_local_pr', np.nan),
            int_dim=s.get('mean_intrinsic_dim', np.nan),
        ))
    df = pd.DataFrame(rows)

    if len(df) == 0:
        ax.text(0.5, 0.5, 'no geometry data',
                ha='center', va='center', transform=ax.transAxes,
                color='#888581')
        ax.axis('off')
        return

    # Plot points; track handles so we can build a legend keyed by modality.
    handles = []
    for _, r in df.iterrows():
        # Marker size proportional to intrinsic dim
        size = 80 + (r['int_dim'] - 2.0) * (240.0 / 8.0)
        size = max(60, min(360, size))
        h = ax.scatter(
            r['global_pr'], r['local_pr'],
            s=size, c=MODALITY_COLOR[r['modality']],
            edgecolors='white', linewidths=1.3,
            alpha=0.95, zorder=3,
            label=MODALITY_LABEL[r['modality']],
        )
        handles.append((r['modality'], h))

    # Legend in the upper-right corner, where the diagonal data layout
    # leaves space. Modality names rendered in their modality colors so
    # the legend doubles as a color key.
    legend = ax.legend(
        [h for _, h in handles],
        [MODALITY_LABEL[m] for m, _ in handles],
        loc='upper right', frameon=True, framealpha=0.92,
        edgecolor='#888888', fontsize=8.5,
        handletextpad=0.5, labelspacing=0.45, borderpad=0.6,
        scatterpoints=1, markerscale=0.55,
    )
    legend.get_frame().set_linewidth(0.5)
    # Color each legend text in its modality color
    for text, (mname, _) in zip(legend.get_texts(), handles):
        text.set_color(MODALITY_COLOR[mname])
        text.set_fontweight('bold')

    # Title as a proper subplot title — no in-axis floating text
    ax.set_title('Manifold geometry across the fleet',
                  fontsize=10.5, fontweight='bold', pad=6, color='#222222')

    ax.set_xlabel('Global participation ratio',  fontsize=8.5)
    ax.set_ylabel('Mean local PR',                fontsize=8.5)

    # Axis limits — slightly tighter since we no longer need huge padding
    # for in-axis labels
    if len(df) >= 2:
        pad_x = max(1.5, 0.10 * (df['global_pr'].max() - df['global_pr'].min()))
        pad_y = max(0.4, 0.15 * (df['local_pr'].max() - df['local_pr'].min()))
        ax.set_xlim(df['global_pr'].min() - pad_x, df['global_pr'].max() + pad_x)
        ax.set_ylim(df['local_pr'].min() - pad_y,  df['local_pr'].max() + pad_y)

    ax.tick_params(labelsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')

    # Visible grid: solid major lines in light gray, faint minor lines.
    # Dotted-low-alpha gridlines from before were invisible at print size.
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='-', linewidth=0.6,
            color='#BFBFBF', alpha=0.9, zorder=0)
    ax.grid(True, which='minor', linestyle='-', linewidth=0.4,
            color='#E0E0E0', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


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
        dry_run_report('Figure 3 — manifold geometry', display)
        return

    apply_rc()

    # Load each modality
    panel_data = {}
    summaries  = {}
    for m in MODALITY_ORDER:
        try:
            probes, geom = load_modality_geometry(m, paths)
        except (FileNotFoundError, ValueError) as e:
            print(f'  [skip] {m}: {e}')
            continue
        panel_data[m] = (probes, geom)
        summaries[m]  = geom

    # Layout — no figure-level title; tight top margin
    fig = plt.figure(figsize=(15.0, 9.0))
    outer = gridspec.GridSpec(
        2, 1, height_ratios=[1, 1],
        hspace=0.18,
        left=0.03, right=0.99, top=0.97, bottom=0.04,
    )
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[0], wspace=0.06)
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1],
        width_ratios=[1.0, 1.0, 0.95], wspace=0.12)

    map_axes_specs = [
        gs_top[0, 0], gs_top[0, 1], gs_top[0, 2],
        gs_bot[0, 0], gs_bot[0, 1],
    ]
    ax_sig_spec = gs_bot[0, 2]

    # Maps
    for j, m in enumerate(MODALITY_ORDER):
        ax = fig.add_subplot(map_axes_specs[j])
        if m not in panel_data:
            ax.set_title(MODALITY_LABEL[m], color=MODALITY_COLOR[m])
            ax.text(0.5, 0.5, 'data not found',
                    ha='center', va='center', transform=ax.transAxes,
                    color='#888581')
            ax.axis('off')
            continue
        probes, geom = panel_data[m]
        draw_complexity_map(ax, probes, m, geom)

    # Geometric signature scatter
    ax_sig = fig.add_subplot(ax_sig_spec)
    draw_geometric_signature(ax_sig, summaries)

    saved = save_figure(fig, 'fig3_manifold_geometry')
    plt.close(fig)
    print('\nSaved:')
    for p_ in saved:
        print(f'  {p_}')


if __name__ == '__main__':
    main()

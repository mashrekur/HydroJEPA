"""
fig5_complementarity.py

Figure 4 — Mini-JEPAs encode information AlphaEarth doesn't.

Two panels, full-width top + bottom (this is a content-shape decision —
Figures 2 and 3 use a 2x3 mirror because they show five maps; Figure 4
has two distinct visual claims, not six).

Panel A (top, larger) — "What does each model see?"
  Horizontal grouped bars, one row per environmental variable.
  Three bars per row:
    - light gray:        AlphaEarth alone
    - modality-colored:  best Mini-JEPA for this variable, alone
    - dark gray:         AE + best Mini-JEPA, jointly
  The "best" Mini-JEPA is the one with the highest joint R² for that
  variable. Reading: on rows where the joint bar extends past the AE
  bar, the Mini-JEPA adds real information. On rows where it doesn't,
  AE alone is sufficient.

Panel B (bottom, smaller) — "Where does each Mini-JEPA add value?"
  7-variable × 5-modality heatmap of the gain column (joint - max(AE, HJ)).
  Diverging colormap centered at zero. Positive cells (Mini-JEPA fills
  an AE gap) get warm colors; negative cells (Mini-JEPA hurts AE) get
  cool colors. Honest about both directions.

Inputs:
  reports/minijepas/<m>/complementarity/hydrojepa_joint_predictive_gain.csv
  (or root-level mirrors)

Run:
  python fig5_complementarity.py --dry-run
  python fig5_complementarity.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (
    apply_rc, save_figure, project_root, dry_run_report,
    MODALITY_ORDER, MODALITY_LABEL, MODALITY_COLOR, AE_COLOR,
    ENVVAR_LABEL, ENVVAR_ORDER, DELTA_CMAP,
)


def first_existing(*candidates: Path) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None


warnings.filterwarnings('ignore')


# Colors for the AE bar and joint bar — neutral grays to keep modality
# color as the visual anchor for "this Mini-JEPA"
AE_BAR_COLOR    = '#B8B5AE'   # warm light gray (matches basemap state edge)
JOINT_BAR_COLOR = '#3D3D3D'   # near-black for the synthesis bar


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def resolve_paths():
    root = project_root()
    paths = {}
    for m in MODALITY_ORDER:
        paths[f'{m}_gain'] = first_existing(
            root / 'reports' / 'minijepas' / m / 'complementarity'
                 / 'hydrojepa_joint_predictive_gain.csv',
            root / 'eval_results_for_review' / f'{m}__joint_predictive_gain.csv',
            root / f'{m}__joint_predictive_gain.csv',
            *((root / 'reports' / 'complementarity'
                    / 'hydrojepa_joint_predictive_gain.csv',)
               if m == 's2_optical' else ()),
        )
    return paths


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_long(paths: dict) -> pd.DataFrame:
    """Stack all per-modality gain CSVs into one long table."""
    rows = []
    for m in MODALITY_ORDER:
        f = paths.get(f'{m}_gain')
        if f is None or not f.exists():
            continue
        df = pd.read_csv(f)
        df['modality'] = m
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def best_modality_per_variable(long_df: pd.DataFrame) -> pd.DataFrame:
    """For each env variable, return the row corresponding to the Mini-JEPA
    that achieves the highest joint R² with AlphaEarth. This is the "winner"
    that the bar chart in Panel A shows.
    """
    if long_df.empty:
        return long_df
    idx = long_df.groupby('variable')['r2_joint_mean'].idxmax()
    return long_df.loc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Panel A: grouped horizontal bars
# ---------------------------------------------------------------------------
def draw_panel_A(ax, long_df: pd.DataFrame):
    """One row per env variable, three bars each (AE / best Mini-JEPA / joint).

    Modality color identifies which Mini-JEPA "won" the joint comparison
    for that variable.
    """
    if long_df.empty:
        ax.text(0.5, 0.5, 'no complementarity data',
                ha='center', va='center', transform=ax.transAxes,
                color='#888581')
        ax.axis('off')
        return

    winners = best_modality_per_variable(long_df)
    winners = winners.set_index('variable').reindex(ENVVAR_ORDER).reset_index()

    n_rows = len(winners)
    y = np.arange(n_rows)
    bar_h = 0.27
    offsets = np.array([-bar_h, 0.0, bar_h])     # AE, MiniJEPA, Joint

    # Plot bars row-by-row so each can carry its modality color in the
    # middle bar
    for i, row in winners.iterrows():
        var = row['variable']
        modality = row['modality']
        mod_color = MODALITY_COLOR.get(modality, '#888888')

        # AE bar (top of trio)
        ax.barh(y[i] + offsets[0], row['r2_ae_mean'],
                height=bar_h, color=AE_BAR_COLOR, edgecolor='none',
                xerr=row['r2_ae_std'], error_kw=dict(elinewidth=0.7,
                                                     ecolor='#666666',
                                                     alpha=0.75),
                zorder=2)
        # Best Mini-JEPA bar (middle)
        ax.barh(y[i] + offsets[1], row['r2_hj_mean'],
                height=bar_h, color=mod_color, edgecolor='none',
                xerr=row['r2_hj_std'], error_kw=dict(elinewidth=0.7,
                                                     ecolor='#666666',
                                                     alpha=0.75),
                zorder=2)
        # Joint bar (bottom)
        ax.barh(y[i] + offsets[2], row['r2_joint_mean'],
                height=bar_h, color=JOINT_BAR_COLOR, edgecolor='none',
                xerr=row['r2_joint_std'], error_kw=dict(elinewidth=0.7,
                                                       ecolor='#666666',
                                                       alpha=0.75),
                zorder=2)

        # Value labels at the end of each bar
        for v, off in [(row['r2_ae_mean'], offsets[0]),
                        (row['r2_hj_mean'], offsets[1]),
                        (row['r2_joint_mean'], offsets[2])]:
            ax.text(v + 0.012, y[i] + off, f'{v:.2f}',
                    va='center', ha='left', fontsize=7.5,
                    color='#333333')

        # Annotate which Mini-JEPA won this row, in modality color, at
        # the right edge of the joint bar
        ax.text(row['r2_joint_mean'] + 0.10, y[i] + offsets[2],
                f'  +  {MODALITY_LABEL[modality]}',
                va='center', ha='left', fontsize=8,
                color=mod_color, fontweight='bold')

        # Gain annotation: show the (joint - max) delta with a sign
        gain = row['gain_over_max']
        gain_color = '#1F7A1F' if gain > 0.005 else (
            '#A33333' if gain < -0.005 else '#888888'
        )
        gain_str = f'Δ = {gain:+.3f}'
        ax.text(1.06, y[i], gain_str,
                va='center', ha='left', fontsize=8.5,
                fontweight='bold', color=gain_color,
                transform=ax.get_yaxis_transform())

    # Style
    ax.set_yticks(y)
    ax.set_yticklabels([ENVVAR_LABEL[v] for v in winners['variable']],
                        fontsize=9.5)
    ax.invert_yaxis()  # first variable at top

    ax.set_xlim(0, 1.04)
    ax.set_xlabel('Cross-validated R²', fontsize=9.5)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=8.5)

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')

    ax.grid(True, axis='x', which='major', linestyle='-',
            linewidth=0.5, color='#E2E2E2', alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    # Title positioned as a subplot title (no figure-level title text)
    ax.set_title(
        'AlphaEarth vs best Mini-JEPA vs joint, per environmental variable',
        fontsize=11, fontweight='bold', pad=8, color='#222222'
    )

    # Legend in the figure (placed via fig.legend in the driver; here
    # we just signal what each bar means)
    legend_handles = [
        Patch(facecolor=AE_BAR_COLOR, edgecolor='none', label='AlphaEarth alone'),
        Patch(facecolor='#888888', edgecolor='none',
              label='Best Mini-JEPA alone  (color = which one)'),
        Patch(facecolor=JOINT_BAR_COLOR, edgecolor='none',
              label='AE + best Mini-JEPA, jointly'),
    ]
    ax.legend(handles=legend_handles, loc='lower right',
              frameon=True, framealpha=0.92, edgecolor='#AAAAAA',
              fontsize=8.5, handletextpad=0.5, labelspacing=0.4,
              borderpad=0.6)


# ---------------------------------------------------------------------------
# Panel B: gain heatmap
# ---------------------------------------------------------------------------
def draw_panel_B(ax, long_df: pd.DataFrame):
    """7-variable × 5-modality heatmap of the gain_over_max column.

    Positive cells = Mini-JEPA fills an AE gap. Negative cells = combining
    hurts AE. Diverging cmap centered at zero.
    """
    if long_df.empty:
        ax.text(0.5, 0.5, 'no complementarity data',
                ha='center', va='center', transform=ax.transAxes,
                color='#888581')
        ax.axis('off')
        return

    pivot = (long_df.pivot_table(index='variable', columns='modality',
                                  values='gain_over_max', aggfunc='mean')
                    .reindex(index=ENVVAR_ORDER, columns=MODALITY_ORDER))
    data = pivot.values

    # Symmetric color scale around zero, capped at the data extreme
    abs_max = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 0.05
    vmax = max(abs_max, 0.01)

    im = ax.imshow(data, cmap=DELTA_CMAP, vmin=-vmax, vmax=vmax, aspect='auto')

    # Cell labels with sign-aware emphasis: bold for clear wins/losses,
    # faint for cells near zero
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            if abs(v) >= 0.015:
                weight = 'bold'
                color  = '#222222' if abs(v) < 0.6 * vmax else 'white'
            else:
                weight = 'normal'
                color  = '#555555'
            ax.text(j, i, f'{v:+.3f}', ha='center', va='center',
                    fontsize=8.5, color=color, fontweight=weight)

    ax.set_xticks(range(len(MODALITY_ORDER)))
    ax.set_xticklabels([MODALITY_LABEL[m] for m in MODALITY_ORDER],
                        rotation=20, ha='right', fontsize=9)
    ax.set_yticks(range(len(ENVVAR_ORDER)))
    ax.set_yticklabels([ENVVAR_LABEL[v] for v in ENVVAR_ORDER], fontsize=9)

    for tick, m in zip(ax.get_xticklabels(), MODALITY_ORDER):
        tick.set_color(MODALITY_COLOR[m])
        tick.set_fontweight('bold')

    ax.set_title(
        'Joint − max(AE, Mini-JEPA)  per (variable, Mini-JEPA)',
        fontsize=11, fontweight='bold', pad=8, color='#222222'
    )

    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    # Colorbar — horizontal under the heatmap to save horizontal space
    cbar = plt.colorbar(im, ax=ax, orientation='vertical',
                         fraction=0.025, pad=0.015, shrink=0.85)
    cbar.set_label('Δ R²  (joint gain over best single)', fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_visible(False)


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
        dry_run_report('Figure 4 — complementarity', display)
        return

    apply_rc()

    long_df = load_long(paths)

    # Layout — two stacked panels, full width
    # Top panel taller (it's the headline); bottom panel compact
    fig = plt.figure(figsize=(13.5, 11.0))
    outer = gridspec.GridSpec(
        2, 1, height_ratios=[1.0, 0.8],
        hspace=0.28,
        left=0.13, right=0.97, top=0.96, bottom=0.06,
    )

    ax_a = fig.add_subplot(outer[0])
    ax_b = fig.add_subplot(outer[1])

    draw_panel_A(ax_a, long_df)
    draw_panel_B(ax_b, long_df)

    saved = save_figure(fig, 'fig5_complementarity')
    plt.close(fig)
    print('\nSaved:')
    for p_ in saved:
        print(f'  {p_}')


if __name__ == '__main__':
    main()

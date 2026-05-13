"""
fig7_experimental_results.py

Figure 6 — Where the routed Mini-JEPA fleet adds value, and the structure
of the evidence.

Three panels:

  Panel A — Per-category effect sizes
    Cohen's d for C5 (AE + fleet) vs C2 (AE-only) across all four
    question categories. The single_modality bar is the headline:
    d ≈ 1.10, p ≈ 0.031 (significant). Other categories are trivial.
    The figure shows all four so the reader sees the effect is
    concentrated, not pervasive. Reads from significance_per_category.csv

  Panel B — Inter-judge calibration
    Same comparisons, broken down by judge (Haiku 4.5 vs GPT-OSS-120B).
    Both judges saw all 40 answers; what did each one conclude?
    Reads from significance_per_judge.csv.

  Panel C — Routing-quality interaction (causal ablation)
    Per-question, was routing a hit or miss? How does answer quality
    differ between hits and misses? This is the actual mechanistic
    ablation: does correct routing translate to higher quality?
    Reads from routing_quality_interaction.csv.

Inputs:
  data/hydrojepa/minijepa_eval/runs/claude-opus-47__opus47_active_n40/
    diagnostics/significance_per_category.csv
    diagnostics/significance_per_judge.csv
    diagnostics/routing_quality_interaction.csv

Run:
  python fig7_experimental_results.py --dry-run
  python fig7_experimental_results.py
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
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (
    apply_rc, save_figure, project_root, dry_run_report,
)


def first_existing(*candidates: Path) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None


warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
CATEGORY_ORDER = ['single_modality', 'multi_modality', 'sar_favorable',
                   'ae_favorable']

CATEGORY_SHORT = {
    'single_modality': 'Single-\nmodality',
    'multi_modality':  'Multi-\nmodality',
    'sar_favorable':   'SAR-\nfavorable',
    'ae_favorable':    'AE-\nfavorable',
}

JUDGE_LABEL = {
    'claude-haiku-4.5': 'Claude Haiku 4.5',
    'gpt-oss-120b':     'GPT-OSS-120B',
}
JUDGE_ORDER = ['claude-haiku-4.5', 'gpt-oss-120b']

COMPARISON_SHORT = {
    'C5 vs C2 (AE+fleet vs AE-only)':       'AE+fleet\nvs AE-only',
    'C5 vs C4 (AE+fleet vs fleet-only)':    'AE+fleet\nvs fleet-only',
    'C4 vs C2 (fleet-only vs AE-only)':     'Fleet-only\nvs AE-only',
}
COMPARISON_ORDER = [
    'C5 vs C2 (AE+fleet vs AE-only)',
    'C4 vs C2 (fleet-only vs AE-only)',
    'C5 vs C4 (AE+fleet vs fleet-only)',
]

# Palette
INK         = '#1A1A1A'
INK_SOFT    = '#555555'
INK_FAINT   = '#888888'
NEUTRAL     = '#B8B5AE'
ACCENT_DARK = '#1F2933'
ACCENT_HIT  = '#2A9D8F'    # routing hit (correct)
ACCENT_MISS = '#E76F51'    # routing miss


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def resolve_paths():
    root = project_root()
    diag = (root / 'data' / 'hydrojepa' / 'minijepa_eval' / 'runs'
                 / 'claude-opus-47__opus47_active_n40' / 'diagnostics')
    return {
        'cat_sig':    diag / 'significance_per_category.csv',
        'judge_sig':  diag / 'significance_per_judge.csv',
        'route_int':  diag / 'routing_quality_interaction.csv',
    }


# ---------------------------------------------------------------------------
# Panel A — per-category effect sizes
# ---------------------------------------------------------------------------
def draw_panel_A(ax, sig_df: pd.DataFrame):
    """Cohen's d (C5 vs C2) per category, with n and significance annotation."""
    if sig_df.empty:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, color='#888581')
        ax.axis('off')
        return

    target_comparison = 'C5 vs C2 (AE+fleet vs AE-only)'
    sub = sig_df[sig_df['comparison'] == target_comparison].copy()
    sub = sub.set_index('category').reindex(CATEGORY_ORDER).reset_index()

    cats = sub['category'].tolist()
    ds   = sub['cohens_d'].values
    ps   = sub['wilcoxon_p'].values
    ns   = sub['n_paired'].values
    sigs = sub['sig_05'].values

    x = np.arange(len(cats))

    # Bars — significant ones get accent color, others get neutral
    bar_colors = [ACCENT_DARK if s else NEUTRAL for s in sigs]
    bars = ax.bar(x, ds, width=0.55, color=bar_colors,
                   edgecolor='none', zorder=3)

    # Zero reference line
    ax.axhline(0, color='#888888', linewidth=0.8, zorder=2)

    # Faint horizontal bands marking magnitude thresholds (Cohen's d
    # conventions: 0.2 small, 0.5 medium, 0.8 large)
    for thresh, label in [(0.2, 'small'), (0.5, 'medium'), (0.8, 'large')]:
        ax.axhline(thresh, color='#DDDDDD', linewidth=0.5,
                   linestyle='--', zorder=1)
        ax.text(len(cats) - 0.4, thresh + 0.018, label,
                fontsize=9.5, color='#999999', ha='right', va='bottom',
                style='italic')

    # Per-bar annotations: d value, p value, n
    for i, (d, p, n, s) in enumerate(zip(ds, ps, ns, sigs)):
        if np.isnan(d):
            continue
        # Value label above bar
        y_top = d + 0.04 if d >= 0 else d - 0.10
        star = ' *' if s else ''
        ax.text(x[i], y_top,
                f'd = {d:.2f}{star}',
                ha='center', va='bottom' if d >= 0 else 'top',
                fontsize=13, fontweight='bold' if s else '500',
                color=INK if s else INK_SOFT)
        ax.text(x[i], y_top + (0.13 if d >= 0 else -0.13),
                f'p = {p:.3f}',
                ha='center', va='bottom' if d >= 0 else 'top',
                fontsize=11, color=INK_SOFT)
        # n at base
        ax.text(x[i], -0.15, f'n = {n}',
                ha='center', va='top', fontsize=11,
                color=INK_FAINT)

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_SHORT[c] for c in cats],
                       fontsize=12, color='#333333')
    ax.set_ylabel("Cohen's d  (AE + fleet vs AE-only)",
                  fontsize=13)
    ax.set_ylim(-0.45, 1.45)
    ax.tick_params(labelsize=11)

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.grid(True, axis='y', linestyle='-', linewidth=0.4,
            color='#EEEEEE', alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title('Effect size by question category',
                 fontsize=14, fontweight='bold', pad=12, color='#222222',
                 loc='left', x=0.0)


# ---------------------------------------------------------------------------
# Panel B — per-judge calibration
# ---------------------------------------------------------------------------
def draw_panel_B(ax, judge_df: pd.DataFrame):
    """Per-judge Cohen's d for three pairwise comparisons.

    Two judges side by side. Three bars per judge. Reader sees how each
    judge separately concluded.
    """
    if judge_df.empty:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, color='#888581')
        ax.axis('off')
        return

    # Pivot: rows = comparison, cols = judge
    pivot = (judge_df.pivot_table(index='comparison', columns='judge_model',
                                   values='cohens_d', aggfunc='mean')
                     .reindex(index=COMPARISON_ORDER, columns=JUDGE_ORDER))
    pvals = (judge_df.pivot_table(index='comparison', columns='judge_model',
                                   values='wilcoxon_p', aggfunc='mean')
                     .reindex(index=COMPARISON_ORDER, columns=JUDGE_ORDER))

    n_comps = len(COMPARISON_ORDER)
    n_judges = len(JUDGE_ORDER)
    x = np.arange(n_comps)
    bar_w = 0.38

    judge_colors = ['#5B7C8E', '#A08661']  # cool blue-gray, warm tan
    judge_colors_dict = dict(zip(JUDGE_ORDER, judge_colors))

    for j, judge in enumerate(JUDGE_ORDER):
        offset = (j - (n_judges - 1) / 2) * bar_w
        ds = pivot[judge].values
        ps = pvals[judge].values

        ax.bar(x + offset, ds, width=bar_w * 0.92,
               color=judge_colors_dict[judge], edgecolor='none',
               zorder=3, label=JUDGE_LABEL[judge])

        # d value labels above each bar
        for i, (d, p) in enumerate(zip(ds, ps)):
            if np.isnan(d):
                continue
            y_top = d + 0.025 if d >= 0 else d - 0.06
            ax.text(x[i] + offset, y_top,
                    f'{d:.2f}',
                    ha='center', va='bottom' if d >= 0 else 'top',
                    fontsize=11.5, fontweight='600', color='#333333')

    # Zero line + magnitude thresholds
    ax.axhline(0, color='#888888', linewidth=0.8, zorder=2)
    for thresh in [0.2]:
        ax.axhline(thresh, color='#DDDDDD', linewidth=0.5,
                   linestyle='--', zorder=1)
        ax.text(n_comps - 0.45, thresh + 0.012, 'small',
                fontsize=9.5, color='#999999', ha='right', va='bottom',
                style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels([COMPARISON_SHORT[c] for c in COMPARISON_ORDER],
                       fontsize=11.5, color='#333333')
    ax.set_ylabel("Cohen's d  (per judge, all 40 questions)",
                  fontsize=13)
    ax.set_ylim(-0.35, 0.55)
    ax.tick_params(labelsize=11)

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.grid(True, axis='y', linestyle='-', linewidth=0.4,
            color='#EEEEEE', alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc='upper right', frameon=True, framealpha=0.95,
              edgecolor='#AAAAAA', fontsize=11.5,
              handletextpad=0.5, labelspacing=0.4, borderpad=0.5)

    ax.set_title('Inter-judge calibration',
                 fontsize=14, fontweight='bold', pad=12, color='#222222',
                 loc='left', x=0.0)


# ---------------------------------------------------------------------------
# Panel C — routing-quality interaction
# ---------------------------------------------------------------------------
def draw_panel_C(ax, route_df: pd.DataFrame):
    """Does routing correctness affect answer quality?

    For each (question, condition) row, we have:
      - hit: did routing pick the expected modalities?
      - cond_score: judge score for the answer
      - ae_only: judge score for the AE-only baseline
      - delta_vs_ae: cond_score - ae_only

    Two grouped bars: hits vs misses, on the delta_vs_ae axis. If
    correct routing yields better answers, delta_vs_ae should be larger
    for hits. This is the causal ablation.
    """
    if route_df.empty:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, color='#888581')
        ax.axis('off')
        return

    # The routing_quality_interaction.csv has rows for both `dual_rag` and
    # `multi_select` conditions. We focus on dual_rag for the headline
    # claim (the operational system).
    df = route_df.copy()
    df = df[df['condition'] == 'dual_rag']

    # Some rows have no expected_modalities (ae_favorable category):
    # those are NOT routing decisions in the testable sense. Drop them.
    # The 'hit' column is then well-defined for the rest.
    df = df[df['expected'].notna() & (df['expected'] != '[]')]

    # If the 'hit' column is missing or all True (per our earlier finding),
    # this panel falls back to showing the distribution of delta_vs_ae.
    if 'hit' not in df.columns or len(df) == 0:
        ax.text(0.5, 0.5, 'no routing-quality data',
                ha='center', va='center', transform=ax.transAxes,
                color=INK_FAINT)
        ax.axis('off')
        return

    df['hit'] = df['hit'].astype(bool)
    hits   = df[df['hit']]['delta_vs_ae'].dropna()
    misses = df[~df['hit']]['delta_vs_ae'].dropna()

    # Compute means and SEMs
    def stats(arr):
        if len(arr) == 0:
            return (np.nan, 0.0, 0)
        return (float(arr.mean()), float(arr.std() / np.sqrt(len(arr))),
                len(arr))

    hit_m, hit_se, hit_n = stats(hits)
    mis_m, mis_se, mis_n = stats(misses)

    # If ALL rows are hits (no variation), this panel can't ablate.
    # Show the overall distribution instead with a clear caption.
    if mis_n == 0:
        # All hits → show distribution of delta_vs_ae for routed questions
        # with a kernel-density-ish histogram and a vertical line at zero
        # to read sign at a glance.
        ax.hist(hits, bins=14, color=ACCENT_HIT, alpha=0.85,
                edgecolor='white', linewidth=0.6, zorder=3)
        ax.axvline(0, color='#888888', linewidth=1.0, zorder=4)
        ax.axvline(hit_m, color=ACCENT_DARK, linewidth=1.6,
                    linestyle='--', zorder=5,
                    label=f'mean = {hit_m:+.3f}')

        # Count how many positive / negative
        n_pos = int((hits > 0).sum())
        n_neg = int((hits < 0).sum())
        n_zero = int((hits == 0).sum())

        # Single tight summary box in upper-right with the counts
        ax.text(0.98, 0.96,
                f'positive: {n_pos}\nnegative: {n_neg}\ntied: {n_zero}',
                transform=ax.transAxes, va='top', ha='right',
                fontsize=11.5, color='#333333', fontweight='600',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                           edgecolor='#DDDDDD', linewidth=0.6, alpha=0.92))

        ax.set_xlabel('Δ score  (dual-RAG − AE-only,  per question)',
                       fontsize=13)
        ax.set_ylabel('Question count', fontsize=13)
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.78),
                   frameon=False, fontsize=11.5)

    else:
        # Standard ablation: two bars
        x = np.array([0, 1])
        means = [hit_m, mis_m]
        sems = [hit_se, mis_se]
        colors = [ACCENT_HIT, ACCENT_MISS]
        labels = [f'Routing hit\n(n = {hit_n})',
                  f'Routing miss\n(n = {mis_n})']
        ax.bar(x, means, width=0.55, color=colors, edgecolor='none',
               yerr=sems,
               error_kw=dict(elinewidth=1.0, ecolor='#444444', capsize=4),
               zorder=3)

        for i, (m, n) in enumerate(zip(means, [hit_n, mis_n])):
            if np.isnan(m):
                continue
            y_top = m + 0.012 if m >= 0 else m - 0.022
            ax.text(x[i], y_top, f'{m:+.3f}',
                    ha='center', va='bottom' if m >= 0 else 'top',
                    fontsize=13, fontweight='bold', color='#222222')

        ax.axhline(0, color='#888888', linewidth=0.8, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12, color='#333333')
        ax.set_ylabel('Δ score  (vs AE-only,  per question)', fontsize=13)

    ax.tick_params(labelsize=11)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.grid(True, axis='y', linestyle='-', linewidth=0.4,
            color='#EEEEEE', alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title('Routing — quality interaction',
                 fontsize=14, fontweight='bold', pad=12, color='#222222',
                 loc='left', x=0.0)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    paths = resolve_paths()

    if args.dry_run:
        display = {k: (v if v.exists() else Path(f'<not found: {v}>'))
                    for k, v in paths.items()}
        dry_run_report('Figure 6 — experimental results', display)
        return

    apply_rc()

    # Load
    cat_sig   = pd.read_csv(paths['cat_sig'])   if paths['cat_sig'].exists()   else pd.DataFrame()
    judge_sig = pd.read_csv(paths['judge_sig']) if paths['judge_sig'].exists() else pd.DataFrame()
    route_int = pd.read_csv(paths['route_int']) if paths['route_int'].exists() else pd.DataFrame()

    # Layout — three panels in a single row (full-width, paper-style)
    # Figure is wider+taller than before to give the bigger fonts room.
    fig = plt.figure(figsize=(16.5, 6.5))
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[1.05, 1.15, 0.95],
        wspace=0.32,
        left=0.05, right=0.98, top=0.88, bottom=0.18,
    )
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    draw_panel_A(ax_a, cat_sig)
    draw_panel_B(ax_b, judge_sig)
    draw_panel_C(ax_c, route_int)

    saved = save_figure(fig, 'fig7_experimental_results')
    plt.close(fig)
    print('\nSaved:')
    for p_ in saved:
        print(f'  {p_}')


if __name__ == '__main__':
    main()

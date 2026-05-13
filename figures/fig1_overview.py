"""
fig1_overview.py

Figure 1 — System overview.

Three stages, two rows. Evaluation lives in the paper text.

      ┌──────────┬──────────────────────────────────┐
      │ 1  Data  │ 2  Pretrain (I-JEPA on ViT-S)   │
      └──────────┴──────────────────────────────────┘
      ┌─────────────────────────────────────────────┐
      │ 3  Mini-JEPA fleet                          │
      └─────────────────────────────────────────────┘

Stage 1 — narrow column. CONUS map dominates as the visual entry point.
A compact list of imagery + label sources sits below.

Stage 2 — wide column. I-JEPA pretraining schema with masking front and
center. Critical: I-JEPA predicts EMBEDDINGS at masked positions, NOT
tokens. The Predictor takes context representations and produces target-
representation predictions in latent space. VICReg regularizes the
embedding distribution to prevent collapse.

Stage 3 — full-width bottom row. Five sensor-specialized Mini-JEPAs.
Each tower shows what makes it unique: which environmental variable it
predicts best (R² from real RF k-fold evaluation). The five "best-at"
labels carry the substantive case for sensor specialization.

The CONUS map slot is intentionally blank in this script. A polished
version is generated separately by fig1_corpus_map.py and overlaid on
the published figure.

Typography: geometric sans-serif (Space Grotesk → Inter → IBM Plex
Sans → fallback) for body, monospace (JetBrains Mono → IBM Plex Mono →
fallback) for technical specs. Mono picks up automatically when
fontfamily='monospace' is passed.

Run:
  python fig1_overview.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, Rectangle, FancyArrowPatch, Circle,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (
    save_figure, output_dir, dry_run_report,
    MODALITY_ORDER, MODALITY_COLOR, AE_COLOR,
)


# ═══════════════════════════════════════════════════════════════════════════
# Typography — futuristic geometric sans + mono for technical specs
# ═══════════════════════════════════════════════════════════════════════════
SANS_STACK = ['Space Grotesk', 'Inter', 'IBM Plex Sans',
              'Helvetica Neue', 'Arial', 'DejaVu Sans']
MONO_STACK = ['JetBrains Mono', 'IBM Plex Mono', 'SF Mono',
              'Consolas', 'DejaVu Sans Mono', 'monospace']


def apply_figure_rc():
    mpl.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': SANS_STACK,
        'font.monospace':  MONO_STACK,
        'font.weight':     400,
        'pdf.fonttype':    42,
        'ps.fonttype':     42,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Canvas
# ═══════════════════════════════════════════════════════════════════════════
FIG_W = 16.0
FIG_H = 14.0

WORK_X0, WORK_X1 = 0.4, FIG_W - 0.4
WORK_Y0, WORK_Y1 = 0.3, FIG_H - 0.3

ROW_GAP = 0.55
COL_GAP_TOP = 0.40

DATA_FRAC = 0.34   # slightly wider so CONUS map has room
ROW_H = (WORK_Y1 - WORK_Y0 - ROW_GAP) / 2

DATA_W = (WORK_X1 - WORK_X0 - COL_GAP_TOP) * DATA_FRAC
PRETRAIN_W = (WORK_X1 - WORK_X0 - COL_GAP_TOP) * (1 - DATA_FRAC)
FLEET_W = WORK_X1 - WORK_X0


def data_bounds():
    x0 = WORK_X0
    y1 = WORK_Y1
    return x0, y1 - ROW_H, x0 + DATA_W, y1


def pretrain_bounds():
    x0 = WORK_X0 + DATA_W + COL_GAP_TOP
    y1 = WORK_Y1
    return x0, y1 - ROW_H, x0 + PRETRAIN_W, y1


def fleet_bounds():
    x0 = WORK_X0
    y1 = WORK_Y1 - ROW_H - ROW_GAP
    return x0, y1 - ROW_H, x0 + FLEET_W, y1


HEADER_OFFSET = 0.60
CONTENT_TOP_OFFSET = 1.25
CONTENT_BOT_OFFSET = 0.30


# ═══════════════════════════════════════════════════════════════════════════
# Palette
# ═══════════════════════════════════════════════════════════════════════════
INK         = '#0E0E12'
INK_SOFT    = '#3C3C44'
INK_FAINT   = '#7A7A82'
EDGE        = '#CFCBC2'
EDGE_STRONG = '#1E1E24'
BG_TINT     = '#FAF9F6'
BG_ACCENT   = '#F1ECE0'
LAND_TONE   = '#EFEBE0'
WHITE       = '#FFFFFF'
ARROW_INK   = '#1E1E24'

# I-JEPA visualization
PATCH_VISIBLE_FILL = '#DDE6EF'
PATCH_VISIBLE_EDGE = '#5C7691'
PATCH_MASKED_FILL  = '#2A2A30'
PATCH_MASKED_EDGE  = '#1A1A1F'
PATCH_TARGET_FILL  = '#D9CFB6'

# Mini-JEPA fleet specializations — from real RF k-fold R² results
# (per-modality best variable + R²). See <modality>__hydrojepa_rf_r2.csv.
MODALITY_SHORT = {
    's2_optical':   'S2 Optical',
    'modis_lst':    'MODIS LST',
    's1_sar':       'S1 SAR',
    'topo_soil':    'Topo + Soil',
    's2_phenology': 'S2 Pheno',
}
MODALITY_INPUT = {
    's2_optical':   '10 bands  ·  optical',
    'modis_lst':    '2 bands  ·  thermal IR',
    's1_sar':       '2 bands  ·  SAR',
    'topo_soil':    '6 bands  ·  topo + soil',
    's2_phenology': '40 bands  ·  quarterly',
}
MODALITY_BEST_AT = {
    's2_optical':   ('Aridity',       'R²  0.73'),
    'modis_lst':    ('Temperature',   'R²  0.97'),
    's1_sar':       ('Precipitation', 'R²  0.62'),
    'topo_soil':    ('Elevation',     'R²  0.97'),
    's2_phenology': ('Precipitation', 'R²  0.81'),
}

# Imagery sources (Stage 1) — short noun labels, mono for the detail
IMAGERY_SOURCES = [
    ('Sentinel-2',       'optical  ·  10 bands',     '#1F77B4'),
    ('Sentinel-1',       'SAR  ·  VV + VH',          '#9467BD'),
    ('MODIS LST',        'thermal  ·  2 bands',      '#D62728'),
    ('Sentinel-2',       'quarterly  ·  40 bands',   '#2CA02C'),
    ('SRTM + SoilGrids', 'topo + soil  ·  6 bands',  '#8C564B'),
]


# ═══════════════════════════════════════════════════════════════════════════
# Drawing primitives
# ═══════════════════════════════════════════════════════════════════════════
def add_rounded_box(ax, x, y, w, h, *, facecolor=BG_TINT, edgecolor=EDGE,
                    linewidth=1.0, rounding=0.06, zorder=2):
    box = FancyBboxPatch(
        (x + rounding, y + rounding), w - 2 * rounding, h - 2 * rounding,
        boxstyle=f'round,pad={rounding},rounding_size={rounding}',
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def add_arrow(ax, x0, y0, x1, y1, *, color=ARROW_INK, lw=1.6, mutation=16,
              linestyle='-', zorder=5):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle='-|>', color=color, lw=lw,
        mutation_scale=mutation, linestyle=linestyle, zorder=zorder,
    ))


def add_stage_header(ax, x0, y1, num, title):
    badge_size = 0.55
    badge_x = x0 + 0.05
    badge_y = y1 - HEADER_OFFSET - badge_size / 2
    ax.add_patch(Rectangle(
        (badge_x, badge_y), badge_size, badge_size,
        facecolor=INK, edgecolor='none', zorder=10,
    ))
    ax.text(badge_x + badge_size / 2, badge_y + badge_size / 2 + 0.01,
            str(num),
            ha='center', va='center', color=WHITE, fontsize=22,
            fontweight='bold', zorder=11)
    ax.text(badge_x + badge_size + 0.24,
            y1 - HEADER_OFFSET, title,
            ha='left', va='center', color=INK,
            fontsize=26, fontweight='bold', zorder=10)


def draw_overlay_slot(ax, x0, y0, w, h, *, label='CONUS overlay'):
    """Empty bordered rectangle. The user replaces this in Inkscape/
    Keynote/PowerPoint with the polished CONUS map from
    fig1_corpus_map.py.
    """
    ax.add_patch(Rectangle(
        (x0, y0), w, h,
        facecolor=LAND_TONE, edgecolor=INK_FAINT, linewidth=1.0,
        linestyle=(0, (4, 3)), zorder=2,
    ))
    ax.text(x0 + w / 2, y0 + h / 2, label,
            ha='center', va='center', color=INK_FAINT,
            fontsize=13, fontfamily='monospace', style='italic',
            zorder=3)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — Data
# CONUS map is the visual entry point; sensor list is supporting detail.
# ═══════════════════════════════════════════════════════════════════════════
def draw_stage_1(ax):
    x0, y0, x1, y1 = data_bounds()
    content_top = y1 - CONTENT_TOP_OFFSET
    content_bot = y0 + CONTENT_BOT_OFFSET
    cx = (x0 + x1) / 2
    cw = x1 - x0

    # ── 1a. CONUS overlay slot (large — the visual anchor of Stage 1) ────
    map_h = 2.20
    map_w = cw - 0.40
    map_x = cx - map_w / 2
    map_y = content_top - map_h
    draw_overlay_slot(ax, map_x, map_y, map_w, map_h, label='CONUS overlay')

    # Compact spec line right below
    ann_y = map_y - 0.32
    ax.text(cx, ann_y,
            '9,704 patches  ·  30 m  ·  2022',
            ha='center', va='top', color=INK_SOFT,
            fontsize=12.5, fontfamily='monospace', zorder=4)

    # ── 1b. Imagery sources ──────────────────────────────────────────────
    # Sensor name on the left half, detail spec on the right half.
    # Takes advantage of horizontal whitespace; both fields read at
    # comfortable size.
    src_top = ann_y - 0.55
    n = len(IMAGERY_SOURCES)
    avail = src_top - content_bot - 0.20   # small bottom margin only
    tile_gap = 0.10
    tile_h_raw = (avail - (n - 1) * tile_gap) / n
    tile_h = max(min(tile_h_raw, 0.62), 0.48)
    tile_w = cw - 0.30
    tile_x = cx - tile_w / 2

    ax.text(tile_x, src_top + 0.10, 'IMAGERY SOURCES',
            ha='left', va='bottom', color=INK_FAINT,
            fontsize=12, fontweight='bold', zorder=5)

    for i, (label, detail, color) in enumerate(IMAGERY_SOURCES):
        y = src_top - (i + 1) * tile_h - i * tile_gap
        # Tile background
        ax.add_patch(Rectangle(
            (tile_x, y), tile_w, tile_h,
            facecolor=WHITE, edgecolor=EDGE, linewidth=0.9, zorder=2,
        ))
        # Modality color stripe on left edge
        stripe_w = 0.10
        ax.add_patch(Rectangle(
            (tile_x, y), stripe_w, tile_h,
            facecolor=color, edgecolor='none', zorder=3,
        ))
        # Sensor name in modality color — vertically centered, left half
        ax.text(tile_x + stripe_w + 0.20, y + tile_h / 2,
                label,
                ha='left', va='center', color=color,
                fontsize=14, fontweight='bold', zorder=4)
        # Detail spec in mono, right-aligned to give visual balance
        ax.text(tile_x + tile_w - 0.20, y + tile_h / 2,
                detail,
                ha='right', va='center', color=INK_SOFT,
                fontsize=11, fontfamily='monospace', zorder=4)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — Pretrain (I-JEPA on ViT-S)
#
# Critical: I-JEPA predicts target REPRESENTATIONS at masked positions,
# not pixels and not tokens. All labels in this stage make that explicit.
# ═══════════════════════════════════════════════════════════════════════════
def draw_stage_2(ax):
    x0, y0, x1, y1 = pretrain_bounds()
    content_top = y1 - CONTENT_TOP_OFFSET
    content_bot = y0 + CONTENT_BOT_OFFSET
    cw = x1 - x0
    cy = (content_top + content_bot) / 2

    # Sub-banner — JEPA framing in plain technical terms
    subhead_y = y1 - HEADER_OFFSET - 0.65
    ax.text(x0 + 0.85, subhead_y,
            'ViT-S backbone   ·   I-JEPA + VICReg   ·   latent-space objective',
            ha='left', va='center', color=INK_SOFT,
            fontsize=13.5, fontfamily='monospace', zorder=10)

    # ── 2a. Patchified input grid with masking pattern ───────────────────
    grid_n = 6
    grid_w = 1.70
    grid_h = grid_w
    grid_x = x0 + 0.40
    grid_y = cy - grid_h / 2
    cell = grid_w / grid_n

    visible_blocks = [
        (1, 1, 2, 2),
        (4, 3, 2, 2),
    ]
    visible_set = set()
    for c0, r0, nc, nr in visible_blocks:
        for r in range(r0, r0 + nr):
            for c in range(c0, c0 + nc):
                visible_set.add((r, c))

    for r in range(grid_n):
        for c in range(grid_n):
            cx_cell = grid_x + c * cell
            cy_cell = grid_y + r * cell
            if (r, c) in visible_set:
                fill, edge = PATCH_VISIBLE_FILL, PATCH_VISIBLE_EDGE
            else:
                fill, edge = PATCH_MASKED_FILL, PATCH_MASKED_EDGE
            ax.add_patch(Rectangle(
                (cx_cell, cy_cell), cell * 0.92, cell * 0.92,
                facecolor=fill, edgecolor=edge, linewidth=0.7, zorder=3,
            ))

    # Mask vs. context legend below the grid
    legend_y = grid_y - 0.30
    sw = 0.20
    legend_x = grid_x + grid_w / 2 - 1.40
    ax.add_patch(Rectangle(
        (legend_x, legend_y - sw / 2), sw, sw,
        facecolor=PATCH_VISIBLE_FILL, edgecolor=PATCH_VISIBLE_EDGE,
        linewidth=0.7, zorder=4,
    ))
    ax.text(legend_x + sw + 0.08, legend_y, 'context',
            ha='left', va='center', color=INK_SOFT,
            fontsize=11, zorder=4)
    ax.add_patch(Rectangle(
        (legend_x + 1.15, legend_y - sw / 2), sw, sw,
        facecolor=PATCH_MASKED_FILL, edgecolor=PATCH_MASKED_EDGE,
        linewidth=0.7, zorder=4,
    ))
    ax.text(legend_x + 1.15 + sw + 0.08, legend_y, 'masked',
            ha='left', va='center', color=INK_SOFT,
            fontsize=11, zorder=4)

    # ── 2b. Two encoder tracks — WIDER boxes, shorter labels ─────────────
    track_top = cy + 1.40
    track_bot = cy - 1.40

    enc_w = 2.00    # slightly narrower to save space
    enc_h = 1.10    # taller to fit 3-line label cleanly
    enc_x = grid_x + grid_w + 0.85   # reduced gap

    ctx_y = track_top - enc_h / 2
    add_rounded_box(ax, enc_x, ctx_y, enc_w, enc_h,
                     facecolor=BG_ACCENT, edgecolor=INK_SOFT, linewidth=1.2)
    ax.text(enc_x + enc_w / 2, ctx_y + enc_h * 0.72, 'Context encoder',
            ha='center', va='center', color=INK,
            fontsize=13, fontweight='bold', zorder=5)
    ax.text(enc_x + enc_w / 2, ctx_y + enc_h * 0.42,
            'ViT-S',
            ha='center', va='center', color=INK_SOFT,
            fontsize=10.5, fontfamily='monospace', zorder=5)
    ax.text(enc_x + enc_w / 2, ctx_y + enc_h * 0.18,
            'visible tokens only',
            ha='center', va='center', color=INK_SOFT,
            fontsize=10.5, fontfamily='monospace', zorder=5)

    tgt_y = track_bot - enc_h / 2
    add_rounded_box(ax, enc_x, tgt_y, enc_w, enc_h,
                     facecolor=BG_ACCENT, edgecolor=INK_SOFT, linewidth=1.2)
    ax.text(enc_x + enc_w / 2, tgt_y + enc_h * 0.72, 'Target encoder',
            ha='center', va='center', color=INK,
            fontsize=13, fontweight='bold', zorder=5)
    ax.text(enc_x + enc_w / 2, tgt_y + enc_h * 0.42,
            'ViT-S  ·  full grid',
            ha='center', va='center', color=INK_SOFT,
            fontsize=10.5, fontfamily='monospace', zorder=5)
    ax.text(enc_x + enc_w / 2, tgt_y + enc_h * 0.18,
            'EMA',
            ha='center', va='center', color=INK_SOFT,
            fontsize=10.5, fontfamily='monospace', zorder=5)

    # Patch grid → encoders
    add_arrow(ax, grid_x + grid_w + 0.04, cy + 0.30,
              enc_x - 0.04, ctx_y + enc_h / 2,
              color=ARROW_INK, lw=1.5, mutation=14)
    add_arrow(ax, grid_x + grid_w + 0.04, cy - 0.30,
              enc_x - 0.04, tgt_y + enc_h / 2,
              color=ARROW_INK, lw=1.5, mutation=14)

    # EMA dashed
    ema_x = enc_x + enc_w + 0.24
    add_arrow(ax, ema_x, ctx_y - 0.04, ema_x, tgt_y + enc_h + 0.04,
              color=INK_FAINT, lw=1.1, mutation=12, linestyle=(0, (3, 2)))
    ax.text(ema_x + 0.10, cy + 0.05, 'EMA',
            ha='left', va='center', color=INK_FAINT,
            fontsize=10.5, fontfamily='monospace', zorder=5)

    # ── 2c. Predictor (top track) — narrower, label makes role explicit ──
    pred_w = 1.55
    pred_h = 1.10
    pred_x = enc_x + enc_w + 0.55
    pred_y = track_top - pred_h / 2
    add_rounded_box(ax, pred_x, pred_y, pred_w, pred_h,
                     facecolor=BG_TINT, edgecolor=INK_SOFT, linewidth=1.1)
    ax.text(pred_x + pred_w / 2, pred_y + pred_h * 0.72, 'Predictor',
            ha='center', va='center', color=INK,
            fontsize=13, fontweight='bold', zorder=5)
    ax.text(pred_x + pred_w / 2, pred_y + pred_h * 0.42,
            'predicts',
            ha='center', va='center', color=INK_SOFT,
            fontsize=10.5, fontfamily='monospace', zorder=5)
    ax.text(pred_x + pred_w / 2, pred_y + pred_h * 0.18,
            'target reps',
            ha='center', va='center', color=INK_SOFT,
            fontsize=10.5, fontfamily='monospace', zorder=5)

    add_arrow(ax, enc_x + enc_w + 0.04, pred_y + pred_h / 2,
              pred_x - 0.04, pred_y + pred_h / 2,
              color=ARROW_INK, lw=1.5, mutation=14)

    # ── 2d. Predicted vs target REPRESENTATIONS ──────────────────────────
    emb_x = pred_x + pred_w + 0.45
    n_bars = 4
    bar_w = 0.20
    bar_gap = 0.07
    bar_h = 0.60

    pred_emb_y = track_top - bar_h / 2
    tgt_emb_y = track_bot - bar_h / 2

    for i in range(n_bars):
        bx = emb_x + i * (bar_w + bar_gap)
        ax.add_patch(Rectangle(
            (bx, pred_emb_y), bar_w, bar_h,
            facecolor='#E5E5E5', edgecolor=INK_SOFT, linewidth=0.7, zorder=4,
        ))
        ax.add_patch(Rectangle(
            (bx, tgt_emb_y), bar_w, bar_h,
            facecolor=PATCH_TARGET_FILL, edgecolor=INK_SOFT,
            linewidth=0.7, zorder=4,
        ))

    total_bars_w = n_bars * bar_w + (n_bars - 1) * bar_gap
    # Compact two-line labels — short enough to fit, clear about what's shown
    ax.text(emb_x + total_bars_w / 2, pred_emb_y + bar_h + 0.16,
            'predicted reps',
            ha='center', va='bottom', color=INK_SOFT,
            fontsize=11, zorder=5)
    ax.text(emb_x + total_bars_w / 2, tgt_emb_y - 0.16,
            'target reps',
            ha='center', va='top', color=INK_SOFT,
            fontsize=11, zorder=5)
    ax.text(emb_x + total_bars_w / 2, tgt_emb_y - 0.42,
            '@ masked positions',
            ha='center', va='top', color=INK_FAINT,
            fontsize=9.5, fontfamily='monospace', zorder=5)

    add_arrow(ax, pred_x + pred_w + 0.04, pred_y + pred_h / 2,
              emb_x - 0.04, pred_emb_y + bar_h / 2,
              color=ARROW_INK, lw=1.5, mutation=14)
    add_arrow(ax, enc_x + enc_w + 0.04, tgt_y + enc_h / 2,
              emb_x - 0.04, tgt_emb_y + bar_h / 2,
              color=ARROW_INK, lw=1.5, mutation=14)

    # ── 2e. Loss bracket ─────────────────────────────────────────────────
    loss_x = emb_x + total_bars_w + 0.20
    bracket_y0 = tgt_emb_y
    bracket_y1 = pred_emb_y + bar_h
    tip = 0.14
    ax.plot([loss_x, loss_x], [bracket_y0, bracket_y1],
            color=INK, lw=1.5, zorder=6)
    ax.plot([loss_x - tip, loss_x], [bracket_y0, bracket_y0],
            color=INK, lw=1.5, zorder=6)
    ax.plot([loss_x - tip, loss_x], [bracket_y1, bracket_y1],
            color=INK, lw=1.5, zorder=6)
    bracket_mid = (bracket_y0 + bracket_y1) / 2
    ax.plot([loss_x, loss_x + tip], [bracket_mid, bracket_mid],
            color=INK, lw=1.5, zorder=6)
    ax.text(loss_x + tip + 0.10, bracket_mid + 0.20,
            'I-JEPA',
            ha='left', va='center', color=INK,
            fontsize=13.5, fontweight='bold', zorder=6)
    ax.text(loss_x + tip + 0.10, bracket_mid - 0.20,
            '+ VICReg',
            ha='left', va='center', color=INK_SOFT,
            fontsize=12, zorder=6)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 — Mini-JEPA fleet
#
# What makes the fleet substantive: each Mini-JEPA EXCELS at a different
# environmental variable. The "BEST AT" row carries the case that
# specialization actually pays off.
# ═══════════════════════════════════════════════════════════════════════════
def draw_stage_3(ax):
    x0, y0, x1, y1 = fleet_bounds()
    content_top = y1 - CONTENT_TOP_OFFSET
    content_bot = y0 + CONTENT_BOT_OFFSET
    cw = x1 - x0

    # Sub-banner — short noun phrase, no narration
    subhead_y = y1 - HEADER_OFFSET - 0.65
    ax.text(x0 + 0.85, subhead_y,
            'five sensor-specialized encoders   ·   shared ViT-S / 22M / 64-d   ·   shared I-JEPA recipe',
            ha='left', va='center', color=INK_SOFT,
            fontsize=13.5, fontfamily='monospace', zorder=10)

    # Five towers spanning the row width
    n = 5
    tower_w = 2.05
    tower_total_w = cw - 1.20
    tower_gap = (tower_total_w - n * tower_w) / (n - 1)
    towers_left = x0 + (cw - tower_total_w) / 2

    # Each tower has three sections (top to bottom):
    #   1. Sensor input header (modality color + sensor name + input spec)
    #   2. ViT-S encoder box with a small "I-JEPA pretrained" icon
    #   3. "BEST AT" specialization block (variable + R²)

    sensor_chip_h = 0.95
    encoder_box_h = 1.10
    output_dot_r = 0.20
    best_block_h = 0.95

    arrow_gap_top = 0.20
    arrow_gap_bot = 0.20
    label_gap = 0.20
    dim_label_h = 0.24

    tower_h = (sensor_chip_h + arrow_gap_top + encoder_box_h + arrow_gap_bot
               + dim_label_h + label_gap + 2 * output_dot_r
               + 0.30 + best_block_h)

    avail = content_top - content_bot - 0.30
    tower_top = content_top - 0.15
    if tower_h < avail:
        slack = avail - tower_h
        tower_top -= slack * 0.20

    for i, m in enumerate(MODALITY_ORDER):
        tx = towers_left + i * (tower_w + tower_gap)
        color = MODALITY_COLOR[m]

        # ── Section 1: sensor input header ───────────────────────────────
        chip_y = tower_top - sensor_chip_h
        ax.add_patch(Rectangle(
            (tx, chip_y), tower_w, sensor_chip_h,
            facecolor=WHITE, edgecolor=EDGE, linewidth=1.0, zorder=3,
        ))
        stripe_w = 0.10
        ax.add_patch(Rectangle(
            (tx, chip_y), stripe_w, sensor_chip_h,
            facecolor=color, edgecolor='none', zorder=4,
        ))
        ax.text(tx + (tower_w + stripe_w) / 2,
                chip_y + sensor_chip_h * 0.65,
                MODALITY_SHORT[m],
                ha='center', va='center', color=color,
                fontsize=14, fontweight='bold', zorder=5)
        ax.text(tx + (tower_w + stripe_w) / 2,
                chip_y + sensor_chip_h * 0.28,
                MODALITY_INPUT[m],
                ha='center', va='center', color=INK_SOFT,
                fontsize=10, fontfamily='monospace', zorder=5)

        # ── Section 2: ViT-S encoder box with I-JEPA masking icon ────────
        enc_top_y = chip_y - arrow_gap_top
        add_arrow(ax, tx + tower_w / 2, chip_y - 0.02,
                  tx + tower_w / 2, enc_top_y + 0.04,
                  color=ARROW_INK, lw=1.4, mutation=12)

        enc_y = enc_top_y - encoder_box_h
        add_rounded_box(ax, tx, enc_y, tower_w, encoder_box_h,
                         facecolor=BG_ACCENT, edgecolor=INK_SOFT,
                         linewidth=1.0, rounding=0.06)

        # Small 3×3 masking icon at left (echoes Stage 2's I-JEPA pattern)
        icon_n = 3
        icon_size = 0.34
        icon_cell = icon_size / icon_n
        icon_x = tx + 0.14
        icon_y = enc_y + (encoder_box_h - icon_size) / 2
        # Mask pattern: visible top-right, masked elsewhere
        visible_cells = {(0, 2), (1, 2), (1, 1)}
        for r in range(icon_n):
            for c in range(icon_n):
                cx_cell = icon_x + c * icon_cell
                cy_cell = icon_y + r * icon_cell
                if (r, c) in visible_cells:
                    fill = PATCH_VISIBLE_FILL
                    edge = PATCH_VISIBLE_EDGE
                else:
                    fill = PATCH_MASKED_FILL
                    edge = PATCH_MASKED_EDGE
                ax.add_patch(Rectangle(
                    (cx_cell, cy_cell), icon_cell * 0.85, icon_cell * 0.85,
                    facecolor=fill, edgecolor=edge, linewidth=0.4, zorder=5,
                ))

        # Encoder label to the right of the icon — pulled in and split across
        # two lines for "I-JEPA pretrained" so both lines stay inside the box
        label_x = icon_x + icon_size + 0.12
        ax.text(label_x, enc_y + encoder_box_h * 0.72,
                'ViT-S encoder',
                ha='left', va='center', color=INK,
                fontsize=11, fontweight='bold', zorder=5)
        ax.text(label_x, enc_y + encoder_box_h * 0.42,
                'I-JEPA',
                ha='left', va='center', color=INK_SOFT,
                fontsize=9.5, fontfamily='monospace', zorder=5)
        ax.text(label_x, enc_y + encoder_box_h * 0.20,
                'pretrained',
                ha='left', va='center', color=INK_SOFT,
                fontsize=9.5, fontfamily='monospace', zorder=5)

        # Arrow down to output dot
        add_arrow(ax, tx + tower_w / 2, enc_y - 0.02,
                  tx + tower_w / 2, enc_y - arrow_gap_bot + 0.04,
                  color=ARROW_INK, lw=1.4, mutation=12)

        # 64-d label
        ax.text(tx + tower_w / 2,
                enc_y - arrow_gap_bot - dim_label_h / 2,
                '64-d',
                ha='center', va='center', color=INK_SOFT,
                fontsize=11.5, fontfamily='monospace', zorder=5)

        # Output dot
        dot_y = (enc_y - arrow_gap_bot - dim_label_h - label_gap
                 - output_dot_r)
        ax.add_patch(Circle(
            (tx + tower_w / 2, dot_y), output_dot_r,
            facecolor=color, edgecolor='none', zorder=5,
        ))

        # ── Section 3: BEST AT specialization block ──────────────────────
        best_top = dot_y - output_dot_r - 0.30
        best_y_bot = best_top - best_block_h
        ax.add_patch(Rectangle(
            (tx, best_y_bot), tower_w, best_block_h,
            facecolor=BG_TINT, edgecolor=EDGE, linewidth=1.0, zorder=3,
        ))
        ax.add_patch(Rectangle(
            (tx, best_top - 0.06), tower_w, 0.06,
            facecolor=color, edgecolor='none', zorder=4,
        ))
        ax.text(tx + tower_w / 2, best_top - 0.22, 'BEST AT',
                ha='center', va='center', color=INK_FAINT,
                fontsize=10, fontweight='bold', zorder=5)
        var_name, r2_str = MODALITY_BEST_AT[m]
        ax.text(tx + tower_w / 2, best_top - 0.50, var_name,
                ha='center', va='center', color=INK,
                fontsize=13, fontweight='bold', zorder=5)
        ax.text(tx + tower_w / 2, best_top - 0.76, r2_str,
                ha='center', va='center', color=color,
                fontsize=12, fontfamily='monospace', fontweight='bold',
                zorder=5)


# ═══════════════════════════════════════════════════════════════════════════
# Inter-cell arrows
# ═══════════════════════════════════════════════════════════════════════════
def draw_intercell_arrows(ax):
    # Data → Pretrain (horizontal, top row gap)
    d_x0, d_y0, d_x1, d_y1 = data_bounds()
    p_x0, _, _, _ = pretrain_bounds()
    sy = (d_y0 + d_y1) / 2
    add_arrow(ax, d_x1 + 0.03, sy, p_x0 - 0.05, sy,
              color=ARROW_INK, lw=2.4, mutation=22)

    # Pretrain → Fleet (vertical, between rows, centered)
    f_x0, _, f_x1, f_y1 = fleet_bounds()
    mid_x = (f_x0 + f_x1) / 2
    add_arrow(ax, mid_x, d_y0 - 0.03, mid_x, f_y1 + 0.05,
              color=ARROW_INK, lw=2.4, mutation=22)


# ═══════════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════════
def build_figure():
    apply_figure_rc()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.add_patch(Rectangle((0, 0), FIG_W, FIG_H,
                            facecolor=WHITE, edgecolor='none', zorder=0))

    d_x0, _, _, d_y1 = data_bounds()
    p_x0, _, _, p_y1 = pretrain_bounds()
    f_x0, _, _, f_y1 = fleet_bounds()
    add_stage_header(ax, d_x0, d_y1, 1, 'Data')
    add_stage_header(ax, p_x0, p_y1, 2, 'Pretrain')
    add_stage_header(ax, f_x0, f_y1, 3, 'Mini-JEPA fleet')

    draw_stage_1(ax)
    draw_stage_2(ax)
    draw_stage_3(ax)
    draw_intercell_arrows(ax)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    if args.dry_run:
        dry_run_report('Figure 1 — overview (no external data)',
                        {'(no data files needed)': Path('/')})
        return

    fig = build_figure()
    saved = save_figure(fig, 'fig1_overview')
    plt.close(fig)
    print('\nSaved:')
    for p_ in saved:
        print(f'  {p_}')


if __name__ == '__main__':
    main()

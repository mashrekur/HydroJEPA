"""
fig6_agent_architecture.py

Figure 5 — System architecture for the Mini-JEPA dual-RAG agent.

Single-page vertical flow, full width. Top-to-bottom stages, straight-line
arrows only. No worked example (per direction).

Layout:

  STAGE 1 — Query encoding
    Question + lat/lon → patch-context encoder → query embedding

  STAGE 2 — Routing (with meta-summary input)
    Router LLM receives:
      - query embedding
      - dimension dictionaries (per Mini-JEPA: which dims encode what)
      - geometric meta-summaries (per Mini-JEPA: PR, intrinsic dim, etc.)
    Router emits a structured tool-call plan (visible as monospaced code).

  STAGE 3 — Parallel retrieval (FAISS)
    Five Mini-JEPA FAISS indexes (one per modality) + one AlphaEarth FAISS.
    Selected Mini-JEPA tools run; unselected don't. AE always runs.

  STAGE 4 — Synthesis
    Synthesis LLM consumes retrieved patches (with modality provenance)
    and produces an answer with citations.

Output: hand-authored SVG (vector, paper-ready). PNG export via cairosvg
if available.

Run:
  python fig6_agent_architecture.py --dry-run
  python fig6_agent_architecture.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (
    project_root, output_dir, dry_run_report,
    MODALITY_ORDER, MODALITY_LABEL, MODALITY_COLOR, AE_COLOR,
)


# ---------------------------------------------------------------------------
# Canvas — wide and full-page
# ---------------------------------------------------------------------------
CANVAS_W = 1500
CANVAS_H = 1100

# Stage Y-anchors (centers)
STAGE1_Y = 100      # query encoding
STAGE2_Y = 290      # routing
STAGE3_Y = 620      # retrieval
STAGE4_Y = 870      # synthesis
STAGE5_Y = 1040     # answer

# Margins
LEFT_MARGIN  = 60
RIGHT_MARGIN = CANVAS_W - 60

# Typography
FONT_STACK = "'Inter', 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif"
FONT_MONO  = "'JetBrains Mono', 'IBM Plex Mono', 'SF Mono', Consolas, monospace"

# Palette — neutrals + one accent + modality colors only where modalities live
INK          = '#1A1A1A'
INK_SOFT     = '#555555'
INK_FAINT    = '#888888'
EDGE         = '#CCCCCC'
EDGE_SOFT    = '#DDDDDD'
BG_PANEL     = '#F8F7F4'
BG_DATA      = '#F1EEE8'         # for "data source" blocks (dictionaries)
BG_CODE      = '#F4F1EC'         # for the inline tool-call snippet
ACCENT       = '#1A1A1A'

STROKE_FINE = 1.0
STROKE_MED  = 1.6
STROKE_BOLD = 2.2


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------
def svg_open(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'style="font-family: {FONT_STACK}; background: white;">\n')

def svg_close():
    return '</svg>\n'

def defs():
    return f'''<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="{INK}"/>
  </marker>
  <marker id="arrow-soft" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="{INK_SOFT}"/>
  </marker>
  <marker id="arrow-faint" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="{INK_FAINT}"/>
  </marker>
</defs>
'''

def esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))

def rect(x, y, w, h, *, fill='white', stroke=EDGE, sw=1.0, rx=4):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="{rx}" ry="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>\n')

def text(x, y, s, *, size=12, color=INK, weight=400, anchor='start',
         family=None, italic=False):
    fam = f' font-family="{family}"' if family else ''
    style = f'font-style: italic;' if italic else ''
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}"{fam} '
            f'style="{style}">{esc(s)}</text>\n')

def hline(x1, x2, y, *, stroke=INK, sw=STROKE_MED, dash=None, marker=None):
    """Strictly horizontal line."""
    d = f' stroke-dasharray="{dash}"' if dash else ''
    m = f' marker-end="url(#{marker})"' if marker else ''
    return (f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}{m}/>\n')

def vline(x, y1, y2, *, stroke=INK, sw=STROKE_MED, dash=None, marker=None):
    """Strictly vertical line."""
    d = f' stroke-dasharray="{dash}"' if dash else ''
    m = f' marker-end="url(#{marker})"' if marker else ''
    return (f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}{m}/>\n')


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def box_with_header(cx, cy, w, h, header, title, lines, *,
                     header_color=INK_FAINT, border=INK, border_sw=1.4,
                     fill='white', accent_left=None):
    """A standard box: small uppercase header, large title, content lines.

    If accent_left is set, draws a colored stripe on the left edge (used
    for modality boxes).
    """
    x = cx - w/2
    y = cy - h/2
    out = [rect(x, y, w, h, fill=fill, stroke=border, sw=border_sw, rx=5)]
    if accent_left:
        out.append(rect(x, y, 5, h, fill=accent_left, stroke='none', rx=0))
        text_x = x + 18
    else:
        text_x = x + 18

    out.append(text(text_x, y + 22, header,
                     size=10, color=header_color, weight=600))
    out.append(text(text_x, y + 44, title,
                     size=14, color=INK, weight=600))
    line_y = y + 64
    for line in lines:
        out.append(text(text_x, line_y, line,
                         size=10.5, color=INK_SOFT, weight=400))
        line_y += 15
    return ''.join(out)


def stage_label(y, num, title, subtitle):
    """Stage banner on the left edge: number, title, subtitle."""
    return (text(LEFT_MARGIN, y - 4,
                  f'STAGE {num}',
                  size=10, color=INK_FAINT, weight=600)
            + text(LEFT_MARGIN, y + 14, title,
                    size=15, color=INK, weight=600)
            + text(LEFT_MARGIN, y + 33, subtitle,
                    size=10.5, color=INK_SOFT, weight=400, italic=True))


# ---------------------------------------------------------------------------
# Diagram
# ---------------------------------------------------------------------------
def draw_diagram():
    svg = []

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 1 — Query encoding
    # ═══════════════════════════════════════════════════════════════════════
    svg.append(stage_label(STAGE1_Y, 1, 'Query',
                            'natural-language question + geographic context'))

    # Two side-by-side blocks: question + encoder
    Q_CX = 520
    Q_W, Q_H = 320, 90
    svg.append(box_with_header(
        Q_CX, STAGE1_Y + 16, Q_W, Q_H,
        header='USER INPUT',
        title='Question + lat/lon',
        lines=['e.g. "Confirm irrigation at 43.4°N, 93.5°W',
               'under summer cloud cover"'],
        fill=BG_PANEL,
    ))

    E_CX = 920
    E_W, E_H = 280, 90
    svg.append(box_with_header(
        E_CX, STAGE1_Y + 16, E_W, E_H,
        header='ENCODE',
        title='Patch-context embedding',
        lines=['64-d vector against patch corpus',
               '(used as FAISS query at Stage 3)'],
    ))

    # Arrow Q → encoder
    svg.append(hline(Q_CX + Q_W/2 + 4, E_CX - E_W/2 - 4, STAGE1_Y + 16,
                      stroke=INK, sw=STROKE_MED, marker='arrow'))

    # Down-arrow from encoder to next stage (straight vertical line)
    svg.append(vline(E_CX, STAGE1_Y + 16 + E_H/2 + 4,
                      STAGE2_Y - 80, stroke=INK, sw=STROKE_MED,
                      marker='arrow'))

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2 — Routing
    # ═══════════════════════════════════════════════════════════════════════
    svg.append(stage_label(STAGE2_Y, 2, 'Route',
                            'Router LLM consults Mini-JEPA meta-summaries '
                            'and emits a tool plan'))

    # Router box (center) — bigger than Stage 1 boxes because it's the hub
    R_CX = E_CX
    R_Y  = STAGE2_Y + 30
    R_W, R_H = 320, 130
    svg.append(rect(R_CX - R_W/2, R_Y - R_H/2, R_W, R_H,
                     fill='white', stroke=INK, sw=1.8, rx=6))
    svg.append(text(R_CX, R_Y - R_H/2 + 24, 'ROUTER',
                     size=10, color=INK_FAINT, weight=600, anchor='middle'))
    svg.append(text(R_CX, R_Y - R_H/2 + 48, 'Claude LLM',
                     size=15, color=INK, weight=600, anchor='middle'))
    svg.append(text(R_CX, R_Y - R_H/2 + 70, 'Reads meta-summaries,',
                     size=11, color=INK_SOFT, weight=400, anchor='middle'))
    svg.append(text(R_CX, R_Y - R_H/2 + 85, 'reasons about which sensor',
                     size=11, color=INK_SOFT, weight=400, anchor='middle'))
    svg.append(text(R_CX, R_Y - R_H/2 + 100, 'physics fits the query,',
                     size=11, color=INK_SOFT, weight=400, anchor='middle'))
    svg.append(text(R_CX, R_Y - R_H/2 + 115, 'emits tool-call plan.',
                     size=11, color=INK_SOFT, weight=400, anchor='middle'))

    # ── Meta-summary block on the left (data source feeding the router) ───
    META_CX = 340
    META_Y  = R_Y
    META_W, META_H = 320, 200
    svg.append(rect(META_CX - META_W/2, META_Y - META_H/2, META_W, META_H,
                     fill=BG_DATA, stroke=INK_FAINT, sw=1.0, rx=5))
    svg.append(text(META_CX, META_Y - META_H/2 + 24, 'META-SUMMARIES',
                     size=10, color=INK_FAINT, weight=600, anchor='middle'))
    svg.append(text(META_CX, META_Y - META_H/2 + 46,
                     'Per-Mini-JEPA reference cards',
                     size=12.5, color=INK, weight=600, anchor='middle'))

    # Bullet entries (one row each)
    meta_lines = [
        ('Dimension dictionary',
         'which dims encode what env variable'),
        ('Geometric profile',
         'global PR, intrinsic dim, local PR'),
        ('Sensor physics summary',
         'what the model can and cannot see'),
        ('Standalone R² table',
         'per env-variable predictive skill'),
    ]
    by = META_Y - META_H/2 + 72
    for head, sub in meta_lines:
        svg.append(rect(META_CX - META_W/2 + 14, by, 6, 6,
                         fill=INK, stroke='none', rx=0))
        svg.append(text(META_CX - META_W/2 + 28, by + 6, head,
                         size=11, color=INK, weight=600))
        svg.append(text(META_CX - META_W/2 + 28, by + 21, sub,
                         size=10, color=INK_SOFT, weight=400, italic=True))
        by += 30

    # Meta → Router arrow (straight horizontal)
    svg.append(hline(META_CX + META_W/2 + 4, R_CX - R_W/2 - 4, R_Y,
                      stroke=INK_SOFT, sw=STROKE_MED, marker='arrow-soft'))

    # ── Tool-call output (right of router) — the agentic layer made literal ─
    TC_CX = 1240
    TC_Y  = R_Y
    TC_W, TC_H = 380, 200
    svg.append(rect(TC_CX - TC_W/2, TC_Y - TC_H/2, TC_W, TC_H,
                     fill=BG_CODE, stroke=INK_FAINT, sw=1.0, rx=5))
    svg.append(text(TC_CX, TC_Y - TC_H/2 + 24, 'TOOL-CALL PLAN',
                     size=10, color=INK_FAINT, weight=600, anchor='middle'))
    svg.append(text(TC_CX, TC_Y - TC_H/2 + 46,
                     'Structured agentic output',
                     size=12.5, color=INK, weight=600, anchor='middle'))

    # Code-like tool calls (monospaced)
    code_lines = [
        'retrieve_s1_sar(q, k=5)',
        'retrieve_s2_phenology(q, k=5)',
        'retrieve_modis_lst(q, k=5)',
        'retrieve_alphaearth(q, k=5)',
    ]
    code_x = TC_CX - TC_W/2 + 24
    code_y = TC_Y - TC_H/2 + 74
    for cl in code_lines:
        svg.append(text(code_x, code_y, cl,
                         size=11.5, color=INK, weight=500,
                         family=FONT_MONO))
        code_y += 22
    # Comment under the code
    svg.append(text(code_x, code_y + 4,
                     '↳ executes in parallel against FAISS',
                     size=10, color=INK_FAINT, weight=400, italic=True))

    # Router → Tool plan arrow
    svg.append(hline(R_CX + R_W/2 + 4, TC_CX - TC_W/2 - 4, R_Y,
                      stroke=INK, sw=STROKE_MED, marker='arrow'))

    # Stage 2 → Stage 3 down-arrow (from below tool-plan box)
    svg.append(vline(TC_CX, TC_Y + TC_H/2 + 4, STAGE3_Y - 110,
                      stroke=INK, sw=STROKE_MED, marker='arrow'))

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 3 — Parallel retrieval (FAISS)
    # ═══════════════════════════════════════════════════════════════════════
    svg.append(stage_label(STAGE3_Y, 3, 'Retrieve',
                            'FAISS k-NN against per-modality embedding indexes; '
                            'AlphaEarth always runs in parallel'))

    # Six FAISS index boxes in a row: 5 Mini-JEPAs + AE
    # Total width fitted to canvas
    FAISS_Y = STAGE3_Y + 90
    FAISS_W = 165
    FAISS_H = 110
    # 5 Mini-JEPAs on the left, gap, then AE on the right
    fleet_total_w = 5 * FAISS_W + 4 * 16   # 4 gaps of 16 px between MJP boxes
    fleet_start_x = LEFT_MARGIN + 160
    ae_start_x    = fleet_start_x + fleet_total_w + 80  # 80 px gap before AE

    # Mini-JEPA FAISS indexes
    faiss_centers = {}
    for i, m in enumerate(MODALITY_ORDER):
        x = fleet_start_x + i * (FAISS_W + 16)
        svg.append(rect(x, FAISS_Y, FAISS_W, FAISS_H,
                         fill='white', stroke=EDGE, sw=1.0, rx=5))
        svg.append(rect(x, FAISS_Y, 4, FAISS_H,
                         fill=MODALITY_COLOR[m], stroke='none', rx=0))
        svg.append(text(x + 14, FAISS_Y + 22,
                         MODALITY_LABEL[m],
                         size=11, color=MODALITY_COLOR[m], weight=600))
        svg.append(text(x + 14, FAISS_Y + 42, 'FAISS index',
                         size=9.5, color=INK_FAINT, weight=600))
        svg.append(text(x + 14, FAISS_Y + 62, '9,704 patches',
                         size=10, color=INK_SOFT, weight=400))
        svg.append(text(x + 14, FAISS_Y + 78, '64-d vectors',
                         size=10, color=INK_SOFT, weight=400))
        svg.append(text(x + 14, FAISS_Y + 95, 'k-NN retrieval',
                         size=9.5, color=INK_FAINT, weight=400, italic=True))
        faiss_centers[m] = (x + FAISS_W/2, FAISS_Y)

    # AlphaEarth FAISS (peer, visually separated)
    svg.append(rect(ae_start_x, FAISS_Y, FAISS_W, FAISS_H,
                     fill='white', stroke=INK_SOFT, sw=1.2, rx=5))
    svg.append(rect(ae_start_x, FAISS_Y, 4, FAISS_H,
                     fill=AE_COLOR, stroke='none', rx=0))
    svg.append(text(ae_start_x + 14, FAISS_Y + 22,
                     'AlphaEarth',
                     size=11, color=INK, weight=600))
    svg.append(text(ae_start_x + 14, FAISS_Y + 42, 'FAISS index',
                     size=9.5, color=INK_FAINT, weight=600))
    svg.append(text(ae_start_x + 14, FAISS_Y + 62, '9,704 patches',
                     size=10, color=INK_SOFT, weight=400))
    svg.append(text(ae_start_x + 14, FAISS_Y + 78, '64-d vectors',
                     size=10, color=INK_SOFT, weight=400))
    svg.append(text(ae_start_x + 14, FAISS_Y + 95, 'always retrieved',
                     size=9.5, color=INK_FAINT, weight=500, italic=True))
    ae_center_x = ae_start_x + FAISS_W/2

    # Tool plan → FAISS row: a single distribution bar
    # Tool plan box is centered around TC_CX in stage 2. Draw a horizontal
    # bus at y = FAISS_Y - 40 spanning from above first MJP to above AE,
    # then vertical drop arrows down into each FAISS box.
    BUS_Y = FAISS_Y - 30

    # The vertical line from tool plan landed at STAGE3_Y - 110 above; now
    # we need to extend it down to the bus_y and then spread horizontally
    svg.append(vline(TC_CX, STAGE3_Y - 110, BUS_Y,
                      stroke=INK, sw=STROKE_MED))

    # Horizontal bus from the leftmost FAISS box center to the AE center
    leftmost_cx = faiss_centers[MODALITY_ORDER[0]][0]
    rightmost_cx = ae_center_x
    svg.append(hline(leftmost_cx, max(rightmost_cx, TC_CX), BUS_Y,
                      stroke=INK, sw=STROKE_MED))

    # Drop arrows from bus into each box
    for m in MODALITY_ORDER:
        cx, _ = faiss_centers[m]
        svg.append(vline(cx, BUS_Y, FAISS_Y - 4,
                          stroke=INK, sw=STROKE_MED, marker='arrow'))
    svg.append(vline(ae_center_x, BUS_Y, FAISS_Y - 4,
                      stroke=INK, sw=STROKE_MED, marker='arrow'))

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 4 — Synthesis
    # ═══════════════════════════════════════════════════════════════════════
    svg.append(stage_label(STAGE4_Y, 4, 'Synthesize',
                            'Synthesis LLM receives retrieved patches with '
                            'modality provenance, produces answer'))

    # Retrieval-gather bus + synthesis box
    # Each FAISS box drops a short vertical line into another bus,
    # then a single trunk descends into the synthesis box.
    GATHER_Y = STAGE4_Y - 30
    for m in MODALITY_ORDER:
        cx, _ = faiss_centers[m]
        svg.append(vline(cx, FAISS_Y + FAISS_H + 4, GATHER_Y,
                          stroke=INK_SOFT, sw=STROKE_MED))
    svg.append(vline(ae_center_x, FAISS_Y + FAISS_H + 4, GATHER_Y,
                      stroke=INK_SOFT, sw=STROKE_MED))

    # Gather bus (horizontal)
    svg.append(hline(leftmost_cx, max(rightmost_cx, TC_CX), GATHER_Y,
                      stroke=INK_SOFT, sw=STROKE_MED))

    # Synthesis box (centered like the router)
    SY_CX = E_CX
    SY_Y  = STAGE4_Y + 70
    SY_W, SY_H = 380, 110
    svg.append(rect(SY_CX - SY_W/2, SY_Y - SY_H/2, SY_W, SY_H,
                     fill='white', stroke=INK, sw=1.8, rx=6))
    svg.append(text(SY_CX, SY_Y - SY_H/2 + 24, 'SYNTHESIS',
                     size=10, color=INK_FAINT, weight=600, anchor='middle'))
    svg.append(text(SY_CX, SY_Y - SY_H/2 + 48, 'Claude LLM',
                     size=15, color=INK, weight=600, anchor='middle'))
    svg.append(text(SY_CX, SY_Y - SY_H/2 + 70,
                     'Receives retrieved patches with',
                     size=11, color=INK_SOFT, weight=400, anchor='middle'))
    svg.append(text(SY_CX, SY_Y - SY_H/2 + 85,
                     'modality provenance tags +',
                     size=11, color=INK_SOFT, weight=400, anchor='middle'))
    svg.append(text(SY_CX, SY_Y - SY_H/2 + 100,
                     'env-variable metadata',
                     size=11, color=INK_SOFT, weight=400, anchor='middle'))

    # Bus → synthesis: vertical down-arrow into synthesis from above its center
    svg.append(vline(SY_CX, GATHER_Y, SY_Y - SY_H/2 - 4,
                      stroke=INK, sw=STROKE_MED, marker='arrow'))

    # Synthesis → Answer
    AN_CX = SY_CX
    AN_Y  = STAGE5_Y
    AN_W, AN_H = 180, 40
    svg.append(rect(AN_CX - AN_W/2, AN_Y - AN_H/2, AN_W, AN_H,
                     fill=BG_PANEL, stroke=EDGE, sw=1.0, rx=20))
    svg.append(text(AN_CX, AN_Y + 5, 'Answer',
                     size=14, color=INK, weight=600, anchor='middle'))
    svg.append(vline(SY_CX, SY_Y + SY_H/2 + 4, AN_Y - AN_H/2 - 4,
                      stroke=INK, sw=STROKE_MED, marker='arrow'))

    return ''.join(svg)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    if args.dry_run:
        dry_run_report('Figure 5 — agent architecture (no external data)',
                        {'(no data files needed)': Path('/')})
        return

    body = [svg_open(CANVAS_W, CANVAS_H), defs(), draw_diagram(), svg_close()]
    svg = ''.join(body)

    out_dir = output_dir()
    svg_path = out_dir / 'fig6_agent_architecture.svg'
    svg_path.write_text(svg)
    print(f'Saved SVG: {svg_path}')

    try:
        import cairosvg
        png_path = out_dir / 'fig6_agent_architecture.png'
        cairosvg.svg2png(bytestring=svg.encode('utf-8'),
                          write_to=str(png_path),
                          output_width=CANVAS_W * 2)
        print(f'Saved PNG: {png_path}')
    except ImportError:
        print('cairosvg not installed; SVG only. (pip install cairosvg)')
    except Exception as e:
        print(f'cairosvg failed: {e}')


if __name__ == '__main__':
    main()

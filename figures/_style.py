"""
figures/_style.py
Shared visual style for the HydroJEPA / Mini-JEPAs paper figures.

One source of truth for:
  - modality color palette
  - typography
  - matplotlib rcParams
  - CONUS map base layer
  - small panel helpers (annotation boxes, effect-size labels)

Every figure script imports from here. Lock once; reuse everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Modality colors  (locked — must match across all 7 figures and the paper text)
# ---------------------------------------------------------------------------
MODALITY_ORDER = ['s2_optical', 'modis_lst', 's1_sar', 'topo_soil', 's2_phenology']

MODALITY_LABEL = {
    's2_optical':   'S2-Optical',
    'modis_lst':    'MODIS-Thermal',
    's1_sar':       'S1-SAR',
    'topo_soil':    'Topo-Soil',
    's2_phenology': 'S2-Phenology',
}

MODALITY_COLOR = {
    's2_optical':   '#1F77B4',  # blue
    'modis_lst':    '#D62728',  # red
    's1_sar':       '#9467BD',  # purple
    'topo_soil':    '#8C564B',  # brown
    's2_phenology': '#2CA02C',  # green
}

# AlphaEarth gets its own neutral color, distinct from the fleet
AE_COLOR = '#555555'

# Environment-variable display labels (used in heatmaps and CONUS maps)
ENVVAR_LABEL = {
    'smap_sm':       'Soil Moisture (SMAP)',
    'elevation':     'Elevation (m)',
    'prism_ppt_mm':  'Precipitation (mm/yr)',
    'prism_tmean_c': 'Temperature Mean (°C)',
    'aridity_proxy': 'Aridity (P/PET)',
    'koppen':        'Köppen Class',
    'nlcd_class':    'NLCD Land Cover',
}

ENVVAR_ORDER = ['prism_ppt_mm', 'aridity_proxy', 'prism_tmean_c',
                'smap_sm', 'elevation', 'koppen', 'nlcd_class']

# Each Mini-JEPA's strongest predictive variable (drives Figure 2)
STRONGEST_VAR = {
    's2_optical':   'aridity_proxy',     # 0.73
    'modis_lst':    'prism_tmean_c',     # 0.97  ⭐ near-1D thermal axis
    's1_sar':       'prism_ppt_mm',      # 0.62  (weakest specialist; broadest fit)
    'topo_soil':    'elevation',         # 0.97  ⭐
    's2_phenology': 'prism_ppt_mm',      # 0.81  (broadly strong)
}

# ---------------------------------------------------------------------------
# Typography  (no matplotlib defaults; reads on print)
# ---------------------------------------------------------------------------
def apply_rc():
    """Install paper-grade rcParams. Call once at start of every figure script."""
    mpl.rcParams.update({
        # Font stack: try a clean sans first, fall back gracefully
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size':         9.5,
        'axes.titlesize':    11,
        'axes.titleweight':  'bold',
        'axes.labelsize':    10,
        'axes.labelweight':  'normal',
        'xtick.labelsize':   8.5,
        'ytick.labelsize':   8.5,
        'legend.fontsize':   8.5,
        'legend.frameon':    False,

        # Axis treatment: thin, no top/right spines unless requested
        'axes.linewidth':       0.7,
        'axes.edgecolor':       '#333333',
        'axes.spines.top':      False,
        'axes.spines.right':    False,
        'axes.grid':            False,

        # Tick treatment
        'xtick.major.width':    0.6,
        'ytick.major.width':    0.6,
        'xtick.major.size':     3.0,
        'ytick.major.size':     3.0,
        'xtick.direction':      'out',
        'ytick.direction':      'out',
        'xtick.color':          '#333333',
        'ytick.color':          '#333333',

        # Output: PDF for paper, PNG for preview, embed fonts
        'pdf.fonttype':         42,   # TrueType, editable in Illustrator
        'ps.fonttype':          42,
        'svg.fonttype':         'none',
        'figure.dpi':           150,
        'savefig.dpi':          300,
        'savefig.bbox':         'tight',
        'savefig.pad_inches':   0.08,
        'savefig.transparent':  False,
        'figure.facecolor':     'white',
        'axes.facecolor':       'white',
    })


# ---------------------------------------------------------------------------
# CONUS basemap
# ---------------------------------------------------------------------------
CONUS_EXTENT = (-125.0, -66.5, 24.5, 49.5)  # (lonW, lonE, latS, latN)

LAND_COLOR   = '#F2F0EB'
STATE_EDGE   = '#B8B5AE'
STATE_LW     = 0.4
COAST_EDGE   = '#888581'
COAST_LW     = 0.6


def _try_geopandas_states():
    """Return a GeoDataFrame of US state outlines, or None if geopandas unavailable.

    Tries Natural Earth via geopandas's bundled dataset URL chain;
    if no network and no cached file, returns None and the basemap
    falls back to a clean CONUS bounding rectangle (still publication-fine).
    """
    try:
        import geopandas as gpd
    except ImportError:
        return None

    # Common cache locations
    candidates = [
        Path.home() / '.cache' / 'naturalearth' / 'ne_50m_admin_1_states_provinces.shp',
        Path('/mnt/data/naturalearth/ne_50m_admin_1_states_provinces.shp'),
    ]
    for c in candidates:
        if c.exists():
            try:
                gdf = gpd.read_file(c)
                return gdf[gdf['admin'] == 'United States of America']
            except Exception:
                continue

    # Try cartopy's bundled shapefile (cartopy ships ne_50m by default)
    try:
        import cartopy.io.shapereader as shpreader
        path = shpreader.natural_earth(
            resolution='50m', category='cultural',
            name='admin_1_states_provinces_lakes',
        )
        gdf = gpd.read_file(path)
        return gdf[gdf['admin'] == 'United States of America']
    except Exception:
        return None


def add_conus_basemap(ax, *, land=True, states=True, lonlat_ticks=False):
    """Draw a clean CONUS basemap onto an existing matplotlib axis.

    Sets extent, fills land, draws state outlines if available,
    strips ticks unless lonlat_ticks=True. Idempotent — safe to call
    after plotting the foreground if you want overlays.
    """
    lonW, lonE, latS, latN = CONUS_EXTENT

    # Land background rectangle (works whether or not geopandas is available)
    if land:
        ax.add_patch(plt.Rectangle((lonW, latS), lonE - lonW, latN - latS,
                                    facecolor=LAND_COLOR, edgecolor='none',
                                    zorder=0))

    if states:
        gdf = _try_geopandas_states()
        if gdf is not None:
            try:
                gdf.boundary.plot(ax=ax, linewidth=STATE_LW,
                                   edgecolor=STATE_EDGE, zorder=1)
            except Exception:
                pass

    ax.set_xlim(lonW, lonE)
    ax.set_ylim(latS, latN)
    ax.set_aspect(1.25)  # rough Mercator-ish aspect for CONUS

    if not lonlat_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    else:
        ax.tick_params(labelsize=7, color='#888581')


# ---------------------------------------------------------------------------
# Annotation primitives
# ---------------------------------------------------------------------------
def annotation_box(ax, text, *, loc='lower left', fontsize=8.5,
                   facecolor='white', edgecolor='#333333', alpha=0.92,
                   pad=0.45, weight='normal'):
    """Drop a clean text box at one of the 4 corners or center.

    Used for inline R²/effect-size annotations on map panels.
    """
    coords = {
        'lower left':  (0.02, 0.02, 'left',   'bottom'),
        'lower right': (0.98, 0.02, 'right',  'bottom'),
        'upper left':  (0.02, 0.98, 'left',   'top'),
        'upper right': (0.98, 0.98, 'right',  'top'),
        'center':      (0.50, 0.50, 'center', 'center'),
    }
    x, y, ha, va = coords[loc]
    ax.text(x, y, text, transform=ax.transAxes,
            ha=ha, va=va, fontsize=fontsize, fontweight=weight,
            bbox=dict(boxstyle=f'round,pad={pad}',
                      facecolor=facecolor, edgecolor=edgecolor,
                      linewidth=0.6, alpha=alpha))


def effect_size_label(ax, *, d, p, n, x=0.5, y=0.98):
    """Standardized effect-size annotation for Figure 7-style panels."""
    sig = '*' if (p is not None and p < 0.05) else ''
    txt = f'd = {d:.2f}{sig}   p = {p:.3f}   n = {n}' if p is not None \
          else f'd = {d:.2f}   n = {n}'
    ax.text(x, y, txt, transform=ax.transAxes,
            ha='center', va='top', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#333333', linewidth=0.7, alpha=0.95))


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------
# Sequential map for prediction skill (R²): perceptually uniform, light→dark
SKILL_CMAP = mpl.colormaps.get_cmap('YlGnBu')

# Diverging map for residuals / deltas: TRUE zero-centered
DELTA_CMAP = mpl.colormaps.get_cmap('RdBu_r')

# Categorical: use modality colors directly via MODALITY_COLOR


def modality_cmap(modality: str, n: int = 256):
    """Per-modality sequential colormap from white → modality color.

    Used for per-modality CONUS scatter where the color identifies the
    Mini-JEPA and intensity encodes the metric.
    """
    base = MODALITY_COLOR[modality]
    return mpl.colors.LinearSegmentedColormap.from_list(
        f'{modality}_seq', ['#FFFFFF', base], N=n,
    )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def project_root() -> Path:
    """Resolve project root from env var or sensible default.

    Allows scripts to run from anywhere without hard-coded absolute paths.
    """
    # 1) explicit override via env var
    env = os.environ.get('HYDROJEPA_ROOT')
    if env:
        return Path(env).expanduser().resolve()

    # 2) walk up from this file looking for the data/hydrojepa marker
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / 'data' / 'hydrojepa').exists():
            return p

    # 3) fallback: the parent of figures/ (i.e. the repo root)
    return here.parent


def output_dir() -> Path:
    out = project_root() / 'figures' / 'output'
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_figure(fig, name: str, *, formats: Sequence[str] = ('pdf', 'png')):
    """Save a figure to figures/output/ in the requested formats."""
    out = output_dir()
    paths = []
    for fmt in formats:
        path = out / f'{name}.{fmt}'
        fig.savefig(path, format=fmt)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Dry-run helper
# ---------------------------------------------------------------------------
def dry_run_report(name: str, paths: dict[str, Path]):
    """Print the data files a figure script would read. Use with --dry-run."""
    print(f'\n=== {name} ===')
    print(f'Project root: {project_root()}')
    for label, path in paths.items():
        marker = '✓' if path.exists() else '✗ MISSING'
        print(f'  [{marker}] {label}: {path}')
    print()

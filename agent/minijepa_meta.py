"""
agent/minijepa_meta.py
Read each Mini-JEPA's evaluation reports and expose a compact summary
the agent router can put in its system prompt.

Each Mini-JEPA has three report directories produced by 6_1_minijepa_evaluation.py:

  reports/minijepas/<modality>/
    interpretability/        — dimension dictionary CSV, summary JSON
    manifold_geometry/       — geometry summary JSON, local PCA CSV
    complementarity/         — complementarity summary JSON, joint predictive gain CSV

The router needs a per-model snapshot small enough to fit in a system prompt:
top-K dimension entries, geometric headline numbers, the variables for which
this model adds joint-predictive-gain when combined with AE.
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path

import pandas as pd

HJ_ROOT     = Path(__file__).resolve().parent.parent
REPORTS_DIR = HJ_ROOT / 'reports' / 'minijepas'

MODALITIES = ['s2_optical', 's1_sar', 's2_phenology', 'modis_lst', 'topo_soil']

# Human-readable display names for the system prompt.
MODALITY_DISPLAY = {
    's2_optical':   'HydroJEPA-S2-Optical (Sentinel-2 annual median, 10 bands)',
    's1_sar':       'HydroJEPA-S1-SAR (Sentinel-1 backscatter, VV+VH)',
    's2_phenology': 'HydroJEPA-S2-Phenology (Sentinel-2 quarterly composites)',
    'modis_lst':    'HydroJEPA-MODIS-Thermal (MODIS LST day+night)',
    'topo_soil':    'HydroJEPA-Topo-Soil (SRTM elevation/slope/aspect + SoilGrids)',
}

# Short tag-line per modality describing what its physics is good for.
# Used in the routing system prompt so the LLM has something to reason from
# beyond the raw geometry numbers.
MODALITY_PHYSICS = {
    's2_optical':   'Spectral reflectance. Vegetation greenness, surface water, broad land cover.',
    's1_sar':       'Microwave backscatter. Sees through clouds. Surface roughness, soil moisture, biomass, inundation.',
    's2_phenology': 'Multi-temporal optical. Seasonal vegetation dynamics, agriculture, deciduous-vs-evergreen.',
    'modis_lst':    'Thermal emission. Urban heat islands, evaporative cooling, frost regime.',
    'topo_soil':    'Static topography and soil texture. Pure geomorphic and pedologic structure, no climate signal.',
}


@dataclass
class MinijepaMeta:
    modality: str
    display_name: str
    physics_tagline: str
    # interpretability
    top_dims: list[dict] = field(default_factory=list)   # [{dim, var, rho, category}]
    sp_rf_agree_pct: float = 0.0
    # geometry
    global_pr: float = 0.0
    intrinsic_dim: float = 0.0
    mean_local_pr: float = 0.0
    pcs_for_80_var: int = 0
    # complementarity
    joint_gains: list[dict] = field(default_factory=list)  # [{variable, gain_over_max}]
    cca_n_components_07: int = 0


def _load_interpretability(modality: str) -> dict:
    """Read interpretability/{summary.json, dimension_dictionary.csv}."""
    d = REPORTS_DIR / modality / 'interpretability'
    out = {'top_dims': [], 'sp_rf_agree_pct': 0.0}

    summ_path = d / 'hydrojepa_summary.json'
    if summ_path.exists():
        s = json.load(open(summ_path))
        # Different scripts have used slightly different keys; tolerate both.
        out['sp_rf_agree_pct'] = float(
            s.get('spearman_rf_agreement_pct',
                  s.get('sp_rf_agreement_pct',
                        s.get('agreement_pct', 0.0))))

    dict_path = d / 'hydrojepa_dimension_dictionary.csv'
    if dict_path.exists():
        df = pd.read_csv(dict_path)
        # Top by absolute primary correlation
        df = df.dropna(subset=['sp_abs_max']).sort_values('sp_abs_max', ascending=False)
        for _, r in df.head(8).iterrows():
            out['top_dims'].append({
                'dim': r.get('dimension', ''),
                'var': r.get('sp_primary', ''),
                'rho': float(r.get('sp_rho', 0.0)),
                'category': r.get('sp_category', ''),
            })
    else:
        logging.warning(f'[{modality}] missing dimension dictionary CSV')
    return out


def _load_geometry(modality: str) -> dict:
    """Read manifold_geometry/hydrojepa_geometry_summary.json."""
    d = REPORTS_DIR / modality / 'manifold_geometry'
    f = d / 'hydrojepa_geometry_summary.json'
    if not f.exists():
        logging.warning(f'[{modality}] missing geometry summary JSON')
        return {}
    s = json.load(open(f))
    return {
        'global_pr':       float(s.get('global_pr', 0.0)),
        'intrinsic_dim':   float(s.get('mean_intrinsic_dim', 0.0)),
        'mean_local_pr':   float(s.get('mean_local_pr', 0.0)),
        'pcs_for_80_var':  int(s.get('pcs_for_80_var', 0)),
    }


def _load_complementarity(modality: str) -> dict:
    """Read complementarity/{complementarity_summary.json, joint_predictive_gain.csv}."""
    d = REPORTS_DIR / modality / 'complementarity'
    out = {'joint_gains': [], 'cca_n_components_07': 0}

    summ_path = d / 'complementarity_summary.json'
    if summ_path.exists():
        s = json.load(open(summ_path))
        cca = s.get('cca_summary', {})
        out['cca_n_components_07'] = int(cca.get('n_components_07', 0))

    gain_path = d / 'joint_predictive_gain.csv'
    if gain_path.exists():
        df = pd.read_csv(gain_path)
        df = df.sort_values('gain_over_max', ascending=False)
        for _, r in df.iterrows():
            out['joint_gains'].append({
                'variable': r.get('label', r.get('variable', '')),
                'gain_over_max': float(r.get('gain_over_max', 0.0)),
                'r2_ae':    float(r.get('r2_ae_mean', 0.0)),
                'r2_hj':    float(r.get('r2_hj_mean', 0.0)),
                'r2_joint': float(r.get('r2_joint_mean', 0.0)),
            })
    return out


def load_one(modality: str) -> MinijepaMeta:
    """Load all three reports for one Mini-JEPA into a single dataclass."""
    meta = MinijepaMeta(
        modality=modality,
        display_name=MODALITY_DISPLAY.get(modality, modality),
        physics_tagline=MODALITY_PHYSICS.get(modality, ''),
    )
    meta.__dict__.update(_load_interpretability(modality))
    meta.__dict__.update(_load_geometry(modality))
    meta.__dict__.update(_load_complementarity(modality))
    return meta


def load_all() -> dict[str, MinijepaMeta]:
    """Load every Mini-JEPA's metadata. Modalities with missing reports get empty entries."""
    return {m: load_one(m) for m in MODALITIES}


# ---------------------------------------------------------------------------
# System-prompt rendering
# ---------------------------------------------------------------------------
def render_for_prompt(metas: dict[str, MinijepaMeta], top_n_dims: int = 5,
                      top_n_gains: int = 4) -> str:
    """
    Compact human-readable rendering the LLM router sees during the routing step.
    Designed to fit in a few hundred tokens, not paragraphs of numbers.
    """
    lines = []
    for m_id, meta in metas.items():
        lines.append(f'## {meta.display_name}')
        lines.append(f'  Physics: {meta.physics_tagline}')
        lines.append(
            f'  Geometry: PR={meta.global_pr:.1f}, intrinsic_dim={meta.intrinsic_dim:.1f}, '
            f'local_PR={meta.mean_local_pr:.1f}'
        )
        if meta.top_dims:
            lines.append('  Strongest encoded variables:')
            for d in meta.top_dims[:top_n_dims]:
                lines.append(
                    f'    - {d["dim"]} -> {d["var"]} (rho={d["rho"]:+.2f}, {d["category"]})'
                )
        if meta.joint_gains:
            lines.append('  Variables where this model + AE beats either alone:')
            shown = 0
            for g in meta.joint_gains:
                if shown >= top_n_gains:
                    break
                if g['gain_over_max'] <= 0:
                    continue
                lines.append(
                    f'    - {g["variable"]}: +{g["gain_over_max"]:.3f} R2 over max(AE,HJ)'
                )
                shown += 1
            if shown == 0:
                lines.append('    - (no positive joint gain on the standard env-var panel)')
        lines.append('')
    return '\n'.join(lines)


def to_json(metas: dict[str, MinijepaMeta]) -> str:
    """Serialize for caching or logging."""
    return json.dumps({k: asdict(v) for k, v in metas.items()}, indent=2, default=float)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    metas = load_all()
    print(render_for_prompt(metas))
    print()
    print('--- raw JSON (truncated) ---')
    j = to_json(metas)
    print(j[:1500] + '...')

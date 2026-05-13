"""
agent/minijepa_query_sets.py
Stratified Q&A generator for the Mini-JEPA LLM evaluation.

Four categories, each scored both for answer quality and for whether the
agent routes to the modality whose physics actually matches the question.

  single_modality (~20)
      Each Mini-JEPA's physics has a clearest fit:
        thermal     -> MODIS-Thermal      (urban heat island, frost regime)
        terrain     -> Topo-Soil          (slope, elevation, geomorphology)
        phenology   -> S2-Phenology       (seasonal vegetation)
        sar         -> S1-SAR             (cloud-obscured surface state)
        spectral    -> S2-Optical         (canopy greenness, surface water)

  multi_modality (~20)
      Genuinely depends on more than one physical signal.
      Each query carries 2+ expected modalities.
      Examples: irrigation under cloud cover (S1-SAR + S2-Phenology),
      snowmelt timing on terrain (MODIS-Thermal + Topo-Soil + S2-Phenology),
      urban-heat exacerbation by impervious surfaces (MODIS-Thermal +
      S2-Optical or Topo-Soil for surface texture).

  ae_favorable (~20)
      Broad characterization questions a generalist embedding handles well.
      Mini-JEPAs may not add anything; AE alone should be competitive.

  sar_favorable (~20)
      Phenomena S1-SAR captures that AE may not: surface roughness change,
      flood inundation, biomass under cloud cover. These test the
      "S1-SAR's value is elsewhere" hypothesis.

Each EvalQuery carries:
  qid, category, sub_type, question, expected_modalities,
  ground_truth (optional, structured), source_data (optional dict),
  judge_rubric_hint (str)

For routing-accuracy scoring we treat expected_modalities as a soft set:
the agent gets credit if its routing list overlaps at least once with the
expected set.
"""

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

HJ_ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR    = HJ_ROOT / 'data' / 'hydrojepa'
LABELS_FILE = DATA_DIR / 'labels.parquet'
EVAL_DIR    = DATA_DIR / 'minijepa_eval'

MODALITIES = ['s2_optical', 's1_sar', 's2_phenology', 'modis_lst', 'topo_soil']

# Default scope of the curated set (medium per the build plan).
TARGET_PER_CATEGORY = {
    'single_modality': 20,
    'multi_modality':  20,
    'ae_favorable':    20,
    'sar_favorable':   20,
}


@dataclass
class EvalQuery:
    qid: str
    category: str            # single_modality | multi_modality | ae_favorable | sar_favorable
    sub_type: str            # within-category tag (thermal, irrigation, etc.)
    question: str
    expected_modalities: list[str] = field(default_factory=list)
    ground_truth: dict       = field(default_factory=dict)
    source_data: dict        = field(default_factory=dict)
    judge_rubric_hint: str   = ''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_loc(lon: float, lat: float) -> str:
    return f'{lat:.2f}°N, {abs(lon):.2f}°W'


def _conus_region(lat: float, lon: float) -> str:
    """
    Assign a (lat, lon) to one of 6 CONUS regions.
    Same boundary scheme used in agent/minijepa_tools.py and the AE-paper
    tools so that the eval set's regional balance is interpretable in the
    same frame the agent uses.
    """
    if lon < -115:
        return 'Pacific_NW' if lat > 42 else 'Southwest'
    if lon < -100:
        if lat > 40:
            return 'Mountain_West'
        return 'Great_Plains' if lat > 32 else 'Southwest'
    if lon < -85:
        return 'Great_Plains' if lat > 40 else 'Southeast'
    return 'Northeast' if lat > 37 else 'Southeast'


def _sample(labels: pd.DataFrame, mask: pd.Series, n: int, rng: np.random.Generator
            ) -> pd.DataFrame:
    """
    Draw `n` rows from `labels[mask]`, stratified across 6 CONUS regions
    so that one regionally-clustered mask (e.g. high elevation, urban,
    wetland) doesn't produce an eval set that lives entirely in one
    corner of the country.

    For each region we draw ceil(n/6) rows; if a region has fewer
    matching patches than that quota, we take what's available and
    redistribute the slack evenly across the remaining regions. If
    the mask is empty in some regions (e.g. 'elevation > 1500m' in the
    Great Plains), those regions naturally drop out and the remaining
    ones absorb the quota. This guarantees we never exceed `n` total.
    """
    sub = labels[mask].dropna(subset=['lon', 'lat']).copy()
    if len(sub) == 0:
        return sub

    sub['_region'] = [
        _conus_region(lat, lon) for lat, lon in zip(sub['lat'], sub['lon'])
    ]
    regions = ['Pacific_NW', 'Mountain_West', 'Great_Plains',
               'Northeast', 'Southeast', 'Southwest']

    # Two-pass quota allocation:
    #   Pass 1: assign each region quota = ceil(n / |regions_with_data|),
    #           cap by available rows, take that much.
    #   Pass 2: if we still have headroom (because some regions had less
    #           than quota), redistribute the remainder across regions
    #           that still have spare patches.
    by_region = {r: sub[sub['_region'] == r] for r in regions}
    nonempty  = [r for r in regions if len(by_region[r]) > 0]
    if not nonempty:
        return sub.iloc[:0]

    target = min(n, len(sub))
    base_quota = (target + len(nonempty) - 1) // len(nonempty)  # ceil
    drawn_chunks: list[pd.DataFrame] = []
    used: dict[str, int] = {}
    for r in nonempty:
        avail = by_region[r]
        take = min(base_quota, len(avail))
        if take > 0:
            chunk = avail.sample(n=take, random_state=int(rng.integers(1e9)))
            drawn_chunks.append(chunk)
            used[r] = take

    drawn = pd.concat(drawn_chunks) if drawn_chunks else sub.iloc[:0]

    # Pass 2: backfill if under target
    deficit = target - len(drawn)
    if deficit > 0:
        spare_pool = []
        for r in nonempty:
            avail = by_region[r]
            taken_ids = set(drawn[drawn['_region'] == r].index) if len(drawn) else set()
            remaining = avail[~avail.index.isin(taken_ids)]
            if len(remaining) > 0:
                spare_pool.append(remaining)
        if spare_pool:
            spare = pd.concat(spare_pool)
            extra = spare.sample(n=min(deficit, len(spare)),
                                 random_state=int(rng.integers(1e9)))
            drawn = pd.concat([drawn, extra])

    # Trim if we over-drew (rare, only with rounding edge cases)
    if len(drawn) > target:
        drawn = drawn.sample(n=target, random_state=int(rng.integers(1e9)))

    return drawn.drop(columns=['_region']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Category-level sampler that allocates across sub-types and regions jointly
# ---------------------------------------------------------------------------
_REGIONS = ['Pacific_NW', 'Mountain_West', 'Great_Plains',
            'Northeast', 'Southeast', 'Southwest']


def _sample_category(labels: pd.DataFrame,
                      sub_specs: list[tuple[str, pd.Series]],
                      n: int,
                      rng: np.random.Generator
                      ) -> dict[str, pd.DataFrame]:
    """
    Allocate `n` queries across multiple sub-types AND across 6 CONUS regions,
    jointly. Returns {sub_type: dataframe-of-sampled-rows}.

    Why this exists: per-sub-type stratification sees only its own mask, so
    when one sub-type's mask is regionally clustered (e.g. snowmelt in the
    West, irrigation in the Plains) the category's regional totals get skewed.
    This function instead computes a feasibility table — how many patches
    each (sub_type, region) cell can supply — and allocates slots so the
    region totals come out as flat as possible given the cells that exist.

    Algorithm:
      1. Build feasibility[sub_type][region] = count of available patches.
      2. Set a target per region (n / 6, rounded), and a target per sub-type
         (n / n_subtypes, rounded), as soft constraints.
      3. Greedy fill: while we have remaining slots, pick the (sub_type, region)
         cell that most increases regional balance subject to feasibility.
      4. Sample randomly within each chosen cell.

    Falls back to per-sub-type _sample when a cell is mostly empty.
    """
    if n <= 0 or not sub_specs:
        return {}

    # Step 1: feasibility — how many patches each (sub_type, region) cell has
    feasibility: dict[str, dict[str, pd.DataFrame]] = {}
    for sub_name, mask in sub_specs:
        sub_df = labels[mask].dropna(subset=['lon', 'lat']).copy()
        if len(sub_df) == 0:
            feasibility[sub_name] = {r: sub_df.iloc[:0] for r in _REGIONS}
            continue
        sub_df['_region'] = [
            _conus_region(lat, lon) for lat, lon in zip(sub_df['lat'], sub_df['lon'])
        ]
        feasibility[sub_name] = {
            r: sub_df[sub_df['_region'] == r] for r in _REGIONS
        }

    # Step 2: targets
    n_sub = len(sub_specs)
    region_target = {r: n // 6 for r in _REGIONS}
    # Distribute the rounding remainder across regions starting from PNW
    remainder = n - sum(region_target.values())
    for i in range(remainder):
        region_target[_REGIONS[i % 6]] += 1
    sub_target = {s[0]: n // n_sub for s in sub_specs}
    remainder = n - sum(sub_target.values())
    for i in range(remainder):
        sub_target[sub_specs[i % n_sub][0]] += 1

    # Step 3: greedy fill. We track allocated[sub][region] = int count chosen.
    allocated = {s[0]: {r: 0 for r in _REGIONS} for s in sub_specs}
    region_remaining = dict(region_target)
    sub_remaining    = dict(sub_target)
    total_remaining  = n

    # Pre-shuffle sub-types and regions to break ties fairly
    # (without RNG here you'd always tie-break in alphabetical order,
    # leaving a slight bias)
    while total_remaining > 0:
        # Find the (sub, region) cell that most needs filling and has supply.
        # Score: (region_deficit, sub_deficit, available - allocated). Pick
        # the cell that maximizes regional deficit first, then sub-type deficit.
        best = None
        best_score = (-1, -1, -1)
        for sub_name, _ in sub_specs:
            if sub_remaining[sub_name] <= 0:
                continue
            for r in _REGIONS:
                if region_remaining[r] <= 0:
                    continue
                avail = len(feasibility[sub_name][r])
                if allocated[sub_name][r] >= avail:
                    continue
                score = (region_remaining[r], sub_remaining[sub_name],
                         avail - allocated[sub_name][r])
                if score > best_score:
                    best_score = score
                    best = (sub_name, r)
        if best is None:
            # Either every region is filled, every sub_type is exhausted,
            # or every cell with remaining demand has 0 supply. Stop.
            break
        sub_name, r = best
        allocated[sub_name][r] += 1
        region_remaining[r]    -= 1
        sub_remaining[sub_name]-= 1
        total_remaining        -= 1

    # If we couldn't hit `n` because of supply constraints, fall back to
    # filling whatever room is left from any non-exhausted sub_type/region
    # without the regional-deficit constraint.
    if total_remaining > 0:
        for sub_name, _ in sub_specs:
            for r in _REGIONS:
                if total_remaining <= 0:
                    break
                avail = len(feasibility[sub_name][r])
                room = avail - allocated[sub_name][r]
                if room > 0:
                    take = min(room, total_remaining)
                    allocated[sub_name][r] += take
                    total_remaining -= take

    # Step 4: actually draw the samples
    out: dict[str, pd.DataFrame] = {}
    for sub_name, _ in sub_specs:
        chunks = []
        for r in _REGIONS:
            k = allocated[sub_name][r]
            pool = feasibility[sub_name][r]
            if k > 0 and len(pool) >= k:
                chunks.append(pool.sample(n=k, random_state=int(rng.integers(1e9))))
        if chunks:
            df = pd.concat(chunks).drop(columns=['_region'], errors='ignore')
            out[sub_name] = df.reset_index(drop=True)
        else:
            out[sub_name] = labels.iloc[:0]
    return out


# ---------------------------------------------------------------------------
# Category generators — each uses _sample_category for joint cross-sub-type,
# cross-region balancing. The pattern: build a list of (sub_type_name, mask)
# pairs, sample once with _sample_category, then build queries per sub-type.
# ---------------------------------------------------------------------------

def gen_single_modality(labels: pd.DataFrame, n: int, rng) -> list[EvalQuery]:
    """Each sub-type has a clear physics fit to one Mini-JEPA."""
    sub_specs = []
    if 'nlcd_class' in labels.columns:
        sub_specs.append(('thermal',  labels['nlcd_class'].isin([22, 23, 24])))
    if 'elevation' in labels.columns:
        sub_specs.append(('terrain',  labels['elevation'] > 1500))
    if 'nlcd_class' in labels.columns:
        sub_specs.append(('phenology', labels['nlcd_class'].isin([81, 82])))
    if 'nlcd_class' in labels.columns and 'aridity_proxy' in labels.columns:
        sub_specs.append(('sar',
            labels['nlcd_class'].isin([90, 95, 11]) | (labels['aridity_proxy'] > 1.5)))
    if 'nlcd_class' in labels.columns:
        sub_specs.append(('spectral', labels['nlcd_class'].isin([41, 42, 43])))

    samples = _sample_category(labels, sub_specs, n, rng)

    out: list[EvalQuery] = []

    for r in samples.get('thermal', pd.DataFrame()).itertuples():
        out.append(EvalQuery(
            qid=f'SM-thermal-{r.patch_id}',
            category='single_modality', sub_type='thermal',
            question=(f'Is the area near {fmt_loc(r.lon, r.lat)} likely to '
                      f'experience pronounced surface heat-island effects? '
                      f'Briefly justify with the physical signal you would use to detect it.'),
            expected_modalities=['modis_lst'],
            source_data={'lon': float(r.lon), 'lat': float(r.lat),
                         'nlcd_class': int(r.nlcd_class)},
            judge_rubric_hint='Should reason about thermal emission / LST, not vegetation indices.',
        ))

    for r in samples.get('terrain', pd.DataFrame()).itertuples():
        out.append(EvalQuery(
            qid=f'SM-terrain-{r.patch_id}',
            category='single_modality', sub_type='terrain',
            question=(f'Characterize the topographic setting at {fmt_loc(r.lon, r.lat)}. '
                      f'How would you confirm whether this is a steep upland or a high plateau '
                      f'using purely geomorphic information?'),
            expected_modalities=['topo_soil'],
            source_data={'lon': float(r.lon), 'lat': float(r.lat),
                         'elevation_m': float(r.elevation)},
            judge_rubric_hint='Should invoke elevation, slope, aspect — not spectral or thermal data.',
        ))

    for r in samples.get('phenology', pd.DataFrame()).itertuples():
        out.append(EvalQuery(
            qid=f'SM-pheno-{r.patch_id}',
            category='single_modality', sub_type='phenology',
            question=(f'Is the cropping pattern at {fmt_loc(r.lon, r.lat)} consistent with '
                      f'a single-season annual crop, double cropping, or perennial pasture? '
                      f'What temporal signal would distinguish these regimes?'),
            expected_modalities=['s2_phenology'],
            source_data={'lon': float(r.lon), 'lat': float(r.lat),
                         'nlcd_class': int(r.nlcd_class)},
            judge_rubric_hint='Should invoke seasonal NDVI/EVI dynamics, not single-date imagery.',
        ))

    for r in samples.get('sar', pd.DataFrame()).itertuples():
        out.append(EvalQuery(
            qid=f'SM-sar-{r.patch_id}',
            category='single_modality', sub_type='sar',
            question=(f'Suppose persistent cloud cover at {fmt_loc(r.lon, r.lat)} prevented '
                      f'optical observation. How would you still detect surface inundation '
                      f'or saturated soil during the wet season at this location?'),
            expected_modalities=['s1_sar'],
            source_data={'lon': float(r.lon), 'lat': float(r.lat)},
            judge_rubric_hint='Should reason about microwave backscatter; cloud-penetration is the key fact.',
        ))

    for r in samples.get('spectral', pd.DataFrame()).itertuples():
        out.append(EvalQuery(
            qid=f'SM-spectral-{r.patch_id}',
            category='single_modality', sub_type='spectral',
            question=(f'Estimate the canopy density and probable forest type at '
                      f'{fmt_loc(r.lon, r.lat)} using a single late-summer optical observation.'),
            expected_modalities=['s2_optical'],
            source_data={'lon': float(r.lon), 'lat': float(r.lat),
                         'nlcd_class': int(r.nlcd_class)},
            judge_rubric_hint='Should invoke vegetation indices like NDVI from spectral bands.',
        ))

    return out[:n]


# ---------------------------------------------------------------------------
# Category 2 — Multi-modality questions
# ---------------------------------------------------------------------------
def gen_multi_modality(labels: pd.DataFrame, n: int, rng) -> list[EvalQuery]:
    """Genuinely depend on 2+ physical signals."""
    sub_specs = []
    if 'nlcd_class' in labels.columns and 'aridity_proxy' in labels.columns:
        sub_specs.append(('irrigation',
            (labels['nlcd_class'] == 82) & (labels['aridity_proxy'] < 1.0)))
    if 'elevation' in labels.columns and 'prism_tmean_c' in labels.columns:
        sub_specs.append(('snowmelt',
            (labels['elevation'] > 1800) & (labels['prism_tmean_c'] < 8)))
    if 'nlcd_class' in labels.columns:
        sub_specs.append(('uhi', labels['nlcd_class'].isin([23, 24])))
    if 'nlcd_class' in labels.columns and 'aridity_proxy' in labels.columns:
        sub_specs.append(('ag_drought',
            labels['nlcd_class'].isin([81, 82]) & (labels['aridity_proxy'] < 0.8)))

    samples = _sample_category(labels, sub_specs, n, rng)
    out: list[EvalQuery] = []

    for r in samples.get('irrigation', pd.DataFrame()).itertuples():
        out.append(EvalQuery(
            qid=f'MM-irrig-{r.patch_id}',
            category='multi_modality', sub_type='irrigation',
            question=(f'How would you confirm active irrigation at {fmt_loc(r.lon, r.lat)} '
                      f'during the growing season, given that summer cloud cover is frequent? '
                      f'Describe the data sources you would combine.'),
            expected_modalities=['s1_sar', 's2_phenology'],
            source_data={'lon': float(r.lon), 'lat': float(r.lat),
                         'aridity_proxy': float(r.aridity_proxy)},
            judge_rubric_hint='Strongest answers combine cloud-penetrating SAR with seasonal vegetation dynamics.',
        ))

    for r in samples.get('snowmelt', pd.DataFrame()).itertuples():
        out.append(EvalQuery(
            qid=f'MM-snow-{r.patch_id}',
            category='multi_modality', sub_type='snowmelt',
            question=(f'How would you estimate the timing and elevation extent of '
                      f'spring snowmelt near {fmt_loc(r.lon, r.lat)}? Identify the '
                      f'physical signals each data source would contribute.'),
            expected_modalities=['modis_lst', 'topo_soil', 's2_phenology'],
            source_data={'lon': float(r.lon), 'lat': float(r.lat),
                         'elevation_m': float(r.elevation)},
            judge_rubric_hint='Land-surface temperature crossing freezing, terrain shading, vegetation green-up.',
        ))

    for r in samples.get('uhi', pd.DataFrame()).itertuples():
        out.append(EvalQuery(
            qid=f'MM-uhi-{r.patch_id}',
            category='multi_modality', sub_type='uhi',
            question=(f'At {fmt_loc(r.lon, r.lat)}, how would you separate the contribution '
                      f'of impervious-surface fraction from the direct thermal signal '
                      f'when assessing urban heat exposure?'),
            expected_modalities=['modis_lst', 's2_optical'],
            source_data={'lon': float(r.lon), 'lat': float(r.lat),
                         'nlcd_class': int(r.nlcd_class)},
            judge_rubric_hint='Pair thermal (LST) with optical (built-up index) reasoning.',
        ))

    for r in samples.get('ag_drought', pd.DataFrame()).itertuples():
        out.append(EvalQuery(
            qid=f'MM-drought-{r.patch_id}',
            category='multi_modality', sub_type='ag_drought',
            question=(f'Identify the early-warning indicators of growing-season drought stress '
                      f'on agriculture at {fmt_loc(r.lon, r.lat)}. What complementary signals '
                      f'would you watch?'),
            expected_modalities=['s2_phenology', 'modis_lst', 's2_optical'],
            source_data={'lon': float(r.lon), 'lat': float(r.lat),
                         'aridity_proxy': float(r.aridity_proxy)},
            judge_rubric_hint='Vegetation stress (NDVI/phenology), surface temperature anomaly, soil moisture.',
        ))

    return out[:n]


# ---------------------------------------------------------------------------
# Category 3 — AE-favorable questions
# ---------------------------------------------------------------------------
def gen_ae_favorable(labels: pd.DataFrame, n: int, rng) -> list[EvalQuery]:
    """Broad descriptive questions a generalist embedding should handle well."""
    sub_specs = []
    if 'nlcd_class' in labels.columns:
        sub_specs.append(('land_cover_general', labels['nlcd_class'].notna()))
    if 'prism_ppt_mm' in labels.columns:
        sub_specs.append(('climate_summary',    labels['prism_ppt_mm'].notna()))
    sub_specs.append(('region_typology',        labels['lon'].notna()))

    samples = _sample_category(labels, sub_specs, n, rng)
    out: list[EvalQuery] = []

    for r in samples.get('land_cover_general', pd.DataFrame()).itertuples():
        out.append(EvalQuery(
            qid=f'AE-lc-{r.patch_id}',
            category='ae_favorable', sub_type='land_cover_general',
            question=(f'Describe the broad land-cover composition near '
                      f'{fmt_loc(r.lon, r.lat)} in plain terms.'),
            expected_modalities=[],
            source_data={'lon': float(r.lon), 'lat': float(r.lat),
                         'nlcd_class': int(r.nlcd_class)},
            judge_rubric_hint='Generalist description; no single Mini-JEPA is uniquely required.',
        ))

    for r in samples.get('climate_summary', pd.DataFrame()).itertuples():
        out.append(EvalQuery(
            qid=f'AE-clim-{r.patch_id}',
            category='ae_favorable', sub_type='climate_summary',
            question=(f'Provide a one-paragraph climate summary for the area around '
                      f'{fmt_loc(r.lon, r.lat)}, covering precipitation regime and temperature.'),
            expected_modalities=[],
            source_data={'lon': float(r.lon), 'lat': float(r.lat)},
            judge_rubric_hint='Broad climate; specialist Mini-JEPAs add little.',
        ))

    for r in samples.get('region_typology', pd.DataFrame()).itertuples():
        out.append(EvalQuery(
            qid=f'AE-region-{r.patch_id}',
            category='ae_favorable', sub_type='region_typology',
            question=(f'Which broad CONUS region or biome best matches the environmental '
                      f'character of {fmt_loc(r.lon, r.lat)}?'),
            expected_modalities=[],
            source_data={'lon': float(r.lon), 'lat': float(r.lat)},
            judge_rubric_hint='Generalist typology; the question should be easy for AE alone.',
        ))

    return out[:n]


# ---------------------------------------------------------------------------
# Category 4 — SAR-favorable questions
# ---------------------------------------------------------------------------
def gen_sar_favorable(labels: pd.DataFrame, n: int, rng) -> list[EvalQuery]:
    """Phenomena S1-SAR captures that AE may not surface as easily."""
    sub_specs = []
    if 'nlcd_class' in labels.columns and 'elevation' in labels.columns:
        sub_specs.append(('flood_inundation',
            labels['nlcd_class'].isin([90, 95, 11]) | (labels['elevation'] < 100)))
    if 'nlcd_class' in labels.columns:
        sub_specs.append(('biomass_under_cloud', labels['nlcd_class'].isin([41, 42, 43])))
    if 'nlcd_class' in labels.columns:
        sub_specs.append(('surface_roughness',   labels['nlcd_class'].isin([81, 82])))

    samples = _sample_category(labels, sub_specs, n, rng)
    out: list[EvalQuery] = []

    for r in samples.get('flood_inundation', pd.DataFrame()).itertuples():
        out.append(EvalQuery(
            qid=f'SAR-flood-{r.patch_id}',
            category='sar_favorable', sub_type='flood_inundation',
            question=(f'How would you map the extent of recent flooding near '
                      f'{fmt_loc(r.lon, r.lat)} during a multi-day storm with persistent '
                      f'cloud cover?'),
            expected_modalities=['s1_sar'],
            source_data={'lon': float(r.lon), 'lat': float(r.lat)},
            judge_rubric_hint='Cloud-penetrating SAR is the obvious tool; optical alone is insufficient.',
        ))

    for r in samples.get('biomass_under_cloud', pd.DataFrame()).itertuples():
        out.append(EvalQuery(
            qid=f'SAR-biom-{r.patch_id}',
            category='sar_favorable', sub_type='biomass_under_cloud',
            question=(f'Suppose multi-week cloud cover prevents optical canopy observation at '
                      f'{fmt_loc(r.lon, r.lat)}. Can you still estimate forest biomass or '
                      f'structural change, and how?'),
            expected_modalities=['s1_sar'],
            source_data={'lon': float(r.lon), 'lat': float(r.lat)},
            judge_rubric_hint='SAR backscatter is sensitive to canopy volume scattering and biomass.',
        ))

    for r in samples.get('surface_roughness', pd.DataFrame()).itertuples():
        out.append(EvalQuery(
            qid=f'SAR-rough-{r.patch_id}',
            category='sar_favorable', sub_type='surface_roughness',
            question=(f'How would you detect a tillage or harvest event at '
                      f'{fmt_loc(r.lon, r.lat)} from satellite data, particularly when '
                      f'spectral changes are subtle or obscured?'),
            expected_modalities=['s1_sar'],
            source_data={'lon': float(r.lon), 'lat': float(r.lat)},
            judge_rubric_hint='Tillage changes surface roughness and dielectric constant; SAR detects this.',
        ))

    return out[:n]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_query_set(seed: int = 42, target_per_category: dict | None = None
                    ) -> list[EvalQuery]:
    rng = np.random.default_rng(seed)
    labels = pd.read_parquet(LABELS_FILE)
    target = target_per_category or TARGET_PER_CATEGORY

    queries: list[EvalQuery] = []
    queries += gen_single_modality(labels, target['single_modality'], rng)
    queries += gen_multi_modality(labels,  target['multi_modality'],  rng)
    queries += gen_ae_favorable(labels,    target['ae_favorable'],    rng)
    queries += gen_sar_favorable(labels,   target['sar_favorable'],   rng)

    logging.info(f'Built {len(queries)} queries')
    by_cat: dict[str, int] = {}
    for q in queries:
        by_cat[q.category] = by_cat.get(q.category, 0) + 1
    for c, n in by_cat.items():
        logging.info(f'  {c}: {n}')

    report_distribution(queries)
    return queries


def report_distribution(queries: list[EvalQuery]):
    """Print regional and category-by-region breakdowns for sanity-checking.
    Each query gets assigned to a CONUS region by its source_data lat/lon."""
    rows = []
    for q in queries:
        sd = q.source_data or {}
        if 'lat' not in sd or 'lon' not in sd:
            continue
        rows.append({
            'category': q.category,
            'sub_type': q.sub_type,
            'region':   _conus_region(float(sd['lat']), float(sd['lon'])),
        })
    if not rows:
        return
    df = pd.DataFrame(rows)

    logging.info('\nRegional distribution (queries per region):')
    region_counts = df['region'].value_counts().sort_index()
    total = len(df)
    for r, n in region_counts.items():
        logging.info(f'  {r:15s}  {n:3d}  ({n / total:.0%})')

    logging.info('\nPer-category × region (counts):')
    pivot = (df.pivot_table(index='category', columns='region',
                            values='sub_type', aggfunc='count', fill_value=0))
    for line in pivot.to_string().split('\n'):
        logging.info(f'  {line}')


def write_query_set(queries: list[EvalQuery], out_dir: Path = EVAL_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / 'qa_set.jsonl'
    csv_path   = out_dir / 'qa_set.csv'

    # Augment each row with a region tag in the CSV for easy review.
    rows = []
    for q in queries:
        d = asdict(q)
        sd = d.get('source_data') or {}
        if 'lat' in sd and 'lon' in sd:
            d['region'] = _conus_region(float(sd['lat']), float(sd['lon']))
        else:
            d['region'] = 'unknown'
        rows.append(d)

    with open(jsonl_path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + '\n')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logging.info(f'Wrote {jsonl_path}')
    logging.info(f'Wrote {csv_path}')


def load_query_set(path: Path | None = None) -> list[EvalQuery]:
    p = path or (EVAL_DIR / 'qa_set.jsonl')
    if not p.exists():
        raise FileNotFoundError(f'{p} missing — run minijepa_query_sets.py first.')
    out: list[EvalQuery] = []
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            d.pop('region', None)  # written for review only, not part of EvalQuery
            out.append(EvalQuery(**d))
    return out


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--small', action='store_true', help='Use ~40 queries instead of ~80')
    p.add_argument('--report_only', action='store_true',
                   help='Skip generation, just report on the existing qa_set.jsonl')
    args = p.parse_args()

    if args.report_only:
        queries = load_query_set()
        report_distribution(queries)
        sys.exit(0)

    target = ({k: v // 2 for k, v in TARGET_PER_CATEGORY.items()}
              if args.small else TARGET_PER_CATEGORY)
    queries = build_query_set(seed=args.seed, target_per_category=target)
    write_query_set(queries)

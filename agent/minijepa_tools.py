"""
agent/minijepa_tools.py
Tool engine for the multi-select Mini-JEPA router.

The agent has the following tools available, all callable by name from
JSON tool-call blocks emitted by the LLM:

  resolve_location(name)
      Geocode a CONUS location string to (lat, lon).
      Reuses the AE-paper LocationResolver if available; otherwise falls
      back to a small CONUS gazetteer + numeric coordinate parsing.

  list_minijepas()
      Returns the per-model metadata summaries (physics tagline, geometry
      headline, top dim-dictionary entries, joint-with-AE predictive gain).
      The agent calls this once during the routing step to decide which
      Mini-JEPA(s) to consult.

  get_minijepa_meta(modality)
      Detailed metadata for one specific Mini-JEPA. Used when the agent
      wants to inspect a single model more deeply before committing.

  retrieve_minijepa(modality, lat, lon, k)
      Top-k embedding-similar patches in `modality`'s embedding space,
      returned with patch_id, lon, lat, distance, and the standard label
      panel (smap_sm, nlcd_class, elevation, prism_ppt_mm, prism_tmean_c,
      aridity_proxy, koppen). One tool, parameterized by modality, so the
      agent can call the same tool for any of the five.

  retrieve_ae(lat, lon, k)
      Same shape as retrieve_minijepa but in the AlphaEarth 64-d space
      (loaded from labels.parquet's A00..A63 columns). Provides the
      generalist-model retrieval condition.

The query-side embedding for retrieve_minijepa is computed by encoding the
target patch on the fly. Since the corpus only contains 9,704 fixed patch
centers, when the user asks about an arbitrary (lat, lon) we first find
the nearest patch geographically, retrieve its precomputed embedding, and
do FAISS k-NN from there. This keeps the retrieval entirely in embedding
space (the FAISS query) while avoiding the cost of running the encoder
on a freshly-downloaded GEE patch every query.
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
import pandas as pd

HJ_ROOT     = Path(__file__).resolve().parent.parent
INDEX_DIR   = HJ_ROOT / 'agent' / 'minijepa_index'
DATA_DIR    = HJ_ROOT / 'data' / 'hydrojepa'
LABELS_FILE = DATA_DIR / 'labels.parquet'

LABEL_PANEL_COLS = [
    'smap_sm', 'nlcd_class', 'elevation',
    'prism_ppt_mm', 'prism_tmean_c', 'aridity_proxy', 'koppen',
]


# ---------------------------------------------------------------------------
# Tool manifest (used by the router prompt)
# ---------------------------------------------------------------------------
MODALITY_LIST = ['s2_optical', 's1_sar', 's2_phenology', 'modis_lst', 'topo_soil']

TOOL_MANIFEST = [
    {
        'name': 'resolve_location',
        'description': 'Resolve a place name or coordinate string to (lat, lon) in CONUS.',
        'parameters': {'name': 'str — e.g. "Boise, Idaho" or "44.5, -116.2"'},
        'returns': '{name, lat, lon, source}',
    },
    {
        'name': 'list_minijepas',
        'description': (
            'Return the metadata summary for every Mini-JEPA in the fleet '
            '(physics, geometry headline, top encoded variables, joint-with-AE '
            'predictive gain). Call this BEFORE selecting which models to consult.'
        ),
        'parameters': {},
        'returns': 'Dict[modality_id -> meta]',
    },
    {
        'name': 'get_minijepa_meta',
        'description': 'Detailed metadata for a single Mini-JEPA modality.',
        'parameters': {'modality': f'str — one of {MODALITY_LIST}'},
        'returns': 'meta dict for that modality',
    },
    {
        'name': 'retrieve_minijepa',
        'description': (
            'Top-k embedding-similar patches in a specific Mini-JEPA\'s embedding '
            'space, with their environmental labels. The neighborhood differs '
            'per modality because each model encodes different physics.'
        ),
        'parameters': {
            'modality': f'str — one of {MODALITY_LIST}',
            'lat':      'float',
            'lon':      'float',
            'k':        'int — default 5',
        },
        'returns': 'List[{patch_id, lon, lat, distance, smap_sm, nlcd_class, ...}]',
    },
    {
        'name': 'retrieve_ae',
        'description': (
            'Top-k embedding-similar patches in the AlphaEarth 64-d generalist '
            'embedding space, with their environmental labels.'
        ),
        'parameters': {
            'lat': 'float', 'lon': 'float', 'k': 'int — default 5',
        },
        'returns': 'List[{patch_id, lon, lat, distance, smap_sm, nlcd_class, ...}]',
    },
]


# ---------------------------------------------------------------------------
# Tool call log
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    tool_name: str
    inputs: dict[str, Any]
    outputs: Any
    success: bool = True
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Lightweight CONUS geocoder
# ---------------------------------------------------------------------------
class SimpleGeocoder:
    """
    Tiny gazetteer for the most common CONUS place names. Falls back to
    parsing literal coordinates. Designed so the router does not depend
    on external geocoding services or the AE-paper LocationResolver.
    """
    GAZETTEER = {
        # major cities — the same set the AE paper uses for benchmarking
        'wichita, kansas':         (37.69, -97.34),
        'omaha, nebraska':         (41.26, -95.93),
        'lubbock, texas':          (33.58, -101.85),
        'fargo, north dakota':     (46.88, -96.79),
        'oklahoma city, oklahoma': (35.47, -97.52),
        'topeka, kansas':          (39.05, -95.68),
        'sioux falls, south dakota': (43.55, -96.73),
        'boise, idaho':            (43.62, -116.21),
        'seattle, washington':     (47.61, -122.33),
        'portland, oregon':        (45.52, -122.68),
        'spokane, washington':     (47.66, -117.43),
        'eugene, oregon':          (44.05, -123.09),
        'denver, colorado':        (39.74, -104.99),
        'salt lake city, utah':    (40.76, -111.89),
        'phoenix, arizona':        (33.45, -112.07),
        'tucson, arizona':         (32.22, -110.97),
        'las vegas, nevada':       (36.17, -115.14),
        'albuquerque, new mexico': (35.08, -106.65),
        'atlanta, georgia':        (33.75, -84.39),
        'miami, florida':          (25.76, -80.19),
        'orlando, florida':        (28.54, -81.38),
        'new orleans, louisiana':  (29.95, -90.07),
        'houston, texas':          (29.76, -95.37),
        'austin, texas':           (30.27, -97.74),
        'dallas, texas':           (32.78, -96.80),
        'memphis, tennessee':      (35.15, -90.05),
        'nashville, tennessee':    (36.16, -86.78),
        'new york, new york':      (40.71, -74.01),
        'boston, massachusetts':   (42.36, -71.06),
        'philadelphia, pennsylvania': (39.95, -75.17),
        'pittsburgh, pennsylvania':(40.44, -79.99),
        'detroit, michigan':       (42.33, -83.05),
        'chicago, illinois':       (41.88, -87.63),
        'minneapolis, minnesota':  (44.98, -93.27),
    }

    def resolve(self, name: str) -> dict:
        s = name.strip().lower()
        # exact gazetteer
        if s in self.GAZETTEER:
            lat, lon = self.GAZETTEER[s]
            return {'name': name, 'lat': lat, 'lon': lon, 'source': 'gazetteer'}
        # numeric coords: "lat, lon" with optional °N/°W
        try:
            lat, lon = self._parse_coords(name)
            return {'name': name, 'lat': lat, 'lon': lon, 'source': 'numeric'}
        except Exception:
            pass
        # partial match — first city or state-suffix that contains the query
        for key, (lat, lon) in self.GAZETTEER.items():
            if s in key or key.startswith(s):
                return {'name': name, 'lat': lat, 'lon': lon, 'source': 'gazetteer_partial'}
        return {'error': f'Could not resolve "{name}". Provide explicit "lat, lon".'}

    @staticmethod
    def _parse_coords(s: str) -> tuple[float, float]:
        import re
        # Pattern A: "44.5°N, 116.2°W"
        m = re.findall(r'(-?\d+\.?\d*)\s*°?\s*([NSEW])', s, re.IGNORECASE)
        if len(m) >= 2:
            vals = []
            for v, dirn in m[:2]:
                v = float(v)
                if dirn.upper() in ('S', 'W'):
                    v = -abs(v)
                vals.append(v)
            return vals[0], vals[1]
        # Pattern B: "lat, lon" plain
        parts = [p for p in s.replace(';', ',').split(',') if p.strip()]
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
        raise ValueError('not parseable')


# ---------------------------------------------------------------------------
# Per-model index
# ---------------------------------------------------------------------------
class ModelIndex:
    """Wraps one (faiss, embeddings, keys) triple for a given model name."""

    def __init__(self, name: str):
        self.name = name
        npy_path = INDEX_DIR / f'{name}.npy'
        idx_path = INDEX_DIR / f'{name}.faiss'
        keys_path = INDEX_DIR / f'{name}_keys.parquet'
        for p in (npy_path, idx_path, keys_path):
            if not p.exists():
                raise FileNotFoundError(
                    f'Missing {p}. Run agent/build_minijepa_indices.py first.')
        self.embs = np.load(npy_path)
        self.index = faiss.read_index(str(idx_path))
        self.keys = pd.read_parquet(keys_path)
        if len(self.keys) != self.embs.shape[0]:
            raise RuntimeError(
                f'{name}: keys ({len(self.keys)}) != embs ({self.embs.shape[0]})')

    def query_by_geo(self, lat: float, lon: float, k: int = 5) -> pd.DataFrame:
        """
        Find the patch geographically closest to (lat, lon), use ITS embedding as
        the query vector, then return top-k embedding neighbors (FAISS).
        Returns the keys rows joined with embedding distances.
        """
        # geographic anchor
        from sklearn.neighbors import BallTree
        tree = BallTree(np.deg2rad(self.keys[['lat', 'lon']].values),
                        metric='haversine')
        _, idx = tree.query(np.deg2rad([[lat, lon]]), k=1)
        anchor_row = int(idx[0][0])
        q = self.embs[anchor_row:anchor_row + 1].astype(np.float32)

        D, I = self.index.search(q, k + 1)  # +1 because anchor itself will appear
        rows = []
        for d, i in zip(D[0], I[0]):
            if i == anchor_row:
                continue
            r = self.keys.iloc[i]
            rows.append({
                'patch_id': r['patch_id'],
                'lon':      float(r['lon']),
                'lat':      float(r['lat']),
                'emb_distance': float(d),
            })
            if len(rows) >= k:
                break
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tool engine
# ---------------------------------------------------------------------------
class ToolEngine:
    """Owns indices, label panel, geocoder, and dispatch."""

    def __init__(self, lazy: bool = True, verbose: bool = False):
        self.verbose = verbose
        self.call_log: list[ToolCall] = []
        self.geocoder = SimpleGeocoder()
        self._labels: Optional[pd.DataFrame] = None
        self._indices: dict[str, ModelIndex] = {}
        # Lazy: defer the heaviest loads until first call. Avoids 300 MB
        # of embeddings sitting in RAM during conditions that don't need them.
        if not lazy:
            self.labels  # noqa
            for m in MODALITY_LIST + ['alphaearth']:
                self._indices[m] = ModelIndex(m)

        self._tools = {
            'resolve_location':  self.resolve_location,
            'list_minijepas':    self.list_minijepas,
            'get_minijepa_meta': self.get_minijepa_meta,
            'retrieve_minijepa': self.retrieve_minijepa,
            'retrieve_ae':       self.retrieve_ae,
        }

    @property
    def labels(self) -> pd.DataFrame:
        if self._labels is None:
            self._labels = pd.read_parquet(LABELS_FILE)
        return self._labels

    def _index(self, name: str) -> ModelIndex:
        if name not in self._indices:
            self._indices[name] = ModelIndex(name)
        return self._indices[name]

    # -----------------------------------------------------------------
    # Tool dispatch
    # -----------------------------------------------------------------
    def call_tool(self, tool: str, **kwargs) -> Any:
        if tool not in self._tools:
            err = f'Unknown tool: {tool}'
            self.call_log.append(ToolCall(tool, kwargs, None, success=False, error=err))
            return {'error': err}
        t0 = time.time()
        try:
            out = self._tools[tool](**kwargs)
            elapsed = (time.time() - t0) * 1000
            self.call_log.append(ToolCall(tool, kwargs, _summarize_for_log(out),
                                          success=True, elapsed_ms=elapsed))
            return out
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            self.call_log.append(ToolCall(tool, kwargs, None, success=False,
                                          error=str(e), elapsed_ms=elapsed))
            return {'error': str(e)}

    # -----------------------------------------------------------------
    # Tool implementations
    # -----------------------------------------------------------------
    def resolve_location(self, name: str, **_) -> dict:
        return self.geocoder.resolve(name)

    def list_minijepas(self, **_) -> dict:
        from minijepa_meta import load_all, render_for_prompt
        metas = load_all()
        # Compact dict-of-dicts keyed by modality
        return {
            'modalities': list(metas.keys()),
            'rendered':   render_for_prompt(metas),
            'raw':        {m: meta_to_dict(v) for m, v in metas.items()},
        }

    def get_minijepa_meta(self, modality: str, **_) -> dict:
        from minijepa_meta import load_one
        if modality not in MODALITY_LIST:
            return {'error': f'Unknown modality {modality}; valid: {MODALITY_LIST}'}
        return meta_to_dict(load_one(modality))

    def retrieve_minijepa(self, modality: str, lat: float, lon: float, k: int = 5, **_) -> list[dict]:
        if modality not in MODALITY_LIST:
            return [{'error': f'Unknown modality {modality}'}]
        idx = self._index(modality)
        nbrs = idx.query_by_geo(float(lat), float(lon), k=int(k))
        return self._enrich_with_labels(nbrs, modality_tag=modality)

    def retrieve_ae(self, lat: float, lon: float, k: int = 5, **_) -> list[dict]:
        idx = self._index('alphaearth')
        nbrs = idx.query_by_geo(float(lat), float(lon), k=int(k))
        return self._enrich_with_labels(nbrs, modality_tag='alphaearth')

    # -----------------------------------------------------------------
    # Internal: join neighbor patch_ids with the standard label panel
    # -----------------------------------------------------------------
    def _enrich_with_labels(self, nbrs: pd.DataFrame, modality_tag: str) -> list[dict]:
        if nbrs.empty:
            return []
        avail_cols = [c for c in LABEL_PANEL_COLS if c in self.labels.columns]
        keep_cols = ['patch_id'] + avail_cols
        joined = nbrs.merge(self.labels[keep_cols], on='patch_id', how='left')
        out = []
        for r in joined.itertuples():
            row = {
                'modality':     modality_tag,
                'patch_id':     getattr(r, 'patch_id'),
                'lon':          float(getattr(r, 'lon')),
                'lat':          float(getattr(r, 'lat')),
                'emb_distance': float(getattr(r, 'emb_distance')),
            }
            for c in avail_cols:
                v = getattr(r, c, None)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    row[c] = None
                else:
                    row[c] = float(v) if isinstance(v, (int, float, np.number)) else v
            out.append(row)
        return out

    # -----------------------------------------------------------------
    # Save call log
    # -----------------------------------------------------------------
    def reset_log(self):
        self.call_log = []

    def dump_log(self) -> list[dict]:
        return [tc.__dict__ for tc in self.call_log]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def meta_to_dict(meta) -> dict:
    """MinijepaMeta -> plain dict (avoid hard import dependency at top)."""
    return {
        'modality':         meta.modality,
        'display_name':     meta.display_name,
        'physics_tagline':  meta.physics_tagline,
        'global_pr':        meta.global_pr,
        'intrinsic_dim':    meta.intrinsic_dim,
        'mean_local_pr':    meta.mean_local_pr,
        'pcs_for_80_var':   meta.pcs_for_80_var,
        'top_dims':         meta.top_dims,
        'sp_rf_agree_pct':  meta.sp_rf_agree_pct,
        'joint_gains':      meta.joint_gains[:6],
        'cca_n_components_07': meta.cca_n_components_07,
    }


def _summarize_for_log(out: Any) -> Any:
    """Truncate large outputs in the call log."""
    if isinstance(out, list):
        return f'<list of {len(out)} items>'
    if isinstance(out, dict) and 'rendered' in out:
        return {k: ('<rendered text>' if k == 'rendered' else
                    f'<{len(v)} entries>' if isinstance(v, dict) else v)
                for k, v in out.items()}
    return out


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    eng = ToolEngine(lazy=True, verbose=True)

    print('--- resolve_location ---')
    print(eng.call_tool('resolve_location', name='Boise, Idaho'))

    print('\n--- list_minijepas (rendered preview) ---')
    res = eng.call_tool('list_minijepas')
    print(res['rendered'][:1200] + '\n...[truncated]')

    print('\n--- retrieve_minijepa (s1_sar near Boise) ---')
    out = eng.call_tool('retrieve_minijepa', modality='s1_sar',
                        lat=43.62, lon=-116.21, k=3)
    for r in out:
        print(' ', r)

    print('\n--- retrieve_ae (near Boise) ---')
    out = eng.call_tool('retrieve_ae', lat=43.62, lon=-116.21, k=3)
    for r in out:
        print(' ', r)

    print(f'\nCall log: {len(eng.call_log)} entries')

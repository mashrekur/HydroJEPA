"""
HydroJEPA data download pipeline (v5: 30m, GeoTIFF transport,
self-computed Köppen, three-pass label fetching).

Why three passes:
  The single-pass label image was a 71-band stack including a heavy
  Köppen subgraph (12 monthly PRISM aggregates, seasonality logic,
  sequential overrides). When sampled at 500 points, GEE silently
  ground for many minutes on first call. Splitting by compute weight
  isolates failures and gives per-chunk progress.

  Pass 1 (light):  SMAP, NLCD, SRTM, PRISM annuals, aridity (7 bands)
  Pass 2 (koppen): Köppen-Geiger alone (1 band, heavy graph)
  Pass 3 (ae):     AlphaEarth embedding (64 bands, lookup only)

Each pass:
  - Has its own chunk size tuned to compute weight
  - Saves an intermediate parquet
  - Resumes automatically on re-run
  - Prints per-chunk progress

Outputs (final):
  data/hydrojepa/patches/<patch_id>.tif
  data/hydrojepa/labels.parquet               # merged final
  data/hydrojepa/manifest.parquet
  data/hydrojepa/labels.light.parquet         # intermediate, kept for resume
  data/hydrojepa/labels.koppen.parquet        # intermediate, kept for resume
  data/hydrojepa/labels.ae.parquet            # intermediate, kept for resume
"""

import ee
import time
import numpy as np
import pandas as pd
import requests
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ID    = 'alpha-mash1'
TARGET_YEAR   = 2022
N_PATCHES     = 10_000
PATCH_SIZE    = 128
PIXEL_SCALE   = 30
N_WORKERS     = 8
SEED          = 42

S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

# Per-pass chunking, tuned to compute weight
CHUNK_LIGHT  = 500
CHUNK_KOPPEN = 250
CHUNK_AE     = 500
RETRIES      = 3
TILE_SCALE   = 4

OUT_DIR     = Path('data/hydrojepa')
PATCH_DIR   = OUT_DIR / 'patches'
LABELS_FILE = OUT_DIR / 'labels.parquet'
MANIFEST    = OUT_DIR / 'manifest.parquet'

LIGHT_PATH  = OUT_DIR / 'labels.light.parquet'
KOPPEN_PATH = OUT_DIR / 'labels.koppen.parquet'
AE_PATH     = OUT_DIR / 'labels.ae.parquet'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
def init_ee():
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
    logging.info(f'EE initialized with project {PROJECT_ID}')


# ---------------------------------------------------------------------------
# (1) Patch center sampling
# ---------------------------------------------------------------------------
def generate_patch_centers(n: int, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_draw = int(n * 3)
    lons = rng.uniform(-125.0, -66.5, size=n_draw)
    lats = rng.uniform(  24.5,  49.5, size=n_draw)

    land_mask = ee.Image('MODIS/006/MOD44W/2015_01_01').select('water_mask').Not()
    keep = []
    chunk = 5000
    for start in range(0, n_draw, chunk):
        pts = [ee.Feature(ee.Geometry.Point([float(lo), float(la)]))
               for lo, la in zip(lons[start:start + chunk], lats[start:start + chunk])]
        sampled = land_mask.sampleRegions(
            collection=ee.FeatureCollection(pts),
            scale=500, geometries=True
        ).getInfo()
        for f in sampled['features']:
            if f['properties'].get('water_mask') == 1:
                c = f['geometry']['coordinates']
                keep.append((c[0], c[1]))
            if len(keep) >= n:
                break
        if len(keep) >= n:
            break

    df = pd.DataFrame(keep, columns=['lon', 'lat'])
    df['patch_id'] = [f'p{i:06d}' for i in range(len(df))]
    df['year']     = TARGET_YEAR
    logging.info(f'Generated {len(df)} land patch centers over CONUS')
    return df


# ---------------------------------------------------------------------------
# (2) Sentinel-2 patch extraction
# ---------------------------------------------------------------------------
def s2_annual_composite(year: int) -> ee.Image:
    conus = ee.Geometry.Rectangle([-125.0, 24.5, -66.5, 49.5])

    def mask_s2(img):
        scl = img.select('SCL')
        bad = (scl.eq(0).Or(scl.eq(1)).Or(scl.eq(3))
               .Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)))
        return (img.updateMask(bad.Not())
                   .divide(10000)
                   .copyProperties(img, ['system:time_start']))

    coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterDate(f'{year}-01-01', f'{year}-12-31')
              .filterBounds(conus)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
              .map(mask_s2)
              .select(S2_BANDS))
    return coll.median().toFloat()


def fetch_patch_tif(image: ee.Image, lon: float, lat: float,
                    out_path: Path) -> str:
    if out_path.exists():
        return 'cached'

    half_m = (PATCH_SIZE / 2) * PIXEL_SCALE
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * np.cos(np.radians(lat)))
    rect = ee.Geometry.Rectangle([lon - dlon, lat - dlat,
                                  lon + dlon, lat + dlat])

    try:
        url = image.getDownloadURL({
            'region': rect,
            'dimensions': f'{PATCH_SIZE}x{PATCH_SIZE}',
            'format': 'GEO_TIFF',
            'bands': S2_BANDS,
        })
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        if len(r.content) < 30_000:
            return 'too_small'
        out_path.write_bytes(r.content)
        return 'ok'
    except Exception as e:
        logging.debug(f'{out_path.stem}: {e}')
        return 'failed'


# ---------------------------------------------------------------------------
# (3a) Three label image builders (one per pass)
# ---------------------------------------------------------------------------
def build_light_labels(year: int) -> ee.Image:
    """SMAP + NLCD + SRTM + PRISM annual + aridity. No monthly aggregation."""
    smap = (ee.ImageCollection('NASA/SMAP/SPL4SMGP/008')
              .filterDate(f'{year}-05-01', f'{year}-09-30')
              .select('sm_surface').mean()
              .rename('smap_sm'))

    nlcd = (ee.ImageCollection('USGS/NLCD_RELEASES/2021_REL/NLCD')
              .filter(ee.Filter.eq('system:index', '2021'))
              .first().select('landcover')
              .rename('nlcd_class'))

    srtm = ee.Image('USGS/SRTMGL1_003').rename('elevation')

    prism = (ee.ImageCollection('OREGONSTATE/PRISM/ANm')
               .filterDate(f'{year}-01-01', f'{year}-12-31'))
    ppt   = prism.select('ppt').sum().rename('prism_ppt_mm')
    tmean = prism.select('tmean').mean().rename('prism_tmean_c')

    aridity = ppt.divide(tmean.add(20).multiply(50)).rename('aridity_proxy')

    return smap.addBands([nlcd, srtm, ppt, tmean, aridity])


def compute_koppen(year: int) -> ee.Image:
    """5-class Köppen-Geiger from PRISM monthly normals (Beck et al. 2018)."""
    monthly = (ee.ImageCollection('OREGONSTATE/PRISM/ANm')
               .filterDate(f'{year}-01-01', f'{year}-12-31'))
    tmean_ic = monthly.select('tmean')
    ppt_ic   = monthly.select('ppt')

    tcold = tmean_ic.min()
    thot  = tmean_ic.max()
    mat   = tmean_ic.mean()
    p_ann = ppt_ic.sum()

    p_summer = ppt_ic.filter(ee.Filter.calendarRange(4, 9, 'month')).sum()
    p_winter = p_ann.subtract(p_summer)
    summer_dom = p_summer.divide(p_ann).gt(0.7)
    winter_dom = p_winter.divide(p_ann).gt(0.7)

    p_th = mat.multiply(2).add(14)
    p_th = p_th.where(summer_dom, mat.multiply(2).add(28))
    p_th = p_th.where(winter_dom.And(summer_dom.Not()), mat.multiply(2))
    arid = p_ann.lt(p_th.multiply(10))

    koppen = ee.Image(4)
    koppen = koppen.where(tcold.gt(-3), 3)
    koppen = koppen.where(tcold.gte(18), 1)
    koppen = koppen.where(arid, 2)
    koppen = koppen.where(thot.lt(10), 5)
    return koppen.rename('koppen').toInt()


def build_ae(year: int) -> ee.Image:
    """AlphaEarth annual embedding, 64 bands A00..A63."""
    return (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
              .filterDate(f'{year}-01-01', f'{year}-12-31')
              .first())


# ---------------------------------------------------------------------------
# (3b) Generic chunked sampler with retry + partial save + resume
# ---------------------------------------------------------------------------
def _sample_one_chunk(image: ee.Image, chunk: pd.DataFrame) -> list[dict]:
    feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                        {'patch_id': r.patch_id})
             for r in chunk.itertuples()]
    sampled = image.reduceRegions(
        collection=ee.FeatureCollection(feats),
        reducer=ee.Reducer.first(),
        scale=PIXEL_SCALE,
        tileScale=TILE_SCALE,
    ).getInfo()
    return [f['properties'] for f in sampled['features']]


def chunked_sample(image: ee.Image, df: pd.DataFrame, partial_path: Path,
                   chunk_size: int, name: str) -> pd.DataFrame:
    """Sample image bands at every patch centroid in df, with full robustness."""
    if partial_path.exists():
        prev = pd.read_parquet(partial_path)
        out_records = prev.to_dict('records')
        completed = set(prev.patch_id) if 'patch_id' in prev.columns else set()
        if len(completed) >= len(df):
            logging.info(f'  [{name}] already complete ({len(completed)} rows)')
            return prev
        logging.info(f'  [{name}] resuming: {len(completed)}/{len(df)} done')
    else:
        out_records = []
        completed = set()

    todo = df[~df.patch_id.isin(completed)].reset_index(drop=True)
    n_chunks = (len(todo) + chunk_size - 1) // chunk_size
    logging.info(f'  [{name}] {len(todo)} to fetch in {n_chunks} chunks of {chunk_size}')

    for ci, start in enumerate(range(0, len(todo), chunk_size), start=1):
        chunk = todo.iloc[start:start + chunk_size]
        t0 = time.time()
        success = False
        for attempt in range(1, RETRIES + 1):
            try:
                records = _sample_one_chunk(image, chunk)
                out_records.extend(records)
                success = True
                break
            except Exception as e:
                msg = str(e).split('\n', 1)[0][:100]
                if attempt < RETRIES:
                    wait = 5 * (2 ** (attempt - 1))
                    logging.info(f'  [{name}] chunk {ci}/{n_chunks} attempt '
                                 f'{attempt} failed ({msg}); retry in {wait}s')
                    time.sleep(wait)
                else:
                    logging.warning(f'  [{name}] chunk {ci}/{n_chunks} '
                                    f'failed permanently: {msg}')

        pd.DataFrame(out_records).to_parquet(partial_path)
        marker = '✓' if success else '✗'
        elapsed = time.time() - t0
        logging.info(f'  [{name}] {marker} chunk {ci}/{n_chunks} '
                     f'({elapsed:.1f}s, {len(out_records):,} total)')

    return pd.DataFrame(out_records)


# ---------------------------------------------------------------------------
# (3c) Three-pass label fetch
# ---------------------------------------------------------------------------
def fetch_labels_threepass(df: pd.DataFrame) -> pd.DataFrame:
    """Run three independent label passes, then merge by patch_id."""
    df = df[['patch_id', 'lon', 'lat']].reset_index(drop=True)

    logging.info('Pass 1/3: light labels (SMAP, NLCD, SRTM, PRISM, aridity)')
    light = chunked_sample(
        build_light_labels(TARGET_YEAR), df, LIGHT_PATH,
        chunk_size=CHUNK_LIGHT, name='light')

    logging.info('Pass 2/3: Köppen-Geiger (heavy compute, smaller chunks)')
    kop = chunked_sample(
        compute_koppen(TARGET_YEAR), df, KOPPEN_PATH,
        chunk_size=CHUNK_KOPPEN, name='koppen')

    logging.info('Pass 3/3: AlphaEarth reference embeddings (64 bands)')
    ae = chunked_sample(
        build_ae(TARGET_YEAR), df, AE_PATH,
        chunk_size=CHUNK_AE, name='ae')

    # Merge — outer joins so partial passes still produce a useful table
    merged = light.merge(kop, on='patch_id', how='outer') \
                  .merge(ae, on='patch_id', how='outer')
    return merged


# ---------------------------------------------------------------------------
# (4) Driver
# ---------------------------------------------------------------------------
def run():
    init_ee()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PATCH_DIR.mkdir(parents=True, exist_ok=True)

    if MANIFEST.exists():
        manifest = pd.read_parquet(MANIFEST)
        logging.info(f'Resuming with {len(manifest)} existing manifest entries')
    else:
        manifest = generate_patch_centers(N_PATCHES)
        manifest['status'] = 'pending'
        manifest.to_parquet(MANIFEST)

    # ---- (A) S2 patches ----
    todo = manifest[manifest.status.isin(['pending', 'failed'])].copy()
    if len(todo) > 0:
        s2 = s2_annual_composite(TARGET_YEAR)
        logging.info(f'Pulling {len(todo)} S2 patches at {PIXEL_SCALE}m '
                     f'with {N_WORKERS} workers')

        def worker(row):
            path = PATCH_DIR / f'{row.patch_id}.tif'
            return row.patch_id, fetch_patch_tif(s2, row.lon, row.lat, path)

        done = 0
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {ex.submit(worker, r): r.patch_id for r in todo.itertuples()}
            for fut in as_completed(futures):
                pid, status = fut.result()
                manifest.loc[manifest.patch_id == pid, 'status'] = status
                done += 1
                if done % 200 == 0:
                    manifest.to_parquet(MANIFEST)
                    ok = (manifest.status == 'ok').sum()
                    logging.info(f'  {done}/{len(todo)} done | total ok: {ok}')
        manifest.to_parquet(MANIFEST)
    else:
        logging.info('S2 phase already complete, skipping to labels')

    ok_df = manifest[manifest.status == 'ok'].copy()
    logging.info(f'S2 phase done: {len(ok_df)} usable patches')

    # ---- (B + C) Three-pass labels and AE embeddings ----
    labels = fetch_labels_threepass(ok_df[['patch_id', 'lon', 'lat']])
    labels = ok_df[['patch_id', 'lon', 'lat', 'year']].merge(
        labels, on='patch_id', how='left')
    labels.to_parquet(LABELS_FILE)
    logging.info(f'Labels saved: {LABELS_FILE} '
                 f'({len(labels)} rows, {labels.shape[1]} cols)')

    # ---- Summary ----
    n_tifs = len(list(PATCH_DIR.glob('*.tif')))
    total_mb = sum(p.stat().st_size for p in PATCH_DIR.glob('*.tif')) / 1e6
    logging.info('')
    logging.info('=' * 60)
    logging.info('HydroJEPA download complete')
    logging.info(f'  Patches (tif):   {n_tifs}')
    logging.info(f'  Disk footprint:  {total_mb:.0f} MB')
    logging.info(f'  Label rows:      {len(labels)}')
    logging.info(f'  AE bands:        '
                 f'{sum(c.startswith("A") and c[1:].isdigit() for c in labels.columns)}')
    logging.info('=' * 60)


if __name__ == '__main__':
    run()

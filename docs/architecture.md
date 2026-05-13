# System Architecture

This document describes how the five Mini-JEPAs, the per-modality reference
cards, the FAISS indexes, and the routing agent fit together. It complements
Section 2 of the paper but is meant to be read alongside the code.

## 1. The fleet

Each Mini-JEPA is an identical ViT-S encoder (12 layers, 6 heads, hidden dim
384, 22M parameters) followed by a linear projection to a 64-dimensional
output. Patches are 128x128 pixels at 30 m resolution; the encoder tokenizes
each patch into 64 tokens of 16x16 px. Training uses I-JEPA (predict masked
target latents from visible context tokens) plus a VICReg variance and
covariance regularizer that prevents representation collapse. 100 epochs,
batch size 64, learning rate 1.5e-4.

The only systematic difference between the five Mini-JEPAs is the satellite
product the encoder sees during pretraining:

| Mini-JEPA      | Source                                  | Channels |
|----------------|-----------------------------------------|----------|
| S2-Optical     | Sentinel-2 SR annual median             | 10       |
| S1-SAR         | Sentinel-1 GRD VV+VH annual median      | 2        |
| MODIS-Thermal  | MODIS LST day+night annual median       | 2        |
| S2-Phenology   | Sentinel-2 SR, 4 quarterly composites   | 40       |
| Topo-Soil      | SRTM elevation + slope + aspect + soil  | 6        |

All five share the same 9,704 patch centers, so any downstream geometric or
predictive difference between Mini-JEPAs is sensor-driven rather than
data-coverage-driven.

## 2. Three-axis per-modality characterization

For each Mini-JEPA, the same three evaluation scripts (`scripts/11`, `12`,
`13`) produce:

1. **Dimension-level physical interpretability** (`scripts/11_*`).
   Spearman correlations between every embedding dimension and every
   environmental variable, Random Forest with permutation importance, and
   spatial-block cross-validated R² per variable. Outputs the dimension
   dictionary that the routing agent later reads.

2. **Manifold geometry** (`scripts/12_*`). Global participation ratio,
   maximum-likelihood intrinsic dimensionality, local PCA at 2,000 probe
   points (the local n_80 statistic), and dominant-dimension maps across
   CONUS. Outputs the geometry summary used by the agent.

3. **Complementarity with AlphaEarth** (`scripts/13_*`). Canonical
   correlation analysis between the Mini-JEPA and AlphaEarth, plus joint
   predictive gain (R² for AE alone, Mini-JEPA alone, and concatenated).
   Outputs the joint-gain table used to score complementarity.

`scripts/6_1_minijepa_evaluation.py` is the driver that runs all three for
every modality. Outputs land under
`reports/minijepas/<modality>/{interpretability,manifold_geometry,complementarity}/`.

## 3. Agent reference cards

`agent/minijepa_meta.py` reads the per-modality reports and assembles a
compact reference card per Mini-JEPA containing:

- The model's dimension dictionary (top entries — which dim encodes which
  variable, derived from interpretability).
- Geometric headline numbers (global PR, intrinsic dim, mean local PR).
- A one-sentence statement of the sensor's physical signal.
- A short table of per-variable cross-validated R² scores.
- The joint-with-AE gain for each variable.

Each card is short by design — it has to fit inside the router LLM's system
prompt alongside the four others.

## 4. FAISS indexes

`agent/build_minijepa_indices.py` writes one FAISS index per Mini-JEPA, plus
one for AlphaEarth. Each index stores the 9,704 mean-pooled 64-d patch
embeddings. About 6 MB per index, 36 MB total on disk.

The indexes use the standard L2 metric. We tried inner-product on
L2-normalized vectors and observed nearly identical retrieval rankings, so
L2 is the default.

## 5. Tools

`agent/minijepa_tools.py` exposes the following tools, each callable by the
agent from a JSON tool-call block:

- `resolve_location(name)` — geocode a CONUS location string to (lat, lon).
- `list_minijepas()` — return all five reference cards as rendered text.
- `get_minijepa_meta(modality)` — return one reference card in detail.
- `retrieve_minijepa(modality, lat, lon, k)` — k-NN against the per-modality
  FAISS index, returning ranked patches with the standard environmental
  label panel and explicit modality provenance.
- `retrieve_ae(lat, lon, k)` — same shape as `retrieve_minijepa` but in the
  AlphaEarth embedding space.

Each retrieval starts by finding the nearest patch geographically to the
query (lat, lon), then doing k-NN in embedding space from that anchor. This
keeps the retrieval entirely in embedding space without re-encoding a
freshly-downloaded GEE patch at every query.

## 6. The router (not in this repository)

The router implementation (`agent/minijepa_router.py`) and the LLM-API client
the paper used (Dartmouth Chat) are **not in this public repository**, because
they depend on an institutional API key that no one outside Dartmouth has. The
five retrieval conditions reported in Section 3.4 of the paper are:

- **C1 `llm_only`** — LLM with no retrieval. Pure parametric baseline.
- **C2 `ae_only`** — LLM + AlphaEarth retrieval. Generalist baseline.
- **C3 `single_fixed`** — LLM + a single fixed Mini-JEPA. Ablates routing.
- **C4 `multi_select`** — LLM + agent-selected Mini-JEPA subset, no AE.
- **C5 `dual_rag`** — LLM + AE + agent-selected Mini-JEPA subset.

For C4 and C5, the router operates in three phases:

1. **Routing** — the LLM reads `list_minijepas()` and emits a JSON tool-call
   plan specifying which Mini-JEPAs to query.
2. **Retrieval** — the engine runs the selected per-modality retrievals
   (and AE retrieval for C5) in parallel against the FAISS indexes built by
   `agent/build_minijepa_indices.py`.
3. **Synthesis** — a second LLM call consumes the ranked retrievals (tagged
   by provenance) and writes the final answer.

The tool layer (`agent/minijepa_tools.py`) and the question set
(`agent/minijepa_query_sets.py`) are public and portable. To reproduce the
agent evaluation against your own LLM provider, write a thin client that
wraps `agent/minijepa_tools.py` and emits the same JSONL response and judge
score schemas that `agent/minijepa_eval_summary.py` and
`agent/minijepa_eval_diagnostics.py` consume. The full set of response and
judge-score JSONLs from the paper's runs are on Zenodo.

## 7. Question set and scoring

`agent/minijepa_query_sets.py` contains the curated 40-question evaluation
set described in Section 2.5 of the paper. Categories:

- **single_modality** — physical signal sits cleanly in one sensor.
- **multi_modality** — requires combining sensors.
- **sar_favorable** — conditions that defeat optical (clouds, surface water
  under vegetation).
- **ae_favorable** — broad characterization questions a generalist handles well.

Each question is paired with an expected-modality set used to score routing
accuracy. The LLM-as-Judge scoring uses two independent judges (paper used
Claude Haiku 4.5 and GPT-OSS-120B) on five rubric items: grounding,
scientific accuracy, completeness, coherence, and practical utility. The
weighted score is the mean across the five.

Sample question/response/judge rows in the format produced by the full
evaluation are in `sample_outputs/evaluation_responses_sample.csv`.

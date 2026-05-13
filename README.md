# HydroJEPA

**Mini-JEPA Foundation Model Fleet Enables Agentic Hydrologic Intelligence**
*Mashrekur Rahman — Dartmouth College*

Code accompanying the manuscript submitted to *Remote Sensing Applications: Society and Environment*. The work introduces a fleet of five small, sensor-specialized Joint Embedding Predictive Architecture (JEPA) foundation models pretrained on Google Earth Engine satellite products, paired with a routing LLM agent that consults each model when its sensor physics matches the hydrologic question.

The trained checkpoints, the 9,704-patch corpus, per-modality FAISS indices, and the full LLM evaluation outputs (responses, judge scores, routing logs) are archived on Zenodo: **[DOI to be inserted on acceptance]**.

---

## What's in this repository

```
HydroJEPA/
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── README.md                              you are here
│
├── scripts/                               data acquisition, pretraining, evaluation
│   ├── 1_hydrojepa_download.py
│   ├── 5_hydrojepa_pretrain.py
│   ├── 6_1_minijepa_evaluation.py
│   ├── 11_hydrojepa_interpretability.py
│   ├── 12_hydrojepa_manifold_geometry.py
│   └── 13_hydrojepa_ae_complementarity.py
│
├── agent/                                 agentic system (no LLM-API runner)
│   ├── build_minijepa_indices.py
│   ├── minijepa_meta.py
│   ├── minijepa_tools.py
│   ├── minijepa_query_sets.py
│   ├── minijepa_eval_summary.py
│   ├── minijepa_eval_diagnostics.py
│   └── build_response_spreadsheet.py
│
├── figures/                               paper-figure code
│   ├── _style.py                          shared palette, fonts, helpers
│   ├── fig1_overview.py
│   ├── fig2_per_modality_skill.py
│   ├── fig3_manifold_geometry.py
│   ├── fig4_manifold_portrait.py
│   ├── fig5_complementarity.py
│   ├── fig6_agent_architecture.py
│   ├── fig7_experimental_results.py
│   └── output/                            populated when figures are run
│
├── docs/
│   └── architecture.md                    code-side companion to paper §2
│
└── sample_outputs/                        small illustrative LLM-eval slice
    ├── README.md
    └── evaluation_responses_sample.csv
```

A note on what's excluded: the LLM-evaluation runner (`minijepa_eval.py`), the router (`minijepa_router.py`), and the Dartmouth Chat API client (`probe_dartmouth_catalog.py`) live in the private working tree because they depend on an institutional API key (Dartmouth Chat) and would not run for anyone else without modification. The post-hoc analysis scripts that consume their JSONL outputs (`minijepa_eval_summary.py`, `minijepa_eval_diagnostics.py`, `build_response_spreadsheet.py`) are included; the full output JSONLs are on Zenodo.

---

## Pipeline overview

The work has four stages:

1. **Data** — sample 10,000 patch centers across CONUS and pull five satellite products at each center (Sentinel-2 optical, Sentinel-1 SAR, MODIS LST, Sentinel-2 quarterly phenology composites, SRTM + SoilGrids). After QC, 9,704 patches survive as the working corpus.
2. **Pretrain** — train five Mini-JEPAs on those products, one per sensor. Same ViT-S architecture (22M parameters), same I-JEPA + VICReg recipe, same 64-d output. Only the input sensor differs.
3. **Per-modality evaluation** — for each Mini-JEPA, compute dimension-level physical interpretability, manifold geometry, and complementarity with AlphaEarth. Each produces a per-modality reference card the routing agent reads.
4. **Agent + LLM evaluation** — build per-modality FAISS indices, write reference cards, route hydrologic questions to the right Mini-JEPA(s), retrieve, synthesize, and score the resulting answers with a cross-model LLM-as-Judge panel.

The directory layout follows that pipeline. Scripts in `scripts/` cover stages 1-3; scripts in `agent/` cover stage 4 (without the API-dependent runner).

---

## Quick start

### 1. Environment

Tested with Python 3.10-3.12 on Ubuntu 22.04 (WSL2) and Windows 11 with an NVIDIA RTX 5090. CPU-only execution is supported but slow for pretraining.

```bash
git clone https://github.com/mashrekur/HydroJEPA.git
cd HydroJEPA

python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

`cartopy`, `faiss-cpu`, and `cairosvg` occasionally need system-level dependencies — see the upstream install notes if pip complains.

### 2. Data acquisition

You need a Google Earth Engine account and a registered project. Authenticate once:

```bash
earthengine authenticate
```

Then run the downloader. The default pulls all five sensor products at the same 9,704 patch centers used in the paper. Expect several hours of wall time depending on GEE quota and network.

```bash
python scripts/1_hydrojepa_download.py
```

Output: `data/hydrojepa/patches/<modality>/*.tif` per-patch GeoTIFFs, plus a single `data/hydrojepa/labels.parquet` of co-located environmental labels (SMAP, NLCD, PRISM, SRTM, Köppen, AlphaEarth embedding).

### 3. Pretrain the fleet

One Mini-JEPA per call. About 2-4 hours per model on an RTX 5090.

```bash
python scripts/5_hydrojepa_pretrain.py --modality s2_optical
python scripts/5_hydrojepa_pretrain.py --modality s1_sar
python scripts/5_hydrojepa_pretrain.py --modality modis_lst
python scripts/5_hydrojepa_pretrain.py --modality s2_phenology
python scripts/5_hydrojepa_pretrain.py --modality topo_soil
```

Checkpoints land at `checkpoints/<modality>_hydrojepa_best.pt`.

### 4. Per-modality evaluation

This drives the three evaluation axes for every Mini-JEPA. The orchestrator wraps scripts 11, 12, 13 and runs them for each modality in turn.

```bash
python scripts/6_1_minijepa_evaluation.py
```

Or each script standalone, against the S2-Optical baseline checkpoint:

```bash
python scripts/11_hydrojepa_interpretability.py        # Spearman + RF + spatial CV
python scripts/12_hydrojepa_manifold_geometry.py       # PR, intrinsic dim, local PCA
python scripts/13_hydrojepa_ae_complementarity.py      # CCA + joint predictive gain
```

Outputs land at `reports/minijepas/<modality>/{interpretability,manifold_geometry,complementarity}/` as CSV summaries and JSON manifests, plus a `geometry_summary.json` and a `dimension_dictionary.csv` per model.

### 5. Build the per-modality FAISS indices

The agent retrieves over per-modality patch embeddings. Each Mini-JEPA encodes all 9,704 patches once; the resulting 64-d mean-pooled vectors go into a FAISS L2 index.

```bash
python agent/build_minijepa_indices.py
```

Output: `data/hydrojepa/faiss/<modality>.index` for each Mini-JEPA, plus an AlphaEarth index for cross-comparison.

### 6. Generate the question set

```bash
python agent/minijepa_query_sets.py
```

Writes `data/hydrojepa/minijepa_eval/qa_set.jsonl` with 40 questions stratified into four categories: `single_modality`, `multi_modality`, `sar_favorable`, `ae_favorable`. Each question carries an `expected_modalities` set used in the routing-accuracy evaluation.

### 7. Run the LLM evaluation

This step uses the Dartmouth Chat API and is **not** in this public repository. It produces three JSONL files per run (one per retrieval condition: `ae_only`, `multi_select`, `dual_rag`) plus a `judge_scores.jsonl` and `routing_log.jsonl`. The full outputs are on Zenodo.

### 8. Summarize and diagnose

These scripts read the JSONL outputs from the previous step (or from the Zenodo archive) and produce the tables and figures reported in the paper.

```bash
python agent/minijepa_eval_summary.py
python agent/minijepa_eval_diagnostics.py
python agent/build_response_spreadsheet.py
```

Outputs: `summary.csv`, `per_category.csv`, `per_criterion.csv`, `routing_behavior.csv`, and `evaluation_responses_for_review.csv` under each run directory.

### 9. Build the paper figures

```bash
python figures/fig1_overview.py
python figures/fig2_per_modality_skill.py
python figures/fig3_manifold_geometry.py
python figures/fig4_manifold_portrait.py
python figures/fig5_complementarity.py
python figures/fig6_agent_architecture.py
python figures/fig7_experimental_results.py
```

Figures are written to `figures/output/` as `.pdf` + `.png` pairs (Figure 6 also writes `.svg`).

---

## File-by-file reference

### `scripts/1_hydrojepa_download.py`
Pulls the 9,704 patch corpus from Google Earth Engine. Uses a hybrid raster + per-point fetcher for efficiency: each sensor product is acquired as 128×128 patches at 30 m resolution centered on the same sampled CONUS points across all five modalities, ensuring sensor-to-sensor co-location. Also pulls co-located environmental labels (SMAP, NLCD, PRISM precipitation/temperature, SRTM elevation, Köppen class, AlphaEarth V1 embedding) into a single `labels.parquet`.

### `scripts/5_hydrojepa_pretrain.py`
The Mini-JEPA pretraining loop. Implements I-JEPA with a VICReg regularizer over a ViT-S backbone (12 transformer layers, 6 attention heads, hidden dim 384) with a linear projection to 64-d output. Context encoder sees 60% of tokens, target encoder is an EMA of the context encoder, and a small predictor maps context latents to target latents at masked positions. 100 epochs, batch size 64, learning rate 1.5e-4. The `--modality` flag selects which sensor product the encoder is trained on; all other hyperparameters are held fixed across the fleet.

### `scripts/6_1_minijepa_evaluation.py`
Driver that runs the three-axis per-modality evaluation (scripts 11, 12, 13) across all five Mini-JEPAs and consolidates outputs under `reports/minijepas/<modality>/`. Useful when you've just finished pretraining and want every characterization computed in one shot.

### `scripts/11_hydrojepa_interpretability.py`
Dimension-level physical interpretability for one Mini-JEPA. Computes per-dimension Spearman rank correlations against every environmental variable; trains Random Forest regressors that map the full 64-d embedding to each variable and records permutation importance per dimension; uses spatial-block cross-validation to guard against autocorrelation-inflated R². Writes a `dimension_dictionary.csv` that ranks every dimension by what it encodes most strongly.

### `scripts/12_hydrojepa_manifold_geometry.py`
Manifold geometry for one Mini-JEPA. Computes the global participation ratio from the embedding covariance eigenvalues; the maximum-likelihood intrinsic dimensionality from k-nearest-neighbor distances; and 2,000 local PCAs across CONUS probe points. Writes `geometry_summary.json`, `hydrojepa_global_covariance.csv`, `hydrojepa_local_pca.csv`, and `hydrojepa_multiscale.csv`.

### `scripts/13_hydrojepa_ae_complementarity.py`
Tests whether a Mini-JEPA carries information AlphaEarth does not already represent. Fits canonical correlation analysis between the Mini-JEPA's embeddings and AlphaEarth's embeddings; trains three Random Forests per environmental variable (AlphaEarth alone, Mini-JEPA alone, joint) and records the gain of the joint model over the better single source. Writes `cca_loadings.csv`, `joint_predictive_gain.csv`, and `primary_dim_comparison.csv`.

### `agent/build_minijepa_indices.py`
Encodes the 9,704-patch corpus once per Mini-JEPA (and once with AlphaEarth) into mean-pooled 64-d vectors and writes a FAISS L2 index per modality. The router reads these at inference time. Total wall time about 30-60 minutes on an RTX 5090.

### `agent/minijepa_meta.py`
Reads `reports/minijepas/<modality>/` and constructs a compact per-modality reference card — dimension dictionary, geometric profile, sensor physics statement, per-variable cross-validated R² table — short enough to fit in the router LLM's prompt. The agent reads these meta-summaries to decide which Mini-JEPAs are appropriate for a given question.

### `agent/minijepa_tools.py`
Tool engine exposed to the router and synthesis LLMs. Provides `resolve_location` (geocoding to lat/lon), `list_minijepas` (catalog with brief sensor physics), `get_minijepa_meta` (full per-modality reference card), `retrieve_minijepa(modality, lat, lon, k)` (kNN retrieval against a per-modality FAISS index), and `retrieve_ae(lat, lon, k)` (parallel retrieval against the AlphaEarth index). Modality-aware so the LLM cannot ask for an embedding from the wrong tool.

### `agent/minijepa_query_sets.py`
Generates the 40-question evaluation set, stratified into four categories that stress different parts of the fleet. Each question carries a hand-curated `expected_modalities` set used for routing-accuracy evaluation. The dataclass model is reused by the runner and the analysis scripts.

### `agent/minijepa_eval_summary.py`
Reads a run directory (`data/hydrojepa/minijepa_eval/runs/<system_model>__<run_tag>/`) and produces the headline summary CSVs: `summary.csv` (aggregate weighted scores by condition), `per_category.csv` (effect sizes by question category), `per_criterion.csv` (mean rubric scores G/A/C/H/U), and `routing_behavior.csv` (modality-selection frequencies per category). Also renders `fig_summary.png` per run.

### `agent/minijepa_eval_diagnostics.py`
Deeper post-hoc diagnostics: per-category significance tables, per-judge calibration, side-by-side response excerpts for qualitative reading, score histograms per condition per judge, and identification of (qid, condition, judge) cells that are missing scores. Outputs go to `<run_dir>/diagnostics/`.

### `agent/build_response_spreadsheet.py`
Merges JSONL responses and judge scores into a single reviewer-facing CSV with one row per (question, condition, system_model). Used to generate the per-paper review CSVs and the Zenodo deposit.

### `figures/_style.py`
Shared figure infrastructure: locked modality palette and order, environmental-variable labels, helper functions for saving PDFs + PNGs, project-root resolution (honoring `HYDROJEPA_ROOT` env var), figure-output directory management, and CONUS basemap helpers. Every figure script imports from here so the visual register stays consistent across the paper.

### `figures/fig1_overview.py`
Figure 1 — schematic of the pipeline. Three panels: patch centers across CONUS with imagery sources, the I-JEPA pretraining loop, and the resulting fleet of five sensor-specialized encoders with each modality's "Best At" R² annotated.

### `figures/fig2_per_modality_skill.py`
Figure 2 — per-modality predictive skill. Five hex maps showing where each Mini-JEPA's RF predictions are accurate (darker hexes = higher within-region R²), plus a 7×5 heatmap of cross-validated R² per (variable, Mini-JEPA). The diagonal pattern — each Mini-JEPA's best variable matches its sensor physics — is the figure's main read.

### `figures/fig3_manifold_geometry.py`
Figure 3 — manifold geometry per Mini-JEPA. Five CONUS maps colored by local participation ratio (per-modality percentile stretch) plus a scatter of (global PR, mean local PR) showing MODIS-Thermal in the high-global / low-local corner and S1-SAR in the opposite corner.

### `figures/fig4_manifold_portrait.py`
Figure 4 — per-Mini-JEPA manifold portrait. Five-row × three-column grid: cumulative variance spectrum (left), local n₈₀ histogram with a shared x-axis (middle), CONUS map of dominant local dimension (right). Directly visualizes the global vs local dimensionality contrast in Section 3.2.

### `figures/fig5_complementarity.py`
Figure 5 — Mini-JEPA complementarity with AlphaEarth. Top panel: per-variable horizontal bars (AlphaEarth alone vs best Mini-JEPA vs joint, with Δ values on the right). Bottom panel: 7×5 ΔR² matrix highlighting Topo-Soil and S2-Phenology as the non-redundant specialists.

### `figures/fig6_agent_architecture.py`
Figure 6 — hand-authored SVG of the agentic system. Four stages from query through routing to per-modality FAISS retrieval to synthesis. Single page, top-to-bottom, full width. Stage labels in a left gutter, FAISS index boxes carry differentiated per-modality sensor physics descriptions and "Best At" R² values. Exports both SVG (vector, paper-ready) and PNG via cairosvg.

### `figures/fig7_experimental_results.py`
Figure 7 — agent evaluation. Three panels: effect size by question category (Cohen's d, with significance stars), inter-judge calibration across the three condition contrasts, and the per-question Δ histogram (dual_rag - ae_only). The single_modality bar at d=1.10 is the figure's headline.

### `sample_outputs/`
A small illustrative slice of the LLM-as-Judge outputs — one question per category × all three conditions, with judge scores. See `sample_outputs/README.md` for the schema. The full evaluation outputs (40 questions × 3 conditions × 2 judges = 240 scored records per run, across two system-model runs) are on Zenodo.

---

## Reproducibility notes

- **Patch centers**: seeded with the same random state used in Rahman 2026 and Rahman, Barrett, Last 2026. Reusing that state regenerates the exact same 10,000 candidate patches; the 9,704 working corpus is the subset that survives QC.
- **Environmental variables**: pulled from public datasets only — SMAP L3 surface soil moisture, PRISM 800 m monthly climate normals, SRTM 30 m DEM, NLCD 2019, Köppen-Geiger 1 km from Beck et al. 2018, and AlphaEarth Foundation V1 annual embeddings (Brown et al. 2025). Source IDs are in the paper's Table 1.
- **Mini-JEPA hyperparameters**: ViT-S, 12 layers, 6 heads, hidden 384, output 64-d, I-JEPA + VICReg, 100 epochs, batch 64, lr 1.5e-4. Held fixed across the fleet; only the input sensor changes.
- **Compute**: a single NVIDIA RTX 5090 workstation suffices for everything in this repository. Total pretraining wall time for the five Mini-JEPAs is roughly 10-20 hours.

---

## Citation

If you use this code or the trained checkpoints, please cite the paper:

> Rahman, M. (2026). Mini-JEPA Foundation Model Fleet Enables Agentic Hydrologic Intelligence. *Remote Sensing Applications: Society and Environment*.

A `CITATION.cff` is provided at the repository root for citation-management tools.

The two prior AlphaEarth studies that this paper extends:

> Rahman, M. (2026). Physically interpretable AlphaEarth foundation model embeddings enable LLM-based land surface intelligence. *arXiv:2602.XXXXX*.

> Rahman, M., Barrett, S. J., Last, C. (2026). Characterizing AlphaEarth embedding geometry for agentic environmental reasoning. *arXiv:2604.XXXXX*.

---

## License

MIT — see `LICENSE`.

---

## Contact

Mashrekur Rahman — mashrekur.rahman@dartmouth.edu

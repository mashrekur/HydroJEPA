# Sample LLM evaluation outputs

This directory contains a small illustrative slice of the LLM-as-Judge
evaluation reported in Section 3.4 of the paper.

The full evaluation outputs (40 questions × three retrieval conditions,
scored independently by two judge models — Claude Haiku 4.5 and
GPT-OSS-120B) are archived on the Zenodo deposit accompanying the paper.
What's in this folder is a **representative sample** of the format,
showing one question from each of the four question categories that
appear in the paper:

| Category | Description | Example QID |
|---|---|---|
| `single_modality` | Signal cleanly in one sensor (e.g. land cover from S2-Optical, LST regime from MODIS-Thermal) | `SM_01`, `SM_04` |
| `sar_favorable` | Conditions that defeat optical imagery (cloud cover, surface water under vegetation) | `SAR_03` |
| `multi_modality` | Requires combining signals from multiple sensors (e.g. irrigation detection from optical + phenology + SAR) | `MM_02` |
| `ae_favorable` | Broad-characterization questions where a planetary-scale generalist is expected to perform well | `AE_07` |

## File format

`evaluation_responses_sample.csv` — one row per (question, retrieval condition),
with the following columns:

| Column | Meaning |
|---|---|
| `qid` | Question identifier, prefix indicates category (`SM_`, `MM_`, `SAR_`, `AE_`) |
| `category`, `sub_type` | Category labels matching the paper's Section 2.5.1 stratification |
| `question` | The hydrologic question posed to the system |
| `latitude`, `longitude` | Geographic anchor for the question (lat/lon of interest) |
| `expected_modalities` | Modalities a domain expert would expect the router to select (for routing-accuracy evaluation) |
| `condition` | Retrieval condition: `ae_only` (AlphaEarth only), `multi_select` (routed Mini-JEPAs only), `dual_rag` (Mini-JEPAs + AlphaEarth) |
| `system_model` | The synthesis LLM used to generate the answer |
| `judge_model` | The LLM judge that scored the response |
| `selected_modalities` | Modalities the router actually selected for this question + condition |
| `response` | The synthesis LLM's final answer |
| `G` | Grounding (1-5) — does the answer cite the retrievals it claims to use? |
| `A` | Scientific accuracy (1-5) — are the claims defensible against domain knowledge? |
| `C` | Completeness (1-5) — does the answer address all parts of the question? |
| `H` | Coherence (1-5) — is the answer internally consistent? |
| `U` | Practical utility (1-5) — would a hydrologist find this answer useful? |
| `weighted_score` | Weighted average: 0.25·G + 0.25·A + 0.20·C + 0.15·H + 0.15·U |

The weighted-score formula mirrors the rubric used in both prior
AlphaEarth papers (Rahman 2026; Rahman, Barrett, Last 2026) for
cross-paper comparability.

## How the full set is generated

See `agent/build_response_spreadsheet.py` in the repo root. The full
pipeline that produces the underlying JSONL records is documented in
the main README under "Section 4 — Agentic system and LLM evaluation".
Note that running the full evaluation pipeline requires a Dartmouth
Chat API key and is therefore **not** in this public repository; only
the post-hoc analysis scripts that consume the JSONL outputs are.

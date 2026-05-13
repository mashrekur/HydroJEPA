"""
build_response_spreadsheet.py
Merge LLM evaluation JSON checkpoints into a reviewer-facing CSV with one
row per (query, condition, system_model) and the five rubric scores from
the judge.

Run from the repo root after the eval runner has populated
`data/hydrojepa/minijepa_eval/runs/`.
"""

import json
import pandas as pd
from pathlib import Path

SONNET_DIR = Path("logs/ablation_named")
OPUS_DIR = Path("logs/ablation_opus")
CONDITIONS = ["full", "no_geometric", "no_confidence", "llm_only", "paper1_deterministic"]

def get_latest_json(folder: Path) -> Path | None:
    """Return the most recent JSON checkpoint in a folder."""
    jsons = sorted(folder.glob("eval_*.json"))
    return jsons[-1] if jsons else None

def load_records(json_path: Path) -> list[dict]:
    """Load records from a checkpoint JSON."""
    with open(json_path) as f:
        return json.load(f)

rows = []
seen = set()

# --- Sonnet ablation ---
for cond in CONDITIONS:
    folder = SONNET_DIR / cond
    if not folder.exists():
        print(f"  Skipping {cond}: folder not found")
        continue
    jp = get_latest_json(folder)
    if jp is None:
        print(f"  Skipping {cond}: no JSON files")
        continue
    records = load_records(jp)
    print(f"  {cond}: {len(records)} records from {jp.name}")
    for r in records:
        key = (r["query_id"], r["condition"], r.get("system_model", "claude-sonnet-4.5"))
        if key in seen:
            continue
        seen.add(key)
        scores = r.get("scores", {})
        rows.append({
            "query_id": r["query_id"],
            "tier": r["tier"],
            "query_text": r.get("query_text", ""),
            "intent": r.get("intent", ""),
            "region": r.get("region", ""),
            "condition": r["condition"],
            "system_model": r.get("system_model", "claude-sonnet-4.5"),
            "judge_model": r.get("judge_model", "gemma-3-27b"),
            "response_text": r.get("response_text", ""),
            "tool_calls": json.dumps(r.get("tool_calls", []))[:500],
            "n_tool_calls": len(r.get("tool_calls", [])),
            "grounding": scores.get("grounding", ""),
            "scientific_accuracy": scores.get("scientific_accuracy", ""),
            "completeness": scores.get("completeness", ""),
            "coherence": scores.get("coherence", ""),
            "practical_utility": scores.get("practical_utility", ""),
            "weighted_score": scores.get("weighted_score", ""),
            "geometric_grounding": scores.get("geometric_grounding", ""),
            "response_time_ms": r.get("response_time_ms", ""),
        })

# --- Also check full_rerun if it exists ---
rerun = SONNET_DIR / "full_rerun"
if rerun.exists():
    jp = get_latest_json(rerun)
    if jp:
        records = load_records(jp)
        print(f"  full_rerun: {len(records)} records from {jp.name}")
        for r in records:
            key = (r["query_id"], r["condition"], r.get("system_model", "claude-sonnet-4.5"))
            if key not in seen:
                seen.add(key)
                scores = r.get("scores", {})
                rows.append({
                    "query_id": r["query_id"],
                    "tier": r["tier"],
                    "query_text": r.get("query_text", ""),
                    "intent": r.get("intent", ""),
                    "region": r.get("region", ""),
                    "condition": r["condition"],
                    "system_model": r.get("system_model", "claude-sonnet-4.5"),
                    "judge_model": r.get("judge_model", "gemma-3-27b"),
                    "response_text": r.get("response_text", ""),
                    "tool_calls": json.dumps(r.get("tool_calls", []))[:500],
                    "n_tool_calls": len(r.get("tool_calls", [])),
                    "grounding": scores.get("grounding", ""),
                    "scientific_accuracy": scores.get("scientific_accuracy", ""),
                    "completeness": scores.get("completeness", ""),
                    "coherence": scores.get("coherence", ""),
                    "practical_utility": scores.get("practical_utility", ""),
                    "weighted_score": scores.get("weighted_score", ""),
                    "geometric_grounding": scores.get("geometric_grounding", ""),
                    "response_time_ms": r.get("response_time_ms", ""),
                })

# --- Opus benchmark ---
for cond_dir in OPUS_DIR.iterdir():
    if not cond_dir.is_dir():
        continue
    jp = get_latest_json(cond_dir)
    if jp is None:
        continue
    records = load_records(jp)
    print(f"  opus/{cond_dir.name}: {len(records)} records from {jp.name}")
    for r in records:
        key = (r["query_id"], r["condition"], r.get("system_model", "claude-opus-4.6"))
        if key in seen:
            continue
        seen.add(key)
        scores = r.get("scores", {})
        rows.append({
            "query_id": r["query_id"],
            "tier": r["tier"],
            "query_text": r.get("query_text", ""),
            "intent": r.get("intent", ""),
            "region": r.get("region", ""),
            "condition": r["condition"],
            "system_model": r.get("system_model", "claude-opus-4.6"),
            "judge_model": r.get("judge_model", "gemma-3-27b"),
            "response_text": r.get("response_text", ""),
            "tool_calls": json.dumps(r.get("tool_calls", []))[:500],
            "n_tool_calls": len(r.get("tool_calls", [])),
            "grounding": scores.get("grounding", ""),
            "scientific_accuracy": scores.get("scientific_accuracy", ""),
            "completeness": scores.get("completeness", ""),
            "coherence": scores.get("coherence", ""),
            "practical_utility": scores.get("practical_utility", ""),
            "weighted_score": scores.get("weighted_score", ""),
            "geometric_grounding": scores.get("geometric_grounding", ""),
            "response_time_ms": r.get("response_time_ms", ""),
        })

# --- Build DataFrame ---
df = pd.DataFrame(rows)
df = df.sort_values(["system_model", "condition", "query_id"]).reset_index(drop=True)

print(f"\nTotal rows: {len(df)}")
print(f"By model: {df.groupby('system_model').size().to_dict()}")
print(f"By condition: {df.groupby('condition').size().to_dict()}")

# Save
df.to_csv("evaluation_responses_for_review.csv", index=False)
print("\nSaved: evaluation_responses_for_review.csv")

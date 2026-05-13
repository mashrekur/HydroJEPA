"""
agent/minijepa_eval_diagnostics.py

Deep diagnostic on a completed eval run. Produces:

  1. Statistical significance for each condition pair
     (paired Wilcoxon signed-rank, since same qids are scored across conditions)
  2. Effect sizes (Cohen's d, paired)
  3. Per-judge calibration: score distributions, perfect-5 rates,
     pairwise paired tests per judge separately
  4. Routing-quality interaction: does the win over ae_only depend on
     whether the agent picked the physics-correct modality?
  5. Per-category significance and effects
  6. Side-by-side: top N questions where dual_rag beat ae_only the most,
     with both responses excerpted for qualitative reading
  7. Score histograms per condition per judge (printed as ASCII bars
     plus saved as PNG)
  8. Identification of qids that didn't get all judge × condition scores

Usage:
    python agent/minijepa_eval_diagnostics.py
    python agent/minijepa_eval_diagnostics.py --run sonnet46_active
    python agent/minijepa_eval_diagnostics.py --top_examples 5

Outputs stored under <run_dir>/diagnostics/.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the run-resolution and load helpers from the summary script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from minijepa_eval_summary import (
    resolve_run, load_scores, load_routing, aggregate_judges, RUNS_ROOT
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ACTIVE_CONDITIONS = ['ae_only', 'multi_select', 'dual_rag']

CONDITION_PAIRS = [
    ('dual_rag',    'ae_only',      'C5 vs C2 (AE+fleet vs AE-only)'),
    ('dual_rag',    'multi_select', 'C5 vs C4 (AE+fleet vs fleet-only)'),
    ('multi_select','ae_only',      'C4 vs C2 (fleet-only vs AE-only)'),
]


# ---------------------------------------------------------------------------
# Loading + per-judge wide format
# ---------------------------------------------------------------------------
def load_responses(run_dir: Path, conditions: list[str]) -> pd.DataFrame:
    """All system responses across active conditions."""
    rows = []
    for cond in conditions:
        p = run_dir / f'responses_{cond}.jsonl'
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                r['_condition'] = cond
                rows.append(r)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def wide_per_judge(per_judge: pd.DataFrame) -> pd.DataFrame:
    """One row per (qid, judge_model). Columns are conditions × weighted score."""
    if per_judge.empty:
        return per_judge
    return per_judge.pivot_table(
        index=['qid', 'judge_model', 'category', 'sub_type'],
        columns='condition', values='weighted', aggfunc='first'
    ).reset_index()


def wide_aggregated(scores_agg: pd.DataFrame) -> pd.DataFrame:
    """One row per qid with one column per condition (judge-averaged)."""
    if scores_agg.empty:
        return scores_agg
    return scores_agg.pivot_table(
        index=['qid', 'category', 'sub_type'],
        columns='condition', values='weighted', aggfunc='first'
    ).reset_index()


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided paired Wilcoxon signed-rank. Skips NaNs (pairs missing in either)."""
    paired = pd.DataFrame({'a': a, 'b': b}).dropna()
    if len(paired) < 5:
        return {'n': len(paired), 'W': float('nan'), 'p': float('nan'),
                'median_diff': float('nan')}
    diff = paired['a'].values - paired['b'].values
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return {'n': len(paired), 'W': 0.0, 'p': 1.0,
                'median_diff': 0.0}
    try:
        from scipy.stats import wilcoxon
        stat = wilcoxon(paired['a'].values, paired['b'].values,
                        zero_method='wilcox', alternative='two-sided')
        return {'n': len(paired), 'W': float(stat.statistic),
                'p': float(stat.pvalue), 'median_diff': float(np.median(diff))}
    except ImportError:
        return {'n': len(paired), 'W': float('nan'), 'p': float('nan'),
                'median_diff': float(np.median(diff)),
                'note': 'scipy not available'}


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired Cohen's d (the standard effect size for matched samples).
    d = mean(diff) / sd(diff). Conventional benchmarks:
      |d| < 0.2  trivial
      0.2-0.5    small
      0.5-0.8    medium
      0.8+       large
    """
    paired = pd.DataFrame({'a': a, 'b': b}).dropna()
    if len(paired) < 3:
        return {'n': len(paired), 'd': float('nan'),
                'mean_diff': float('nan'), 'sd_diff': float('nan')}
    diff = paired['a'].values - paired['b'].values
    sd = float(np.std(diff, ddof=1))
    mean = float(np.mean(diff))
    return {'n': len(paired), 'd': mean / sd if sd > 0 else float('nan'),
            'mean_diff': mean, 'sd_diff': sd}


def magnitude_label(d: float) -> str:
    if pd.isna(d):
        return '?'
    ad = abs(d)
    if ad < 0.2:    return 'trivial'
    if ad < 0.5:    return 'small'
    if ad < 0.8:    return 'medium'
    return 'large'


# ---------------------------------------------------------------------------
# Per-judge ceiling diagnostics
# ---------------------------------------------------------------------------
def judge_calibration(per_judge: pd.DataFrame) -> pd.DataFrame:
    """For each (judge × condition): n, mean, sd, % perfect-5, % at-or-above-4.5,
    % below 4. Helps spot ceiling-pinned judges (gemma at 4.95+) vs
    discriminating judges (gpt-oss spread across the range)."""
    rows = []
    for (jm, cond), sub in per_judge.groupby(['judge_model', 'condition']):
        if 'weighted' not in sub.columns or sub.empty:
            continue
        w = sub['weighted'].values
        rows.append({
            'judge_model':       jm,
            'condition':         cond,
            'n':                 len(w),
            'mean':              float(np.mean(w)),
            'sd':                float(np.std(w, ddof=1)) if len(w) > 1 else float('nan'),
            'min':               float(np.min(w)),
            'max':               float(np.max(w)),
            'pct_perfect_5':     float(np.mean(w >= 4.95)),
            'pct_at_least_4_5': float(np.mean(w >= 4.5)),
            'pct_below_4':       float(np.mean(w < 4.0)),
        })
    return pd.DataFrame(rows)


def per_judge_pairwise_tests(per_judge: pd.DataFrame) -> pd.DataFrame:
    """Run the paired Wilcoxon test SEPARATELY per judge so we can see whether
    the result holds at each judge or is driven by one judge's signal."""
    rows = []
    for jm in sorted(per_judge['judge_model'].unique()):
        sub = per_judge[per_judge['judge_model'] == jm]
        wide = sub.pivot_table(index='qid', columns='condition',
                                values='weighted', aggfunc='first')
        for a_cond, b_cond, label in CONDITION_PAIRS:
            if a_cond not in wide.columns or b_cond not in wide.columns:
                continue
            a = wide[a_cond].values
            b = wide[b_cond].values
            test = wilcoxon_paired(a, b)
            d = cohens_d_paired(a, b)
            rows.append({
                'judge_model': jm,
                'comparison':  label,
                'n_paired':    test['n'],
                'mean_diff':   d['mean_diff'],
                'cohens_d':    d['d'],
                'd_magnitude': magnitude_label(d['d']),
                'wilcoxon_p':  test['p'],
                'sig_05':      (test['p'] < 0.05) if not pd.isna(test['p']) else None,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Routing × quality interaction
# ---------------------------------------------------------------------------
def routing_quality_interaction(scores_agg: pd.DataFrame,
                                  routing: pd.DataFrame,
                                  responses: pd.DataFrame) -> pd.DataFrame:
    """For multi_select and dual_rag: does the answer-quality win over ae_only
    correlate with whether the router picked the expected modalities?

    For each (qid, condition) pair we compute:
      - selected modalities (from the routing log)
      - expected modalities (from the original query, via responses jsonl)
      - hit:  bool(set(selected) & set(expected) != empty), if expected is non-empty
      - delta vs ae_only on weighted score
    Then aggregate by (condition, hit) to see whether the agent's hits drive
    the answer-quality lift.
    """
    if responses.empty or scores_agg.empty:
        return pd.DataFrame()

    # Pull expected modalities per qid
    expected_map: dict[str, list[str]] = {}
    if 'expected_modalities' in responses.columns:
        for _, r in responses.iterrows():
            em = r.get('expected_modalities')
            if isinstance(em, list) and em:
                expected_map[r['qid']] = em
    if not expected_map:
        return pd.DataFrame()

    # Pull selected modalities per qid per condition
    selected_map: dict[tuple[str, str], list[str]] = {}
    if not responses.empty:
        col = 'selected_modalities' if 'selected_modalities' in responses.columns else None
        if col is not None:
            for _, r in responses.iterrows():
                sm = r.get(col)
                if isinstance(sm, list):
                    selected_map[(r['qid'], r['_condition'])] = sm

    if not selected_map:
        return pd.DataFrame()

    # Build per-question delta vs ae_only
    wide = wide_aggregated(scores_agg)
    if 'ae_only' not in wide.columns:
        return pd.DataFrame()

    rows = []
    for cond in ['multi_select', 'dual_rag']:
        if cond not in wide.columns:
            continue
        sub = wide[['qid', 'category', 'sub_type', 'ae_only', cond]].dropna()
        for _, r in sub.iterrows():
            qid = r['qid']
            expected = expected_map.get(qid, [])
            selected = selected_map.get((qid, cond), [])
            if not expected:
                continue
            hit = bool(set(expected) & set(selected))
            rows.append({
                'qid': qid,
                'condition': cond,
                'category': r['category'],
                'sub_type': r['sub_type'],
                'expected': expected,
                'selected': selected,
                'hit': hit,
                'ae_only': r['ae_only'],
                'cond_score': r[cond],
                'delta_vs_ae': r[cond] - r['ae_only'],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Side-by-side example puller
# ---------------------------------------------------------------------------
def find_top_dual_rag_wins(scores_agg: pd.DataFrame,
                            responses: pd.DataFrame,
                            top_n: int = 5) -> list[dict]:
    """Find the qids where dual_rag beat ae_only by the largest delta.
    Return a list of dicts with question, both responses, the delta."""
    if scores_agg.empty or responses.empty:
        return []
    wide = wide_aggregated(scores_agg)
    if 'ae_only' not in wide.columns or 'dual_rag' not in wide.columns:
        return []

    wide['delta'] = wide['dual_rag'] - wide['ae_only']
    wide = wide.dropna(subset=['delta'])
    top = wide.nlargest(top_n, 'delta')

    out = []
    for _, r in top.iterrows():
        ae   = responses[(responses['qid'] == r['qid']) &
                          (responses['_condition'] == 'ae_only')]
        drag = responses[(responses['qid'] == r['qid']) &
                          (responses['_condition'] == 'dual_rag')]
        if ae.empty or drag.empty:
            continue
        ae_row = ae.iloc[0]
        dr_row = drag.iloc[0]
        out.append({
            'qid':          r['qid'],
            'category':     r['category'],
            'sub_type':     r['sub_type'],
            'delta':        float(r['delta']),
            'ae_only_score':  float(r['ae_only']),
            'dual_rag_score': float(r['dual_rag']),
            'question':     ae_row.get('question', '?'),
            'expected_modalities': ae_row.get('expected_modalities', []),
            'ae_only_response':  ae_row.get('response', ''),
            'dual_rag_response': dr_row.get('response', ''),
            'dual_rag_selected':  dr_row.get('selected_modalities', []),
        })
    return out


def find_dual_rag_losses(scores_agg: pd.DataFrame,
                          responses: pd.DataFrame,
                          top_n: int = 3) -> list[dict]:
    """Inverse: questions where dual_rag scored WORSE than ae_only.
    These are diagnostic for understanding when adding fleet hurts."""
    if scores_agg.empty or responses.empty:
        return []
    wide = wide_aggregated(scores_agg)
    if 'ae_only' not in wide.columns or 'dual_rag' not in wide.columns:
        return []
    wide['delta'] = wide['dual_rag'] - wide['ae_only']
    losses = wide.nsmallest(top_n, 'delta')
    losses = losses[losses['delta'] < 0]

    out = []
    for _, r in losses.iterrows():
        ae   = responses[(responses['qid'] == r['qid']) &
                          (responses['_condition'] == 'ae_only')]
        drag = responses[(responses['qid'] == r['qid']) &
                          (responses['_condition'] == 'dual_rag')]
        if ae.empty or drag.empty:
            continue
        ae_row = ae.iloc[0]
        dr_row = drag.iloc[0]
        out.append({
            'qid':          r['qid'],
            'category':     r['category'],
            'sub_type':     r['sub_type'],
            'delta':        float(r['delta']),
            'ae_only_score':  float(r['ae_only']),
            'dual_rag_score': float(r['dual_rag']),
            'question':     ae_row.get('question', '?'),
            'ae_only_response':  ae_row.get('response', ''),
            'dual_rag_response': dr_row.get('response', ''),
        })
    return out


# ---------------------------------------------------------------------------
# Missing-record diagnostics
# ---------------------------------------------------------------------------
def find_missing_records(per_judge: pd.DataFrame) -> pd.DataFrame:
    """Identify (qid, condition, judge_model) tuples that don't have scores.
    For our 80-question, 3-condition, 2-judge run we expect 480 rows; we have 478."""
    if per_judge.empty:
        return pd.DataFrame()

    judges = sorted(per_judge['judge_model'].unique())
    conditions = sorted(per_judge['condition'].unique())
    all_qids = sorted(per_judge['qid'].unique())

    have = set(zip(per_judge['qid'], per_judge['condition'], per_judge['judge_model']))
    missing = []
    for qid in all_qids:
        for cond in conditions:
            for jm in judges:
                if (qid, cond, jm) not in have:
                    missing.append({'qid': qid, 'condition': cond, 'judge_model': jm})
    return pd.DataFrame(missing)


# ---------------------------------------------------------------------------
# ASCII histograms
# ---------------------------------------------------------------------------
def ascii_hist(values: list[float], bins: int = 10, width: int = 40,
               vmin: float = 1.0, vmax: float = 5.0) -> str:
    """A quick terminal-friendly histogram so we can see distribution shapes."""
    if len(values) == 0:
        return '(no data)'
    edges = np.linspace(vmin, vmax, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    if counts.max() == 0:
        return '(empty)'
    scale = width / counts.max()
    out = []
    for i, c in enumerate(counts):
        bar = '#' * int(c * scale)
        out.append(f'  {edges[i]:.2f}-{edges[i+1]:.2f}  {bar} ({c})')
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Excerpt helper for side-by-side
# ---------------------------------------------------------------------------
def excerpt(text: str, n_chars: int = 600) -> str:
    text = (text or '').strip()
    if len(text) <= n_chars:
        return text
    return text[:n_chars] + '...'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default=None,
                   help='Run tag or directory name. Defaults to most recent.')
    p.add_argument('--top_examples', type=int, default=5,
                   help='Number of side-by-side examples to print.')
    p.add_argument('--excerpt_chars', type=int, default=800,
                   help='How much of each response to print in side-by-side.')
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = resolve_run(args.run)
    diag_dir = run_dir / 'diagnostics'
    diag_dir.mkdir(exist_ok=True)

    print(f'Diagnosing run: {run_dir}')
    print('=' * 80)

    # Load
    per_judge  = load_scores(run_dir)
    routing    = load_routing(run_dir)
    responses  = load_responses(run_dir, ACTIVE_CONDITIONS)
    scores_agg = aggregate_judges(per_judge)

    if per_judge.empty:
        print('No judge scores found. Aborting.')
        return

    # Restrict to active conditions
    per_judge  = per_judge[per_judge['condition'].isin(ACTIVE_CONDITIONS)]
    scores_agg = scores_agg[scores_agg['condition'].isin(ACTIVE_CONDITIONS)]

    # ---- 1. Missing records -------------------------------------------------
    print('\n[1] Missing (qid, condition, judge) tuples')
    print('-' * 80)
    missing = find_missing_records(per_judge)
    if missing.empty:
        print('  None — full coverage.')
    else:
        print(f'  {len(missing)} missing tuples:')
        print(missing.to_string(index=False))
        missing.to_csv(diag_dir / 'missing_tuples.csv', index=False)
        print(f'  Wrote {diag_dir}/missing_tuples.csv')

    # ---- 2. Per-judge calibration ------------------------------------------
    print('\n[2] Per-judge calibration (score distribution per judge × condition)')
    print('-' * 80)
    cal = judge_calibration(per_judge)
    if not cal.empty:
        cal_disp = cal.copy()
        for c in ['mean','sd','min','max']:
            cal_disp[c] = cal_disp[c].round(3)
        for c in ['pct_perfect_5','pct_at_least_4_5','pct_below_4']:
            cal_disp[c] = (cal_disp[c] * 100).round(1).astype(str) + '%'
        print(cal_disp.to_string(index=False))
        cal.to_csv(diag_dir / 'judge_calibration.csv', index=False)

    # ---- 2b. ASCII histogram per judge --------------------------------------
    print('\n[2b] Score distribution per judge (across all conditions)')
    print('-' * 80)
    for jm in sorted(per_judge['judge_model'].unique()):
        vals = per_judge[per_judge['judge_model'] == jm]['weighted'].values
        print(f'\n  {jm}  (n={len(vals)}, mean={np.mean(vals):.2f}, sd={np.std(vals, ddof=1):.2f})')
        print(ascii_hist(list(vals), bins=10, width=40, vmin=1, vmax=5))

    # ---- 3. Significance: aggregate across judges ---------------------------
    print('\n[3] Significance & effect sizes (judge-averaged scores)')
    print('-' * 80)
    wide = wide_aggregated(scores_agg)
    sig_rows = []
    for a_cond, b_cond, label in CONDITION_PAIRS:
        if a_cond not in wide.columns or b_cond not in wide.columns:
            continue
        a = wide[a_cond].values
        b = wide[b_cond].values
        test = wilcoxon_paired(a, b)
        d = cohens_d_paired(a, b)
        sig_rows.append({
            'comparison':  label,
            'n_paired':    test['n'],
            'mean_diff':   d['mean_diff'],
            'sd_diff':     d['sd_diff'],
            'cohens_d':    d['d'],
            'd_magnitude': magnitude_label(d['d']),
            'wilcoxon_W':  test['W'],
            'wilcoxon_p':  test['p'],
            'sig_05':      (test['p'] < 0.05) if not pd.isna(test['p']) else None,
            'sig_10':      (test['p'] < 0.10) if not pd.isna(test['p']) else None,
        })
    sig = pd.DataFrame(sig_rows)
    if not sig.empty:
        for c in ['mean_diff','sd_diff','cohens_d','wilcoxon_W']:
            sig[c] = sig[c].round(4)
        sig['wilcoxon_p'] = sig['wilcoxon_p'].apply(
            lambda x: f'{x:.4f}' if not pd.isna(x) else 'nan')
        print(sig.to_string(index=False))
        sig.to_csv(diag_dir / 'significance_aggregate.csv', index=False)

    # ---- 4. Significance per judge ------------------------------------------
    print('\n[4] Significance per judge (does each judge produce the same direction?)')
    print('-' * 80)
    pj_sig = per_judge_pairwise_tests(per_judge)
    if not pj_sig.empty:
        for c in ['mean_diff','cohens_d']:
            pj_sig[c] = pj_sig[c].round(4)
        pj_sig['wilcoxon_p'] = pj_sig['wilcoxon_p'].apply(
            lambda x: f'{x:.4f}' if not pd.isna(x) else 'nan')
        print(pj_sig.to_string(index=False))
        pj_sig.to_csv(diag_dir / 'significance_per_judge.csv', index=False)

    # ---- 5. Per-category significance --------------------------------------
    print('\n[5] Per-category significance (judge-averaged)')
    print('-' * 80)
    cat_rows = []
    for cat in sorted(scores_agg['category'].unique()):
        cat_scores = scores_agg[scores_agg['category'] == cat]
        cat_wide = wide_aggregated(cat_scores) if 'category' in cat_scores.columns else None
        if cat_wide is None or cat_wide.empty:
            continue
        for a_cond, b_cond, label in CONDITION_PAIRS:
            if a_cond not in cat_wide.columns or b_cond not in cat_wide.columns:
                continue
            a = cat_wide[a_cond].values
            b = cat_wide[b_cond].values
            test = wilcoxon_paired(a, b)
            d = cohens_d_paired(a, b)
            cat_rows.append({
                'category':    cat,
                'comparison':  label,
                'n_paired':    test['n'],
                'mean_diff':   d['mean_diff'],
                'cohens_d':    d['d'],
                'd_magnitude': magnitude_label(d['d']),
                'wilcoxon_p':  test['p'],
                'sig_05':      (test['p'] < 0.05) if not pd.isna(test['p']) else None,
            })
    cat_sig = pd.DataFrame(cat_rows)
    if not cat_sig.empty:
        for c in ['mean_diff','cohens_d']:
            cat_sig[c] = cat_sig[c].round(4)
        cat_sig['wilcoxon_p'] = cat_sig['wilcoxon_p'].apply(
            lambda x: f'{x:.4f}' if not pd.isna(x) else 'nan')
        print(cat_sig.to_string(index=False))
        cat_sig.to_csv(diag_dir / 'significance_per_category.csv', index=False)

    # ---- 6. Routing-quality interaction ------------------------------------
    print('\n[6] Routing-quality interaction')
    print('     For multi_select / dual_rag: does the win over ae_only depend on')
    print('     whether the agent hit the expected modalities?')
    print('-' * 80)
    interaction = routing_quality_interaction(scores_agg, routing, responses)
    if interaction.empty:
        print('  Could not compute (missing routing or expected_modalities data).')
    else:
        summary = interaction.groupby(['condition', 'hit']).agg(
            n=('qid', 'count'),
            mean_delta=('delta_vs_ae', 'mean'),
            sd_delta=('delta_vs_ae', 'std'),
        ).round(3).reset_index()
        print(summary.to_string(index=False))
        interaction.to_csv(diag_dir / 'routing_quality_interaction.csv', index=False)

    # ---- 7. Side-by-side: top dual_rag wins --------------------------------
    print(f'\n[7] Top {args.top_examples} questions where dual_rag beat ae_only')
    print('-' * 80)
    wins = find_top_dual_rag_wins(scores_agg, responses, top_n=args.top_examples)
    for i, w in enumerate(wins, 1):
        print(f'\n--- Example {i}: {w["qid"]} ({w["category"]} / {w["sub_type"]}) ---')
        print(f'    delta: dual_rag {w["dual_rag_score"]:.2f} - ae_only {w["ae_only_score"]:.2f} = +{w["delta"]:.2f}')
        print(f'    expected modalities: {w["expected_modalities"]}')
        print(f'    dual_rag selected:   {w["dual_rag_selected"]}')
        print(f'\n    Q: {w["question"]}')
        print(f'\n    AE_ONLY response (excerpt):')
        for line in excerpt(w["ae_only_response"], args.excerpt_chars).split('\n'):
            print(f'      | {line}')
        print(f'\n    DUAL_RAG response (excerpt):')
        for line in excerpt(w["dual_rag_response"], args.excerpt_chars).split('\n'):
            print(f'      | {line}')

    # Persist examples to disk for closer reading
    if wins:
        with open(diag_dir / 'top_dual_rag_wins.json', 'w') as f:
            json.dump(wins, f, indent=2, default=str)
        print(f'\n  Full responses saved to {diag_dir}/top_dual_rag_wins.json')

    # ---- 8. Side-by-side: dual_rag losses ----------------------------------
    print(f'\n[8] Questions where dual_rag scored WORSE than ae_only')
    print('-' * 80)
    losses = find_dual_rag_losses(scores_agg, responses, top_n=3)
    if not losses:
        print('  None found.')
    for i, w in enumerate(losses, 1):
        print(f'\n--- Loss {i}: {w["qid"]} ({w["category"]} / {w["sub_type"]}) ---')
        print(f'    delta: dual_rag {w["dual_rag_score"]:.2f} - ae_only {w["ae_only_score"]:.2f} = {w["delta"]:.2f}')
        print(f'\n    Q: {w["question"]}')
        print(f'\n    AE_ONLY response (excerpt):')
        for line in excerpt(w["ae_only_response"], args.excerpt_chars).split('\n'):
            print(f'      | {line}')
        print(f'\n    DUAL_RAG response (excerpt):')
        for line in excerpt(w["dual_rag_response"], args.excerpt_chars).split('\n'):
            print(f'      | {line}')

    if losses:
        with open(diag_dir / 'top_dual_rag_losses.json', 'w') as f:
            json.dump(losses, f, indent=2, default=str)

    # ---- 9. Per-question delta distribution --------------------------------
    print('\n[9] Per-question delta distributions')
    print('-' * 80)
    if 'ae_only' in wide.columns and 'dual_rag' in wide.columns:
        d_dr = (wide['dual_rag'] - wide['ae_only']).dropna()
        print(f'\n  dual_rag - ae_only  (n={len(d_dr)})')
        print(f'    mean={d_dr.mean():+.3f}  median={d_dr.median():+.3f}  sd={d_dr.std():.3f}')
        print(f'    pct positive: {(d_dr > 0).mean()*100:.1f}%   '
              f'pct zero: {(d_dr == 0).mean()*100:.1f}%   '
              f'pct negative: {(d_dr < 0).mean()*100:.1f}%')
        print(ascii_hist(list(d_dr), bins=12, width=40, vmin=-1.0, vmax=1.0))
    if 'multi_select' in wide.columns and 'ae_only' in wide.columns:
        d_ms = (wide['multi_select'] - wide['ae_only']).dropna()
        print(f'\n  multi_select - ae_only  (n={len(d_ms)})')
        print(f'    mean={d_ms.mean():+.3f}  median={d_ms.median():+.3f}  sd={d_ms.std():.3f}')
        print(f'    pct positive: {(d_ms > 0).mean()*100:.1f}%   '
              f'pct zero: {(d_ms == 0).mean()*100:.1f}%   '
              f'pct negative: {(d_ms < 0).mean()*100:.1f}%')
        print(ascii_hist(list(d_ms), bins=12, width=40, vmin=-1.0, vmax=1.0))

    # ---- 10. Save the wide judge-averaged frame so we can grep / sort it ---
    if not wide.empty:
        wide.to_csv(diag_dir / 'per_question_wide.csv', index=False)
        print(f'\n  Per-question wide frame saved to {diag_dir}/per_question_wide.csv')
        print('  (One row per qid with one column per condition; useful for sorting/spot-checking.)')

    # ---- Summary recommendations -------------------------------------------
    print('\n' + '=' * 80)
    print('Diagnostic summary recommendations')
    print('=' * 80)

    notes = []

    if not cal.empty:
        ceiling_judges = cal[cal['pct_perfect_5'] > 0.5]['judge_model'].unique()
        if len(ceiling_judges) > 0:
            notes.append(
                f'Ceiling-pinned judges (>50% perfect-5 ratings): '
                f'{list(ceiling_judges)}. These contribute little discrimination.')

    if not sig.empty:
        any_sig = sig['sig_05'].any() if 'sig_05' in sig.columns else False
        if not any_sig:
            notes.append(
                'No condition pair achieves p < 0.05 on judge-averaged scores. '
                'Effect may be too small to claim significance with the current judges.')

    if not pj_sig.empty:
        # Check that BOTH judges show same-sign effect
        for label in pj_sig['comparison'].unique():
            sub = pj_sig[pj_sig['comparison'] == label]
            signs = set(np.sign(sub['mean_diff'].dropna()))
            if len(signs) > 1:
                notes.append(f'Judges disagree on direction for "{label}".')

    if notes:
        for n in notes:
            print(f'  - {n}')
    else:
        print('  No major flags. Significance and judge agreement look ok.')

    print(f'\nAll diagnostic CSVs and JSON examples written to:\n  {diag_dir}')


if __name__ == '__main__':
    main()

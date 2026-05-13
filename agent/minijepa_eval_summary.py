"""
agent/minijepa_eval_summary.py
Tables and headline figure for the Mini-JEPA LLM evaluation.

Reads:
  data/hydrojepa/minijepa_eval/judge_scores.jsonl
  data/hydrojepa/minijepa_eval/routing_log.jsonl

Writes:
  reports/llm_eval/exp_minijepa_summary.csv         per-condition mean ± std
  reports/llm_eval/exp_minijepa_per_category.csv    condition × category
  reports/llm_eval/exp_minijepa_per_criterion.csv   condition × criterion (G/A/C/H/U)
  reports/llm_eval/exp_minijepa_routing_behavior.csv  routing-decision summary
  reports/llm_eval/fig_minijepa_summary.png         headline 4-panel figure

The novel comparisons the paper reports on:
  C5 vs C2  — does the fleet improve over AE alone?
  C4 vs C3  — does multi-select routing beat a single fixed Mini-JEPA?
  C5 vs C4  — does AE add complementary signal on top of the routed fleet?

Plus a per-category breakdown so we can say *where* the fleet helps,
and a routing-behavior table so we can say *what the agent picks* and
how often it agrees with the expected modalities.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HJ_ROOT     = Path(__file__).resolve().parent.parent
EVAL_ROOT   = HJ_ROOT / 'data' / 'hydrojepa' / 'minijepa_eval'
RUNS_ROOT   = EVAL_ROOT / 'runs'

CONDITION_ORDER = ['llm_only', 'ae_only', 'single_fixed', 'multi_select', 'dual_rag']
CONDITION_LABEL = {
    'llm_only':     'LLM only',
    'ae_only':      'AE only',
    'single_fixed': 'Single Mini-JEPA',
    'multi_select': 'Multi-select fleet',
    'dual_rag':     'AE + fleet',
}
CONDITION_COLOR = {
    'llm_only':     '#999999',
    'ae_only':      '#7B4DB7',
    'single_fixed': '#1F77B4',
    'multi_select': '#3A7D44',
    'dual_rag':     '#D62728',
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------------
# Run dir resolution
# ---------------------------------------------------------------------------
def find_latest_run() -> Path:
    """Return the most recently modified run directory."""
    if not RUNS_ROOT.exists():
        raise FileNotFoundError(f'No runs directory at {RUNS_ROOT}')
    runs = sorted([d for d in RUNS_ROOT.iterdir() if d.is_dir()],
                  key=lambda d: d.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError('No runs found.')
    return runs[0]


def resolve_run(run_arg: str | None) -> Path:
    """If `run_arg` is None -> latest. Otherwise treat as either a full path or
    a directory name under runs/."""
    if run_arg is None:
        return find_latest_run()
    p = Path(run_arg)
    if p.is_absolute() and p.exists():
        return p
    candidate = RUNS_ROOT / run_arg
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f'Run not found: {run_arg}')


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_scores(run_dir: Path) -> pd.DataFrame:
    """Load the raw per-judge score frame. One row per (qid, condition, judge_model)."""
    p = run_dir / 'judge_scores.jsonl'
    if not p.exists():
        raise FileNotFoundError(f'No judge scores at {p}')
    df = pd.read_json(p, lines=True)
    if 'judge_model' not in df.columns:
        # Backwards compat: old runs had a single judge per (qid, condition)
        df['judge_model'] = 'unknown'
    return df


def aggregate_judges(per_judge: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-judge rows to one row per (qid, condition) by averaging
    each criterion (G, A, C, H, U) and the weighted score across judges.
    Carries over category and sub_type metadata from the first judge's row.
    """
    if per_judge.empty:
        return per_judge.copy()

    crit_cols = [c for c in ['G', 'A', 'C', 'H', 'U', 'weighted'] if c in per_judge.columns]
    meta_cols = [c for c in ['category', 'sub_type'] if c in per_judge.columns]

    # Mean across judges
    agg = (per_judge.groupby(['qid', 'condition'])[crit_cols]
                    .mean().reset_index())
    if meta_cols:
        meta = (per_judge.groupby(['qid', 'condition'])[meta_cols]
                          .first().reset_index())
        agg = agg.merge(meta, on=['qid', 'condition'])
    # Carry the number of judges that successfully scored each pair
    n_judges = (per_judge.groupby(['qid', 'condition']).size()
                          .rename('n_judges').reset_index())
    agg = agg.merge(n_judges, on=['qid', 'condition'])
    return agg


def per_judge_summary(per_judge: pd.DataFrame) -> pd.DataFrame:
    """For each (judge_model, condition) the mean weighted score."""
    if per_judge.empty:
        return pd.DataFrame()
    rows = []
    for (jm, cond), sub in per_judge.groupby(['judge_model', 'condition']):
        rows.append({
            'judge_model':    jm,
            'condition':      cond,
            'n':              len(sub),
            'weighted_mean':  sub['weighted'].mean(),
            'weighted_std':   sub['weighted'].std(),
        })
    return pd.DataFrame(rows)


def judge_agreement(per_judge: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation of weighted scores between every pair of judges,
    measured over the (qid, condition) tuples both judges scored.
    A diagnostic on whether the judge panel is internally consistent.
    """
    if per_judge.empty or per_judge['judge_model'].nunique() < 2:
        return pd.DataFrame()
    # Pivot to wide: rows = (qid, condition), columns = judge_model
    wide = per_judge.pivot_table(
        index=['qid', 'condition'], columns='judge_model',
        values='weighted', aggfunc='first')
    judges = sorted(wide.columns)
    rows = []
    for i, j1 in enumerate(judges):
        for j2 in judges[i + 1:]:
            paired = wide[[j1, j2]].dropna()
            if len(paired) < 3:
                rho = float('nan')
            else:
                rho = paired[j1].corr(paired[j2])
            rows.append({
                'judge_a': j1, 'judge_b': j2,
                'n_paired': len(paired),
                'pearson_r': rho,
            })
    return pd.DataFrame(rows)


def load_routing(run_dir: Path) -> pd.DataFrame:
    p = run_dir / 'routing_log.jsonl'
    if not p.exists():
        return pd.DataFrame()
    return pd.read_json(p, lines=True)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def summary_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Per-condition mean ± std of weighted score and each criterion."""
    rows = []
    for cond in CONDITION_ORDER:
        sub = scores[scores.condition == cond]
        if sub.empty:
            continue
        row = {
            'condition': cond,
            'label':     CONDITION_LABEL[cond],
            'n':         len(sub),
            'weighted_mean': sub['weighted'].mean(),
            'weighted_std':  sub['weighted'].std(),
        }
        for k in ['G', 'A', 'C', 'H', 'U']:
            row[f'{k}_mean'] = sub[k].mean() if k in sub.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def per_category_table(scores: pd.DataFrame) -> pd.DataFrame:
    """condition × category mean weighted score."""
    if 'category' not in scores.columns or scores['category'].isna().all():
        return pd.DataFrame()
    pivot = (scores.pivot_table(index='condition', columns='category',
                                values='weighted', aggfunc='mean')
                    .reindex(index=CONDITION_ORDER))
    return pivot


def per_criterion_table(scores: pd.DataFrame) -> pd.DataFrame:
    """condition × criterion mean."""
    rows = []
    for cond in CONDITION_ORDER:
        sub = scores[scores.condition == cond]
        if sub.empty:
            continue
        row = {'condition': cond, 'label': CONDITION_LABEL[cond]}
        for k in ['G', 'A', 'C', 'H', 'U']:
            if k in sub.columns:
                row[k] = sub[k].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Routing-behavior analysis
# ---------------------------------------------------------------------------
def routing_behavior_table(routing: pd.DataFrame) -> pd.DataFrame:
    """
    Per (condition, category) routing summary:
      mean number of modalities selected,
      hit rate against expected_modalities (any-overlap),
      most common modality picked.
    """
    if routing.empty:
        return pd.DataFrame()
    rows = []
    for (cond, cat), sub in routing.groupby(['condition', 'category']):
        n = len(sub)
        n_selected = sub['selected_modalities'].apply(len)
        # hit rate: any overlap with expected_modalities (skip rows where expected is empty)
        with_exp = sub[sub['expected_modalities'].apply(lambda x: len(x) > 0)]
        if not with_exp.empty:
            hit = with_exp.apply(
                lambda r: int(bool(set(r['selected_modalities']) & set(r['expected_modalities']))),
                axis=1,
            )
            hit_rate = hit.mean()
        else:
            hit_rate = np.nan

        # most common modality
        all_picked = [m for sel in sub['selected_modalities'] for m in sel]
        mc = pd.Series(all_picked).value_counts()
        most_common = mc.index[0] if len(mc) else ''
        rows.append({
            'condition': cond, 'category': cat, 'n': n,
            'mean_selected':       n_selected.mean(),
            'expected_hit_rate':   hit_rate,
            'most_common_pick':    most_common,
        })
    return pd.DataFrame(rows)


def modality_pick_frequency(routing: pd.DataFrame) -> pd.DataFrame:
    """How often each Mini-JEPA gets picked, broken down by category."""
    if routing.empty:
        return pd.DataFrame()
    rows = []
    for cat, sub in routing.groupby('category'):
        flat = [m for sel in sub['selected_modalities'] for m in sel]
        s = pd.Series(flat).value_counts(normalize=True)
        rows.append({'category': cat, **s.to_dict()})
    return pd.DataFrame(rows).fillna(0.0)


# ---------------------------------------------------------------------------
# Headline figure
# ---------------------------------------------------------------------------
def make_summary_figure(summary: pd.DataFrame, per_cat: pd.DataFrame,
                        routing: pd.DataFrame, out_path: Path):
    fig = plt.figure(figsize=(15, 9.5), dpi=130)
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)

    # (a) Overall weighted score per condition with error bars
    a = fig.add_subplot(gs[0, 0])
    if not summary.empty:
        x = range(len(summary))
        colors = [CONDITION_COLOR[c] for c in summary['condition']]
        a.bar(x, summary['weighted_mean'], yerr=summary['weighted_std'],
              capsize=4, color=colors, alpha=0.9)
        a.set_xticks(list(x))
        a.set_xticklabels(summary['label'], rotation=15, ha='right', fontsize=9)
        a.set_ylabel('Weighted score (1–5)')
        a.set_ylim(1, 5)
        a.set_title('(a) Overall weighted score by condition')
        a.axhline(3.0, color='gray', lw=0.5, ls='--')
    else:
        a.axis('off')

    # (b) Per-category heatmap
    b = fig.add_subplot(gs[0, 1])
    if not per_cat.empty:
        data = per_cat.values
        im = b.imshow(data, cmap='RdYlGn', vmin=1, vmax=5, aspect='auto')
        b.set_yticks(range(len(per_cat.index)))
        b.set_yticklabels([CONDITION_LABEL.get(c, c) for c in per_cat.index], fontsize=9)
        b.set_xticks(range(len(per_cat.columns)))
        b.set_xticklabels(per_cat.columns, rotation=20, ha='right', fontsize=9)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    b.text(j, i, f'{v:.2f}', ha='center', va='center',
                           fontsize=8, color='black')
        b.set_title('(b) Weighted score by category × condition')
        plt.colorbar(im, ax=b, fraction=0.04, pad=0.02)
    else:
        b.axis('off')

    # (c) Routing-pick frequency per category
    c = fig.add_subplot(gs[1, 0])
    if not routing.empty:
        freq = modality_pick_frequency(routing)
        if not freq.empty:
            cats = freq['category'].tolist()
            freq_no_cat = freq.drop(columns=['category']).fillna(0.0)
            modalities = list(freq_no_cat.columns)
            x = np.arange(len(cats))
            bottom = np.zeros(len(cats))
            cmap = plt.cm.tab10
            for i, m in enumerate(modalities):
                vals = freq_no_cat[m].values
                c.bar(x, vals, bottom=bottom, label=m,
                      color=cmap(i / max(len(modalities), 1)))
                bottom += vals
            c.set_xticks(x)
            c.set_xticklabels(cats, rotation=15, ha='right', fontsize=9)
            c.set_ylabel('Selection frequency')
            c.set_title('(c) Agent routing pick distribution by category')
            c.legend(fontsize=7, loc='upper right', ncol=2)
            c.set_ylim(0, 1.05)
    else:
        c.text(0.5, 0.5, 'No routing log (multi_select / dual_rag not run yet)',
               ha='center', va='center', transform=c.transAxes, color='gray')
        c.axis('off')

    # (d) Routing hit rate per category (any-overlap with expected)
    d = fig.add_subplot(gs[1, 1])
    if not routing.empty:
        rb = routing_behavior_table(routing)
        # average across conditions for the headline figure
        if not rb.empty:
            agg = (rb.dropna(subset=['expected_hit_rate'])
                     .groupby('category')['expected_hit_rate'].mean()
                     .sort_values(ascending=False))
            if not agg.empty:
                d.bar(range(len(agg)), agg.values, color='#3A7D44', alpha=0.85)
                d.set_xticks(range(len(agg)))
                d.set_xticklabels(agg.index, rotation=15, ha='right', fontsize=9)
                d.set_ylabel('Routing hit rate (any overlap)')
                d.set_ylim(0, 1.05)
                d.axhline(1.0, color='gray', lw=0.5, ls='--')
                d.set_title('(d) Routing accuracy vs expected modalities')
            else:
                d.text(0.5, 0.5, 'No expected_modalities to score against',
                       ha='center', va='center', transform=d.transAxes, color='gray')
                d.axis('off')
    else:
        d.axis('off')

    fig.suptitle('Mini-JEPA fleet — LLM evaluation summary', fontsize=14, fontweight='bold')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'Saved {out_path}')


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--run', default=None,
                   help='Run directory name under data/hydrojepa/minijepa_eval/runs/, '
                        'or absolute path. Defaults to latest.')
    p.add_argument('--list', action='store_true',
                   help='List available runs and exit')
    args = p.parse_args()

    if args.list:
        if not RUNS_ROOT.exists():
            print('No runs directory yet.')
            return
        print(f'Runs in {RUNS_ROOT}:')
        for d in sorted(RUNS_ROOT.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if d.is_dir():
                manifest = d / 'manifest.json'
                tag = ''
                if manifest.exists():
                    try:
                        m = json.load(open(manifest))
                        judges = m.get('judge_models') or [m.get('judge_model', '?')]
                        n_judges = len(judges) if isinstance(judges, list) else 1
                        tag = (f' [system={m.get("system_model", "?")} '
                               f'judges={n_judges} '
                               f'cond={m.get("condition", "?")} '
                               f'n={m.get("n_queries", "?")}]')
                    except Exception:
                        pass
                print(f'  {d.name}{tag}')
        return

    run_dir = resolve_run(args.run)
    logging.info(f'Summarizing run: {run_dir}')

    per_judge = load_scores(run_dir)
    routing   = load_routing(run_dir)

    n_judges = per_judge['judge_model'].nunique() if not per_judge.empty else 0
    logging.info(f'Loaded {len(per_judge)} per-judge scores '
                 f'({n_judges} judges, {per_judge.condition.nunique()} conditions)')

    # Aggregate per-judge -> per (qid, condition) by averaging across judges
    scores = aggregate_judges(per_judge)
    logging.info(f'Aggregated to {len(scores)} (qid, condition) pairs')

    summary  = summary_table(scores)
    per_cat  = per_category_table(scores)
    per_crit = per_criterion_table(scores)
    routing_table = routing_behavior_table(routing)

    # Multi-judge specific tables
    pj_summary = per_judge_summary(per_judge)
    agreement  = judge_agreement(per_judge)

    # Write tables into the run dir so a single run is self-contained
    summary.to_csv(run_dir / 'summary.csv', index=False)
    if not per_cat.empty:
        per_cat.to_csv(run_dir / 'per_category.csv')
    per_crit.to_csv(run_dir / 'per_criterion.csv', index=False)
    if not routing_table.empty:
        routing_table.to_csv(run_dir / 'routing_behavior.csv', index=False)
    if not pj_summary.empty:
        pj_summary.to_csv(run_dir / 'per_judge_summary.csv', index=False)
    if not agreement.empty:
        agreement.to_csv(run_dir / 'judge_agreement.csv', index=False)

    print('\nOverall summary (averaged across judges)')
    print(summary.to_string(index=False))

    if not pj_summary.empty:
        print('\nPer-judge × condition (mean weighted score)')
        pivot = (pj_summary.pivot_table(
            index='judge_model', columns='condition', values='weighted_mean')
                 .reindex(columns=[c for c in CONDITION_ORDER
                                   if c in pj_summary['condition'].unique()]))
        print(pivot.round(3).to_string())

    if not agreement.empty:
        print('\nJudge-judge agreement (Pearson r on weighted scores)')
        print(agreement.round(3).to_string(index=False))

    if not per_cat.empty:
        print('\nPer-category × condition mean weighted score')
        print(per_cat.round(2).to_string())

    if not routing_table.empty:
        print('\nRouting behavior')
        print(routing_table.round(2).to_string(index=False))

    # Headline comparisons for this paper (across-judge means)
    present = set(summary['condition'])

    print('\n--- Headline comparisons (mean across judges) ---')
    if {'ae_only', 'dual_rag'}.issubset(present):
        ae = summary.loc[summary.condition == 'ae_only', 'weighted_mean'].iloc[0]
        dr = summary.loc[summary.condition == 'dual_rag', 'weighted_mean'].iloc[0]
        print(f'  C5 (AE + fleet) vs C2 (AE only):           '
              f'{dr:.3f} vs {ae:.3f}  (delta = {dr - ae:+.3f})')

    if {'multi_select', 'dual_rag'}.issubset(present):
        ms = summary.loc[summary.condition == 'multi_select', 'weighted_mean'].iloc[0]
        dr = summary.loc[summary.condition == 'dual_rag', 'weighted_mean'].iloc[0]
        print(f'  C5 (AE + fleet) vs C4 (fleet only):        '
              f'{dr:.3f} vs {ms:.3f}  (delta = {dr - ms:+.3f})')

    if {'ae_only', 'multi_select'}.issubset(present):
        ae = summary.loc[summary.condition == 'ae_only', 'weighted_mean'].iloc[0]
        ms = summary.loc[summary.condition == 'multi_select', 'weighted_mean'].iloc[0]
        print(f'  C4 (fleet only) vs C2 (AE only):           '
              f'{ms:.3f} vs {ae:.3f}  (delta = {ms - ae:+.3f})')

    # Routing-ablation comparison only printed if C3 was actually run
    if {'single_fixed', 'multi_select'}.issubset(present):
        sf = summary.loc[summary.condition == 'single_fixed', 'weighted_mean'].iloc[0]
        ms = summary.loc[summary.condition == 'multi_select', 'weighted_mean'].iloc[0]
        print(f'  C4 (multi-select) vs C3 (single fixed):    '
              f'{ms:.3f} vs {sf:.3f}  (delta = {ms - sf:+.3f})')

    fig_in_run = run_dir / 'fig_summary.png'
    make_summary_figure(summary, per_cat, routing, fig_in_run)

    # Also drop a copy under reports/llm_eval/ tagged by run name, for the
    # paper-figure pile.
    out_root = HJ_ROOT / 'reports' / 'llm_eval'
    out_root.mkdir(parents=True, exist_ok=True)
    fig_in_reports = out_root / f'fig_minijepa_summary__{run_dir.name}.png'
    make_summary_figure(summary, per_cat, routing, fig_in_reports)


if __name__ == '__main__':
    main()

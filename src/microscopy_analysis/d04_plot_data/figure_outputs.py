"""Post-processing helpers for stats tables and figure-data archival.

Three small utilities used by the figure-producing notebooks:

- ``tag_stats``: prepend dataset / group-variable / compared-group
  columns to a stats dataframe so it can be filtered by figure or by
  comparison rather than relying on the concatenated ``comparison``
  string.
- ``print_stats_summary``: pretty-print the stats for a single
  comparison so the relevant numbers appear inline beneath the figure
  cell.
- ``save_plot_data``: write the long-form dataframe consumed by a
  plotting helper to xlsx alongside the SVG, so every published figure
  has a self-contained, exactly-what-was-plotted data archive.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Stats-table tagging
# ---------------------------------------------------------------------------

_TAG_COLS = ['dataset', 'group_variable', 'groups', 'group_a', 'group_b']


def tag_stats(stats_df, dataset, group_variable, groups):
    """Prepend dataset / group identifier columns to a stats dataframe.

    Parameters
    ----------
    stats_df : pandas.DataFrame
        Output of ``run_stats.run_repeated_measures_stats`` or
        ``run_stats_lmm.run_lmm_stats``.
    dataset : str
        The figure prefix, naming the dataset / panel the comparison
        belongs to (e.g. ``'ctrl_actinint'``).
    group_variable : str
        Name of the column whose levels are being compared (e.g.
        ``'tx'`` or ``'region'``).
    groups : sequence of str
        The compared group labels in plot order. ``group_a`` and
        ``group_b`` are populated only for two-group comparisons; for
        three-or-more-group comparisons they are left as ``NaN`` and
        the full ordered list is recorded in ``groups``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``stats_df`` with five new columns inserted at the
        front.
    """
    out = stats_df.copy()
    # Drop any prior tag columns so re-tagging produces clean output.
    out = out.drop(columns=[c for c in _TAG_COLS if c in out.columns])

    groups_list = list(groups)
    out.insert(0, 'dataset', dataset)
    out.insert(1, 'group_variable', group_variable)
    out.insert(2, 'groups', ' vs '.join(map(str, groups_list)))
    if len(groups_list) == 2:
        out.insert(3, 'group_a', groups_list[0])
        out.insert(4, 'group_b', groups_list[1])
    else:
        out.insert(3, 'group_a', np.nan)
        out.insert(4, 'group_b', np.nan)
    return out


# ---------------------------------------------------------------------------
# Stats summary printing
# ---------------------------------------------------------------------------

_MAIN_KEYS = {'main_time', 'main_group', 'interaction'}


def _fmt_p(v):
    if v is None:
        return ' n/a '
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ' n/a '
    if not np.isfinite(f):
        return ' n/a '
    return f'{f:.3g}'


def print_stats_summary(stats_df, comparison_label, indent='  '):
    """Print a compact human-readable summary of one comparison's stats.

    Renders the main-effect / interaction rows first, then any
    per-timepoint contrast rows with their normality and Wilcoxon
    sensitivity columns when present.
    """
    name = comparison_label.get('name', '?') if isinstance(comparison_label, dict) else str(comparison_label)
    print(f'--- Stats: {name} ---')

    if stats_df is None or len(stats_df) == 0:
        print(f'{indent}(no stats rows)')
        return

    timepoint_str = stats_df['timepoint'].astype(str)
    main = stats_df[timepoint_str.isin(_MAIN_KEYS)]
    posthoc = stats_df[~timepoint_str.isin(_MAIN_KEYS)]

    if len(main) > 0:
        print(f'{indent}Main effects:')
        for _, row in main.iterrows():
            print(
                f'{indent}  {str(row["timepoint"]):12s}  p = {_fmt_p(row.get("p_value"))}'
                f'  sig = {row.get("significant")}'
                f'  ({row.get("test", "")})'
            )

    if len(posthoc) > 0:
        print(f'{indent}Per-timepoint contrasts:')
        for _, row in posthoc.iterrows():
            tp = row['timepoint']
            parts = [f't = {tp}']
            n = row.get('paired_diffs_n')
            if n is not None and pd.notna(n):
                parts.append(f'n = {int(n)}')
            parts.append(f'p = {_fmt_p(row.get("p_value"))}')
            parts.append(f'p_fdr = {_fmt_p(row.get("p_value_fdr"))}')
            shap = row.get('shapiro_p')
            if shap is not None and pd.notna(shap):
                parts.append(f'shapiro_p = {_fmt_p(shap)}')
            parts.append(f'sig = {row.get("significant")}')
            test = row.get('test', '')
            if test:
                parts.append(f'({test})')
            print(f'{indent}  ' + '  '.join(parts))


# ---------------------------------------------------------------------------
# Stats-table filtering for figure annotations
# ---------------------------------------------------------------------------


def filter_significant_rows(stats_df, comparison_name, rename_to=None):
    """Keep only timepoints with at least one significant pairwise contrast.

    For each numeric timepoint, returns a single row marked
    ``significant=True`` if any pairwise contrast at that timepoint was
    significant; drops the timepoint entirely otherwise. Main-effect /
    interaction / non-numeric rows are dropped (the figure-annotation
    logic in ``plot_timelapse_lines`` already filters these by the
    timepoint string check).

    Designed to be passed to ``plot_timelapse_lines(..., stats_df=...)``
    so that non-significant timepoints receive no annotation at all
    (instead of the default behavior of drawing the p-value as text). For
    multi-group comparisons (e.g. DMSO vs. latrunculin-A vs.
    jasplakinolide) the timepoint is annotated with a single asterisk
    when *any* pairwise contrast at that timepoint is significant.

    Parameters
    ----------
    stats_df : pandas.DataFrame
        Output of ``run_stats.run_repeated_measures_stats`` /
        ``run_stats_lmm_r.run_lmm_stats`` / ``run_stats_lmm.run_lmm_stats``.
    comparison_name : str
        Match value for the ``comparison`` column.
    rename_to : str, optional
        When provided, the ``comparison`` column on the kept rows is
        rewritten to this value. Useful when the same underlying
        comparison is being plotted under a graph name that differs
        from the original (e.g. ``'<name>_rmanova'`` vs.
        ``'<name>_lmm'`` — ``plot_timelapse_lines`` matches stats rows
        to its figure via ``comparison == graphname``, so the rename
        keeps the match working).

    Returns
    -------
    pandas.DataFrame
        Possibly empty dataframe with at most one row per significant
        timepoint, all marked ``significant=True``. The ``comparison``
        column is set to ``rename_to`` when provided, otherwise left as
        ``comparison_name``.
    """
    if stats_df is None or len(stats_df) == 0:
        return stats_df

    sub = stats_df[stats_df['comparison'] == comparison_name].copy()
    sub['_tp_num'] = pd.to_numeric(sub['timepoint'], errors='coerce')
    sub = sub.dropna(subset=['_tp_num'])
    if sub.empty:
        out = stats_df.iloc[0:0].copy()
        return out

    keep = []
    for _tp, grp in sub.groupby('_tp_num', sort=True):
        if bool(grp['significant'].any()):
            row = grp.iloc[0].copy()
            row['significant'] = True
            keep.append(row)

    if not keep:
        return stats_df.iloc[0:0].copy()
    out = pd.DataFrame(keep).drop(columns=['_tp_num']).reset_index(drop=True)
    if rename_to is not None:
        out['comparison'] = rename_to
    return out


# ---------------------------------------------------------------------------
# Plot data archival
# ---------------------------------------------------------------------------

def save_plot_data(df_long, savepath, columns=None):
    """Write the exact long-form dataframe consumed by a plot to CSV.

    Parameters
    ----------
    df_long : pandas.DataFrame
        The long-form dataframe passed to the plotting helper. Already
        filtered to the rows actually drawn in the figure.
    savepath : str or pathlib.Path
        Output ``.csv`` path. The parent directory is created if
        missing.
    columns : sequence of str, optional
        Subset of columns to retain. Missing columns are dropped
        silently. If ``None``, the full frame is written.
    """
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        keep = [c for c in columns if c in df_long.columns]
        df_to_save = df_long[keep]
    else:
        df_to_save = df_long
    df_to_save.to_csv(savepath, index=False)

"""Linear mixed model fitting via R (lme4 + lmerTest + emmeans), driven
as a subprocess from Python. Drop-in replacement for
``run_stats_lmm.run_lmm_stats``: same call signature, same output schema.

Why subprocess instead of pymer4
--------------------------------
The published, reviewer-recognized pymer4 0.8.x API requires Python
3.11; the in-progress rewrite has a different undocumented API. Driving
``Rscript`` directly avoids the Python-R bridge entirely, runs in any
Python environment, and produces results identical to what published
``lmerTest``-based papers report. The R script is plain text and easy
for a reviewer to inspect or re-run in vanilla R.

R-side requirements
-------------------
``R``, ``lme4``, ``lmerTest``, ``emmeans``, ``jsonlite``. Install once::

    install.packages(c('lme4', 'lmerTest', 'emmeans', 'jsonlite'))

If ``Rscript`` is not on ``PATH``, the function raises ``RuntimeError``
with installation guidance.

Model
-----
``response ~ time * group + (1|subject) + (1|subject:cell)``, fit with
REML. Time and group are treated as categorical factors. Main-effect
and interaction p-values are Type-III ANOVA with Satterthwaite degrees
of freedom (``lmerTest``). Per-timepoint pairwise group contrasts are
extracted with ``emmeans``; FDR adjustment is applied here in Python
across timepoints to match the convention used by
``run_stats.repeated_measures_with_posthoc``.
"""

import json
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import microscopy_analysis.d04_plot_data.plot_timelapse_data as ptd

R_SCRIPT = Path(__file__).parent / 'fit_lmm.R'

_MAIN_LABELS = {'main_time', 'main_group', 'interaction'}


def _clean(name):
    """Sanitize a column name to keep it identifier-friendly inside R."""
    return ptd.clean_column_name(name)


def _check_rscript():
    if shutil.which('Rscript') is None:
        raise RuntimeError(
            "Rscript not found on PATH. Install R from https://cran.r-project.org "
            "and required packages: install.packages(c('lme4','lmerTest','emmeans','jsonlite'))"
        )
    if not R_SCRIPT.exists():
        raise RuntimeError(f"R fitting script missing: {R_SCRIPT}")


def _empty_stats(comparison_name, reason):
    return pd.DataFrame([
        {'timepoint': lbl, 'comparison': comparison_name,
         'p_value': np.nan, 'p_value_fdr': np.nan, 'significant': False,
         'test': f'LMM (R, {reason})'}
        for lbl in ('main_time', 'main_group', 'interaction')
    ])


def _stats_from_r(r_out, comparison_name, time_term, group_term, alpha):
    """Convert the R script's JSON output to the standard stats_df schema."""
    rows = []

    # Main-effect rows. Map R term names back to the canonical labels
    # used elsewhere in the pipeline.
    for me in r_out.get('main_effects', []) or []:
        term = me.get('term', '')
        if term == '(Intercept)':
            continue
        if ':' in term:
            label = 'interaction'
        elif term == time_term:
            label = 'main_time'
        elif term == group_term:
            label = 'main_group'
        else:
            label = f'effect:{term}'
        p = me.get('p_value')
        rows.append({
            'timepoint': label,
            'comparison': comparison_name,
            'p_value': p,
            'p_value_fdr': np.nan,
            'significant': bool(p is not None and np.isfinite(p) and p <= alpha),
            'test': 'LMM (R, Satterthwaite F)',
            'F_value': me.get('F_value'),
            'df_num': me.get('df_num'),
            'df_den': me.get('df_den'),
        })

    # Per-timepoint pairwise contrasts. Cast timepoint back to numeric
    # when possible so the figure-annotation logic in plot_timelapse_lines
    # can match it against numeric x-axis values.
    for ph in r_out.get('posthoc', []) or []:
        tp_raw = ph.get('timepoint')
        try:
            tp = float(tp_raw)
        except (ValueError, TypeError):
            tp = tp_raw
        rows.append({
            'timepoint': tp,
            'comparison': comparison_name,
            'p_value': ph.get('p_value'),
            'p_value_fdr': np.nan,
            'significant': False,
            'test': 'LMM (R, Satterthwaite t)',
            'estimate': ph.get('estimate'),
            'SE': ph.get('SE'),
            'df': ph.get('df'),
            't_ratio': ph.get('t_ratio'),
            'contrast': ph.get('contrast'),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return _empty_stats(comparison_name, 'no rows in R output')

    # FDR-Benjamini-Hochberg across per-timepoint contrast rows.
    posthoc_mask = ~out['timepoint'].astype(str).isin(_MAIN_LABELS)
    valid = posthoc_mask & out['p_value'].notna()
    if valid.any():
        rej, p_adj, _, _ = multipletests(
            out.loc[valid, 'p_value'].astype(float).values, method='fdr_bh',
        )
        out.loc[valid, 'p_value_fdr'] = p_adj
        out.loc[valid, 'significant'] = rej

    # Stable column order. Schema columns first (matches the other LMM /
    # RM-ANOVA outputs), then LMM-specific extras.
    front = ['timepoint', 'comparison', 'p_value', 'p_value_fdr', 'significant', 'test']
    extras = [c for c in out.columns if c not in front]
    return out[front + extras]


def run_lmm_stats(
    df_cells,
    subject,
    time,
    group_category,
    ycol,
    comparison_label,
    melt_df=True,
    cell_id=None,
    alpha=0.05,
):
    """Fit an LMM in R and return the standard stats_df.

    Drop-in replacement for ``run_stats_lmm.run_lmm_stats``: identical
    signature, identical output schema (with extra LMM-only columns
    appended).

    Parameters
    ----------
    df_cells : pandas.DataFrame
        Cell-level (NOT replicate-level) measurements.
    subject : str
        Column identifying the biological replicate. Random intercept.
    time : str
        Column with the timepoint. Cast to factor in R.
    group_category : str
        Column or, when ``melt_df=True``, the variable name to assign
        after melting ``comparison_label['columns']``.
    ycol : str
        Measurement column to model. When ``melt_df=True`` this is the
        name assigned to the melted value.
    comparison_label : dict
        ``comparison_label['name']`` is written to the output
        ``comparison`` column so the figure-annotation logic can match
        rows to figures.
    melt_df : bool, default True
        ``True`` for paired-within-cell comparisons (each cohort is a
        separate column of ``df_cells``); the frame is melted before
        modeling. ``False`` for between-cell comparisons.
    cell_id : str, optional
        Column identifying the cell. When provided, a cell-within-
        ``subject`` random intercept is added to the model.
    alpha : float, default 0.05
        Significance threshold applied to FDR-adjusted per-timepoint
        p-values.

    Returns
    -------
    pandas.DataFrame
        Columns ``timepoint, comparison, p_value, p_value_fdr,
        significant, test`` plus LMM-only columns (``F_value``,
        ``df_num``, ``df_den`` on main-effect rows;
        ``estimate``, ``SE``, ``df``, ``t_ratio``, ``contrast`` on
        per-timepoint rows).
    """
    _check_rscript()

    # --- 1. Optional melt to long form ---
    if melt_df:
        id_vars = [subject, time]
        if cell_id is not None and cell_id in df_cells.columns:
            id_vars = [cell_id] + id_vars
        df_long = df_cells.melt(
            id_vars=id_vars,
            value_vars=list(comparison_label['columns']),
            var_name=group_category,
            value_name=ycol,
        )
    else:
        df_long = df_cells.copy()

    # --- 2. Sanitize column names so the R formula is well-formed ---
    rename = {c: _clean(c) for c in [subject, time, group_category, ycol]}
    if cell_id is not None and cell_id in df_long.columns:
        rename[cell_id] = _clean(cell_id)
    df_long = df_long.rename(columns=rename)
    subject_c = rename[subject]
    time_c = rename[time]
    group_c = rename[group_category]
    ycol_c = rename[ycol]
    cell_c = rename[cell_id] if (cell_id is not None and cell_id in rename) else None

    keep = ([cell_c] if cell_c else []) + [subject_c, time_c, group_c, ycol_c]
    df_long = df_long[keep].dropna(subset=[ycol_c]).copy()

    if df_long.empty:
        return _empty_stats(comparison_label['name'], 'empty input')

    # --- 3. Write CSV, call Rscript, read JSON ---
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        csv_path = td / 'data.csv'
        json_path = td / 'results.json'
        df_long.to_csv(csv_path, index=False)

        cmd = [
            'Rscript', '--vanilla', str(R_SCRIPT),
            '--csv', str(csv_path),
            '--response', ycol_c,
            '--time', time_c,
            '--group', group_c,
            '--subject', subject_c,
            '--output', str(json_path),
        ]
        if cell_c:
            cmd += ['--cell', cell_c]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not json_path.exists():
            warnings.warn(
                f"R LMM fit failed for {comparison_label['name']!r} "
                f"(return code {result.returncode}). stderr:\n{result.stderr.strip()}"
            )
            return _empty_stats(comparison_label['name'], 'fit failed')

        try:
            r_out = json.loads(json_path.read_text())
        except Exception as e:
            warnings.warn(
                f"Could not parse R output JSON for {comparison_label['name']!r}: {e}"
            )
            return _empty_stats(comparison_label['name'], 'parse failed')

    # --- 4. Build stats_df with the standard schema ---
    return _stats_from_r(r_out, comparison_label['name'],
                         time_term=time_c, group_term=group_c, alpha=alpha)

"""Mixed-effects-model equivalent of ``run_stats.run_repeated_measures_stats``.

Fits cell-level data with biological replicate (and optionally cell within
replicate) as random intercepts. Returns a stats dataframe with the same
schema as ``run_stats.repeated_measures_with_posthoc`` so the same
plotting helpers in ``plot_timelapse_data`` can consume the output without
modification.

Backend
-------
``statsmodels.regression.mixed_linear_model`` (REML, L-BFGS, Powell
fallback). p-values are Wald (asymptotic chi-square) and may be
anti-conservative at small replicate counts; this should be reported in
methods. For Satterthwaite-corrected denominator degrees of freedom,
substitute an R/lme4 backend (e.g. ``pymer4``) — the public function
signature here is designed to make that swap local to this module.

Models fit
----------
- Full model: ``y ~ C(time) * C(group)`` with random intercepts on
  ``subject`` (and on ``cell`` within ``subject`` when ``cell_id`` is
  supplied). Joint Wald tests on the model terms produce the
  ``main_time``, ``main_group``, and ``interaction`` rows of the output.
- Per-timepoint contrasts: ``y ~ C(group)`` fit independently at each
  timepoint with the same random-effect structure. The Wald p-value on
  the ``C(group)`` term is then Benjamini-Hochberg FDR-adjusted across
  timepoints to populate ``p_value_fdr`` and ``significant``.

Output schema
-------------
``timepoint, comparison, p_value, p_value_fdr, significant, test``. All
non-``main``/``interaction`` rows carry ``comparison_label['name']`` so
``plot_timelapse_data.plot_timelapse_lines`` can match figure annotations
to stats rows.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

import microscopy_analysis.d04_plot_data.plot_timelapse_data as ptd


def _clean(name):
    """Sanitize a column name for safe use inside Patsy formula strings."""
    return ptd.clean_column_name(name)


def _fit_silent(formula, data, groups, vc_formula=None):
    """Fit a MixedLM with REML, suppressing convergence warnings.

    Tries L-BFGS first and falls back to Powell. Returns the fitted
    results object, or ``None`` if both optimizers fail.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        kwargs = dict(formula=formula, data=data, groups=groups, re_formula='1')
        if vc_formula:
            kwargs['vc_formula'] = vc_formula
        try:
            md = smf.mixedlm(**kwargs)
        except Exception:
            return None
        for method in ('lbfgs', 'powell'):
            try:
                return md.fit(reml=True, method=method)
            except Exception:
                continue
        return None


def _term_joint_p(res, term_prefix):
    """Joint Wald test that all coefficients with names starting with
    ``term_prefix`` are simultaneously zero. Returns ``np.nan`` if the
    test cannot be evaluated.

    Uses an explicit restriction matrix so coefficient names containing
    brackets, colons, or dots (e.g. ``C(group)[T.latA]:C(time)[T.4.0]``)
    do not need to be re-parsed as formula strings.
    """
    if res is None:
        return np.nan
    param_names = list(res.params.index)
    matching_idx = [i for i, name in enumerate(param_names) if name.startswith(term_prefix)]
    if not matching_idx:
        return np.nan
    R = np.zeros((len(matching_idx), len(param_names)))
    for row, col in enumerate(matching_idx):
        R[row, col] = 1.0
    for kwargs in ({'scalar': True}, {}):
        try:
            wt = res.wald_test(R, **kwargs)
            return float(np.squeeze(wt.pvalue))
        except TypeError:
            continue
        except Exception:
            return np.nan
    return np.nan


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
    """Mixed-effects equivalent of ``run_repeated_measures_stats``.

    Parameters
    ----------
    df_cells : pandas.DataFrame
        Cell-level (NOT biological-replicate-level) measurements.
    subject : str
        Column identifying the biological replicate. Modeled as a random
        intercept.
    time : str
        Column with the timepoint.
    group_category : str
        Column holding the categorical comparison factor. When
        ``melt_df=True`` this is the variable name assigned to the
        cohort axis after melting ``comparison_label['columns']``.
    ycol : str
        Measurement column to model. When ``melt_df=True`` this is the
        name assigned to the melted value column.
    comparison_label : dict
        Same structure as in ``run_stats.run_repeated_measures_stats``.
        ``comparison_label['name']`` is written to the output
        ``comparison`` column so plotting helpers can match figures to
        stats rows.
    melt_df : bool, default True
        ``True`` for paired-within-cell comparisons where each cohort is
        a separate column of ``df_cells``; the frame is melted before
        modeling. ``False`` for between-cell comparisons where the
        cohort factor is already a single column of ``df_cells``.
    cell_id : str, optional
        Column identifying the cell. When provided, a cell-within-
        ``subject`` random intercept is added so within-cell pairing
        across cohorts and timepoints is modeled explicitly.
    alpha : float, default 0.05
        Significance threshold applied to FDR-adjusted per-timepoint
        p-values to populate the ``significant`` flag.

    Returns
    -------
    pandas.DataFrame
        Columns ``timepoint, comparison, p_value, p_value_fdr,
        significant, test``. Rows in this order: ``main_time``,
        ``main_group``, ``interaction``, then one row per timepoint
        (FDR-Benjamini-Hochberg-adjusted across timepoints).
    """
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

    # --- 2. Sanitize column names for Patsy formulae ---
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

    # --- 3. Full model: y ~ C(time) * C(group) ---
    full_formula = f'{ycol_c} ~ C({time_c}) * C({group_c})'
    vc = {cell_c: f'0 + C({cell_c})'} if cell_c else None
    res_full = _fit_silent(full_formula, df_long, df_long[subject_c], vc_formula=vc)

    main_time_p = _term_joint_p(res_full, f'C({time_c})')
    main_group_p = _term_joint_p(res_full, f'C({group_c})')
    inter_p = _term_joint_p(res_full, f'C({time_c}):C({group_c})')

    rows = [
        {'timepoint': 'main_time',  'p_value': main_time_p,
         'p_value_fdr': np.nan,
         'significant': bool(np.isfinite(main_time_p) and main_time_p <= alpha),
         'test': 'LMM (Wald, joint)'},
        {'timepoint': 'main_group', 'p_value': main_group_p,
         'p_value_fdr': np.nan,
         'significant': bool(np.isfinite(main_group_p) and main_group_p <= alpha),
         'test': 'LMM (Wald, joint)'},
        {'timepoint': 'interaction', 'p_value': inter_p,
         'p_value_fdr': np.nan,
         'significant': bool(np.isfinite(inter_p) and inter_p <= alpha),
         'test': 'LMM (Wald, joint)'},
    ]

    # --- 4. Per-timepoint contrasts: y ~ C(group) at each timepoint ---
    time_levels = sorted(df_long[time_c].dropna().unique())
    for t in time_levels:
        d = df_long[df_long[time_c] == t]
        groups_here = pd.unique(d[group_c].dropna())
        if len(groups_here) < 2:
            rows.append({'timepoint': t, 'p_value': np.nan,
                         'p_value_fdr': np.nan, 'significant': False,
                         'test': 'n/a (<2 groups)'})
            continue

        f_t = f'{ycol_c} ~ C({group_c})'
        vc_t = {cell_c: f'0 + C({cell_c})'} if cell_c else None
        res_t = _fit_silent(f_t, d, d[subject_c], vc_formula=vc_t)
        p = _term_joint_p(res_t, f'C({group_c})')
        rows.append({'timepoint': t, 'p_value': p,
                     'p_value_fdr': np.nan, 'significant': False,
                     'test': 'LMM (Wald, per-timepoint)'})

    out = pd.DataFrame(rows)

    # --- 5. FDR correction across per-timepoint contrasts ---
    posthoc_mask = ~out['timepoint'].isin(['main_time', 'main_group', 'interaction'])
    valid = posthoc_mask & out['p_value'].notna() & np.isfinite(out['p_value'])
    if valid.any():
        rej, p_adj, _, _ = multipletests(out.loc[valid, 'p_value'], method='fdr_bh')
        out.loc[valid, 'p_value_fdr'] = p_adj
        out.loc[valid, 'significant'] = rej

    # --- 6. Tag every row with the figure-matching comparison name ---
    out['comparison'] = comparison_label['name']

    return out[['timepoint', 'comparison', 'p_value', 'p_value_fdr', 'significant', 'test']]

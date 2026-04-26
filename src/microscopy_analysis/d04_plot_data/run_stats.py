import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

import microscopy_analysis.d04_plot_data.plot_timelapse_data as ptd

def clean_vars_for_stats(df, subject, time, group, value):

    cols_to_clean = [subject, time, group, value]

    # build mapping {old: cleaned}
    col_mapping = {col: ptd.clean_column_name(col) for col in cols_to_clean}

    # rename using dict
    df = df.rename(columns=col_mapping)

    # unpack cleaned names
    subject, time, group, value = col_mapping.values()

    return df, subject, time, group, value

def run_repeated_measures_stats(df, subject, time, group_category, ycol, comparison_label, melt_df=True):
    if melt_df:
        df_long = df.melt(
            id_vars=[subject, time],
            value_vars=comparison_label['columns'],
            var_name=group_category,
            value_name=ycol
        )
    else:
        df_long = df

    posthoc_df = repeated_measures_with_posthoc(
        df_long=df_long,
        subject=subject,
        time=time,
        group=group_category,
        value=ycol
    )

    posthoc_df['comparison'] = comparison_label['name']
    return df_long, posthoc_df
#
# def repeated_measures_with_posthoc(df_long, subject, time, group, value, alpha=0.05):
#     """
#     RM-ANOVA (within: time, group) + per-timepoint posthocs (paired if matched subjects; else Welch).
#     Returns both overall interaction ANOVA and timepoint-specific post-hoc results.
#     """
#     df_long, subject, time, group, value = clean_vars_for_stats(df_long, subject, time, group, value)
#     df_long = df_long[[subject, time, group, value]].copy()
#     time_levels = sorted(df_long[time].unique())
#
#     # RM-ANOVA
#     aov = AnovaRM(df_long, depvar=value, subject=subject, within=[time, group]).fit()
#     at = aov.anova_table
#     inter_key = f'{time}:{group}' if f'{time}:{group}' in at.index else f'{group}:{time}'
#     inter_p = at.loc[inter_key, 'Pr > F']
#
#     # Record overall interaction ANOVA result
#     output_rows = [{
#         'timepoint': 'interaction',
#         'comparison': f'{time} × {group}',
#         'p_value': inter_p,
#         'p_value_fdr': np.nan,
#         'significant': inter_p <= alpha,
#         'test': 'RM-ANOVA',
#     }]
#
#     # Skip post-hocs if not significant
#     if not np.isfinite(inter_p) or inter_p > alpha:
#         return pd.DataFrame(output_rows)
#
#     # Post-hoc per timepoint
#     for t in time_levels:
#         d = df_long[df_long[time] == t]
#         groups_here = pd.unique(d[group].dropna())
#         if len(groups_here) != 2:
#             output_rows.append({
#                 'timepoint': t,
#                 'comparison': ' / '.join(groups_here),
#                 'p_value': np.nan,
#                 'p_value_fdr': np.nan,
#                 'significant': False,
#                 'test': 'n/a (!=2 groups)'
#             })
#             continue
#
#         g1, g2 = groups_here
#         comp_str = f"{g1} vs {g2}"
#
#         # Try paired test
#         pvt = d.pivot_table(index=subject, columns=group, values=value, aggfunc='mean')
#         s1, s2 = pvt.get(g1), pvt.get(g2)
#         if s1 is not None and s2 is not None:
#             paired_idx = s1.dropna().index.intersection(s2.dropna().index)
#             if len(paired_idx) >= 2:
#                 _, p = stats.ttest_rel(s1.loc[paired_idx], s2.loc[paired_idx], nan_policy='omit')
#                 output_rows.append({
#                     'timepoint': t,
#                     'comparison': comp_str,
#                     'p_value': p,
#                     'test': 'paired t-test',
#                     'n_paired': len(paired_idx)
#                 })
#                 continue
#
#         # Fallback: Welch
#         v1 = d.loc[d[group] == g1, value].dropna().values
#         v2 = d.loc[d[group] == g2, value].dropna().values
#         p = stats.ttest_ind(v1, v2, equal_var=False, nan_policy='omit')[1] if (len(v1) >= 2 and len(v2) >= 2) else np.nan
#         output_rows.append({
#             'timepoint': t,
#             'comparison': comp_str,
#             'p_value': p,
#             'test': 'unpaired t-test (Welch)'
#         })
#
#     out = pd.DataFrame(output_rows)
#
#     # FDR correction for all timepoint post-hocs only (exclude 'interaction' row)
#     mask = out['timepoint'] != 'interaction'
#     valid = out.loc[mask, 'p_value'].notna() & np.isfinite(out.loc[mask, 'p_value'])
#     if valid.any():
#         rej, p_adj, _, _ = multipletests(out.loc[mask & valid, 'p_value'], method='fdr_bh')
#         out.loc[mask & valid, 'p_value_fdr'] = p_adj
#         out.loc[mask & valid, 'significant'] = rej
#     else:
#         out.loc[mask, 'p_value_fdr'] = np.nan
#         out.loc[mask, 'significant'] = False
#
#     # Clean column order
#     cols = ['timepoint', 'comparison', 'p_value', 'p_value_fdr', 'significant', 'test']
#     if 'n_paired' in out.columns:
#         cols.append('n_paired')
#     return out[cols]


def repeated_measures_with_posthoc(df_long, subject, time, group, value, alpha=0.05):
    """
    RM-ANOVA (within: time, group) + per-timepoint posthocs (paired if matched subjects; else Welch).
    Returns main effect p-values, interaction p-value, and timepoint-specific post-hoc results.
    """
    df_long, subject, time, group, value = clean_vars_for_stats(df_long, subject, time, group, value)
    df_long = df_long[[subject, time, group, value]].copy()
    time_levels = sorted(df_long[time].unique())

    # RM-ANOVA
    aov = AnovaRM(df_long, depvar=value, subject=subject, within=[time, group]).fit()
    at = aov.anova_table
    inter_key = f'{time}:{group}' if f'{time}:{group}' in at.index else f'{group}:{time}'
    inter_p = at.loc[inter_key, 'Pr > F']
    time_p = at.loc[time, 'Pr > F']
    group_p = at.loc[group, 'Pr > F']

    # Record main effects + interaction
    output_rows = [
        {'timepoint': 'main_time', 'comparison': f'{time}', 'p_value': time_p, 'p_value_fdr': np.nan, 'significant': time_p <= alpha, 'test': 'RM-ANOVA'},
        {'timepoint': 'main_group', 'comparison': f'{group}', 'p_value': group_p, 'p_value_fdr': np.nan, 'significant': group_p <= alpha, 'test': 'RM-ANOVA'},
        {'timepoint': 'interaction', 'comparison': f'{time} × {group}', 'p_value': inter_p, 'p_value_fdr': np.nan, 'significant': inter_p <= alpha, 'test': 'RM-ANOVA'}
    ]

    # Skip post-hocs if interaction not significant
    if not np.isfinite(inter_p) or inter_p > alpha:
        return pd.DataFrame(output_rows)

    # Per-timepoint post-hocs
    for t in time_levels:
        d = df_long[df_long[time] == t]
        groups_here = pd.unique(d[group].dropna())
        if len(groups_here) != 2:
            output_rows.append({
                'timepoint': t,
                'comparison': ' / '.join(groups_here),
                'p_value': np.nan,
                'p_value_fdr': np.nan,
                'significant': False,
                'test': 'n/a (!=2 groups)'
            })
            continue

        g1, g2 = groups_here
        comp_str = f"{g1} vs {g2}"

        # Try paired test
        pvt = d.pivot_table(index=subject, columns=group, values=value, aggfunc='mean')
        s1, s2 = pvt.get(g1), pvt.get(g2)
        if s1 is not None and s2 is not None:
            paired_idx = s1.dropna().index.intersection(s2.dropna().index)
            if len(paired_idx) >= 2:
                _, p = stats.ttest_rel(s1.loc[paired_idx], s2.loc[paired_idx], nan_policy='omit')
                output_rows.append({
                    'timepoint': t,
                    'comparison': comp_str,
                    'p_value': p,
                    'test': 'paired t-test',
                    'n_paired': len(paired_idx)
                })
                continue

        # Fallback: Welch
        v1 = d.loc[d[group] == g1, value].dropna().values
        v2 = d.loc[d[group] == g2, value].dropna().values
        p = stats.ttest_ind(v1, v2, equal_var=False, nan_policy='omit')[1] if (len(v1) >= 2 and len(v2) >= 2) else np.nan
        output_rows.append({
            'timepoint': t,
            'comparison': comp_str,
            'p_value': p,
            'test': 'unpaired t-test (Welch)'
        })

    out = pd.DataFrame(output_rows)

    # FDR correction (post-hoc timepoints only)
    mask = ~out['timepoint'].isin(['main', 'interaction'])
    valid = out.loc[mask, 'p_value'].notna() & np.isfinite(out.loc[mask, 'p_value'])
    if valid.any():
        rej, p_adj, _, _ = multipletests(out.loc[mask & valid, 'p_value'], method='fdr_bh')
        out.loc[mask & valid, 'p_value_fdr'] = p_adj
        out.loc[mask & valid, 'significant'] = rej
    else:
        out.loc[mask, 'p_value_fdr'] = np.nan
        out.loc[mask, 'significant'] = False

    # Column order
    cols = ['timepoint', 'comparison', 'p_value', 'p_value_fdr', 'significant', 'test']
    if 'n_paired' in out.columns:
        cols.append('n_paired')
    return out[cols]

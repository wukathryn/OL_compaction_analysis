"""Per-timepoint normality diagnostics for the RM-ANOVA stats produced
by ``run_stats.run_repeated_measures_stats``.

Two complementary checks are exposed:

- ``add_normality_checks``: augments the stats dataframe with the
  Shapiro-Wilk test on the per-replicate *paired differences* at each
  timepoint. This is the assumption the paired t-test actually makes
  (normality of the differences, not of the raw values). At small N the
  test has low power; treat a non-significant ``shapiro_p`` as "no
  evidence of non-normality," not as evidence of normality.
- ``save_qq_plots``: writes a per-comparison Q-Q plot grid (one panel
  per timepoint) of the same paired differences. At small N (the
  regime where formal tests and non-parametric alternatives are largely
  uninformative) visual inspection of the Q-Q points against the
  reference line is the actually-informative diagnostic.

Rows of the stats dataframe that are not per-timepoint paired t-tests
(main-effect / interaction rows, fallback Welch rows, ``n/a`` rows) are
left untouched by ``add_normality_checks``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from microscopy_analysis.d04_plot_data.run_stats import clean_vars_for_stats


_AUGMENT_COLS = [
    'paired_diffs_n',
    'shapiro_W',
    'shapiro_p',
]


def _paired_diffs_at_t(df_long, time_c, t, subject_c, group_c, value_c):
    """Return the per-subject paired-difference vector at timepoint ``t``.

    Returns ``(diffs, n)`` where ``diffs`` is ``np.ndarray`` (possibly
    empty) and ``n`` is the number of paired subjects contributing.
    """
    d = df_long[df_long[time_c] == t]
    groups_here = pd.unique(d[group_c].dropna())
    if len(groups_here) != 2:
        return np.array([]), 0
    g1, g2 = groups_here
    pvt = d.pivot_table(index=subject_c, columns=group_c, values=value_c, aggfunc='mean')
    if g1 not in pvt or g2 not in pvt:
        return np.array([]), 0
    s1 = pvt[g1].dropna()
    s2 = pvt[g2].dropna()
    paired_idx = s1.index.intersection(s2.index)
    if len(paired_idx) == 0:
        return np.array([]), 0
    return (s1.loc[paired_idx] - s2.loc[paired_idx]).values, len(paired_idx)


def add_normality_checks(stats_df, df_long, subject, time, group, value, alpha=0.05):
    """Augment a RM-ANOVA stats dataframe with Shapiro-Wilk columns.

    Parameters
    ----------
    stats_df : pandas.DataFrame
        Output of ``run_stats.run_repeated_measures_stats`` (the second
        element of the returned tuple).
    df_long : pandas.DataFrame
        Long-form dataframe used for that RM-ANOVA call (the first
        element of the returned tuple).
    subject, time, group, value : str
        Same column names passed to ``run_repeated_measures_stats``.
        ``clean_vars_for_stats`` is applied so the resolved names match.
    alpha : float, default 0.05
        Reserved for future per-row significance flags.

    Returns
    -------
    pandas.DataFrame
        Copy of ``stats_df`` with three additional columns
        (``paired_diffs_n``, ``shapiro_W``, ``shapiro_p``). Rows that
        are not per-timepoint paired-t-test rows have ``NaN`` in the
        new columns.
    """
    df_long, subject_c, time_c, group_c, value_c = clean_vars_for_stats(
        df_long.copy(), subject, time, group, value,
    )

    out = stats_df.copy()
    for col in _AUGMENT_COLS:
        if col not in out.columns:
            out[col] = np.nan

    for idx, row in out.iterrows():
        if str(row.get('test', '')) != 'paired t-test':
            continue
        try:
            t = float(row['timepoint'])
        except (ValueError, TypeError):
            continue

        diffs, n = _paired_diffs_at_t(df_long, time_c, t, subject_c, group_c, value_c)
        out.loc[idx, 'paired_diffs_n'] = n
        if n < 3:
            # Shapiro-Wilk requires n >= 3.
            continue
        if not np.isfinite(diffs).all() or np.ptp(diffs) == 0:
            # Shapiro-Wilk is undefined for constant input. This arises
            # at the normalization timepoint of per-cell-normalized
            # series, where every paired difference is exactly zero.
            continue

        try:
            sw_W, sw_p = stats.shapiro(diffs)
            out.loc[idx, 'shapiro_W'] = float(sw_W)
            out.loc[idx, 'shapiro_p'] = float(sw_p)
        except Exception:
            pass

    return out


def save_qq_plots(
    df_long,
    comparison_label,
    savepath,
    subject,
    time,
    group,
    value,
    ncols=4,
    panel_size=(2.0, 2.0),
    show=True,
):
    """Write a Q-Q plot grid of per-replicate paired differences.

    One panel per timepoint, arranged in a grid. Each panel is a
    quantile-quantile plot of the paired differences against the
    standard normal, with the reference line drawn. A subtitle on each
    panel reports n and the Shapiro-Wilk p-value.

    Parameters
    ----------
    df_long : pandas.DataFrame
        Long-form dataframe used by the RM-ANOVA call (one row per
        subject x timepoint x group).
    comparison_label : dict
        Same dict passed to ``run_repeated_measures_stats``;
        ``comparison_label['name']`` is used as the figure title.
    savepath : str or pathlib.Path
        Output ``.svg`` (or other matplotlib-supported) path. The
        parent directory is created if missing.
    subject, time, group, value : str
        Column names. ``clean_vars_for_stats`` is applied so the
        resolved names match.
    ncols : int, default 4
        Number of columns in the panel grid.
    panel_size : (float, float), default ``(2.0, 2.0)``
        Per-panel figure size in inches.
    show : bool, default ``True``
        When ``True``, the figure is rendered inline (in a notebook
        context) before being closed. Set ``False`` for batch /
        non-interactive use.

    Notes
    -----
    Timepoints with fewer than three paired replicates are skipped
    (Shapiro-Wilk and Q-Q plots are not informative below n = 3).
    Returns silently without writing a file if no timepoint passes the
    threshold.
    """
    df_long, subject_c, time_c, group_c, value_c = clean_vars_for_stats(
        df_long.copy(), subject, time, group, value,
    )

    timepoints = sorted(pd.unique(df_long[time_c].dropna()))

    panels = []
    for t in timepoints:
        diffs, n = _paired_diffs_at_t(df_long, time_c, t, subject_c, group_c, value_c)
        if n < 3:
            continue
        if not np.isfinite(diffs).all() or np.ptp(diffs) == 0:
            # Constant or non-finite differences (e.g., the
            # normalization timepoint of per-cell-normalized series)
            # produce a degenerate Q-Q plot and an undefined Shapiro
            # statistic; skip rather than draw a misleading panel.
            continue
        try:
            _, sw_p = stats.shapiro(diffs)
        except Exception:
            sw_p = np.nan
        panels.append((t, diffs, n, sw_p))

    if not panels:
        return

    n_panels = len(panels)
    nrows = int(np.ceil(n_panels / ncols))
    figsize = (panel_size[0] * ncols, panel_size[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for ax_idx, (t, diffs, n, sw_p) in enumerate(panels):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        # probplot draws scatter points and the reference line.
        stats.probplot(diffs, dist='norm', plot=ax)
        sw_str = 'n/a' if not np.isfinite(sw_p) else f'{sw_p:.2g}'
        ax.set_title(f't = {t} (n={n}, Shapiro p={sw_str})', fontsize=8)
        ax.set_xlabel('Theoretical quantiles', fontsize=7)
        ax.set_ylabel('Paired difference', fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused panels.
    for ax_idx in range(n_panels, nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].axis('off')

    name = comparison_label.get('name', '?') if isinstance(comparison_label, dict) else str(comparison_label)
    fig.suptitle(f'Q-Q of paired differences: {name}', fontsize=10)
    fig.tight_layout()

    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

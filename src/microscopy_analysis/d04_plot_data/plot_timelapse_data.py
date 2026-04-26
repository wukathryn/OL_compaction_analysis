import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import textwrap
import re

import microscopy_analysis.d04_plot_data.plot_settings as ps

def clean_column_name(name):
    """Sanitize column name for use in filenames."""
    return name.lower().replace(' ', '_').replace('/', '').replace('(', '').replace(')', '')

def wrap_text(text, width=15):
    """Wrap text longer than specified width"""
    return '\n'.join(textwrap.wrap(text, width=width))


def style_standard_plot(ax):

    sns.despine(ax=ax)

    # Set y-axis starting at 0 if no negative y values
    ymins, ymaxs = ax.get_ylim()
    if ymins >= 0:
        ax.set_ylim(bottom=0)

    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


def plot_individual_tp(df_long, stats_df, tp, xcol, ycol, group_col, comparison_label, labels, palette=None, savepath=None, figsize=ps.scatter_figsize):

    df_tp = df_long[df_long[xcol] == tp].copy()

    sig_row = stats_df[
        (stats_df['timepoint'] == tp) &
        (stats_df['comparison'] == comparison_label)
    ]

    # Relabel group names
    colnames = df_tp[group_col].unique()
    if len(colnames) == 2:
        mapping = dict(zip(colnames, labels))
        df_tp[group_col] = df_tp[group_col].map(mapping)

    fig, ax = plt.subplots(figsize=figsize)

    sns.stripplot(
        data=df_tp, x=group_col, y=ycol,
        jitter=True, size=7,
        ax=ax, hue=group_col, palette=palette, legend=False
    )

    sns.pointplot(
        data=df_tp, x=group_col, y=ycol,
        linestyle='', errorbar='se',
        marker='_', markersize=25, markeredgewidth=2.5,
        zorder=3, color='k', capsize=0.1,
        err_kws={'linewidth': 1}, ax=ax
    )

    # Add asterisks for significant p-values (or p-values for NS values)
    if not sig_row.empty:
        y_max = df_tp[ycol].max()
        line_height = y_max * 1.05

        # Draw bracket between groups
        ax.plot([0, 1], [line_height, line_height], lw=0.75, color='k')
        tick_height = y_max * 0.01
        ax.plot([0, 0], [line_height - tick_height, line_height], lw=0.75, color='k')
        ax.plot([1, 1], [line_height - tick_height, line_height], lw=0.75, color='k')

        # Add significance marker or p-value
        text_height = line_height * 1.02

        if sig_row.iloc[0]['significant']:
            ax.text(0.5, text_height, '*', ha='center', va='center', fontsize=16)
        else:
            pval = sig_row.iloc[0]['p_value_fdr']
            if pd.notnull(pval):
                label = f"{pval:.2g}"
                ax.text(0.5, text_height, label, ha='center', va='bottom', fontsize=10)

    sns.despine()
    ax.set_xlim(-0.4, 1.4)
    ymin, ymax = ax.get_ylim()
    if ymin > 0:
        ax.set_ylim(0, ymax*1.1)

    plt.tight_layout()
    plt.show()

    if savepath:
        fig.savefig(savepath, format='svg', bbox_inches='tight')

def plot_timelapse_lines(df_long, xcol, ycol, hue, hue_order, palette=None, group_labels=None, tx_line=None, graphname=None, save_dir=None, stats_df=None, figsize=ps.timelapse_figsize):
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the lines
    sns.lineplot(data=df_long, x=xcol, y=ycol, errorbar='se', hue=hue, hue_order=hue_order, palette=palette, ax=ax)

    if tx_line is not None:
        ax.axvline(x=tx_line, color='k', linewidth=0.2, linestyle='dashed')

    # Calculate upper SEM bounds (for asterisk placement)
    sems = df_long.groupby([xcol, hue])[ycol].agg(['mean', 'sem'])
    sems['upperSEM'] = sems['mean'] + sems['sem']
    upper_sems = sems.groupby(xcol)['upperSEM'].max().reset_index()
    mean_uppersem = upper_sems['upperSEM'].mean()
    spacing_above_sem = 0.05 * mean_uppersem
    upper_sems['y_star'] = upper_sems['upperSEM'] + spacing_above_sem

    # Get previous and next y_star values
    prev_y_star = upper_sems['y_star'].shift(1)
    next_y_star = upper_sems['y_star'].shift(-1)

    # Replace y_star with max of [self, prev, next]
    upper_sems['y_star'] = np.maximum.reduce([
        upper_sems['y_star'],
        prev_y_star.fillna(0),
        next_y_star.fillna(0)
    ])

    # Add stars for significant timepoints
    if stats_df is not None:
        sig_rows = stats_df[(stats_df['comparison'] == graphname) & (stats_df['timepoint'] != 'interaction')].copy()

        # Keep only numeric timepoints
        sig_rows['timepoint_num'] = pd.to_numeric(sig_rows['timepoint'], errors='coerce')
        sig_rows = sig_rows.dropna(subset=['timepoint_num'])

        for _, row in sig_rows.iterrows():
            tp = float(row['timepoint_num'])

            y_star_series = upper_sems.loc[upper_sems[xcol] == tp, 'y_star']
            if not y_star_series.empty:
                y_star = float(y_star_series.values[0])
            else:
                y_star = df_long.loc[df_long[xcol] == tp, ycol].max()

            if row['significant'] is True:
                ax.text(tp, y_star, '*', ha='center', va='center', fontsize=14, color='k')
            else:
                pval = row.get('p_value_fdr', np.nan)
                if pd.notnull(pval):
                    ax.text(tp, y_star, f"{pval:.2g}", ha='center', va='bottom', fontsize=7, color='k')

    ylabel = edit_ycol(ycol)
    ylabel = wrap_text(ylabel, 30)
    ax.set_ylabel(ylabel)

    # Rename legend labels
    if group_labels is not None:
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=group_labels)

    style_standard_plot(ax)
    plt.show()

    if not ((graphname is None) or (save_dir is None)):
        fig.savefig(save_dir / f'{graphname}.svg', format='svg', bbox_inches='tight')


def clean_column_name(colname):
    if colname is None:
        return ''
    colname = colname.replace('compact', 'cmp')
    colname = colname.replace('%', 'perc')
    colname = colname.replace('-', '')
    colname = re.sub(r'\W+', '_', colname)
    colname = colname.strip('_')
    colname = colname.replace('compact', 'cmp')
    return colname.lower()


def edit_ycol(label):
    label = label.replace('change in', 'Δ').strip()
    if not '(μm²)' in label:
        label = label.replace('area', 'area (μm²)').strip()
    label = label.replace('-positive', '+').strip()
    label = label.replace('num', '#').strip()
    return label

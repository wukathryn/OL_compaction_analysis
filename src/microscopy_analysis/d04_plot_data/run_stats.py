import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def style_standard_plot(ax):
    """Standard formatting for plots."""
    sns.despine(ax=ax)
    ymins, ymaxs = ax.get_ylim()
    if ymins >= 0:
        ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

def plot_individual_tp(
    df_long, stats_df, tp, xcol, ycol, group_col,
    comparison_label, labels, palette=None, savepath=None
):
    """Plot scatter + mean ± SEM for a specific timepoint, with significance annotation."""
    df_tp = df_long[df_long[xcol] == tp].copy()
    sig_row = stats_df[
        (stats_df['timepoint'] == tp) & (stats_df['comparison'] == comparison_label)
    ]

    # Relabel groups
    colnames = df_tp[group_col].unique()
    if len(colnames) == 2:
        mapping = dict(zip(colnames, labels))
        df_tp[group_col] = df_tp[group_col].map(mapping)

    fig, ax = plt.subplots(figsize=(2.5, 3))
    sns.stripplot(data=df_tp, x=group_col, y=ycol, jitter=True, size=7,
                  ax=ax, hue=group_col, palette=palette, legend=False)
    sns.pointplot(data=df_tp, x=group_col, y=ycol, linestyle='', errorbar='se',
                  marker='_', markersize=25, markeredgewidth=2.5,
                  zorder=3, color='k', capsize=0.1,
                  err_kws={'linewidth': 1}, ax=ax)

    # Significance
    if not sig_row.empty:
        y_max = df_tp[ycol].max()
        line_height = y_max * 1.05
        tick_height = y_max * 0.01
        ax.plot([0, 1], [line_height, line_height], lw=0.75, color='k')
        ax.plot([0, 0], [line_height - tick_height, line_height], lw=0.75, color='k')
        ax.plot([1, 1], [line_height - tick_height, line_height], lw=0.75, color='k')
        text_height = line_height * 1.001
        if sig_row.iloc[0]['significant']:
            ax.text(0.5, text_height, '*', ha='center', va='bottom', fontsize=16)
        else:
            pval = sig_row.iloc[0]['p_value_fdr']
            if pd.notnull(pval):
                ax.text(0.5, text_height, f"{pval:.2g}", ha='center', va='bottom', fontsize=10)

    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(ycol)
    style_standard_plot(ax)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, format='svg', bbox_inches='tight')
    plt.show()


def plot_timelapse_lines(df_long, xcol, ycol, hue, hue_order, palette=None,
                         group_labels=None, graphname=None, save_dir=None,
                         stats_df=None, figsize=(5, 3)):
    """Plot timelapse lines with significance stars or FDR values."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=df_long, x=xcol, y=ycol, errorbar='se',
                 hue=hue, hue_order=hue_order, palette=palette, ax=ax)

    # Calculate SEM-based star height
    sems = df_long.groupby([xcol, hue])[ycol].agg(['mean', 'sem'])
    sems['upperSEM'] = sems['mean'] + sems['sem']
    upper_sems = sems.groupby(xcol)['upperSEM'].max().reset_index()
    spacing = 0.7 * upper_sems['upperSEM'].mean()
    upper_sems['y_star'] = upper_sems['upperSEM'] + spacing

    # Annotate
    if stats_df is not None:
        sig_rows = stats_df[(stats_df['comparison'] == graphname) & (stats_df['timepoint'] != 'interaction')]
        for _, row in sig_rows.iterrows():
            tp = row['timepoint']
            y_star_row = upper_sems[upper_sems[xcol] == tp]
            if not y_star_row.empty:
                y_star = y_star_row['y_star'].values[0]
                if row['significant']:
                    ax.text(tp, y_star, '*', ha='center', va='bottom', fontsize=14, color='k')
                else:
                    label = f"{row['p_value_fdr']:.2g}" if pd.notnull(row['p_value_fdr']) else ''
                    ax.text(tp, y_star, label, ha='center', va='bottom', fontsize=7, color='k')

    if group_labels is not None:
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=group_labels)

    style_standard_plot(ax)
    ax.set_ylabel(ycol)
    plt.tight_layout()
    if graphname and save_dir:
        fig.savefig(save_dir / f"{graphname}.svg", format='svg', bbox_inches='tight')
    plt.show()
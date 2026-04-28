"""Shared plotting style for publication figures.

Defines the rc parameters, figure sizes, and color palettes used by every
figure-generating notebook in this repository so that figures share a single,
consistent visual language. The same rc parameters are mirrored in
``paper_figures.mplstyle`` at the repository root; notebooks can either import
the constants from this module or load the style file with
``plt.style.use('paper_figures.mplstyle')``.

Importing this module applies the style globally via
``sns.set(rc=rc)`` and ``sns.set_style("ticks")``.
"""

import seaborn as sns

# --- Font and line settings ----------------------------------------------
dpi = 150
xtick_fontsize = 10
ytick_fontsize = 10
axislabel_fontsize = 11
ticks_linewidth = 1
axes_linewidth = 1
axes_labelpad = 7.5

# Fill/edge colors for individual-cell scatter points plotted underneath
# biological-replicate means.
smallpts_fillcolor = '#DBDBDB'
smallpts_edgecolor = '#AFAFAF'

rc = {
    'svg.fonttype': 'none',
    'font.family': 'Arial',
    'figure.dpi': dpi,
    'axes.linewidth': axes_linewidth,
    'axes.labelweight': 'bold',
    'axes.labelsize': axislabel_fontsize,
    'axes.labelpad': axes_labelpad,
    'xtick.major.width': ticks_linewidth,
    'ytick.major.width': ticks_linewidth,
    'xtick.labelsize': xtick_fontsize,
    'ytick.labelsize': ytick_fontsize,
}

# --- Figure sizes (inches) -----------------------------------------------
timelapse_figsize = (3.5, 3)
scatter_figsize = (2.5, 3)

# --- Color palettes ------------------------------------------------------
# CAAX-positive vs. compacted region comparisons.
cmp_color = '#A218A2'
caax_color = '#116F11'
cmpreg_palette = [caax_color, cmp_color]

# Two-condition control / latrunculin-A comparisons.
ctrl_color = '#dd8452'
lat_color = '#4c72b0'
ctrllat_palette = [ctrl_color, lat_color]

# Two-condition control / DeAct (actin-disrupting construct) comparisons.
# Shares its hex code with ``lat_color`` but is named separately so that
# figure code documents which biological perturbation is being plotted.
deact_color = '#4c72b0'
ctrldeact_palette = [ctrl_color, deact_color]

# Three-condition drug-treatment palette (DMSO / latrunculin-A / jasplakinolide).
# Uses muted Set2-style hues so all three categories remain distinguishable
# in print and to viewers with color-vision deficiencies.
drugtx_ctrl_color = '#8da0cb'
drugtx_lat_color = '#66c2a5'
drugtx_jasp_color = '#fc8d62'
drugtx_palette = [drugtx_ctrl_color, drugtx_lat_color, drugtx_jasp_color]

# --- Apply global seaborn style ------------------------------------------
sns.set(rc=rc)
sns.set_style("ticks")

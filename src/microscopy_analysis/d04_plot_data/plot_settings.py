import seaborn as sns

# Font and style settings
dpi = 150
smallpts_fillcolor = '#DBDBDB'
smallpts_edgecolor = '#AFAFAF'
xtick_fontsize = 10
ytick_fontsize = 10
axislabel_fontsize = 11
ticks_linewidth = 1
axes_linewidth = 1
axes_labelpad = 7.5

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
    'ytick.labelsize': ytick_fontsize
}

# Figure sizes
timelapse_figsize = (3.5, 3)
scatter_figsize = (2.5, 3)

caax_color = '#A218A2'
cmp_color = '#116F11'
cmpreg_palette = [caax_color, cmp_color]

ctrl_color = '#dd8452'
lat_color = '#4c72b0'
ctrllat_palette = [ctrl_color, lat_color]

# Apply global seaborn style
sns.set(rc=rc)
sns.set_style("ticks")

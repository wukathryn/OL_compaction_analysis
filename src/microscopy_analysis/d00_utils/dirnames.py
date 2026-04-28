"""Standard directory names used across the per-experiment processing tree.

A typical processed experiment lives at ``<experiment>/img_processing/`` and
contains subdirectories for each pipeline stage (raw OME-TIFFs, aligned stacks,
background-subtracted images, segmentations, masks, exclusion analysis, tables,
graphs, and so on). Pipeline modules read these constants instead of hard-coded
strings so directory layouts stay consistent across notebooks.
"""

CZI_dirname = 'CZI'
aligned_dirname = 'aligned'
orig_unaligned_dirname = 'orig_unaligned'
raw_ometif_dirname = 'raw_ometifs'
raw_ometif_chsubset_dirname = 'raw_ometifs_chsubset'
stack_dirname = 'caax_cell_stack'
init_rescale_dirname = 'init_rescaled'
bg_sub_dirname = 'bg_subtracted'
labbkit_seg_dirname = 'labkit_seg'
user_input_dirname = 'user_input_tables'
excluded_dirname = 'excluded'
masked_imgs_dirname = 'masked_imgs'
masked_compaction_dirname = 'masked_compaction'
corr_dirname = 'correlations'
ch_aligned_dirname = 'ch_aligned'
excl_analysis_dirname = 'exclusion_analysis'
cellmask_dirname = 'cellmasks'
proc_dirname = 'img_processing'
analyses_dirname = 'analyses'
ratiometric_dirname = 'ratiometric'
int_dirname = 'intensities'
masks_dirname = 'masks'
polygon_ROI_dirname = 'polygonmasks'
refinedmasks_dirname = 'refinedmasks'
figs_dirname = 'figs'
tables_dirname = 'tables'
graphs_dirname = 'graphs'

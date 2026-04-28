"""Image-analysis pipeline for fluorescence microscopy of compaction events.

Subpackages
-----------
``d00_utils``
    Path conventions, OME metadata helpers, dataframe save / column helpers,
    and notebook-side import setup.
``d01_init_proc``
    Initial processing of CZI data: file conversion, channel and timepoint
    alignment, polygon-ROI mask application, background subtraction, and
    visualization / rescaling utilities.
``d02_metrics``
    Per-image area and intensity readouts from segmented OME-TIFF stacks;
    derives cell area, percent compaction, and integrated densities used
    by the plotting notebooks.
``d04_plot_data``
    Plot construction, statistical tests (including linear mixed models via
    R), and figure layout helpers used by the manuscript figure notebooks.
"""

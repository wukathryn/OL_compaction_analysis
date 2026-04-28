"""Metric computation from segmented multi-channel image stacks.

Functions in this subpackage assume per-image OME-TIFF stacks with the
dimension order (T, C, Z, Y, X). The compaction analysis layers two
binary segmentation channels on top of the raw fluorescence channels
(`compacted` and `CAAX-positive`); the helpers here count pixels in
those segmentations to obtain real-world areas, compute mean and median
intensities of fluorescence channels under arbitrary masks, and derive
per-cell summary metrics (cell area, % compaction, integrated density).
"""

from microscopy_analysis.d02_metrics.compute_metrics import (
    compute_areas,
    compute_int,
    derive_compaction_metrics,
)

__all__ = ["compute_areas", "compute_int", "derive_compaction_metrics"]

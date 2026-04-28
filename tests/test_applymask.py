"""Unit tests for the pure morphological helpers in d01_init_proc.applymask."""

from __future__ import annotations

import numpy as np

from microscopy_analysis.d01_init_proc import applymask as am


# ---------------------------------------------------------------------------
# mask_img
# ---------------------------------------------------------------------------

def test_mask_img_zeros_pixels_outside_mask():
    img = np.arange(16, dtype=np.uint16).reshape(1, 1, 1, 4, 4) + 1
    mask_2d = np.zeros((4, 4), dtype=np.uint8)
    mask_2d[1:3, 1:3] = 1  # 2x2 centered

    out = am.mask_img(img, mask_2d)

    # Outside the mask: zeroed.
    assert out[0, 0, 0, 0, 0] == 0
    # Inside the mask: original values preserved.
    assert out[0, 0, 0, 1, 1] == img[0, 0, 0, 1, 1]
    assert out[0, 0, 0, 2, 2] == img[0, 0, 0, 2, 2]


def test_mask_img_handles_already_5d_mask():
    img = np.ones((1, 2, 1, 3, 3), dtype=np.uint16)
    mask_5d = np.zeros((1, 1, 1, 3, 3), dtype=np.uint8)
    mask_5d[0, 0, 0, 1, 1] = 1
    out = am.mask_img(img, mask_5d)
    # Only the (1,1) pixel survives; mask broadcasts across all channels.
    assert out[0, 0, 0, 1, 1] == 1
    assert out[0, 1, 0, 1, 1] == 1
    assert out.sum() == 2


# ---------------------------------------------------------------------------
# remove_small_objects (wrapper applied per timepoint)
# ---------------------------------------------------------------------------

def test_remove_small_objects_drops_blobs_below_min_size():
    bin_img = np.zeros((1, 1, 1, 6, 6), dtype=bool)
    # Big blob: 3x3 = 9 pixels
    bin_img[0, 0, 0, 0:3, 0:3] = True
    # Tiny blob: 1 pixel
    bin_img[0, 0, 0, 5, 5] = True

    out = am.remove_small_objects(bin_img, min_size=5)

    # 1-pixel blob is removed; 9-pixel blob retained.
    assert not out[0, 0, 0, 5, 5]
    assert out[0, 0, 0, 0, 0]


# ---------------------------------------------------------------------------
# remove_small_holes (wrapper applied per timepoint)
# ---------------------------------------------------------------------------

def test_remove_small_holes_fills_small_interior_holes():
    bin_img = np.ones((1, 1, 1, 5, 5), dtype=bool)
    # Single-pixel hole in a sea of True
    bin_img[0, 0, 0, 2, 2] = False

    out = am.remove_small_holes(bin_img, area_threshold=5)

    # The 1-pixel hole should be filled (i.e., turned True).
    assert out[0, 0, 0, 2, 2]


# ---------------------------------------------------------------------------
# remove_unconnected_areas (keeps the N largest connected components)
# ---------------------------------------------------------------------------

def test_remove_unconnected_areas_keeps_only_largest():
    bin_img = np.zeros((1, 1, 1, 8, 8), dtype=bool)
    # Big blob: 4x4 = 16 px
    bin_img[0, 0, 0, 0:4, 0:4] = True
    # Tiny blob: 2x2 = 4 px (well-separated from the big blob)
    bin_img[0, 0, 0, 6:8, 6:8] = True

    out = am.remove_unconnected_areas(bin_img, num_areas=1)

    # Big blob retained
    assert out[0, 0, 0, 0, 0]
    # Tiny blob dropped
    assert not out[0, 0, 0, 7, 7]

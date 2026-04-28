"""Unit tests for d01_init_proc.subtractbg pure helpers.

Tests the numpy-only subtraction, smoothing, clipping, and parameter-parsing
helpers. The notebook-driven `subtract_bg` orchestration function is not
covered here because it requires CSV input and writes OME-TIFFs to disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from microscopy_analysis.d01_init_proc import subtractbg as sb


# ---------------------------------------------------------------------------
# clip_upper_outliers
# ---------------------------------------------------------------------------

def test_clip_upper_outliers_clips_extremes_only():
    # (T=1, C=1, Z=1, Y=1, X=10) frame: nine ones and one outlier of 1000.
    img = np.ones((1, 1, 1, 1, 10), dtype=np.uint16)
    img[0, 0, 0, 0, 9] = 1000

    # `np.nanpercentile(..., method='higher')` for n=10 with q=80 picks the
    # value at sorted index ceil(0.80 * 9) = 8 -> 1. Setting the threshold to
    # 1 clips the outlier (1000) down to 1.
    out = sb.clip_upper_outliers(img, outlier_perc=80)

    assert out[0, 0, 0, 0, 9] == 1
    assert (out[0, 0, 0, 0, :9] == 1).all()
    assert out.dtype == img.dtype


def test_clip_upper_outliers_ignores_zeros():
    """Zero pixels are treated as background (NaN'd out before percentile)."""
    img = np.zeros((1, 1, 1, 1, 4), dtype=np.uint16)
    img[0, 0, 0, 0, :] = [0, 0, 1, 100]

    out = sb.clip_upper_outliers(img, outlier_perc=50)
    # Percentile is computed over the nonzero values [1, 100]; 50th percentile
    # ('higher' interpolation) -> 100. So clipping at 100 leaves 100 intact.
    assert out[0, 0, 0, 0, 3] == 100
    # Zeros stay zero.
    assert (out[0, 0, 0, 0, :2] == 0).all()


# ---------------------------------------------------------------------------
# subtract_background
# ---------------------------------------------------------------------------

def test_subtract_background_subtracts_per_channel_threshold():
    img = np.array([[10, 20], [30, 40]], dtype=np.uint16).reshape(1, 1, 1, 2, 2)
    thresholds = np.array([[[15]]])  # (T=1, C=1, Z=1)

    out = sb.subtract_background(img, thresholds)

    expected = np.array([[0, 5], [15, 25]], dtype=np.uint16).reshape(1, 1, 1, 2, 2)
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == np.uint16


def test_subtract_background_floors_at_zero():
    """Pixels below threshold become 0, never negative."""
    img = np.array([1, 2, 3], dtype=np.uint16).reshape(1, 1, 1, 1, 3)
    thresholds = np.array([[[10]]])
    out = sb.subtract_background(img, thresholds)
    assert (out == 0).all()


# ---------------------------------------------------------------------------
# gaussian_smoothing
# ---------------------------------------------------------------------------

def test_gaussian_smoothing_preserves_zero_pixels():
    """Background pixels (value=0) must remain zero after smoothing."""
    img = np.zeros((1, 1, 1, 5, 5), dtype=float)
    img[0, 0, 0, 2, 2] = 100.0  # single bright pixel in the center

    out = sb.gaussian_smoothing(img.copy(), sigma=1.0)

    # The center pixel (which was nonzero) is allowed to change.
    # All originally-zero pixels stay zero, even though Gaussian filtering
    # would have spread some signal into them.
    zero_mask = img == 0
    assert (out[zero_mask] == 0).all()


# ---------------------------------------------------------------------------
# subtract_median
# ---------------------------------------------------------------------------

def test_subtract_median_subtracts_nonzero_median_per_frame():
    # Frame: [0, 1, 2, 3, 4]. Nonzero median over [1,2,3,4] = 2.5.
    img = np.array([0, 1, 2, 3, 4], dtype=np.uint16).reshape(1, 1, 1, 1, 5)

    out = sb.subtract_median(img)

    # 0 - 2.5 -> 0 (floor); 1 - 2.5 -> 0; 2 - 2.5 -> 0; 3 - 2.5 -> 0 (cast);
    # 4 - 2.5 -> 1 (cast). uint16 cast truncates toward zero.
    assert out[0, 0, 0, 0, 0] == 0
    assert out[0, 0, 0, 0, 4] == 1


# ---------------------------------------------------------------------------
# param_to_list
# ---------------------------------------------------------------------------

def test_param_to_list_scalar_returned_as_list():
    assert sb.param_to_list(5) == [5]
    assert sb.param_to_list(2.5) == [2.5]


def test_param_to_list_list_returned_as_list():
    assert sb.param_to_list([1, 2, 3]) == [1, 2, 3]


def test_param_to_list_string_parsed_via_literal_eval():
    assert sb.param_to_list("[1, 2, 3]") == [1, 2, 3]


def test_param_to_list_rejects_unsupported_types():
    # A dict is neither a string (so ast.literal_eval is skipped), nor a
    # numeric scalar, nor a list, so the final assertion fires.
    with pytest.raises(AssertionError, match="number or a list of numbers"):
        sb.param_to_list({"key": 1})


def test_param_to_list_raises_on_non_literal_string():
    # An unparseable string flows into ast.literal_eval, which raises
    # ValueError (not AssertionError).
    with pytest.raises(ValueError):
        sb.param_to_list("hello")

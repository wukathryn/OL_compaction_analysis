"""Unit tests for d02_metrics.compute_metrics.

Tests are intentionally pure: they construct synthetic numpy arrays and
dataframes so they can run without any image files on disk and without
manual input. Run with `pytest` from the repo root.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microscopy_analysis.d02_metrics import (
    compute_areas,
    compute_int,
    derive_compaction_metrics,
)


# ---------------------------------------------------------------------------
# compute_areas
# ---------------------------------------------------------------------------

def _make_seg_two_channels():
    """Two segmentation channels of known size, T=2 frames, Z=1, 4x4 spatial."""
    seg = np.zeros((2, 2, 1, 4, 4), dtype=bool)
    # frame 0: ch0 covers a 2x2 square (4 px), ch1 covers a 3x3 square (9 px)
    seg[0, 0, 0, 0:2, 0:2] = True
    seg[0, 1, 0, 0:3, 0:3] = True
    # frame 1: ch0 covers 1 px, ch1 covers 16 px (full frame)
    seg[1, 0, 0, 0, 0] = True
    seg[1, 1, 0, :, :] = True
    return seg


def test_compute_areas_pixel_counts_and_scaling():
    seg = _make_seg_two_channels()
    df = pd.DataFrame({"image name": ["A", "A"], "t": [0, 1]})
    df_idc = [0, 1]
    pixel_area = 0.25  # microns squared per pixel

    out = compute_areas(seg, ["compacted", "CAAX-positive"], df_idc, df, pixel_area)

    assert out is df  # mutates and returns the same object
    assert list(out["compacted area"]) == [4 * pixel_area, 1 * pixel_area]
    assert list(out["CAAX-positive area"]) == [9 * pixel_area, 16 * pixel_area]


def test_compute_areas_unit_pixel_area():
    """With pixel_area=1, areas equal raw pixel counts."""
    seg = _make_seg_two_channels()
    df = pd.DataFrame({"t": [0, 1]})
    out = compute_areas(seg, ["a", "b"], [0, 1], df, pixel_area=1.0)
    assert list(out["a area"]) == [4, 1]
    assert list(out["b area"]) == [9, 16]


def test_compute_areas_empty_mask_yields_zero():
    seg = np.zeros((1, 1, 1, 4, 4), dtype=bool)
    df = pd.DataFrame({"x": [None]})
    out = compute_areas(seg, ["only"], [0], df, pixel_area=1.0)
    assert out["only area"].iloc[0] == 0


def test_compute_areas_label_mismatch_raises():
    seg = np.zeros((1, 2, 1, 4, 4), dtype=bool)
    df = pd.DataFrame({"x": [None]})
    with pytest.raises(ValueError, match="2 channels but 1 labels"):
        compute_areas(seg, ["only-one-label"], [0], df, pixel_area=1.0)


# ---------------------------------------------------------------------------
# compute_int
# ---------------------------------------------------------------------------

def test_compute_int_no_mask_full_frame_mean():
    # T=1, C=1, Z=1, 2x2 frame with values [[1, 2], [3, 4]] -> mean = 2.5, median = 2.5
    img = np.array([[1, 2], [3, 4]], dtype=float).reshape(1, 1, 1, 2, 2)
    df = pd.DataFrame({"x": [None]})
    out = compute_int(img, ["raw"], [0], df, mask=None)
    assert out["mean raw int"].iloc[0] == pytest.approx(2.5)
    assert out["median raw int"].iloc[0] == pytest.approx(2.5)


def test_compute_int_with_mask_restricts_pixels():
    # Same frame, but mask covers only the top row -> mean over [1, 2] = 1.5
    img = np.array([[1, 2], [3, 4]], dtype=float).reshape(1, 1, 1, 2, 2)
    mask = np.zeros((1, 1, 1, 2, 2), dtype=bool)
    mask[0, 0, 0, 0, :] = True  # top row only
    df = pd.DataFrame({"x": [None]})
    out = compute_int(img, ["raw"], [0], df, mask=mask, mask_label="top")
    assert out["mean raw int (top)"].iloc[0] == pytest.approx(1.5)
    assert out["median raw int (top)"].iloc[0] == pytest.approx(1.5)


def test_compute_int_mask_broadcasts_across_channels():
    # T=1, C=2: ch0 = constant 10 inside mask, ch1 = constant 20 inside mask.
    img = np.zeros((1, 2, 1, 3, 3), dtype=float)
    img[0, 0, 0, 1, 1] = 10
    img[0, 1, 0, 1, 1] = 20
    mask = np.zeros((1, 1, 1, 3, 3), dtype=bool)
    mask[0, 0, 0, 1, 1] = True

    df = pd.DataFrame({"x": [None]})
    out = compute_int(img, ["a", "b"], [0], df, mask=mask, mask_label="center")
    assert out["mean a int (center)"].iloc[0] == pytest.approx(10.0)
    assert out["mean b int (center)"].iloc[0] == pytest.approx(20.0)


def test_compute_int_label_mismatch_raises():
    img = np.zeros((1, 2, 1, 2, 2), dtype=float)
    df = pd.DataFrame({"x": [None]})
    with pytest.raises(ValueError, match="2 channels but 1 labels"):
        compute_int(img, ["just-one"], [0], df, mask=None)


# ---------------------------------------------------------------------------
# derive_compaction_metrics
# ---------------------------------------------------------------------------

def test_derive_compaction_metrics_basic_arithmetic():
    df = pd.DataFrame({
        "compacted area": [3.0, 0.0, 5.0],
        "CAAX-positive area": [7.0, 4.0, 0.0],
        "mean caax int (cell)": [2.0, 0.5, 1.0],
        "mean actin int (cell)": [3.0, 1.5, 2.0],
    })
    out = derive_compaction_metrics(df, integrated_density_channels=["caax", "actin"])

    # cell area = compacted + CAAX-positive
    assert list(out["cell area"]) == [10.0, 4.0, 5.0]
    # % compaction = compacted / (compacted + CAAX-positive)
    assert out["% compaction"].iloc[0] == pytest.approx(0.3)
    assert out["% compaction"].iloc[1] == pytest.approx(0.0)
    assert out["% compaction"].iloc[2] == pytest.approx(1.0)
    # integrated density = mean intensity * cell area
    assert list(out["caax integrated density"]) == [20.0, 2.0, 5.0]
    assert list(out["actin integrated density"]) == [30.0, 6.0, 10.0]


def test_derive_compaction_metrics_zero_cell_area_yields_nan():
    df = pd.DataFrame({
        "compacted area": [0.0],
        "CAAX-positive area": [0.0],
        "mean caax int (cell)": [1.0],
        "mean actin int (cell)": [1.0],
    })
    out = derive_compaction_metrics(df)
    # 0 / 0 -> NaN
    assert np.isnan(out["% compaction"].iloc[0])


def test_derive_compaction_metrics_skips_missing_intensity_column():
    """When a requested integrated-density channel has no `mean ... (cell)` column,
    the function silently skips it rather than raising."""
    df = pd.DataFrame({
        "compacted area": [2.0],
        "CAAX-positive area": [3.0],
        "mean caax int (cell)": [1.5],
        # no 'mean actin int (cell)' column
    })
    out = derive_compaction_metrics(df, integrated_density_channels=["caax", "actin"])
    assert "caax integrated density" in out.columns
    assert "actin integrated density" not in out.columns
    assert out["caax integrated density"].iloc[0] == pytest.approx(7.5)

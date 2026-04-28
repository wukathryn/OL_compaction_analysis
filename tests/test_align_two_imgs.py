"""Unit tests for the pure helpers in d01_init_proc.align_two_imgs.

The full ``align_2imgs`` orchestration writes OME-TIFFs to disk and is not
covered here. The private helpers tested below are pure numpy and pure
string parsing, suitable for fast unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from microscopy_analysis.d01_init_proc import align_two_imgs as a2i


# ---------------------------------------------------------------------------
# _parse_channel_list
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inp", [None, "", "all", "ALL"])
def test_parse_channel_list_returns_none_for_keep_all_sentinels(inp):
    assert a2i._parse_channel_list(inp) is None


def test_parse_channel_list_parses_comma_separated():
    out = a2i._parse_channel_list("0,1,2")
    assert isinstance(out, np.ndarray)
    assert out.dtype.kind == "i"
    assert out.tolist() == [0, 1, 2]


def test_parse_channel_list_parses_space_separated():
    assert a2i._parse_channel_list("0 1 2").tolist() == [0, 1, 2]


def test_parse_channel_list_handles_mixed_whitespace_and_commas():
    assert a2i._parse_channel_list("0, 1,  2 ,3").tolist() == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# _crop_imgs_to_same_xy
# ---------------------------------------------------------------------------

def test_crop_imgs_to_same_xy_crops_to_minimum_dims():
    img1 = np.zeros((1, 2, 1, 10, 12))
    img2 = np.zeros((1, 2, 1, 8, 14))
    out1, out2 = a2i._crop_imgs_to_same_xy(img1, img2)
    # Both end up at min(Y) x min(X) = 8 x 12.
    assert out1.shape == (1, 2, 1, 8, 12)
    assert out2.shape == (1, 2, 1, 8, 12)


def test_crop_imgs_to_same_xy_no_op_when_already_matching():
    img1 = np.ones((1, 1, 1, 5, 5))
    img2 = np.ones((1, 1, 1, 5, 5))
    out1, out2 = a2i._crop_imgs_to_same_xy(img1, img2)
    assert out1.shape == img1.shape
    assert out2.shape == img2.shape


# ---------------------------------------------------------------------------
# _subset_channels
# ---------------------------------------------------------------------------

def test_subset_channels_selects_requested_channels():
    img = np.arange(2 * 4 * 1 * 1 * 1).reshape(2, 4, 1, 1, 1)
    out = a2i._subset_channels(img, np.array([0, 2]))
    assert out.shape == (2, 2, 1, 1, 1)
    # Channel 0 of the subset must equal channel 0 of the original
    np.testing.assert_array_equal(out[:, 0], img[:, 0])
    # Channel 1 of the subset must equal channel 2 of the original
    np.testing.assert_array_equal(out[:, 1], img[:, 2])


def test_subset_channels_none_returns_input_unchanged():
    img = np.zeros((1, 3, 1, 4, 4))
    out = a2i._subset_channels(img, None)
    assert out is img

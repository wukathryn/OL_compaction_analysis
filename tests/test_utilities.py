"""Unit tests for d00_utils.utilities — pure helpers that need no image data."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microscopy_analysis.d00_utils import utilities as utils


# ---------------------------------------------------------------------------
# search_name
# ---------------------------------------------------------------------------

def test_search_name_finds_token():
    elems = np.array(["CE029", "div5", "tx-DMSO", "sc12", "ROI3"])
    assert utils.search_name(elems, "div") == "div5"
    assert utils.search_name(elems, "ROI") == "ROI3"


def test_search_name_returns_nan_when_missing():
    elems = np.array(["CE029", "tx-DMSO"])
    assert pd.isna(utils.search_name(elems, "div"))


# ---------------------------------------------------------------------------
# extract_img_info
# ---------------------------------------------------------------------------

def test_extract_img_info_full_filename():
    name = "CE029_div5_tx-DMSO_sc12_ROI3.ome.tif"
    info = utils.extract_img_info(name)
    assert info["Image Name"].iloc[0] == name
    assert info["Experiment"].iloc[0] == "CE029"
    assert info["DIV"].iloc[0] == "div5"
    assert info["Tx"].iloc[0] == "tx-DMSO"
    assert info["Scene"].iloc[0] == "sc12"
    assert info["ROI"].iloc[0] == "ROI3"
    assert info["UID"].iloc[0] == "sc12_ROI3"


def test_extract_img_info_handles_missing_tokens():
    """Filenames missing some tokens still parse, with NaN in the missing fields."""
    name = "CE029_sc1_ROI2.ome.tif"
    info = utils.extract_img_info(name)
    assert info["Experiment"].iloc[0] == "CE029"
    assert info["Scene"].iloc[0] == "sc1"
    assert info["ROI"].iloc[0] == "ROI2"
    assert pd.isna(info["DIV"].iloc[0])
    assert pd.isna(info["Tx"].iloc[0])


# ---------------------------------------------------------------------------
# get_pixel_area
# ---------------------------------------------------------------------------

def test_get_pixel_area_with_known_sizes():
    sizes = SimpleNamespace(X=0.5, Y=0.4, Z=1.0)
    assert utils.get_pixel_area(sizes) == pytest.approx(0.2)


def test_get_pixel_area_returns_none_when_x_missing():
    sizes = SimpleNamespace(X=None, Y=0.4, Z=1.0)
    assert utils.get_pixel_area(sizes) is None


# ---------------------------------------------------------------------------
# move_columns_to_front
# ---------------------------------------------------------------------------

def test_move_columns_to_front_reorders_correctly():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
    out = utils.move_columns_to_front(df, ["c", "a"])
    assert list(out.columns) == ["c", "a", "b", "d"]


def test_move_columns_to_front_ignores_unknown_columns():
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = utils.move_columns_to_front(df, ["does_not_exist", "b"])
    assert list(out.columns) == ["b", "a"]


def test_move_columns_to_front_preserves_data():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    out = utils.move_columns_to_front(df, ["y"])
    assert list(out["y"]) == [3, 4]
    assert list(out["x"]) == [1, 2]


# ---------------------------------------------------------------------------
# safe_save_csv
# ---------------------------------------------------------------------------

def test_safe_save_csv_writes_new_file(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    out_path = tmp_path / "table.csv"
    utils.safe_save_csv(df, out_path)
    assert out_path.is_file()
    assert pd.read_csv(out_path).equals(df)


def test_safe_save_csv_renames_existing_to_prev(tmp_path):
    out_path = tmp_path / "table.csv"
    pd.DataFrame({"a": [99]}).to_csv(out_path, index=False)

    new_df = pd.DataFrame({"a": [1, 2]})
    utils.safe_save_csv(new_df, out_path)

    prev_path = tmp_path / "table_prev.csv"
    assert prev_path.is_file()
    assert pd.read_csv(prev_path)["a"].tolist() == [99]
    assert pd.read_csv(out_path)["a"].tolist() == [1, 2]

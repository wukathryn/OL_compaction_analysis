"""Unit tests for pure helpers in d04_plot_data.

Plotting itself is hard to unit-test; the helpers covered here are pure
text / dataframe utilities that other parts of the pipeline depend on.
"""

from __future__ import annotations

import pandas as pd

from microscopy_analysis.d04_plot_data import plot_timelapse_data as ptd
from microscopy_analysis.d04_plot_data import restructure_data as rd


# ---------------------------------------------------------------------------
# clean_column_name
# ---------------------------------------------------------------------------

def test_clean_column_name_lowercases_and_replaces_spaces():
    assert ptd.clean_column_name("Mean Caax Int") == "mean_caax_int"


def test_clean_column_name_collapses_non_word_chars_to_underscore():
    # Spaces and parentheses both fall under \W+ and collapse to a single underscore.
    assert ptd.clean_column_name("mean caax int (cell)") == "mean_caax_int_cell"
    # The `/` is non-word and is replaced by an underscore.
    assert ptd.clean_column_name("a/b") == "a_b"


def test_clean_column_name_substitutes_domain_tokens():
    # `clean_column_name` performs domain-specific rewrites: `%` becomes `perc`
    # and the literal substring `compact` becomes `cmp`.
    assert ptd.clean_column_name("% compaction") == "perc_cmpion"
    # The hyphen is stripped before `\W+` collapse, so "CAAX-positive" loses
    # the dash entirely rather than gaining an underscore.
    assert ptd.clean_column_name("CAAX-positive") == "caaxpositive"


def test_clean_column_name_handles_none():
    assert ptd.clean_column_name(None) == ""


# ---------------------------------------------------------------------------
# wrap_text
# ---------------------------------------------------------------------------

def test_wrap_text_inserts_newline_for_long_strings():
    out = ptd.wrap_text("aaa bbb ccc", width=4)
    assert "\n" in out


def test_wrap_text_no_newline_when_under_width():
    assert ptd.wrap_text("short", width=20) == "short"


# ---------------------------------------------------------------------------
# list_cells_to_omit
# ---------------------------------------------------------------------------

def test_list_cells_to_omit_picks_y_marked_rows():
    df = pd.DataFrame({"UID": ["a", "a", "b", "c"], "omit": ["Y", "Y", "N", "Y"]})
    out = rd.list_cells_to_omit(df, "omit")
    assert sorted(out) == ["a", "c"]


def test_list_cells_to_omit_returns_empty_when_column_missing():
    df = pd.DataFrame({"UID": ["a", "b"]})
    assert rd.list_cells_to_omit(df, "nonexistent_col") == []


def test_list_cells_to_omit_dedupes_uids():
    df = pd.DataFrame({"UID": ["x", "x", "x"], "omit": ["Y", "Y", "Y"]})
    out = rd.list_cells_to_omit(df, "omit")
    assert out == ["x"]

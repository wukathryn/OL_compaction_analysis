"""Per-image area and intensity metrics from segmented OME-TIFF stacks.

The functions in this module operate on 5D arrays with the bioio
convention `(T, C, Z, Y, X)`. They write into a long-format pandas
DataFrame indexed by (image, frame). The same helpers serve both the
single-timepoint pipeline (T=1) and the timelapse pipeline (T>1).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import numpy.ma as ma
import pandas as pd


def compute_areas(
    seg: np.ndarray,
    seg_labels: Sequence[str],
    df_idc: Sequence[int],
    df: pd.DataFrame,
    pixel_area: float,
) -> pd.DataFrame:
    """Count nonzero pixels in each segmentation channel and write per-frame areas.

    Parameters
    ----------
    seg
        Binary segmentation stack, shape `(T, C, Z, Y, X)`. Each channel
        carries one region's binary mask. Non-boolean inputs are
        evaluated as truthy/zero.
    seg_labels
        One label per channel of `seg`. The output columns are named
        ``f"{label} area"``.
    df_idc
        Row indices into `df` that correspond to this image's `T` frames,
        in temporal order.
    df
        Long-format dataframe to write into; must already contain the
        rows referenced by `df_idc`.
    pixel_area
        Physical area of one pixel, typically `physical_pixel_sizes.X *
        physical_pixel_sizes.Y` in microns squared. Output areas are in
        the same units as `pixel_area`.

    Returns
    -------
    pd.DataFrame
        The same `df`, with one new column per segmentation channel.
    """
    if seg.shape[1] != len(seg_labels):
        raise ValueError(
            f"seg has {seg.shape[1]} channels but {len(seg_labels)} labels were given"
        )

    # count_nonzero over the (Y, X) plane gives a (T, C, Z) count per frame.
    region_areas = np.count_nonzero(seg, axis=(3, 4)) * pixel_area

    for c, label in enumerate(seg_labels):
        df.loc[df_idc, f"{label} area"] = region_areas[:, c, :]

    return df


def compute_int(
    img: np.ndarray,
    ch_labels: Sequence[str],
    df_idc: Sequence[int],
    df: pd.DataFrame,
    mask: Optional[np.ndarray] = None,
    mask_label: Optional[str] = None,
) -> pd.DataFrame:
    """Compute mean and median intensity per channel under an optional mask.

    When `mask` is supplied it is broadcast against `img` and pixels
    where the mask is zero are excluded from both statistics. The mask
    is treated as a region of interest, not an exclusion zone.

    Parameters
    ----------
    img
        Fluorescence stack, shape `(T, C, Z, Y, X)`.
    ch_labels
        One label per channel of `img`. Output columns are named
        ``f"mean {label} int{suffix}"`` and ``f"median {label} int{suffix}"``,
        where `suffix` is ``f" ({mask_label})"`` when a mask is used and
        empty otherwise.
    df_idc, df
        As in :func:`compute_areas`.
    mask
        Optional binary mask, shape broadcastable to `img`. Typically
        `(T, 1, Z, Y, X)` so it broadcasts across all channels.
    mask_label
        Label inserted into the column names; ignored when `mask` is None.

    Returns
    -------
    pd.DataFrame
        The same `df`, with two new columns per channel.
    """
    if img.shape[1] != len(ch_labels):
        raise ValueError(
            f"img has {img.shape[1]} channels but {len(ch_labels)} labels were given"
        )

    if mask is not None:
        # masked_array uses True for "ignore"; invert the ROI mask accordingly.
        mask_b = np.broadcast_to(mask, img.shape)
        img = ma.masked_array(img, mask_b == 0)
        suffix = f" ({mask_label})"
    else:
        suffix = ""

    mean_int = ma.mean(img, axis=(3, 4))
    median_int = ma.median(img, axis=(3, 4))

    for c, label in enumerate(ch_labels):
        df.loc[df_idc, f"mean {label} int{suffix}"] = mean_int[:, c, :].squeeze()
        df.loc[df_idc, f"median {label} int{suffix}"] = median_int[:, c, :].squeeze()

    return df


def derive_compaction_metrics(
    df: pd.DataFrame,
    integrated_density_channels: Iterable[str] = ("caax", "actin"),
) -> pd.DataFrame:
    """Add cell area, % compaction, and per-channel integrated densities.

    Assumes `compute_areas` has populated ``"compacted area"`` and
    ``"CAAX-positive area"``, and that `compute_int` has populated
    ``f"mean {ch} int (cell)"`` for each channel listed in
    `integrated_density_channels`.

    Definitions
    -----------
    * ``cell area = "compacted area" + "CAAX-positive area"``
    * ``"% compaction" = "compacted area" / "cell area"``
    * ``f"{ch} integrated density" = f"mean {ch} int (cell)" * "cell area"``

    Rows where ``cell area == 0`` produce NaN/inf in ``"% compaction"``;
    callers are expected to filter or interpret those.

    Parameters
    ----------
    df
        Long-format dataframe with the prerequisite columns.
    integrated_density_channels
        Channel labels for which to compute integrated density.

    Returns
    -------
    pd.DataFrame
        The same `df`, mutated in place and returned for chaining.
    """
    df["cell area"] = df["CAAX-positive area"] + df["compacted area"]
    df["% compaction"] = df["compacted area"] / df["cell area"]

    for ch in integrated_density_channels:
        cell_int_col = f"mean {ch} int (cell)"
        if cell_int_col in df.columns:
            df[f"{ch} integrated density"] = df[cell_int_col] * df["cell area"]

    return df

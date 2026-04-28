"""Two-image alignment via phase correlation, with optional within-stack drift correction.

Estimates the (dy, dx) translation between two OME-TIFF stacks using a single
(t, z) slice from each, then applies that shift to all of `img2` so it aligns
to `img1`. Optional preprocessing steps correct for stage drift across
timepoints and constant optical offsets between channels before the
between-image shift is estimated. Channel-subset selection is independent for
each input stack; the aligned output OME-TIFF concatenates the two subsets
along the channel axis.

Phase-correlation reference: Evangelidis & Psarakis, IEEE TPAMI 30(10), 2008.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

import bioio_ome_tiff
from bioio import BioImage
from bioio.writers import OmeTiffWriter

from microscopy_analysis.d00_utils import dirnames as dn
from microscopy_analysis.d00_utils import utilities as utils
from microscopy_analysis.d01_init_proc import alignment_core
from microscopy_analysis.d01_init_proc import vis_and_rescale

# Filenames written into aligned output directory
ALIGNMENT_CSV_NAME = "alignment_info.csv"
ERROR_CSV_NAME = "alignment_errors.csv"


def _parse_channel_list(s: Optional[str]) -> Optional[np.ndarray]:
    """
    Parse a channel subset string into a 1D integer array.

    Accepted inputs
    ---------------
    - None / "" / "all" -> None   (meaning: keep all channels)
    - "0,1,2" or "0 1 2" -> np.array([0,1,2], dtype=int)

    Notes
    -----
    Subsets are interpreted as channel indices into the input OME-TIFF.
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() == "all":
        return None
    parts = [p for p in s.replace(",", " ").split() if p != ""]
    return np.array([int(p) for p in parts], dtype=int)


def _crop_imgs_to_same_xy(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure two stacks share the same spatial size by cropping to the smallest Y/X.

    Expects arrays where the final axes are (..., Y, X).
    For BioIO OME-TIFF stacks, this is typically (T, C, Z, Y, X).

    This keeps the pixel grid consistent so that estimated shifts are meaningful and
    concatenation along the channel axis is always valid.
    """
    min_y = min(img1.shape[-2], img2.shape[-2])
    min_x = min(img1.shape[-1], img2.shape[-1])
    return img1[..., :min_y, :min_x], img2[..., :min_y, :min_x]


def _subset_channels(img: np.ndarray, ch_subset: Optional[np.ndarray]) -> np.ndarray:
    """
    Return a view/copy of img restricted to a subset of channels.

    Parameters
    ----------
    img
        5D stack shaped (T, C, Z, Y, X).
    ch_subset
        1D integer array of channel indices, or None to keep all channels.
    """
    if ch_subset is None:
        return img
    ch_subset = np.asarray(ch_subset, dtype=int)
    return img[:, ch_subset, ...]


def align_2imgs(
    imgpath1: Union[str, Path],
    imgpath2: Union[str, Path],
    aligned_dirpath: Union[str, Path],
    img1_ch_align: int = 0,
    img2_ch_align: int = 0,
    z_align: int = 0,
    tp_align: int = -1,
    max_translation: int = 200,
    align_time: bool = False,
    align_channels: bool = False,
    time_ref_ch1: Optional[int] = None,
    time_ref_ch2: Optional[int] = None,
    channel_ref1: Optional[int] = None,
    channel_ref2: Optional[int] = None,
    img1_ch_subset: Optional[np.ndarray] = None,
    img2_ch_subset: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Align two OME-TIFF images and write a merged aligned OME-TIFF.

    Overview
    --------
    1) Read img1/img2 as arrays shaped (T, C, Z, Y, X).
    2) Optionally correct within-stack drift:
       - time alignment: per-timepoint translation correction
       - channel alignment: per-channel translation correction
    3) Estimate inter-image shift using a single (t, z) slice from each stack.
    4) Apply the shift to all of img2 (all T/C/Z).
    5) Subset channels *separately* for img1 and img2 (if requested).
    6) Concatenate along the channel axis: [img1_subset, img2_subset].
    7) Write an OME-TIFF and return a one-row DataFrame with dy/dx.

    Important
    ---------
    The direction matters: the computed (dy, dx) is applied to img2 so it aligns to img1.

    Parameters
    ----------
    img1_ch_align, img2_ch_align
        Channels used to estimate the between-image shift.
    z_align, tp_align
        Plane/timepoint used to estimate shift. tp_align supports negative indexing
        (e.g., -1 for last timepoint).
    max_translation
        Safety limit (pixels). Large shifts are rejected and replaced with (0, 0).
    align_time
        If True, align timepoints within each stack before between-image alignment.
    align_channels
        If True, align channels within each stack before between-image alignment.
    time_ref_ch1/time_ref_ch2
        Reference channels for time alignment (defaults to img*_ch_align).
    channel_ref1/channel_ref2
        Reference channels for channel alignment (defaults to img*_ch_align).
    img1_ch_subset/img2_ch_subset
        Channels to retain in the merged output for img1 and img2, respectively.
        If None, all channels from that image are kept.

    Returns
    -------
    pd.DataFrame
        Single-row table with columns: img1, img2, dy, dx
    """
    imgpath1 = Path(imgpath1)
    imgpath2 = Path(imgpath2)
    aligned_dirpath = Path(aligned_dirpath)
    aligned_dirpath.mkdir(parents=True, exist_ok=True)

    # Read images (BioIO returns a 5D array in expected (T, C, Z, Y, X) order for OME-TIFF).
    img_file1 = BioImage(str(imgpath1), reader=bioio_ome_tiff.Reader)
    img1 = img_file1.data

    img_file2 = BioImage(str(imgpath2), reader=bioio_ome_tiff.Reader)
    img2 = img_file2.data

    # Crop out black borders and keep both images on the same pixel grid so estimated shifts are meaningful.
    # img1 = utils.crop_black_borders(img1)
    # img2 = utils.crop_black_borders(img2)
    img1, img2 = _crop_imgs_to_same_xy(img1, img2)

    # Optional within-timelapse alignment for each stack.
    # This is useful if stage drift exists (time alignment) or if there is a constant
    # optical offset between channels (channel alignment).
    if align_time:
        ref1 = img1_ch_align if time_ref_ch1 is None else int(time_ref_ch1)
        ref2 = img2_ch_align if time_ref_ch2 is None else int(time_ref_ch2)
        img1, _ = alignment_core.align_timepoints(
            img1, ch_ref=ref1, z=z_align, tp_ref=0, max_translation=max_translation
        )
        img2, _ = alignment_core.align_timepoints(
            img2, ch_ref=ref2, z=z_align, tp_ref=0, max_translation=max_translation
        )

    if align_channels:
        ref1 = img1_ch_align if channel_ref1 is None else int(channel_ref1)
        ref2 = img2_ch_align if channel_ref2 is None else int(channel_ref2)
        img1, _ = alignment_core.align_channels(img1, c_ref=ref1, z=z_align, t_ref=0, max_translation=max_translation)
        img2, _ = alignment_core.align_channels(img2, c_ref=ref2, z=z_align, t_ref=0, max_translation=max_translation)

    ref = img1[tp_align, np.newaxis, img1_ch_align, np.newaxis, z_align, np.newaxis, :, :]
    ref, _ = vis_and_rescale.rescale_img(ref, target_perc_grayval=50, standardize_scaling=False, scaling_fact=None, conv_to_8bit=True)
    ref = ref.squeeze()

    mov = img2[tp_align, np.newaxis, img2_ch_align, np.newaxis, z_align, np.newaxis, :, :]
    mov, _ = vis_and_rescale.rescale_img(mov, target_perc_grayval=50, standardize_scaling=False, scaling_fact=None, conv_to_8bit=True)
    mov = mov.squeeze()

    # # Compute between-image shift on 2D slices from the chosen alignment channels.
    dy, dx = alignment_core.phase_correlation_shift(ref, mov)

    # Guard against catastrophic misregistration (e.g., poor signal, wrong channel).
    if abs(dy) > max_translation or abs(dx) > max_translation:
        print(
            f"[WARN] Rejected large shift for {imgpath2.name} vs {imgpath1.name}: "
            f"(dy,dx)=({dy},{dx}) exceeds max_translation={max_translation}. Using (0,0)."
        )
        dy, dx = 0, 0


    # Apply shift to the full img2 stack (all T/C/Z).
    img2_aligned = alignment_core.shift_xy_integer(img2, dy=dy, dx=dx)

    # Subset channels separately, then concatenate in the requested final order:
    # [img1_subset channels, img2_subset channels]
    img1_out = _subset_channels(img1, img1_ch_subset)
    img2_out = _subset_channels(img2_aligned, img2_ch_subset)
    merged = np.concatenate([img1_out, img2_out], axis=1)

    # Write aligned output.
    # Metadata: we start from img1 metadata and update dimensions via helper.
    ome_xml = utils.construct_ome_metadata(merged, img_file1)
    out_name = f"{imgpath1.stem}_aligned.ome.tif"
    out_path = aligned_dirpath / out_name
    OmeTiffWriter.save(merged, out_path, ome_xml=ome_xml)

    return pd.DataFrame({"img1": [imgpath1.name], "img2": [imgpath2.name], "dy": [dy], "dx": [dx]})

def batch_align_2imgs(
    aligndir: Union[str, Path],
    img1_ch_align: int,
    img2_ch_align: int,
    img1_ch_subset: Optional[np.ndarray] = None,
    img2_ch_subset: Optional[np.ndarray] = None,
    tp_align: int = -1,
    z_align: int = 0,
    max_translation: int = 200,
    align_time: bool = False,
    align_channels: bool = False,
    img2_dir_uniquestring: Optional[str] = None,
) -> None:
    """
    Align all matching image pairs from two experiment directories inside `aligndir`.

    How img1 vs img2 is chosen
    --------------------------
    - If `img2_dir_uniquestring` is provided, the directory whose name contains this string
      (case-insensitive) is assigned to img2 (the moving image), and another directory is
      chosen as img1 (the fixed reference).
    - If `img2_dir_uniquestring` is not provided, the first two experiment directories are
      sorted alphabetically; the first becomes img1 and the second img2.

    Notes
    -----
    Experiment directories are detected by the presence of:
        <exp_dir> / dn.proc_dirname / dn.raw_ometif_dirname

    Pairing between directories is performed by matching the parsed `Scene` field from
    utils.extract_img_info() for each filename.

    Side effects
    ------------
    Writes aligned OME-TIFFs into the aligned output directory and appends alignment
    info to ALIGNMENT_CSV_NAME / ERROR_CSV_NAME.
    """
    aligndir = Path(aligndir)

    # ------------------------------------------------------------
    # Discover candidate experiment directories
    # ------------------------------------------------------------
    exp_dirs = []
    for d in aligndir.iterdir():
        if not d.is_dir():
            continue
        raw_dir = d / dn.proc_dirname / dn.raw_ometif_dirname
        if raw_dir.exists():
            exp_dirs.append(d)

    if len(exp_dirs) < 2:
        raise AssertionError(
            f"Found {len(exp_dirs)} experiment dirs under {aligndir} containing "
            f"{dn.proc_dirname}/{dn.raw_ometif_dirname}. Need at least 2."
        )

    # ------------------------------------------------------------
    # Choose img1 vs img2 (img2 is shifted to align to img1)
    # ------------------------------------------------------------
    if img2_dir_uniquestring is not None:
        key = str(img2_dir_uniquestring).lower()
        matches = [d for d in exp_dirs if key in d.name.lower()]

        if len(matches) == 0:
            raise AssertionError(
                f"No experiment directory under {aligndir} contains '{img2_dir_uniquestring}'. "
                f"Candidates: {[d.name for d in exp_dirs]}"
            )
        if len(matches) > 1:
            raise AssertionError(
                f"Multiple experiment directories under {aligndir} contain '{img2_dir_uniquestring}': "
                f"{[d.name for d in matches]}. Please use a more specific string."
            )

        img2_dir = matches[0]
        others = sorted([d for d in exp_dirs if d != img2_dir])
        if len(others) == 0:
            raise AssertionError(
                f"Could not determine img1 directory after selecting img2='{img2_dir_uniquestring}'. "
                f"Candidates: {[d.name for d in exp_dirs]}"
            )
        img1_dir = others[0]

        if len(exp_dirs) > 2:
            ignored = [d.name for d in exp_dirs if d not in {img1_dir, img2_dir}]
            print(
                f"[WARN] Found {len(exp_dirs)} experiment dirs under {aligndir}. "
                f"Using img1='{img1_dir.name}' and img2='{img2_dir.name}'. "
                f"Other dirs ignored: {ignored}"
            )
    else:
        exp_dirs = sorted(exp_dirs)
        if len(exp_dirs) > 2:
            print(
                f"[WARN] Found {len(exp_dirs)} experiment dirs under {aligndir}. "
                f"Aligning the first two after sorting: {exp_dirs[0].name}, {exp_dirs[1].name}"
            )
        img1_dir, img2_dir = exp_dirs[0], exp_dirs[1]

    print(f"[INFO] Using img1 (reference) dir: {img1_dir.name}")
    print(f"[INFO] Using img2 (moving)    dir: {img2_dir.name}")

    # ------------------------------------------------------------
    # Resolve raw OME-TIFF directories
    # ------------------------------------------------------------
    dir1_raw = img1_dir / dn.proc_dirname / dn.raw_ometif_dirname
    dir2_raw = img2_dir / dn.proc_dirname / dn.raw_ometif_dirname
    assert dir1_raw.exists(), f"Please process {img1_dir.name} into raw ome-tiffs."
    assert dir2_raw.exists(), f"Please process {img2_dir.name} into raw ome-tiffs."

    # Enumerate files
    imgnames1 = sorted([p.name for p in dir1_raw.glob("*.ome.tif")])
    imgnames2 = sorted([p.name for p in dir2_raw.glob("*.ome.tif")])
    total_count1 = len(imgnames1)

    # ------------------------------------------------------------
    # Construct aligned output directory
    # ------------------------------------------------------------
    multiexp_dirpath = aligndir.parent
    aligned_dirpath = (
        multiexp_dirpath
        / dn.aligned_dirname
        / f"{img1_dir.name}_and_{img2_dir.name}_aligned"
        / dn.proc_dirname
        / dn.raw_ometif_dirname
    )
    aligned_dirpath.mkdir(parents=True, exist_ok=True)

    # Skip images already aligned (based on naming convention).
    aligned_imgnames = [
        (p.name[: p.name.find("_aligned")] + ".ome.tif")
        for p in aligned_dirpath.glob("*.ome.tif")
        if "_aligned" in p.name
    ]
    imgnames1 = [n for n in imgnames1 if n not in aligned_imgnames]
    print(f"[INFO] Skipping {total_count1 - len(imgnames1)} previously aligned images.")

    # ------------------------------------------------------------
    # Load bookkeeping, if present
    # ------------------------------------------------------------
    aligned_csv_path = aligned_dirpath / ALIGNMENT_CSV_NAME
    error_csv_path = aligned_dirpath / ERROR_CSV_NAME

    if aligned_csv_path.is_file():
        align_df = pd.read_csv(aligned_csv_path)
        if {"dy", "dx"}.issubset(align_df.columns):
            align_df = align_df.dropna(axis=0, subset=["dy", "dx"])
    else:
        align_df = pd.DataFrame()

    if error_csv_path.is_file():
        error_df = pd.read_csv(error_csv_path)
    else:
        error_df = pd.DataFrame({"image name": [], "error time": [], "error": []})

    # ------------------------------------------------------------
    # Match up images by scene parsed from filename
    # ------------------------------------------------------------
    imgs1_df = pd.concat([utils.extract_img_info(n) for n in imgnames1], ignore_index=True) if imgnames1 else pd.DataFrame()
    imgs2_df = pd.concat([utils.extract_img_info(n) for n in imgnames2], ignore_index=True) if imgnames2 else pd.DataFrame()

    if imgs1_df.empty or imgs2_df.empty:
        print("[WARN] No images found to align after filtering.")
        return

    # Inner join: align only scenes present in both experiments.
    align_df_new = imgs1_df.merge(imgs2_df, how="inner", on="Scene")
    align_df_new["dy"] = np.nan
    align_df_new["dx"] = np.nan

    # Append new work to any previous alignment log.
    start_idx = align_df.shape[0]
    align_df = pd.concat([align_df, align_df_new], ignore_index=True)

    # ------------------------------------------------------------
    # Align each paired scene and save outputs incrementally
    # ------------------------------------------------------------
    for i, row in align_df.loc[start_idx:].iterrows():
        imgpath1 = dir1_raw / row["Image Name_x"]
        imgpath2 = dir2_raw / row["Image Name_y"]

        try:
            single_df = align_2imgs(
                imgpath1=imgpath1,
                imgpath2=imgpath2,
                aligned_dirpath=aligned_dirpath,
                img1_ch_align=img1_ch_align,
                img2_ch_align=img2_ch_align,
                z_align=z_align,
                tp_align=tp_align,
                max_translation=max_translation,
                align_time=align_time,
                align_channels=align_channels,
                img1_ch_subset=img1_ch_subset,
                img2_ch_subset=img2_ch_subset,
            )
            align_df.at[i, "dy"] = single_df.at[0, "dy"]
            align_df.at[i, "dx"] = single_df.at[0, "dx"]

        except Exception as e:
            print(f"[ERROR] Failed alignment for {imgpath1.name} vs {imgpath2.name}: {e}")
            error_df = pd.concat(
                [
                    error_df,
                    pd.DataFrame(
                        {
                            "image name": [imgpath1.name],
                            "error time": [pd.Timestamp.now()],
                            "error": [str(e)],
                        }
                    ),
                ],
                ignore_index=True,
            )

        # Save after each pair so the run can be resumed safely.
        align_df.to_csv(aligned_csv_path, index=False)
        error_df.to_csv(error_csv_path, index=False)

    # ------------------------------------------------------------
    # Move original, unaligned directory aside after processing
    # ------------------------------------------------------------
    unaligned_parent_dirpath = multiexp_dirpath / dn.orig_unaligned_dirname
    unaligned_parent_dirpath.mkdir(parents=True, exist_ok=True)
    shutil.move(str(aligndir), str(unaligned_parent_dirpath))

    print(f"[INFO] Done. Images saved to {aligned_dirpath}")


def multiexp_align(
    multiexp_dir: Union[str, Path],
    img1_ch_align: int,
    img2_ch_align: int,
    img1_ch_subset: Optional[np.ndarray] = None,
    img2_ch_subset: Optional[np.ndarray] = None,
    tp_align: int = -1,
    z_align: int = 0,
    max_translation: int = 500,
    align_time: bool = False,
    align_channels: bool = False,
    img2_dir_uniquestring: Optional[str] = None,
) -> None:
    """
    Align all eligible alignment-group directories inside a multi-experiment directory.

    Directory structure expectation
    ------------------------------
    multiexp_dir/
      <alignment_group_1>/
        <exp_A>/proc/raw_ometif/*.ome.tif
        <exp_B>/proc/raw_ometif/*.ome.tif
      <alignment_group_2>/
        ...

    Exclusions
    ----------
    Skips:
      - dn.aligned_dirname: output directory created by this pipeline
      - dn.orig_unaligned_dirname: storage for already-processed original directories

    img1 vs img2 selection
    ----------------------
    Delegated to batch_align_2imgs(). If img2_dir_uniquestring is provided, the experiment
    directory whose name contains that substring (case-insensitive) is treated as img2
    (moving image).

    Output channel order
    --------------------
    The merged output is always:
        [img1 subset channels, img2 subset channels]
    where each subset is independently controlled by img1_ch_subset/img2_ch_subset.
    """
    multiexp_dir = Path(multiexp_dir)
    assert multiexp_dir.exists(), f"multiexp_dir does not exist: {multiexp_dir}"

    # Ensure top-level output parents exist.
    (multiexp_dir / dn.aligned_dirname).mkdir(parents=True, exist_ok=True)
    (multiexp_dir / dn.orig_unaligned_dirname).mkdir(parents=True, exist_ok=True)

    # Candidate alignment-group directories: subdirectories that are not output folders.
    aligndirs = [
        d
        for d in multiexp_dir.iterdir()
        if d.is_dir() and d.name not in {dn.orig_unaligned_dirname, dn.aligned_dirname}
    ]
    aligndirs.sort()

    if len(aligndirs) == 0:
        print(f"[WARN] No alignment-group directories found under {multiexp_dir}.")
        return

    for i, aligndir in enumerate(aligndirs, start=1):
        print(f"[INFO] Starting to align {aligndir.name} ({i}/{len(aligndirs)})")

        batch_align_2imgs(
            aligndir=aligndir,
            img1_ch_align=img1_ch_align,
            img2_ch_align=img2_ch_align,
            img1_ch_subset=img1_ch_subset,
            img2_ch_subset=img2_ch_subset,
            tp_align=tp_align,
            z_align=z_align,
            max_translation=max_translation,
            align_time=align_time,
            align_channels=align_channels,
            img2_dir_uniquestring=img2_dir_uniquestring,
        )

    print("[INFO] Done with multi-experiment alignment.")

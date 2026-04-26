"""
Core image alignment utilities.

This module implements fast, reusable routines for rigid XY alignment of
microscopy images. All functions operate directly on NumPy arrays and are
independent of file I/O, enabling easy reuse in analysis pipelines and
unit testing.

Key design principles
--------------------
- Alignment is performed using phase correlation (FFT-based), which is robust
  to global intensity scaling and suitable for biological images with moderate
  structural changes.
- Shifts are applied as integer translations with zero padding (no wrap-around),
  preserving spatial consistency across channels and z-planes.
- Computation is restricted to the minimal data required (typically 2D slices),
  avoiding unnecessary processing of full 5D stacks.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from microscopy_analysis.d01_init_proc import vis_and_rescale

def phase_correlation_shift(img1: np.ndarray, img2: np.ndarray):
    """
    Estimate integer pixel translation (dy, dx) to align img2 to img1.

    Parameters
    ----------
    img1, img2 : np.ndarray
        2D arrays (Y, X) representing the reference and moving images.

    Returns
    -------
    dy, dx : int
        Translation required to apply to img2 to align it with img1.

    Notes
    -----
    Phase correlation computes the cross-power spectrum between images and
    identifies the translation as the location of the peak in the inverse FFT.
    """
    img1 = img1.astype(np.float32, copy=False)
    img2 = img2.astype(np.float32, copy=False)

    f1 = fft2(img1)
    f2 = fft2(img2)

    denom = np.abs(f1) * np.abs(f2)
    denom[denom == 0] = 1  # prevent division by zero

    ir = np.abs(ifft2((f1 * f2.conjugate()) / denom))
    t0, t1 = np.unravel_index(np.argmax(ir), img1.shape)

    # Convert peak location to signed shift
    if t0 > img1.shape[0] // 2:
        t0 -= img1.shape[0]
    if t1 > img1.shape[1] // 2:
        t1 -= img1.shape[1]

    print(t0, t1)
    return int(t0), int(t1)

def shift_xy_integer(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """
    Apply integer XY translation with zero padding.

    Parameters
    ----------
    arr : np.ndarray
        Input array with shape (..., Y, X).
    dy, dx : int
        Translation along Y and X axes.

    Returns
    -------
    shifted : np.ndarray
        Array of the same shape with translated content.

    Notes
    -----
    This implementation avoids costly padding operations by directly copying
    overlapping regions into a preallocated output array.
    """
    out = np.zeros_like(arr)
    y, x = arr.shape[-2], arr.shape[-1]

    y0_dst = max(0, dy)
    y1_dst = min(y, y + dy)
    x0_dst = max(0, dx)
    x1_dst = min(x, x + dx)

    y0_src = max(0, -dy)
    y1_src = y0_src + (y1_dst - y0_dst)
    x0_src = max(0, -dx)
    x1_src = x0_src + (x1_dst - x0_dst)

    if (y1_dst > y0_dst) and (x1_dst > x0_dst):
        out[..., y0_dst:y1_dst, x0_dst:x1_dst] = arr[..., y0_src:y1_src, x0_src:x1_src]
    return out


# def prep_for_shift(img2d: np.ndarray, percentile: int = 50):
#     """
#     Lightweight intensity normalization for alignment.
#
#     Parameters
#     ----------
#     img2d : np.ndarray
#         2D image.
#     percentile : int
#         Percentile used for scaling.
#
#     Returns
#     -------
#     normalized_img : np.ndarray
#
#     Notes
#     -----
#     This normalization reduces sensitivity to global intensity changes
#     (e.g., bleaching) while preserving spatial structure.
#     """
#     img = img2d.astype(np.float32, copy=False)
#     p = np.percentile(img, percentile)
#     if p > 0:
#         img = img / p
#     return img


def align_timepoints(img, ch_ref=0, z=0, tp_ref=0, max_translation=200):
    """
    Align a timelapse across time using a reference channel.

    Parameters
    ----------
    img : np.ndarray
        Image stack with shape (T, C, Z, Y, X).
    ch_ref : int
        Channel used to estimate shifts.
    z : int
        Z-plane used for alignment.
    tp_ref : int
        Reference timepoint.
    max_translation : int
        Maximum allowed shift magnitude; larger values are rejected.

    Returns
    -------
    aligned : np.ndarray
        Time-aligned image stack.
    shifts : list of tuple
        List of (dy, dx) shifts applied per timepoint.

    Notes
    -----
    All channels and z-planes are shifted together to preserve internal
    structure within each frame.
    """
    T = img.shape[0]
    aligned = np.empty_like(img)

    img_rescaled, _ = vis_and_rescale.rescale_img(img, target_perc_grayval=50, standardize_scaling=False,
                                               scaling_fact=None, conv_to_8bit=True)
    ref = img_rescaled[tp_ref, ch_ref, z]
    shifts = []

    for t in range(T):
        if t == tp_ref:
            aligned[t] = img[t]
            shifts.append((0, 0))
            continue

        mov = img_rescaled[t, ch_ref, z]
        dy, dx = phase_correlation_shift(ref, mov)

        if abs(dy) > max_translation or abs(dx) > max_translation:
            dy, dx = 0, 0

        aligned[t] = shift_xy_integer(img[t], dy, dx)
        shifts.append((dy, dx))

    return aligned, shifts


def align_channels(img, ch_ref=0, z=0, t_ref=0, max_translation=200):
    """
    Align channels assuming a constant offset across time.

    Parameters
    ----------
    img : np.ndarray
        Image stack with shape (T, C, Z, Y, X).
    ch_ref : int
        Reference channel.
    z : int
        Z-plane used for alignment.
    t_ref : int
        Timepoint used for estimating channel offsets.

    Returns
    -------
    aligned : np.ndarray
        Channel-aligned image stack.
    shifts : list of tuple
        List of (dy, dx) shifts per channel.

    Notes
    -----
    Channel alignment is typically time-invariant (optical offset), so shifts
    are computed once and applied to all frames.
    """
    T, C = img.shape[:2]
    aligned = np.copy(img)

    img_rescaled, _ = vis_and_rescale.rescale_img(img, target_perc_grayval=50, standardize_scaling=False,
                                               scaling_fact=None, conv_to_8bit=True)
    ref = img_rescaled[t_ref, ch_ref, z]
    shifts = []

    for c in range(C):
        if c == ch_ref:
            shifts.append((0, 0))
            continue

        mov = img_rescaled[t_ref, c, z]
        dy, dx = phase_correlation_shift(ref, mov)

        if abs(dy) > max_translation or abs(dx) > max_translation:
            dy, dx = 0, 0

        aligned[:, c] = shift_xy_integer(img[:, c], dy, dx)
        shifts.append((dy, dx))

    return aligned, shifts

"""Legacy region-decomposition analyses (overlap, percent-area, exclusion channel).

Pixel-area and per-channel intensity helpers used by the active pipeline
have moved to :mod:`microscopy_analysis.d02_metrics`. This module retains
the older percent-area and overlap visualizations that were used in
earlier analyses and are kept for archival reproducibility.
"""

from bioio.writers import OmeTiffWriter
import numpy as np

from microscopy_analysis.d00_utils import utilities as utils
# Pixel-counting helpers were extracted to d02_metrics; re-exported here for
# any legacy caller that imports them from this module by name.
from microscopy_analysis.d02_metrics import compute_areas, compute_int

from matplotlib import pyplot as plt

# Output directory name and rounding precision used by ``calc_overlap``.
overlap_dirname = "overlap"
num_digits = 4

# t and z dimension indices used by ``add_exclusion_channel``.
tp = -1
z = 0


def add_exclusion_channel(binary_regions, cell_ch, caax_ch):
    size_c = binary_regions.shape[1]
    caax_ch = check_and_convert_to_list(caax_ch, size_c)
    for ch in caax_ch:
        exclusion_region = binary_regions[tp, cell_ch, z, :, :] & ~binary_regions[tp, caax_live_ch, z, :, :]
        exclusion_region = np.expand_dims(exclusion_region, axis=(0, 1, 2))
        binary_regions = np.concatenate([binary_regions, exclusion_region], axis=1)
    return binary_regions

def check_and_convert_to_list(chs, size_c):
    if chs=='all':
        chs = np.arange(size_c)
    else:
        assert isinstance(chs, (int, list, tuple))
        if isinstance(chs, int):
            chs = [chs]
    return chs


def calc_overlap(ch_1st, ch_2nd, binary_regions, saveinfo, idv_overlap_d):
    (overlap_dirpath, imgname, ch_labels, ch_save_abbr, physical_pixel_sizes) = saveinfo

    binary_ch_1st = binary_regions[-1:, ch_1st, 0:1, :, :]
    binary_ch_2nd = binary_regions[-1:, ch_2nd, 0:1, :, :]

    intersect_img = binary_ch_1st & binary_ch_2nd
    union_img = binary_ch_1st | binary_ch_2nd
    binary_ch_1st_only = binary_ch_1st & ~binary_ch_2nd
    binary_ch_2nd_only = binary_ch_2nd & ~binary_ch_1st

    binaryoverlap = np.stack([intersect_img, union_img, binary_ch_1st_only, binary_ch_2nd_only], axis=1)

    binaryoverlap_dirpath = overlap_dirpath / f'binary_{ch_save_abbr[ch_1st]}_{ch_save_abbr[ch_2nd]}'
    binaryoverlap_dirpath.mkdir(parents=True, exist_ok=True)
    ome_metadata = utils.construct_ome_metadata(binaryoverlap, physical_pixel_sizes, ch_labels)
    OmeTiffWriter.save(binaryoverlap, binaryoverlap_dirpath / imgname, ome_xml=ome_metadata)

    areas = np.count_nonzero(binaryoverlap, axis=(0, 2, 3, 4))

    # area of intersection / area of union
    overlap = round(areas[0] / areas[1], 4)
    idv_overlap_d.update({f'{ch_labels[ch_1st]}_{ch_labels[ch_2nd]}_overlap': overlap})

    # visualize overlap
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(binary_ch_1st.squeeze(), cmap='gray', interpolation=None)
    axs[0].set_title(f'{ch_labels[ch_1st]}')
    axs[1].imshow(binary_ch_2nd.squeeze(), cmap='gray', interpolation=None)
    axs[1].set_title(f'{ch_labels[ch_2nd]}')
    axs[2].imshow(union_img.squeeze(), cmap='Greys', interpolation=None)
    axs[2].imshow(intersect_img.squeeze(), cmap='Blues', alpha=0.5, interpolation=None)
    axs[2].set_title(f'gray: union, \nblue: overlap\n{round(overlap * 100, 4)}%')
    axs[3].imshow(binary_ch_1st_only.squeeze(), cmap='Greens', interpolation=None)
    axs[3].imshow(binary_ch_2nd_only.squeeze(), cmap='Purples', alpha=0.5, interpolation=None)
    axs[3].set_title(f'green: {ch_labels[ch_1st]} only, \npurple: {ch_labels[ch_2nd]} only')
    for ax in axs:
        ax.axis('off')
    plt.rcParams.update({'font.size': 7})
    plt.tight_layout()
    plt.show()

    base_imgname = imgname.split('.ome.tif')[0]
    fig.suptitle(imgname)

    # Save matplotlib figure
    fig_dirpath = overlap_dirpath / f'fig_{ch_save_abbr[ch_1st]}_{ch_save_abbr[ch_2nd]}'
    fig_dirpath.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dirpath / f'{base_imgname}.png')

    return idv_overlap_d

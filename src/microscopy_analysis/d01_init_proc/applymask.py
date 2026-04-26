import numpy as np
from pathlib import Path
from bioio import BioImage
import bioio_ome_tiff
from bioio.writers import OmeTiffWriter
from skimage import morphology, segmentation
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from skimage.draw import polygon2mask
from matplotlib import pyplot as plt
import numpy.ma as ma

import sys
src_path = str(Path.cwd().parent)
if src_path not in sys.path:
    sys.path.append(src_path)
import microscopy_analysis.d00_utils.dirnames as dn
import microscopy_analysis.d00_utils.utilities as utils

z = 0

def apply_ROI_masks(input_dirpath, mask_src, add_cellmask=True, bgsub_cell_ch=None, subset=None):
    '''
    Masks ome-tif images using polygon ROIs and saves these into a new directory

    Parameters:
        input_dirpath (str): string path for folder containing images to be masked
        mask_type (str): 'polygonmasks' or 'metadata'
        add_cellmask (bool) : if True, will also mask image based on the cell channel (cell_ch variable)
        cell_ch (int) : int representing the channel corresponding to the cell (e.g. membrane dye)

    Returns: None
    '''

    input_dirpath = Path(input_dirpath)

    # Gets directory paths for input masks
    proc_dir = utils.get_proc_dirpath(input_dirpath)

    if add_cellmask:
        assert bgsub_cell_ch is not None, "Please input the channel corresponding to the full cell"
        cellmasks_dirpath = masks_dirpath.parent / 'cell_masks'
        cellmasks_dirpath.mkdir(parents=True, exist_ok=True)

    masked_imgs_dirpath = proc_dir / dn.masked_imgs_dirname
    masked_imgs_dirpath.mkdir(parents=True, exist_ok=True)

    # Get list of image paths
    imgpaths = [path for path in Path(input_dirpath).glob('*.ome.tif')]
    imgpaths.sort()

    if subset is None:
        subset = list(np.arange(len(imgpaths)))
    else:
        imgpaths = [imgpaths[i] for i in subset]

    print(f'Applying masks to {len(imgpaths)} images')

    # Apply masks to each image
    for i, imgpath in enumerate(imgpaths):

        img_basename = imgpath.name.split('.ome.tif')[0]
        img_file = BioImage(imgpath, reader=bioio_ome_tiff.Reader)
        img = img_file.data
        ome_metadata = img_file.ome_metadata

        rois, t_list, _, _ = extract_metadata_ROIs(ome_metadata, img.shape)

        print(f'Processing {img_basename} ({len(rois)} found)')
        for i, roi in enumerate(rois):

            imgname_roi = img_basename + f'_ROI{i}'

            if add_cellmask:
                bgsub_cell = img[t_list[i], bgsub_cell_ch, z, :, :]
                bgsub_cell = ma.masked_array(bgsub_cell, ~roi)
                cellmask = refine_cellmask(bgsub_cell, min_size=100000)
                cellmask = cellmask.squeeze().astype('uint8') * np.iinfo('uint8').max
                OmeTiffWriter(cellmask).save(cellmasks_dirpath / (imgname_roi + '.png'))
                roi = cellmask

            roi_exp = np.broadcast_to(roi, img.shape)
            img_masked = ma.masked_array(img, ~roi_exp)

            ome_metadata = utils.construct_ome_metadata(img_masked, img_file.physical_pixel_sizes)
            OmeTiffWriter.save(img_masked, masked_imgs_dirpath / (imgname_roi + '.ome.tif'),
                               ome_xml=ome_metadata)
    print('Done!')


def fill_holes_and_remove_uncon_areas(bin_img, num_areas=1):
    # fill in holes
    bin_img_edited = ndi.binary_fill_holes(bin_img, axes=[3, 4])
    bin_img_edited = remove_unconnected_areas(bin_img_edited, num_areas=num_areas)

    return bin_img_edited

def remove_unconnected_areas(bin_img, dtype='uint16', num_areas=1):
    # remove unconnected areas
    size_t = bin_img.shape[0]
    bin_filt = np.zeros_like(bin_img).astype('int')
    for t in range(size_t):
        labels_slice = label(bin_img[t, 0, 0, :, :])
        regions = regionprops(labels_slice)

        regions = sorted(regions, key=lambda x: x.area, reverse=True)
        for i, reg in enumerate(regions[:num_areas]):
            new_label = i+1
            bin_filt[t, 0, 0, :, :] = np.where(labels_slice == reg.label, new_label,
                                                      bin_filt[t, 0, 0, :, :])
    bin_filt = (bin_filt > 0).astype(dtype)
    return bin_filt

def remove_small_holes(bin_img, area_threshold=5):
    bin_img = bin_img.astype('bool')
    size_t = bin_img.shape[0]
    for t in range(size_t):
        bin_img[t, :, :, :, :] = morphology.remove_small_holes(bin_img[t, :, :, :, :], area_threshold=area_threshold)
    return bin_img

def remove_small_objects(bin_img, min_size=5):
    bin_img = bin_img.astype('bool')
    size_t = bin_img.shape[0]
    for t in range(size_t):
        bin_img[t, :, :, :, :] = morphology.remove_small_objects(bin_img[t, :, :, :, :], min_size=min_size)
    return bin_img

def remove_cmp_near_cellborders(seg_cell, seg_cmp, iter_dil=3):
    seg_cell_bord = np.zeros_like(seg_cell)
    size_t = seg_cell.shape[0]
    for t in range(size_t):
        seg_cell_bord[t, :, :, :, :] = segmentation.find_boundaries(seg_cell[t, :, :, :, :], connectivity=1, mode="thick", background=0)
        seg_cell_bord[t, :, :, :, :] = ndi.binary_dilation(seg_cell_bord[t, :, :, :, :], iterations=iter_dil)
    seg_cmp_edited = np.where(seg_cell_bord, 0, seg_cmp)
    return seg_cmp_edited

def mask_img(img, mask):
    mask = (mask > 0).astype('int')
    if len(mask.shape)==2:
        mask = np.expand_dims(mask, (0, 1, 2))
    mask_exp = np.broadcast_to(mask, img.shape)
    img_masked = (img * mask_exp)
    return img_masked

def extract_metadata_ROIs(metadata, img_shape):
    roi_labels = []
    rois = []
    roi_coords = []
    t_list = []

    for roi_data in metadata.rois:

        # Extract roi label
        roi_labels.append(roi_data.union[0].text)

        # Extract polygon coordinates
        coords = roi_data.union[0].points.split(' ')
        for i, coord in enumerate(coords):
            coord = coord.split(',')
            coords[i] = [int(coord[0]), int(coord[1])]

        # Convert polygon coordinates into a mask
        coords_T = [[e[1], e[0]] for e in coords]
        roi = polygon2mask(img_shape[3:], coords_T)
        plt.imshow(roi)
        rois.append(roi.astype(int))
        roi_coords.append(coords)

        # Get timeslice for polygon mask
        t = metadata.rois[0].union[0].the_t
        if t is None:
            t=0
        t_list.append(t)

    return rois, t_list, roi_labels, roi_coords


def subtract_nuclei_from_cellmasks(nuclei_dirpath):
    nuclei_dirpath = Path(nuclei_dirpath)

    # Get directory path for cell masks
    masks_dirpath = nuclei_dirpath.parent
    cellmask_dirpath = masks_dirpath / dn.cellmask_dirname

    # Check that both nuclei and cell mask directories exist
    assert nuclei_dirpath.exists()
    assert cellmask_dirpath.exists()

    # Create an output path to save cell masks without nuclei
    output_dirpath = masks_dirpath / dn.cellmask_nonuc_dirname
    output_dirpath.mkdir(parents=True, exist_ok=True)

    # Iterate through all tif files in the cell masks directory
    cellmaskpaths = [path for path in cellmask_dirpath.glob('*.tif')]
    for cellmaskpath in cellmaskpaths:

        # Find the matching nucleus mask path
        nucleipaths = [path for path in nuclei_dirpath.glob(f'{cellmaskpath.name}*')]
        print(f'{len(nucleipaths)} nuclei found for {cellmaskpath.name}')

        for nucleipath in nucleipaths:
            cellmask_file = BioImage(cellmaskpath)
            cellmask = cellmask_file.data.squeeze().astype('bool')

            nuclei_file = BioImage(nucleipath, reader=bioio_ome_tiff.Reader)
            nuclei = nuclei_file.data.squeeze().astype('bool')

            # Create a cell mask without the nucleus
            cellmask_nonuc = cellmask * ~nuclei

            # Rescale and save the cell mask without the nucleus
            cellmask_nonuc = cellmask_nonuc.astype('uint8') * np.iinfo('uint8').max
            OmeTiffWriter(cellmask_nonuc).save(output_dirpath / nucleipath.name)

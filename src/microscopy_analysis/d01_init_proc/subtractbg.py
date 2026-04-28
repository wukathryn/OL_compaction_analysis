"""Otsu-based background subtraction with smoothing and outlier clipping.

Pipeline used by notebooks 04 and 06: for each image and parameter set,
clip upper-intensity outliers, Gaussian-smooth the image, run Otsu on
nonzero pixels to obtain a per-frame / per-channel / per-z-slice
background threshold, and subtract that threshold from a smoothed copy
of the original image. Results are written as OME-TIFF stacks plus a
binary visualization, with a parameter-by-image table for downstream
selection.
"""

from pathlib import Path
import numpy as np
from bioio import BioImage
import bioio_ome_tiff
from bioio.writers import OmeTiffWriter
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
import pandas as pd
from microscopy_analysis.d00_utils import dirnames as dn
from microscopy_analysis.d00_utils import utilities as utils
from . import vis_and_rescale
import itertools
import ast


def clip_upper_outliers(img, outlier_perc):
    size_t, size_c, size_z, size_y, size_x = img.shape
    img_nan = np.where(img == 0, np.nan, img)
    outlier_thresh_ch = np.nanpercentile(img_nan, q=outlier_perc, axis=(3, 4), method='higher', keepdims=True)
    img_outliers_clipped = np.clip(img, a_min=None, a_max=outlier_thresh_ch)

    return img_outliers_clipped.astype(img.dtype)


def get_otsu_thresholds(img):
    # flatten 2D images into 1D vectors
    [size_t, size_c, size_z, size_x, size_y] = img.shape

    otsu_thresholds = np.zeros([size_t, size_c, size_z])

    for t in range(size_t):
        for ch in range(size_c):
            for z in range(size_z):
                imgslice = img[t, ch, z, :]
                imgslice = imgslice[imgslice > 0]
                otsu_thresholds[t, ch, z] = threshold_otsu(imgslice, nbins=65536)

    return otsu_thresholds

# Subtract threshold from image

def subtract_background(img, otsu_thresholds):
    otsu_thresholds = np.expand_dims(otsu_thresholds, axis=(3, 4))
    img_bg_subtract = (img.astype('float') - otsu_thresholds).astype('float')
    img_bg_subtract[img_bg_subtract < 0] = 0
    return img_bg_subtract.astype('uint16')

def gaussian_smoothing(img, sigma):
    excluded = np.where(img == 0)
    img = gaussian_filter(img, sigma=sigma, axes=(3,4))
    img[excluded] = 0
    return img

def subtract_bg(input_dirpath, bgsub_list_path):
    '''
    Estimates background intensity via OTSU thresholding and subtracts it from the image

    Parameters:
        imgpath (str): full path for image
        bgsub_df

    Returns:
    '''
    # Get input dirpath, create a directory for bg subtracted images
    input_dirpath = Path(input_dirpath)
    assert input_dirpath.exists(), "Please input a valid directory path"

    assert bgsub_list_path.is_file(), "Please input a valid path for the background subtract list"
    bgsub_df = pd.read_csv(bgsub_list_path)

    # Check whether there's an existing list of thresholds
    bs_thresh_path = bgsub_list_path.parent / (bgsub_list_path.name.split('.csv')[0] + '_thresh.csv')
    if bs_thresh_path.is_file():
        bs_thresh_df = pd.read_csv(bs_thresh_path)
    else:
        bs_thresh_df = pd.DataFrame()

    # Iterate through images
    for img_idx, row in bgsub_df.iterrows():
        if row['BG subtract?']:
            imgname = row['input image name']
            imgpath = Path(input_dirpath / imgname)

            bgsb_dirpath = Path(row[f'init bgsb dirpath'])
            bgsb_dirpath.mkdir(parents=True, exist_ok=True)
            bin_dirpath = Path(row[f'init bin dirpath'])
            fig_dirpath = bin_dirpath / dn.figs_dirname
            fig_dirpath.mkdir(parents=True, exist_ok=True)

            if imgpath.is_file():
                print(f'Processing file {img_idx}/{len(bgsub_df) - 1}: {imgname}')

                # Open image
                img_file = BioImage(imgpath, reader=bioio_ome_tiff.Reader)
                orig_img = img_file.data
                ome_metadata = img_file.ome_metadata
                basename = imgpath.name.split('.')[0]

                # Extract channels to be processed
                ch_to_process = param_to_list(row['ch_to_process'])
                img_chsubset = orig_img[:, ch_to_process, :, :, :]

                # Get parameters from df
                p_prev = row['# prev tested params']
                rescale_perc_grayvals = param_to_list(row['rescale_perc_grayval'])
                outlier_percentiles = param_to_list(row['outlier_percentiles'])
                sigmas_smoothing = param_to_list(row['sigmas_smoothing'])

                # get combinations of parameters
                params_list = [params for params in (itertools.product(rescale_perc_grayvals, outlier_percentiles,
                                                                       sigmas_smoothing))]

                # create p_idx, a unique identifier for the parameter set used
                p_idx = p_prev

                rescaled_img_chsubset, _ = vis_and_rescale.rescale_img(img_chsubset, target_perc_grayval=30)
                bgsb_for_fig = [rescaled_img_chsubset]
                pset_for_fig = ['orig img, rescaled']
                for params in params_list:

                    perc_grayval = params[0]
                    outlier_perc = params[1]
                    sigma = params[2]

                    p_idx = p_idx + 1

                    # Pre-process image via smoothing and clipping outliers
                    img_preprocessed = utils.crop_black_borders(img_chsubset)
                    img_preprocessed = clip_upper_outliers(img_preprocessed, outlier_perc)
                    img_preprocessed = gaussian_smoothing(img_preprocessed, sigma)

                    # Use OTSU to obtain background thresholds from pre-processed image
                    otsu_thresholds = get_otsu_thresholds(img_preprocessed)

                    # Smooth original image and subtract background
                    bg_sb_chsubset = img_chsubset.copy()
                    bg_sb_chsubset = gaussian_smoothing(bg_sb_chsubset, sigma)
                    bg_sb_chsubset = subtract_background(bg_sb_chsubset, otsu_thresholds)

                    # store info for generating figure
                    bgsb_for_fig.append((bg_sb_chsubset > 0).astype('bool'))
                    pset_for_fig.append(f'p{p_idx}')

                    # concatenate bg subtracted image to image subset and save image
                    bg_sb_img = np.concatenate((img_chsubset, bg_sb_chsubset), axis=1)
                    bgsub_imgname = f'{basename}_p{p_idx}'
                    bs_ome_metadata = utils.construct_ome_metadata(bg_sb_img, img_file)
                    OmeTiffWriter.save(bg_sb_img, bgsb_dirpath / (bgsub_imgname + '.ome.tif'), ome_xml=bs_ome_metadata)

                    # concatenate binary image to image subset and save image
                    bin_chsubset = ((bg_sb_chsubset > 0).astype('int') *
                                    np.iinfo(img_file.dtype).max).astype(img_file.dtype)
                    bin_img = np.concatenate((img_chsubset, bin_chsubset), axis=1)
                    OmeTiffWriter.save(bin_img, bin_dirpath / (bgsub_imgname + '.ome.tif'), ome_xml=bs_ome_metadata)

                    # append parameters and thresholds
                    thresh_df_idx = len(bs_thresh_df)+1
                    bs_thresh_df.at[thresh_df_idx, 'image name'] = bgsub_imgname
                    bs_thresh_df.at[thresh_df_idx, 'img idx'] = img_idx
                    bs_thresh_df.at[thresh_df_idx, 'input imgname'] = imgpath
                    bs_thresh_df.at[thresh_df_idx, 'input dirpath'] = input_dirpath
                    bs_thresh_df.at[thresh_df_idx, f'init bgsb {ch_to_process} dirpath'] = bgsb_dirpath
                    bs_thresh_df.at[thresh_df_idx, f'init bin {ch_to_process} dirpath'] = bin_dirpath
                    bs_thresh_df.at[thresh_df_idx, 'channel'] = ch_to_process
                    bs_thresh_df.at[thresh_df_idx, 'param set'] = p_idx
                    bs_thresh_df.at[thresh_df_idx, 'rescale_perc_grayval'] = perc_grayval
                    bs_thresh_df.at[thresh_df_idx, 'outlier percentile'] = outlier_perc
                    bs_thresh_df.at[thresh_df_idx, 'sigma smoothing'] = sigma

                    bs_thresh_df.at[thresh_df_idx, f'ch{ch_to_process} threshold'] = ''
                    bs_thresh_df[f'ch{ch_to_process} threshold'] = bs_thresh_df[f'ch{ch_to_process} threshold'].astype('object')
                    bs_thresh_df.at[thresh_df_idx, f'ch{ch_to_process} threshold'] = otsu_thresholds.squeeze().tolist()
                    #bs_thresh_df.drop_duplicates()
                    bs_thresh_df.to_csv(bs_thresh_path, index=False)

                # update params df
                bgsub_df.at[img_idx, 'BG subtract?'] = False
                bgsub_df.at[img_idx, '# prev tested params'] = p_idx

                # save params df
                bgsub_df.to_csv(bgsub_list_path, index=False)

                # create figure for easy visualization
                rescaled_img_chsubset, _ = vis_and_rescale.rescale_img(img_chsubset, target_perc_grayval=30)
                fig = vis_and_rescale.create_fig(bgsb_for_fig, imglabels=pset_for_fig,
                                                 figtitle=imgname, show=True)
                fig.savefig(fig_dirpath / (basename + '.png'))

            else:
                print(f'{imgname} not found.')
    return bgsub_df

def param_to_list(param):
# if the param is not in list format, convert it to a lsit
    if isinstance(param, str):
        param = ast.literal_eval(param)
    if isinstance(param, (int, float, complex)):
        param = [param]
    assert isinstance(param, list), \
        'Please include a number or a list of numbers for this parameter'
    return param


def subtract_median(img):
    img_ma = np.ma.masked_array(img, img == 0)
    median = np.ma.median(img_ma, axis=(3, 4))
    median = np.expand_dims(median, axis=(3,4))
    median = np.broadcast_to(median, img.shape).astype('float')
    img_bgsb = img.astype('float') - median
    img_bgsb[img_bgsb < 0] = 0

    return img_bgsb.astype('uint16')

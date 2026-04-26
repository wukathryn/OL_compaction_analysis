from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import pandas as pd
from bioio import BioImage
import bioio_ome_tiff
from bioio.writers import OmeTiffWriter
from microscopy_analysis.d00_utils import utilities as utils
from microscopy_analysis.d00_utils import dirnames as dn
from tqdm import tqdm


dpi = 100

# Obtain scaling factor for each image based on target percent grayvalue for the 50th percentile pixel
def get_scaling_fact(img, target_perc_grayval, percentile=50):
    # Calculate the target grayvalue for the image type
    target_grayval = (target_perc_grayval / 100) * np.iinfo(img.dtype).max

    # Calculate scaling factor
    img = img.astype('float')
    img[img == 0] = 'nan'
    orig_grayvals = np.nanpercentile(img, q=percentile, axis=(2, 3, 4))
    scaling_fact = target_grayval / orig_grayvals
    scaling_fact = np.expand_dims(scaling_fact, axis=(2, 3, 4))

    return scaling_fact


# Rescale image by scaling factor. Any values exceeding the max value for the image type will be clipped to the max value
def multiply_by_scaling_fact(img, scaling_fact):
    dtype = img.dtype
    img_rescaled = (img * scaling_fact)
    img_rescaled = np.clip(img_rescaled, a_min=None, a_max=np.iinfo(dtype).max)
    return img_rescaled.astype(dtype)


# Converts img from uint16 to uint8, scaling pixel intensities accordingly
def convert_to_8bit(img):
    assert str(img.dtype) == 'uint16', f'Can only convert uint16 images to uint8'
    scaling_factor = np.iinfo('uint8').max / np.iinfo('uint16').max
    img_rescaled = (img * scaling_factor)
    img_rescaled = np.clip(img_rescaled, a_min=None, a_max=np.iinfo('uint8').max)
    return img_rescaled.astype('uint8')


def rescale_img(img, target_perc_grayval=30,
                standardize_scaling=False, scaling_fact=None, conv_to_8bit=True):
    if standardize_scaling is False or scaling_fact is None:
        scaling_fact = get_scaling_fact(img, target_perc_grayval)
    img_vis = multiply_by_scaling_fact(img, scaling_fact)
    if conv_to_8bit is True:
        img_vis = convert_to_8bit(img_vis)
    return img_vis, scaling_fact

def create_fig(imgs, imglabels=None, figtitle=None, show=True):
    # note: imgs must be a list of imgs to display, and these imgs must have the same dimensions

    size_t, size_c, _, size_y, size_x = imgs[0].shape
    n_imgs = len(imgs)

    # set up figure size
    fig_y_size = 4
    fig_x_size = 4 * (size_x / size_y)
    fig_y_padding = 1
    fig_x_padding = 0.75

    fig, ax = plt.subplots(size_t, n_imgs * size_c, figsize=(fig_x_size * n_imgs * size_c + fig_x_padding,
                                                    fig_y_size * size_t + fig_y_padding),
                           squeeze=False, layout='constrained')
    for ch in range(size_c):
        for t in range(size_t):
            for i in range(n_imgs):
                img = imgs[i]
                ax[t, ch + i].imshow(img[t, ch, 0, :, :], cmap='gray', interpolation=None)
                ax[t, ch + i].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

                # add column headings
                if (imglabels is not None) and (t == 0):
                    ax[t, ch + i].set_title(f'Ch{ch}: {imglabels[i]}')

                # add timelapse frame label
                if ((ch==0) & (i==0)):
                    ax[t, ch + i].set_ylabel(f'T={t}', rotation=0, labelpad=15)

    if figtitle is not None:
        fig.suptitle(figtitle)

    if show:
        plt.show()

    return fig

def create_cmp_fig(seg_cmp, seg_caax, cmp_val=255, caax_val=100):
    cmp_fig = seg_caax * 100 + seg_cmp * 255
    cmp_fig = cmp_fig.astype('uint8')
    return cmp_fig

#TODO: consider replacing this fxn with more general create_fig fxn above
def generate_subtractbg_fig(orig_img, bgsb_img, bgsub_imgname, params, target_perc_grayval=30, index=None, ome_metadata=None):
    size_t, size_c, _, _, _ = orig_img.shape

    rescaled_orig_img, scaling_fact = rescale_img(orig_img, target_perc_grayval=target_perc_grayval,
                                                  standardize_scaling=False, conv_to_8bit=True)
    binary_bgsb_img = (bgsb_img > 0).astype('uint8')

    # # Get ROIs from metadata
    # if ome_metadata is not None:
    #     rois, roi_labels, roi_coords = extract_ROIs(ome_metadata, orig_img.shape)

    num_imgs = 2
    fig = plt.figure(figsize=(2 + num_imgs * size_c * 3, size_t * 3.5), constrained_layout=True)
    plt.rcParams.update({'font.size': 8})

    subfigs = fig.subfigures(1, size_c, squeeze=False)

    for c in range(size_c):
        subfig = subfigs[0, c]
        #subfig.suptitle(f'Channel {c + 1}')

        subfigs_imgtype = subfig.subfigures(1, num_imgs)
        subfigs_imgtype[0].suptitle('Rescaled orig image')
        subfigs_imgtype[1].suptitle('Bg subtracted binary')

        axs_img = subfigs_imgtype[0].subplots(size_t, squeeze=False)
        axs_binary = subfigs_imgtype[1].subplots(size_t, squeeze=False)

        for t in range(size_t):
            axs_img[t, 0].imshow(rescaled_orig_img[t, c, 0, :, :], cmap='gray', interpolation=None)
            axs_img[t, 0].axis('off')
            axs_img[t, 0].set_title(f'Timepoint: {t + 1}')
            axs_binary[t, 0].imshow(binary_bgsb_img[t, c, 0, :, :], cmap='gray', interpolation=None)
            axs_binary[t, 0].axis('off')
            axs_binary[t, 0].set_title(f'Timepoint: {t + 1}')

            if ome_metadata is not None:
                for i, coords in enumerate(roi_coords):
                    # Add ROI outline
                    roi_patch = mpatches.Polygon(coords, fill=False, edgecolor='blue', linewidth=1)
                    axs_binary[t, 0].add_patch(roi_patch)

                    # Add label
                    coords = np.array(coords)
                    [max_x, max_y] = np.max(coords, axis=0)
                    [min_x, min_y] = np.min(coords, axis=0)
                    axs_binary[t, 0].text((max_x + min_x) / 2, min_y, i, fontsize=12, color='blue')

    fig.suptitle(f'Idx: {index}, {bgsub_imgname}\n{str(params)}')
    plt.show()

    return fig

def batch_rescale_create_fig(input_dirpath, output_dirpath=None, target_perc_grayval=30, suffix='.ome.tif'):
    imgpaths = [path for path in input_dirpath.glob(f'*{suffix}')]
    imgpaths.sort()

    if output_dirpath is None:
        output_dirpath = input_dirpath / dn.figs_dirname
        output_dirpath.mkdir(exist_ok=True)

    for imgpath in tqdm(imgpaths):
        imgname = imgpath.name.split(suffix)[0]
        figsavepath = output_dirpath / (imgname + '.png')

        # Only generate figure if it doesn't exist
        if not figsavepath.is_file():
            img = BioImage(imgpath, reader=bioio_ome_tiff.Reader).data

            rescaled_img, _ = rescale_img(img, target_perc_grayval=target_perc_grayval,
                                                          standardize_scaling=False, conv_to_8bit=True)

            fig = create_fig([rescaled_img], imglabels=[f'rescaled (target_perc_grayval:{target_perc_grayval}'],
                             figtitle=imgname, show=True)

            fig.savefig(figsavepath)

    return

def display_fig(img):
    height, width, depth = img.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img, cmap='gray')
    plt.show()

def show_figs(fig_dirpath, selection_phrase=None, subset=None):
    imgpaths = [imgpath for imgpath in fig_dirpath.glob('.png')]
    imgpaths.sort()

    if subset is not None:
        imgpaths = [imgpaths[e] for e in subset]
    if selection_phrase is not None:
        imgpaths = [path for path in imgpaths if selection_phrase in path.name]

    print(f'{len(imgpaths)} images found')

    for imgpath in imgpaths:
        img = mpimg.imread(imgpath)
        display_fig(img)

def batch_rescale_imgs(input_dirpath, target_perc_grayval=30):
    input_dirpath = Path(input_dirpath)
    assert input_dirpath.exists(), "Please input a valid directory path"
    vis_dirpath = input_dirpath.parent / dn.init_rescale_dirname
    vis_dirpath.mkdir(parents=True, exist_ok=True)

    imgpaths = [path for path in input_dirpath.glob('*.ome.tif')]
    if len(imgpaths)==0:
        print("No images found.")
    print(f'Rescaling {len(imgpaths)} images')

    rescale_params_df = pd.DataFrame()
    scaling_fact = None
    for i, imgpath in enumerate(imgpaths):
        img_file = BioImage(imgpath, reader=bioio_ome_tiff.Reader)
        img = img_file.data
        img_vis, scaling_fact = rescale_img(img, target_perc_grayval=target_perc_grayval, standardize_scaling=False,
                                            scaling_fact=scaling_fact, conv_to_8bit=False)

        ome_metadata = utils.construct_ome_metadata(img_vis, img_file)
        OmeTiffWriter.save(img_vis, Path(vis_dirpath) / imgpath.name, ome_xml=ome_metadata)

        img_rescale_params = pd.DataFrame({'Image name': [imgpath.name]})

        size_c = scaling_fact.shape[1]
        for c in range(size_c):
            img_rescale_params[f'Scaling_fact_ch{c}'] = [scaling_fact[:, c, :, :, :].squeeze()]
        rescale_params_df = pd.concat([rescale_params_df, img_rescale_params], ignore_index=True)

    rescale_params_df.to_csv(Path(vis_dirpath) / 'rescale_params.csv')
    print("Done!")

    return vis_dirpath

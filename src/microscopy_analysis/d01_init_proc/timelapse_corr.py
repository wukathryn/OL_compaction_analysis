from pathlib import Path
from bioio import BioImage
import bioio_ome_tiff
from bioio.writers import OmeTiffWriter
from skimage.exposure import match_histograms
import microscopy_analysis.d00_utils.utilities as utils
import numpy as np

# Might not be useful
def match_histograms_dir(img_dirpath, ref_imgpath):
    ref_img = BioImage(ref_imgpath, reader=bioio_ome_tiff.Reader).data

    img_dirpath = Path(img_dirpath)
    imgpaths = [p for p in img_dirpath.glob('*.ome.tif')]
    print(f'{len(imgpaths)} images found in {img_dirpath.name}.')

    if len(imgpaths) > 0:
        output_dirpath = img_dirpath.parent / (img_dirpath.name + '_histomatched')
        output_dirpath.mkdir(parents=True, exist_ok=True)

    for p in imgpaths:
        img_file = BioImage(p, reader=bioio_ome_tiff.Reader)
        img = img_file.data
        matched_img = match_histograms(img, ref_img)
        ome_metadata = utils.construct_ome_metadata(matched_img, img_file.physical_pixel_sizes)
        OmeTiffWriter.save(matched_img, (output_dirpath / p.name), ome_xml=ome_metadata)

def timelapse_histomatch(img, ch=None, t_ref=0):
    if ch is None:
        size_c = img.shape[1]
        ch = np.arange(size_c)
    if not isinstance(ch, list):
        ch = [ch]
    img_chsubset = img[:, ch, :, :, :]

    ref_img = img[t_ref, :, :, :]

    size_t = img.shape[0]
    chsubset_histomatched = img_chsubset
    for t in range(size_t):
        if t!=t_ref:
            chsubset_histomatched[t, :, :, :, :] = match_histograms(img_chsubset[t, :, :, :, :], ref_img)
    img_histomatched = img

    img_histomatched[:, ch, :, :, :] = chsubset_histomatched
    return img_histomatched


def timelapse_simpleratio(img, t_ref=0):
    # calculate the ratio of intensities between each frame and the first timeframe
    img = np.ma.masked_array(img, img == 0)
    mean_int = np.ma.mean(img, axis=(3, 4))

    simple_ratio = mean_int[t_ref, np.newaxis, :, :] / mean_int
    simple_ratio = np.expand_dims(simple_ratio, (3, 4))
    img_corr = np.broadcast_to(simple_ratio, img.shape) * img

    return img_corr

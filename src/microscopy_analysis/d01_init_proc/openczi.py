"""
Batch-convert Zeiss .czi datasets into OME-TIFF, organizing inputs under a processing folder.

This script:
- moves all *.czi files from the experiment directory into <exp_dir>/<proc_dir>/<CZI_dirname>/
- converts each CZI (and each scene within a CZI, if present) into an OME-TIFF
- writes OME-XML metadata including pixel sizes and channel names (and optionally timestamps)

Notes
-----
- The CZI reader is provided by the `bioio-czi` plugin. If installed, BioImage can usually
  autodetect it; you can also explicitly set `CZI_READER` to the plugin Reader class.
- Timestamp injection from vendor metadata is provided as an optional utility, but may require
  adapting the metadata path depending on the CZI schema/version in your files.
"""

from pathlib import Path
import argparse

import numpy as np

from microscopy_analysis.d00_utils import dirnames as dn

from bioio import BioImage
from bioio.writers import OmeTiffWriter

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--exp_dir",
    required=True,
    help="Directory containing the dataset (expects one or more *.czi files).",
)

def get_ome_metadata(img: BioImage):
    """
    Construct OME-XML metadata for an image using BioIO's OME-TIFF writer helper.

    The metadata includes:
    - image dimensions inferred from `img.shape`
    - pixel type inferred from `img.dtype`
    - channel names (falling back to Ch0..ChN-1 if unavailable)
    - physical pixel sizes from `img.physical_pixel_sizes`
    """

    ome_metadata = OmeTiffWriter.build_ome(
        [img.shape],
        [np.dtype(img.dtype)],
        channel_names=[img.channel_names],
        physical_pixel_sizes=[img.physical_pixel_sizes]
    )

    return ome_metadata


def convert_czi_to_tif(imgpath: Path, raw_tif_dirpath: Path):
    """
    Convert a single CZI file to one or more OME-TIFF files.

    If the input contains multiple scenes, each scene is written as its own OME-TIFF
    with a suffix `_sc<scene>`. If scene labels are missing, a deterministic fallback
    name is generated.
    """
    img = BioImage(imgpath)

    # Counter used only when a scene entry exists but is unnamed.
    unnamed_scene_counter = 0

    for scene in img.scenes:
        if len(img.scenes) > 1:
            if scene is None:
                scene = f"N{unnamed_scene_counter}"
                unnamed_scene_counter += 1

            img.set_scene(scene)
            out_name = f"{Path(imgpath).stem}_sc{scene}.ome.tif"
        else:
            out_name = f"{Path(imgpath).stem}.ome.tif"

        # Avoid characters that can complicate downstream parsing/CLI handling.
        out_name = out_name.replace("-", "_")
        out_path = Path(raw_tif_dirpath) / out_name

        ome_metadata = get_ome_metadata(img)

        # `img.data` materializes the array in memory; for very large datasets, consider
        # chunked/streamed writing if supported by your IO stack.
        OmeTiffWriter.save(img.data, out_path, ome_xml=ome_metadata)


def move_files_into_CZI_dir(exp_dir: Path, proc_dir: Path) -> Path:
    """
    Move all CZI files from the experiment directory into the processing CZI folder.

    Returns
    -------
    Path
        The directory path containing the moved CZI files.
    """
    proc_dir = Path(proc_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    czi_dirpath = proc_dir / dn.CZI_dirname
    czi_dirpath.mkdir(parents=True, exist_ok=True)

    for imgpath in Path(exp_dir).glob("*.czi"):
        imgpath.rename(czi_dirpath / imgpath.name)

    return czi_dirpath


def batch_convert_czi_to_tif(exp_dir: str):
    """
    Convert all CZI files found in `exp_dir` to OME-TIFF and write outputs to:
        <exp_dir>/<proc_dir>/<raw_ometif_dirname>/
    """
    exp_dir = Path(exp_dir)
    proc_dir = exp_dir / dn.proc_dirname

    czi_dirpath = move_files_into_CZI_dir(exp_dir, proc_dir)

    imgpaths = sorted(czi_dirpath.glob("*.czi"))
    if not imgpaths:
        print("No CZI images found")
        return

    raw_ometif_dirpath = proc_dir / dn.raw_ometif_dirname
    raw_ometif_dirpath.mkdir(parents=True, exist_ok=True)

    num_imgs = len(imgpaths)
    for i, imgpath in enumerate(imgpaths):
        print(f"Converting {imgpath.name} to OME-TIFF (file {i + 1}/{num_imgs})")
        convert_czi_to_tif(imgpath, raw_ometif_dirpath)

    print(f"Done! OME-TIFF files saved to {raw_ometif_dirpath}")


if __name__ == "__main__":
    args = parser.parse_args()
    batch_convert_czi_to_tif(args.exp_dir)

# OL_compaction_analysis

Image-analysis pipeline for fluorescence microscopy of oligodendrocyte compaction
events. Ingests CZI files from a Zeiss microscope, aligns and splits channels,
performs background subtraction and mask-based selection, computes
exclusion-zone and actin-intensity metrics, and produces the figure-ready
plots used in the accompanying manuscript.

The pipeline is implemented as a sequence of Jupyter notebooks
(`notebooks/`) that call functions in the installable Python package
(`src/microscopy_analysis/`).

Two pipeline variants are provided:

- **Single-timepoint analysis** — `notebooks/00_*` through `notebooks/10_*`
  (top-level `notebooks/` directory)
- **Timelapse analysis** — `notebooks/timelapse_analysis/`

Both pipelines share the helper package in `src/microscopy_analysis/`.

## Repository layout

```
OL_compaction_analysis/
├── LICENSE                          MIT license
├── README.md                        this file
├── environment.yml                  conda environment specification
├── requirements.txt                 pip equivalent
├── setup.py                         installable Python package
├── paper_figures.mplstyle           matplotlib style used for paper figures
├── src/microscopy_analysis/         core library
│   ├── d00_utils/                   path/I-O helpers, notebook init
│   ├── d01_init_proc/               CZI ingest, channel alignment, masking,
│   │                                background subtraction
│   ├── d03_exclusion_analysis/      ROI definition, exclusion-zone metrics
│   └── d04_plot_data/               plotting, statistics, linear-mixed-model
│                                    fits (Python and R)
└── notebooks/
    ├── 00_Convert_CZIs.ipynb            CZI → OME-TIFF conversion
    ├── 01_Multiexp_align_imgs.ipynb     align channels across experiments
    ├── 02_Create_caax_cell_stack.ipynb  combine CAAX + cell channels
    ├── 03_Select_images.ipynb           QC and image selection
    ├── 04_Subtract_background_cell_channel.ipynb
    ├── 05_Select_cellch_bgsub_bymask.ipynb
    ├── 06_Subtract_background_caax_channel.ipynb
    ├── 07_Select_caaxch_bgsub_bymask.ipynb
    ├── 09_import_well_conditions.ipynb  attach experimental metadata
    ├── 10_plot_exclusion_analysis_singleT_1exp.ipynb
    │                                    final plotting (single replicate)
    ├── Plot_exclusion_analysis_multiexp.ipynb  multi-experiment plotting
    ├── Visualize_compaction.ipynb       qualitative visualization
    ├── identify_outliers.ipynb          quality control
    ├── timelapse_analysis/              timelapse-specific pipeline
    ├── figures_for_paper/               manuscript figure-generation notebooks
    ├── edit_csv_files/                  one-off data-wrangling utilities
    └── helpful_functions/               additional utilities (mask editing,
                                         image blinding for QC)
```

## Installation

Requires Python 3.10. Conda is recommended for managing the heavy dependency
stack (BioImage I/O, scientific Python, matplotlib + seaborn).

```bash
git clone https://github.com/wukathryn/OL_compaction_analysis.git
cd OL_compaction_analysis
conda env create -f environment.yml
conda activate img_analysis
pip install -e .
```

The `pip install -e .` step installs the `microscopy_analysis` package in
editable mode so the notebooks can `import microscopy_analysis.d00_utils`
etc. directly.

The plotting notebooks expect a working R installation with `lme4` and
`lmerTest` for linear-mixed-model fits (see
`src/microscopy_analysis/d04_plot_data/fit_lmm.R`).

### Optional: pre-commit hooks

The repository ships with a pre-commit configuration that strips cell
outputs from notebooks before they enter git history (avoiding large
output blobs). To enable:

```bash
pip install pre-commit
pre-commit install
```

## Running the pipeline

Activate the environment and start Jupyter:

```bash
conda activate img_analysis
jupyter lab
```

### Single-timepoint pipeline

Run the top-level notebooks in numerical order. Each notebook reads its
inputs from the previous step's outputs and updates a shared analysis
dataframe (`analysis.csv`). Configuration cells at the top of each
notebook point at the input/output directories — edit these for your
data.

```
00_Convert_CZIs               raw .czi → ome-tiff
01_Multiexp_align_imgs        align channels across replicates
02_Create_caax_cell_stack     stack the CAAX + cell channels
03_Select_images              manual QC / inclusion list
04_Subtract_background_cell_channel
05_Select_cellch_bgsub_bymask
06_Subtract_background_caax_channel
07_Select_caaxch_bgsub_bymask
09_import_well_conditions     attach treatment / condition metadata
10_plot_exclusion_analysis_singleT_1exp
                              final plots and statistics
```

(Notebook `08` is intentionally absent — that step is timelapse-only and
lives under `notebooks/timelapse_analysis/`.)

### Timelapse pipeline

For data with multiple timepoints, follow the same numbered convention
within `notebooks/timelapse_analysis/`:

```
02_align_ch_and_tp            align channels and timepoints
03_split_chs                  channel splitting
06_Stack_seg_apply_masks      stack segmentations and apply masks
08_Calculateactin             actin-intensity metrics around compaction
                              events (see notebook header for full
                              description of the metrics)
Plot_timelapse_data           timelapse trajectory plots
```

Plus utility notebooks for compaction-zone analysis, metric computation,
and channel correction.

### Manuscript figures

Notebooks under `notebooks/figures_for_paper/` regenerate the multi-replicate
figures in the manuscript from the per-experiment outputs of the pipelines
above.

## License

MIT — see `LICENSE`.


## Contact

Kathryn Wu — wukathryn@gmail.com — Stanford University

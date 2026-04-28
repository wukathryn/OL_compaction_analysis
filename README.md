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

- **Single-timepoint analysis** — `notebooks/single_tp_analysis/`
- **Timelapse analysis** — `notebooks/timelapse_analysis/`

Both pipelines share the helper package in `src/microscopy_analysis/`.

## Repository layout

```
OL_compaction_analysis/
├── LICENSE                          MIT license
├── README.md                        this file
├── environment.yml                  conda environment specification
├── requirements.txt                 pip equivalent
├── pyproject.toml                   Python package metadata
├── paper_figures.mplstyle           matplotlib style used for paper figures
├── src/microscopy_analysis/         core library
│   ├── d00_utils/                   path/I-O helpers, notebook init
│   ├── d01_init_proc/               CZI ingest, channel alignment, masking,
│   │                                background subtraction
│   ├── d02_metrics/                 area and intensity computation
│   └── d04_plot_data/               plotting, statistics, linear-mixed-model
│                                    fits (Python and R)
└── notebooks/
    ├── single_tp_analysis/              single-timepoint pipeline (steps 00–11)
    ├── timelapse_analysis/              timelapse pipeline (steps 02–10)
    └── helpful_functions/               shared utilities: mask editing,
                                         image blinding, dataframe wrangling,
                                         well-condition standardization,
                                         outlier identification
```

## Installation

Requires Python 3.13. Conda is recommended for managing the heavy dependency
stack (BioImage I/O, scientific Python, matplotlib + seaborn).

```bash
git clone https://github.com/wukathryn/OL_compaction_analysis.git
cd OL_compaction_analysis
conda env create -f environment.yml
conda activate cmp_analysis
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
conda activate cmp_analysis
jupyter lab
```

### Single-timepoint pipeline

Run the notebooks in `notebooks/single_tp_analysis/` in numerical order.
Each notebook reads its inputs from the previous step's outputs and
updates a shared analysis dataframe (`analysis.csv`). Configuration cells
at the top of each notebook point at the input/output directories — edit
these for your data.

```
00_Convert_CZIs                              raw .czi → ome-tiff
01_Multiexp_align_imgs                       align channels across replicates
02_Create_caax_cell_stack                    stack the CAAX + cell channels
03_Select_images                             manual QC / inclusion list
04_Subtract_background_cell_channel
05_Select_cellch_bgsub_bymask
06_Subtract_background_caax_channel
07_Select_caaxch_bgsub_bymask
08_compute_metrics                           per-image areas, intensities, % compaction
09_import_well_conditions                    attach treatment / condition metadata
10_plot_exclusion_analysis_singleT_1exp      single-replicate plots and statistics
11_plot_exclusion_analysis_multirep          multi-replicate manuscript figure
```

### Timelapse pipeline

For data with multiple timepoints, run the notebooks in
`notebooks/timelapse_analysis/` in numerical order:

```
02_align_ch_and_tp                           align channels and timepoints
03_split_chs                                 channel splitting
06_Stack_seg_apply_masks                     stack segmentations and apply masks
08_Calculateactin                            actin-intensity metrics around
                                             compaction events (see notebook
                                             header for the full metric definitions)
compute_metrics                              per-frame areas, intensities, % compaction
10_plot_timelapse_data_multirep              multi-replicate trajectory figure
```

Plus utility notebooks (`Plot_timelapse_data`, `analyze_indiv_cmpzones`,
`correct_actin_ch`) for single-replicate plotting, per-zone analysis,
and channel correction.

### Shared utilities

Notebooks in `notebooks/helpful_functions/` are not part of either
sequential pipeline. They cover one-off needs: editing cell masks,
blinding image names for QC, merging or standardizing analysis
dataframes, importing well-condition metadata, identifying outliers, and
visualizing compaction qualitatively.

## License

MIT — see `LICENSE`.


## Contact

Kathryn Wu — wukathryn@gmail.com — Zuchero lab, Stanford University

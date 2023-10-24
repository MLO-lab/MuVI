# MuVI

A multi-view latent variable model with domain-informed structured sparsity, that integrates noisy domain expertise in terms of feature sets.

[Examples](examples/1_basic_tutorial.ipynb) | [Paper](https://proceedings.mlr.press/v206/qoku23a/qoku23a.pdf) | [BibTeX](citation.bib)

[![Build](https://github.com/mlo-lab/muvi/actions/workflows/build.yml/badge.svg)](https://github.com/mlo-lab/muvi/actions/workflows/build.yml/)
[![Coverage](https://codecov.io/gh/mlo-lab/muvi/branch/master/graph/badge.svg)](https://codecov.io/gh/mlo-lab/muvi)

## Basic usage

The `MuVI` class is the main entry point for loading the data and performing the inference:

```py
import numpy as np
import pandas as pd
import anndata as ad
import mudata as md
import muvi

# Load processed input data (missing values are allowed)
# Matrix of dimensions n_samples x n_rna_features
rna_df = pd.read_csv(...)
# Matrix of dimensions n_samples x n_prot_features
prot_df = pd.read_csv(...)

# Load prior feature sets, e.g. gene sets
gene_sets = muvi.fs.from_gmt(...)
# Binary matrix of dimensions n_gene_sets x n_rna_features
gene_sets_mask = gene_sets.to_mask(rna_df.columns)

# Create a MuVI object by passing both input data and prior information
model = muvi.MuVI(
    observations={"rna": rna_df, "prot": prot_df},
    prior_masks={"rna": gene_sets_mask},
    ...
    device=device,
)

# Alternatively, create a MuVI model from AnnData (single-view)
rna_adata = ad.AnnData(rna_df, dtype=np.float32)
rna_adata.varm['gene_sets_mask'] = gene_sets_mask.T
model = muvi.tl.from_adata(
    adata, 
    prior_mask_key="gene_sets_mask", 
    ..., 
    device=device
)

# Alternatively, create a MuVI model from MuData (multi-view)
mdata = md.MuData({"rna": rna_adata, "prot": prot_adata})
model = muvi.tl.mdata(
    mdata, 
    prior_mask_key="gene_sets_mask", 
    ..., 
    device=device
)

# Fit the model for a given number of training epochs
model.fit(batch_size, n_epochs, ...)

# Continue with the downstream analysis (see below)
```

## Submodules

The package consists of three additional submodules for analysing the results post-training:

- [`muvi.tl`](muvi/tools/utils.py) provides tools for downstream analysis, e.g.,
  - compute `muvi.tl.variance_explained` across all factors and views
  - `muvi.tl.test` the significance between the prior feature sets and the inferred factors
  - apply clustering on the latent space such as `muvi.tl.leiden`
  - `muvi.tl.save` the model in order to `muvi.tl.load` it at a later point in time
- [`muvi.pl`](muvi/tools/plotting.py) works in tandem with `muvi.tl` by providing visualization methods such as
  - `muvi.pl.variance_explained` (see above)
  - plotting the latent space via `muvi.pl.tsne`, `muvi.pl.scatter` or `muvi.pl.stripplot`
  - investigating factors in terms of their inferred loadings with `muvi.pl.inspect_factor`
- [`muvi.fs`](muvi/tools/feature_sets.py) serves the data structure and methods for loading, processing and storing the prior information from feature sets

## Tutorials

Check out our [basic tutorial](examples/1_basic_tutorial.ipynb) to get familiar with `MuVI`, or jump straight to a [single-cell multiome](examples/3a_single-cell_multi-omics_integration.ipynb) analysis!

`R` users can readily export a trained `MuVI` model into `R` with a single line of code and resume the analysis with the [`MOFA2`](https://biofam.github.io/MOFA2) package.

```py
muvi.ext.save_as_hdf5(model, "muvi.hdf5", save_metadata=True)
```

See [this vignette](https://raw.githack.com/MLO-lab/MuVI/master/examples/4_single-cell_multi-omics_integration_R.html) for more details!

## Installation

We suggest using [conda](https://docs.conda.io/en/latest/miniconda.html) to manage your environments, and [pip](https://pypi.org/project/pip/) to install `muvi` as a python package. Follow these steps to get `muvi` up and running!

1. Create a python environment in `conda`:

```bash
conda create -n muvi python=3.9
```

2. Activate freshly created environment:

```bash
source activate muvi
```

3. Install `muvi` with `pip`:

```bash
python3 -m pip install muvi
```

4. Alternatively, install the latest version with `pip`:

```bash
python3 -m pip install git+https://github.com/MLO-lab/MuVI.git
```

Make sure to install a GPU version of [PyTorch](https://pytorch.org/) to significantly speed up the inference.

## Citation

If you use `MuVI` in your work, please use this [BibTeX](citation.bib) entry:

> **Encoding Domain Knowledge in Multi-view Latent Variable Models: A Bayesian Approach with Structured Sparsity**
>
> Arber Qoku and Florian Buettner
>
> _International Conference on Artificial Intelligence and Statistics (AISTATS)_ 2023
>
> <https://proceedings.mlr.press/v206/qoku23a.html>

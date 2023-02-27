# MuVI
A multi-view latent variable model with domain-informed structured sparsity, that integrates noisy domain expertise in terms of feature sets.

## Setup
We suggest using [conda](https://docs.conda.io/en/latest/miniconda.html) to manage your environments, and [poetry](https://python-poetry.org/) to install `muvi` as a python package. Follow these steps to get `muvi` up and running!

1. Clone repository:
```bash
git clone https://github.com/MLO-lab/MuVI.git
```
2. Create a python environment in conda:
```bash
conda create -n muvi python=3.8
```
3. Activate freshly created environment:
```bash
source activate muvi
```
4. Install `muvi` with poetry:
```bash
cd MuVI
poetry install
```
5. Check out [basic tutorial](examples/1_basic_tutorial.ipynb) to get familiar with MuVI!

import os

import numpy as np
import pandas as pd
import anndata as ad

from spexlvm import data


def test_shapes(data_generator):

    data_generator.generate()
    assert data_generator.x.shape == (data_generator.n_samples, data_generator.n_factors)
    assert data_generator.w.shape == (data_generator.n_factors, data_generator.n_features)
    assert data_generator.y.shape == (data_generator.n_samples, data_generator.n_features)


def test_w_mask(data_generator):

    data_generator.generate()

    w = data_generator.w
    w_mask = data_generator.w_mask

    assert w.shape == w_mask.shape
    assert ((np.abs(w) > 0.5) == w_mask).all()


def test_noisy_w_mask(data_generator):
    data_generator.generate()

    noise_fraction = 0.1
    w_mask = data_generator.w_mask.astype(bool)
    noisy_w_mask = data_generator.get_noisy_w_mask(noise_fraction=noise_fraction).astype(bool)

    p = w_mask.sum(1)
    tp = (noisy_w_mask & w_mask).sum(1)
    tpr = (tp / p).mean()

    assert tpr > 0.88
    assert tpr < 0.92


def test_missing_y(data_generator):
    data_generator.generate()

    missing_fraction = 0.2
    data_generator.generate_missingness_mask(missing_fraction=missing_fraction)

    y = data_generator.y
    missing_y = data_generator.missing_y
    presence_mask = data_generator.presence_mask

    assert y.shape == missing_y.shape
    assert y.shape == presence_mask.shape
    assert (~np.isnan(missing_y) == presence_mask).all()


def test_to_adata(data_generator):
    data_generator.generate()
    data_generator.generate_missingness_mask(missing_fraction=0.2)

    adata = data_generator.to_adata(missing=False)
    missing_adata = data_generator.to_adata(missing=True)

    np.testing.assert_almost_equal(adata.X, data_generator.y, decimal=5)
    np.testing.assert_almost_equal(missing_adata.X, data_generator.missing_y, decimal=5)

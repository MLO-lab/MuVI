import logging

from pathlib import Path

import h5py
import numpy as np

from muvi.tools.utils import variance_explained


logger = logging.getLogger(__name__)


def save_as_hdf5(
    model,
    path,
    save_metadata=True,
):
    if not model._trained:
        raise ValueError(
            "Cannot save an untrained model, call `fit` first to train a MuVI model."
        )

    path = Path(path)
    if path.exists():
        logger.warning(f"`{path}` already exists, overwriting.")

    default_group_name = "group_0"
    logger.info(
        f"Setting default group name to `{default_group_name}` for single group data."
    )
    logger.info("Computing variance explained.")
    r2_view, r2_factor, _ = variance_explained(
        model, subsample=0 if model.n_samples < 5000 else 1000
    )

    f = h5py.File(path, "w")

    # data
    f_data = f.create_group("data")

    for vn in model.view_names:
        f_data.create_group(vn).create_dataset(
            default_group_name, data=model.get_observations()[vn]
        )

    # samples_metadata
    if save_metadata:
        metadata = model._cache.factor_adata.obs
        if not metadata.empty:
            f_metadata = f.create_group("samples_metadata/group_0")
            f_metadata.create_dataset("sample", data=model.sample_names.tolist())
            for col in metadata.columns:
                f_metadata.create_dataset(col, data=metadata[col].to_numpy())

    # views
    f.create_group("views").create_dataset("views", data=model.view_names.tolist())

    # groups
    f.create_group("groups").create_dataset("groups", data=[default_group_name])

    # features
    f_features = f.create_group("features")

    for vn in model.view_names:
        f_features.create_dataset(vn, data=model.feature_names[vn].tolist())

    # samples
    f.create_group("samples").create_dataset(
        default_group_name, data=model.sample_names.tolist()
    )

    # sorted factors
    r2_order = r2_factor.sum(1).argsort().to_numpy()[::-1]
    f.create_group("factors").create_dataset(
        default_group_name, data=model.factor_names[r2_order].tolist()
    )

    # model_options

    model_options = {
        "ard_weights": [True],
        "spikeslab_weights": [True],
        "ard_factors": [False],
        "spikeslab_factors": [False],
        "likelihoods": [
            {"normal": "gaussian"}.get(model.likelihoods[vn], model.likelihoods[vn])
            for vn in model.view_names
        ],
    }

    f_model_options = f.create_group("model_options")
    for name, value in model_options.items():
        f_model_options.create_dataset(name, data=value)

    # training_opts
    training_opts = [
        # maxiter
        model._training_log["n_epochs"],
        # freqELBO
        1,
        # start_elbo
        1,
        # gpu_mode
        int(str(model.device) != "cpu"),
        # stochastic
        int(model._training_log["batch_size"] < model.n_samples),
        # seed
        model._training_log["seed"],
    ]

    f.create_dataset("training_opts", data=training_opts)

    # expectations
    f_w = f.create_group("expectations/W")
    f_z = f.create_group("expectations/Z")

    for vn in model.view_names:
        f_w.create_dataset(vn, data=model.get_factor_loadings()[vn][r2_order, :])
    f_z.create_dataset(
        default_group_name, data=model.get_factor_scores()[:, r2_order].T
    )

    # intercepts
    f_intercepts = f.create_group("intercepts")

    for vn in model.view_names:
        f_intercepts.create_group(vn).create_dataset(
            default_group_name, data=np.zeros(model.n_features[vn])
        )

    # training stats
    n_iter = model._training_log["n_iter"]
    training_stats = {
        "elbo": model._training_log["history"],
        "number_factors": model.n_factors * np.ones(n_iter),
        "time": np.zeros(n_iter),
    }

    f_training_stats = f.create_group("training_stats")
    for name, value in training_stats.items():
        f_training_stats.create_dataset(name, data=value)

    # variance_explained
    f.create_group("variance_explained/r2_total").create_dataset(
        default_group_name, data=[r2_view[vn] for vn in model.view_names]
    )
    f.create_group("variance_explained/r2_per_factor").create_dataset(
        default_group_name, data=r2_factor.iloc[r2_order, :].to_numpy().T
    )

    return f.close()

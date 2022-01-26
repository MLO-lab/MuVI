"""Data module."""
import math
import os
from typing import Collection, Iterable, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from gsea_api.molecular_signatures_db import (
    GeneSet,
    GeneSets,
    MolecularSignaturesDatabase,
)

from spexlvm import config

# logging stuff
logger = config.logger


class Pathways(GeneSets):
    """A collection of pathways/gene sets, wraps GeneSets."""

    def __init__(self, gene_sets: Collection[GeneSet], **kwargs):
        """Initialise Pathways.

        Parameters
        ----------
        gene_sets : Collection[GeneSet]
        """
        super().__init__(gene_sets=gene_sets, **kwargs)

    def _to_gmt(self, f: str):
        for gene_set in self.gene_sets:
            f.write(
                gene_set.name
                + "\t"
                + gene_set.description
                + "\t"
                + "\t".join(gene_set.genes)
                + "\n"
            )

    def info(self, verbose: int = 0):
        """Get an overview of this pathway collection.

        Parameters
        ----------
        verbose : int, optional
            Level of verbosity, by default 0

        Returns
        -------
        str

        Raises
        ------
        ValueError
            Raised on negative verbosity level
        """
        if verbose < 0:
            raise ValueError(
                "Invalid verbosity level of %s, please use 0, 1 or 2." % verbose
            )

        info = str(self) + "\n"

        if verbose == 1:
            info += "Following gene sets are stored:\n"
            info += "\n".join([gs.name for gs in self.gene_sets])
        elif verbose == 2:
            info += "Following gene sets (with genes) are stored:\n"
            # double list comprehension is not readable
            for gene_sets in self.gene_sets:
                info += (
                    gene_sets.name
                    + ": "
                    + ", ".join([gene for gene in gene_sets.genes])
                    + "\n"
                )

        return info

    def find(self, partial_gene_set_names: Iterable[str]):
        """Perform a simple search given a list of (partial) gene set names.

        Parameters
        ----------
        partial_gene_set_names : Iterable[str]
            Collection of gene set names

        Returns
        -------
        dict
            Search results as a dictionary of
            {partial_gene_set_names[0]: [GeneSet], ...}
        """
        search_results = {partial_gsn: [] for partial_gsn in partial_gene_set_names}
        for partial_gsn in partial_gene_set_names:
            search_results[partial_gsn] = [
                full_gs for full_gs in self.gene_sets if partial_gsn in full_gs.name
            ]

        return search_results

    def remove(self, gene_set_names: Iterable[str]):
        """Remove specific pathways.

        Parameters
        ----------
        gene_sets : Iterable[str]
            List of names (str) of unwanted pathways

        Returns
        -------
        Pathways
        """
        return Pathways(
            {
                GeneSet(name=gene_set.name, genes=gene_set.genes)
                for gene_set in self.gene_sets
                if gene_set.name not in gene_set_names
            }
        )

    def subset(
        self,
        genes: Iterable[str],
        fraction_available: float = 0.5,
        min_gene_count: int = 0,
        max_gene_count: int = 0,
        keep: Iterable[str] = None,
    ):
        """Extract a subset of pathways available in a collection of genes.

        Parameters
        ----------
        genes : Iterable[str]
            List of genes
        fraction_available : float, optional
            What fraction of the pathway genes should be available
            in the genes collection to insert the pathway into the subset,
            by default 0.5 (half of genes of a pathway must be present)
        min_gene_count : int, optional
            Minimal number of pathway genes available in the data
            for the pathway to be considered in the subset
        max_gene_count : int, optional
            Maximal number of pathway genes available in the data
            for the pathway to be considered in the subset
        keep : Iterable[str]
            List of pathways to keep regardless of filters

        Returns
        -------
        Pathways
        """
        if keep is None:
            keep = []

        if not isinstance(genes, set):
            genes = set(genes)

        pathways_subset = set()
        for gene_set in self.gene_sets:
            gene_intersection = gene_set.genes & genes  # intersection
            available_genes = len(gene_intersection)
            gene_fraction = available_genes / len(gene_set.genes)
            if gene_set.name in keep:
                logger.info(
                    "Keeping a %s out of %s genes (%.2f) from '%s'.",
                    available_genes,
                    len(gene_set.genes),
                    gene_fraction,
                    gene_set.name,
                )
            if gene_set.name in keep or (
                gene_fraction >= fraction_available
                and available_genes >= min_gene_count
            ):
                if max_gene_count == 0 or available_genes <= max_gene_count:
                    pathways_subset.add(
                        GeneSet(
                            name=gene_set.name,
                            genes=gene_intersection,
                            warn_if_empty=False,
                        )
                    )

        return Pathways(pathways_subset)

    def to_mask(self, genes: Iterable[str], sort: bool = False):
        """Generate a binary matrix of pathways x genes.

        Parameters
        ----------
        genes : Iterable[str]
            List of genes
        sort : bool, optional
            Whether to sort alphabetically, by default False

        Returns
        -------
        ndarray
        """
        gene_sets_list = list(self.gene_sets)
        if sort:
            gene_sets_list = sorted(gene_sets_list, key=lambda gs: gs.name)
        # probably faster than calling list.index() for every gene in the pathways
        gene_to_idx = {k: v for k, v in zip(genes, range(len(genes)))}

        mask = np.zeros((len(gene_sets_list), len(genes)))

        for i, gene_sets in enumerate(gene_sets_list):
            for gene in gene_sets.genes:
                if gene in gene_to_idx:
                    mask[i, gene_to_idx[gene]] = 1.0

        return mask, gene_sets_list


def load_pathways(keep: List[str] = None):
    """Load pathways from the existing msigdb.

    Parameters
    ----------
    keep : list, optional
        List of gene set collections, by default None

    Returns
    -------
    Pathways
    """
    if keep is None:
        keep = ["hallmark", "reactome"]
    # load msigdb files located at ./msigdb (.gmt extension)
    msigdb = MolecularSignaturesDatabase(os.path.join("..", "msigdb"), version=7.4)
    # relevant gene sets dictionary
    gene_sets = {
        "hallmark": "h.all",
        "kegg": "c2.cp.kegg",
        "reactome": "c2.cp.reactome",
    }
    # load relevant pathways
    pathway_dict = {
        k: msigdb.load(v, id_type="symbols") for k, v in gene_sets.items() if k in keep
    }

    # concatenate pathways
    pathways = Pathways(
        sum([pathway_dict[k].gene_sets for k in pathway_dict.keys()], ())
    )

    return pathways


def load_dataset(
    dataset: str,
    subsample_size: int = 0,
    n_top_genes: int = 0,
    center: bool = True,
    as_adata: bool = True,
):
    """Load dataset.

    Parameters
    ----------
    dataset : str
        Name of the dataset, e.g. mesc, retina_large
    subsample_size : int, optional
        Size of a random sample from the dataset, by default 0 (all)
    n_top_genes : int, optional
        Number of the most variable genes, by default 0 (all)
    center : bool, optional
        Whether to center the data, by default True
    as_adata : bool, optional
        Whether to return an AnnData or a dictionary, by default True

    Returns
    -------
    object
        ad.AnnData or dictionary of "Y", "label", "batch"
    """
    # lambda allows for lazy loading..
    dataset_dict = {
        "mesc": lambda: load_mesc,
        "retina_large": lambda: load_retina_large,
        "retina_small": lambda: load_retina_small,
        "retina_rod": lambda: load_retina_rod,
    }
    Y, labels, batch = dataset_dict.get(dataset)()()

    if n_top_genes > 0:
        Y_var = Y.var()
        top_var_col_indices = Y_var.argsort()[-n_top_genes:]
        logger.info("Using %s most variable genes", n_top_genes)
        Y = Y.iloc[:, top_var_col_indices]
    if center:
        Y_mean = Y.mean()
    if subsample_size > 0:
        logger.info("Using a random subsample of %s", subsample_size)
        subsample_indices = np.random.choice(Y.shape[0], subsample_size, replace=False)
        Y = Y.iloc[subsample_indices]
        labels = labels[subsample_indices]
        if batch is not None:
            batch = batch[subsample_indices]

    # center data column-wise, ignoring last columns (labels)
    if center:
        Y = Y - Y_mean

    # all genes have uppercase in pathways
    Y.columns = Y.columns.str.upper()

    result = {"Y": Y, "label": labels, "batch": batch}

    if as_adata:
        result = ad.AnnData(Y)
        if labels is not None:
            result.obs["label"] = labels
        if batch is not None:
            result.obs["batch"] = batch

    return result


def load_mesc():
    Y = pd.read_csv(
        os.path.join(config.DATASET_DIR, "Buettneretal.csv.gz"), compression="gzip"
    )

    return Y, [{1: "G1", 2: "S", 3: "G2/M"}[i] for i in Y.index], None


def load_retina_large():
    # https://data.humancellatlas.org/explore/projects/8185730f-4113-40d3-9cc3-929271784c2b/project-matrices
    # load data from storage
    dataset_dir = os.path.join(
        "/",
        "data",
        "aqoku",
        "projects",
        "spexlvm",
        "processed",
    )
    Y = pd.read_pickle(
        os.path.join(
            dataset_dir,
            "retina.pkl",
        )
    )
    labels = pd.read_csv(
        os.path.join(
            dataset_dir,
            "WongRetinaCelltype.csv",
        )
    )

    labels = labels["annotated_cell_identity.ontology_label"]
    batch = Y["batch"]
    return Y.drop("batch", axis=1), labels.values, batch.values


def load_retina_rod():
    Y, labels, batch = load_retina_large()
    # remove dominant cluster
    subsample_indices = labels == "retinal rod cell"
    Y = Y.iloc[subsample_indices, :]
    if batch is not None:
        batch = batch[subsample_indices]
    labels = labels[subsample_indices]

    return Y, labels, batch


def load_retina_small():
    Y, labels, batch = load_retina_large()
    # remove dominant cluster
    subsample_indices = labels != "retinal rod cell"
    Y = Y.iloc[subsample_indices, :]
    if batch is not None:
        batch = batch[subsample_indices]
    labels = labels[subsample_indices]

    return Y, labels, batch


class DataGenerator:
    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 200,
        n_factors: int = 10,
        likelihood: str = "normal",
        factor_size_params: Tuple[float] = None,
        factor_size_dist: str = "uniform",
        n_active_factors: float = 1.0,
    ) -> None:
        """Generate synthetic data.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples, by default 1000
        n_features : int, optional
            Number of features, by default 200
        n_factors : int, optional
            Number of latent factors, by default 10
        likelihood : str, optional
            Likelihood, either "normal" or "bernoulli", by default "normal"
        factor_size_params : Tuple[float], optional
            Parameters for the distribution of the number
            of active factor loadings for the latent factors,
            by default None
        factor_size_dist : str, optional
            Distribution of the number of active factor loadings,
            either "uniform" or "gamma",
            by default "uniform"
        n_active_factors : float, optional
            Number or fraction of active factors, by default 1.0 (all)
        """

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_factors = n_factors
        self.likelihood = likelihood
        if factor_size_params is None:
            if factor_size_dist == "uniform":
                logger.warning(
                    "Using a uniform distribution with parameters 0.05 and 0.15 "
                    "for generating the number of active factor loadings."
                )
                factor_size_params = (0.05, 0.15)
            elif factor_size_dist == "gamma":
                logger.warning(
                    "Using a uniform distribution with shape of 1 and scale of 50 "
                    "for generating the number of active factor loadings."
                )
                factor_size_params = (1.0, 50.0)

        self.factor_size_params = factor_size_params
        self.factor_size_dist = factor_size_dist

        if n_active_factors <= 1.0:
            # if fraction of active factors convert to int
            n_active_factors = int(n_active_factors * self.n_factors)

        self.n_active_factors = n_active_factors

        # set upon data generation
        self.x = None
        self.w = None
        self.y = None
        self.w_mask = None
        self.noisy_w_mask = None
        self.active_factor_indices = None

        # set when introducing missingness
        self.presence_mask = None

    @property
    def missing_y(self):
        if self.y is None:
            logger.warning("Generate data first by calling `generate`.")
            return self.y
        if self.presence_mask is None:
            logger.warning(
                "Introduce missing data first by calling `generate_missingness_mask`."
            )
            return self.y

        return self.y * self._mask_to_nan()

    def _mask_to_nan(self):
        nan_mask = np.array(self.presence_mask, dtype=np.float32, copy=True)
        nan_mask[nan_mask == 0] = np.nan
        return nan_mask

    def _mask_to_bool(self):
        return self.presence_mask == 1.0

    def normalise(self, with_std=False):
        if self.likelihood == "normal":
            y = np.array(self.y, copy=True)
            y -= y.mean(axis=0)
            if with_std:
                y_std = y.std(axis=0)
                y = np.divide(y, y_std, out=np.zeros_like(y), where=y_std != 0)
            self.y = y

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def generate(self, seed: int = None, overwrite: bool = False) -> None:

        rng = np.random.default_rng()

        if seed is not None:
            rng = np.random.default_rng(seed)

        if self.y is not None and not overwrite:
            logger.warning(
                "Data has already been generated, "
                "to generate new data please set `overwrite` to True."
            )
            return rng

        # generate factor scores which lie in the latent space
        x = rng.standard_normal((self.n_samples, self.n_factors))

        # generate factor loadings
        w_shape = (self.n_factors, self.n_features)
        w = rng.standard_normal(w_shape) * 4.0

        w_mask = np.zeros(w_shape)

        active_factor_indices = sorted(
            rng.choice(
                self.n_factors,
                size=math.ceil(self.n_active_factors),
                replace=False,
            )
        )

        fraction_active_features = {
            "gamma": lambda shape, scale: (rng.gamma(shape, scale, self.n_factors) + 20)
            / self.n_features,
            "uniform": lambda low, high: rng.uniform(low, high, self.n_factors),
        }[self.factor_size_dist](self.factor_size_params[0], self.factor_size_params[1])

        for factor_idx, faft in enumerate(fraction_active_features):
            if factor_idx in active_factor_indices:
                w_mask[factor_idx] = rng.choice(2, self.n_features, p=[1 - faft, faft])

        # set small values to zero
        w_mask[np.abs(w) < 0.5] = 0.0
        w = w_mask * w
        # add some noise to avoid exactly zero values
        w = np.where(np.abs(w) < 0.5, w + rng.standard_normal(w_shape) / 20, w)
        # assert ((np.abs(w) > 0.5)*1.0 == w_mask).all()

        # generate feature sigmas
        sigma = 1.0 / np.sqrt(rng.gamma(5.0, 0.5, self.n_features))
        loc = np.matmul(x, w)

        if self.likelihood == "normal":
            y = rng.normal(loc=loc, scale=sigma)
        else:
            y = rng.binomial(1, self.sigmoid(loc))

        self.x = x
        self.w = w
        self.w_mask = w_mask
        self.sigma = sigma
        self.y = y
        self.active_factor_indices = active_factor_indices

        return rng

    def get_noisy_w_mask(self, rng=None, noise_fraction=0.1):
        if rng is None:
            rng = np.random.default_rng()

        noisy_w_mask = np.array(self.w_mask, copy=True)

        for factor_idx in range(self.n_factors):

            active_cell_indices = noisy_w_mask[factor_idx, :].nonzero()[0]
            inactive_cell_indices = (noisy_w_mask[factor_idx, :] == 0).nonzero()[0]
            n_noisy_cells = int(noise_fraction * len(active_cell_indices))
            swapped_indices = zip(
                rng.choice(len(active_cell_indices), n_noisy_cells, replace=False),
                rng.choice(len(inactive_cell_indices), n_noisy_cells, replace=False),
            )

            for on_idx, off_idx in swapped_indices:
                noisy_w_mask[factor_idx, active_cell_indices[on_idx]] = 0.0
                noisy_w_mask[factor_idx, inactive_cell_indices[off_idx]] = 1.0

        self.noisy_w_mask = noisy_w_mask
        return self.noisy_w_mask

    def generate_missingness_mask(
        self,
        missing_fraction: float = 0.0,
        seed=None,
    ):
        rng = np.random.default_rng()
        if seed is not None:
            rng = np.random.default_rng(seed)
        # remove random fraction
        self.presence_mask = rng.choice(
            [0, 1], self.y.shape, p=[missing_fraction, 1 - missing_fraction]
        )
        return rng

    def permute_features(self, new_feature_order):
        if len(new_feature_order) != self.n_features:
            raise ValueError(
                "Length of new order list must equal the number of features."
            )

        new_feature_order = np.array(new_feature_order)

        self.w = np.array(self.w[:, new_feature_order], copy=True)
        self.w_mask = np.array(self.w_mask[:, new_feature_order], copy=True)
        if self.noisy_w_mask is not None:
            self.noisy_w_mask = np.array(
                self.noisy_w_mask[:, new_feature_order], copy=True
            )

        self.sigma = np.array(self.sigma[new_feature_order], copy=True)
        self.y = np.array(self.y[:, new_feature_order], copy=True)
        if self.presence_mask is not None:
            self.missing_y = np.array(self.missing_y[:, new_feature_order], copy=True)
            self.presence_mask = np.array(
                self.presence_mask[:, new_feature_order], copy=True
            )

    def permute_factors(self, new_factor_order):
        if len(new_factor_order) != self.n_factors:
            raise ValueError(
                "Length of new order list must equal the number of factors."
            )

        self.x = np.array(self.x[:, new_factor_order], copy=True)
        self.w = np.array(self.w[new_factor_order, :], copy=True)
        self.w_mask = np.array(self.w_mask[new_factor_order, :], copy=True)
        if self.noisy_w_mask is not None:
            self.noisy_w_mask = np.array(
                self.noisy_w_mask[new_factor_order, :], copy=True
            )

        # TODO: what about active factor indices?

    def to_adata(self, missing=False, **kwargs):
        y = self.y
        if missing:
            y = self.missing_y

        return ad.AnnData(
            pd.DataFrame(
                y,
                columns=[f"feature_{j}" for j in range(self.n_features)],
                index=[f"sample_{i}" for i in range(self.n_samples)],
            ),
            dtype="float32",
        )


if __name__ == "__main__":
    n_samples = 100
    n_clusters = 1
    cluster_type = "blobs"
    n_features = 200
    n_factors = 10
    n_active_features = (0.05, 0.15)
    n_active_factors = 1.0

    dg = DataGenerator(
        n_samples,
        n_clusters,
        cluster_type,
        n_features,
        n_factors,
        likelihood="normal",
        n_active_features=n_active_features,
        factor_size_dist="uniform",
        n_active_factors=n_active_factors,
    )

    dg.generate(overwrite=True)

    assert dg.x.shape[0] == n_samples
    assert dg.x.shape[1] == dg.n_factors

    assert dg.w.shape[0] == dg.n_factors
    assert dg.w.shape[1] == n_features

    assert dg.ys.shape[0] == n_samples
    assert dg.ys.shape[1] == n_features

    dg.generate_missingness_mask(
        fraction_missing=0.1,
        n_partial_samples=10,
        n_partial_features=10,
        n_features=10,
    )
    # test r2_score
    from sklearn.metrics import r2_score

    x = dg.x
    w = dg.w

    y_true = dg.y
    y_pred_tot = x @ w
    r2 = r2_score(y_true, y_pred_tot)
    factor_r2_scores = []
    for k in range(n_factors):
        y_pred = np.outer(x[:, k], w[k, :])

        factor_r2_scores.append(r2_score(y_true, y_pred))

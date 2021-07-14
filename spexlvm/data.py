"""Data module."""
import math
import os
from typing import Collection, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
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
            raise ValueError("Invalid verbosity level of %s, please use 0, 1 or 2." % verbose)

        info = str(self) + "\n"

        if verbose == 1:
            info += "Following gene sets are stored:\n"
            info += "\n".join([gs.name for gs in self.gene_sets])
        elif verbose == 2:
            info += "Following gene sets (with genes) are stored:\n"
            # double list comprehension is not readable
            for gene_sets in self.gene_sets:
                info += gene_sets.name + ": " + ", ".join([gene for gene in gene_sets.genes]) + "\n"

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
            Search results as a dictionary of {partial_gene_set_names[0]: [GeneSet], ...}
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
                    "Keeping a %s out of %s genes (%.2f) from the special gene set '%s'.",
                    available_genes,
                    len(gene_set.genes),
                    gene_fraction,
                    gene_set.name,
                )
            if gene_set.name in keep or (
                gene_fraction >= fraction_available and available_genes >= min_gene_count
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
        torch.Tensor
        """
        gene_sets_list = list(self.gene_sets)
        if sort:
            gene_sets_list = sorted(gene_sets_list, key=lambda gs: gs.name)
        # probably faster than calling list.index() for every gene in the pathways
        gene_to_idx = {k: v for k, v in zip(genes, range(len(genes)))}

        mask = torch.zeros(len(gene_sets_list), len(genes))

        for i, gene_sets in enumerate(gene_sets_list):
            for gene in gene_sets.genes:
                mask[i, gene_to_idx[gene]] = 1.0

        return mask, gene_sets_list


def load_pathways(keep=None):
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
    print(msigdb)
    # relevant gene sets dictionary
    gene_sets = {
        "hallmark": "h.all",
        "kegg": "c2.cp.kegg",
        "reactome": "c2.cp.reactome",
    }
    # gene_sets = {"hallmark": "h.all"}
    # load relevant pathways
    pathway_dict = {k: msigdb.load(v, "symbols") for k, v in gene_sets.items() if k in keep}

    # concatenate pathways
    pathways = Pathways(sum([pathway_dict[k].gene_sets for k in pathway_dict.keys()], ()))

    return pathways


def load_dataset(dataset, subsample_size=0, n_top_genes=0, center=True):
    # lambda allows for lazy loading..
    dataset_dict = {
        "mesc": lambda: load_mesc,
        "retina_small": lambda: load_retina_small,
        "retina_rod": lambda: load_retina_rod,
        "retina_large": lambda: load_retina_large,
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
    # Y = Y.rename(str.upper, axis='columns')
    return Y, labels, batch


def load_mesc():
    Y = pd.read_csv(os.path.join(config.DATASET_DIR, "Buettneretal.csv.gz"), compression="gzip")

    return Y, Y.index, None


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


def generate_toy_dataset(
    n_samples: int = 10000,
    n_features: int = 200,
    n_factors: int = 40,
    n_active_features: float = 0.1,
    n_active_factors: float = 0.5,
    constant_weight: float = 4.0,
):
    """Generate toy dataset for simulated evaluation.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples, by default 10000
    n_features : int, optional
        Number of features (genes), by default 200
    n_factors : int, optional
        Number of factors, by default 40
    n_active_features : float, optional
        Number or fraction of active genes per factor, by default 0.1
    n_active_factors : float, optional
        Number of fraction of active factors, by default 0.5
    constant_weight : float, optional
        A constant weight to fill in the non-zero elements, by default 4.0

    Returns
    -------
    tuple
        w, mask, active factor indices, x, y
    """
    if isinstance(n_active_features, float):
        n_active_features = (n_active_features, n_active_features)

    # convert active features and factors into fractions if > 1.0
    n_active_features = tuple(
        naft / n_features if naft > 1.0 else naft for naft in n_active_features
    )
    min_n_active_features, max_n_active_features = n_active_features
    if n_active_factors > 1.0:
        n_active_factors /= n_factors

    w_shape = [n_factors, n_features]
    x_shape = [n_samples, n_factors]
    true_mask = torch.zeros(w_shape)
    constant_w = constant_weight * torch.ones(w_shape)
    for factor_idx, naft in enumerate(
        np.random.uniform(min_n_active_features, max_n_active_features, n_factors)
    ):
        true_mask[factor_idx] = torch.multinomial(
            torch.tensor([1 - naft, naft]),
            w_shape[1],
            replacement=True,
        )
    # generate small random values around 0
    random_noise = torch.normal(
        mean=torch.zeros(w_shape), std=constant_weight / 50 * torch.ones(w_shape)
    )
    true_w = true_mask * constant_w + random_noise
    true_x = torch.normal(mean=torch.zeros(x_shape), std=torch.ones(x_shape))

    active_factor_indices = sorted(
        np.random.choice(
            range(n_factors),
            size=math.ceil(n_factors * n_active_factors),
            replace=False,
        )
    )
    for row_idx in range(n_factors):
        if row_idx not in active_factor_indices:
            true_w[row_idx, :] = torch.normal(
                torch.zeros(n_features),
                std=constant_weight / 50 * torch.ones(n_features),
            )
    return (
        true_w,
        true_mask,
        active_factor_indices,
        true_x,
        torch.matmul(true_x, true_w),
    )

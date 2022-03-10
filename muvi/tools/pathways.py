import logging
from typing import Collection, Iterable, List, Union

import numpy as np
import pandas as pd
from gsea_api.molecular_signatures_db import (
    GeneSet,
    GeneSets,
    MolecularSignaturesDatabase,
)
from tqdm import tqdm

from muvi.tools import config

logger = logging.getLogger(__name__)


class Pathways(GeneSets):
    """A collection of pathways/gene sets, wraps GeneSets."""

    def __init__(
        self,
        gene_sets: Collection[GeneSet],
        name="",
        allow_redundant=False,
        remove_empty=True,
        path=None,
    ):
        super().__init__(gene_sets, name, allow_redundant, remove_empty, path)

    def _to_gmt(self, f):
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

    def merge_duplicates(self):
        unique_gene_sets = {}
        for current_gs in self.gene_sets:
            cgsn = current_gs.name
            if cgsn in unique_gene_sets:
                existing_gs = unique_gene_sets[cgsn]
                new_gs = GeneSet(
                    name=cgsn,
                    genes=frozenset().union(*[existing_gs.genes, current_gs.genes]),
                )
                unique_gene_sets[cgsn] = new_gs
            else:
                unique_gene_sets[cgsn] = current_gs

        return Pathways(unique_gene_sets.values())

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
                GeneSet(
                    name=gene_set.name,
                    description=gene_set.description,
                    genes=gene_set.genes,
                )
                for gene_set in self.gene_sets
                if gene_set.name not in gene_set_names
            }
        )

    def keep(self, gene_set_names: Iterable[str]):
        """Keep specific pathways.

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
                GeneSet(
                    name=gene_set.name,
                    description=gene_set.description,
                    genes=gene_set.genes,
                )
                for gene_set in self.gene_sets
                if gene_set.name in gene_set_names
            }
        )

    def subset(
        self,
        genes: Iterable[str],
        fraction_available: float = 0.5,
        min_gene_count: int = 0,
        max_gene_count: int = -1,
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
            if available_genes == 0:
                continue
            if gene_set.name in keep:
                logger.info(
                    "Keeping a %s out of %s genes (%.2f) from the special gene set '%s'.",
                    available_genes,
                    len(gene_set.genes),
                    gene_fraction,
                    gene_set.name,
                )
            if gene_set.name in keep or (
                gene_fraction >= fraction_available
                and available_genes >= min_gene_count
            ):
                if max_gene_count < 0 or available_genes <= max_gene_count:
                    pathways_subset.add(
                        GeneSet(
                            name=gene_set.name,
                            description=gene_set.description,
                            genes=gene_intersection,
                            warn_if_empty=False,
                        )
                    )

        return Pathways(pathways_subset)

    def to_mask(self, genes: Iterable[str], sort: bool = True):
        """Generate a binary matrix of pathways x genes.

        Parameters
        ----------
        genes : Iterable[str]
            List of genes
        sort : bool, optional
            Whether to sort alphabetically, by default True

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

    # assume format: COLLECTION_SOME_LONG_GENE_SET_NAME
    # return format: Some Long...Name (C)
    def prettify(self, str_len_threshold=40):
        new_gene_set_names = set()
        for gene_set in self.gene_sets:
            gsn_parts = [part.capitalize() for part in gene_set.name.split("_")]
            gsn_parts[0] = "(" + gsn_parts[0][0] + ")"
            # put (C) after the rest of the name
            gsn_parts = gsn_parts[1:] + gsn_parts[:1]

            new_gene_set_name = " ".join(gsn_parts)
            if len(new_gene_set_name) > str_len_threshold:
                half_str_len_threshold = (str_len_threshold - 4) // 2
                new_gene_set_name = (
                    new_gene_set_name[:half_str_len_threshold]
                    + "..."
                    + new_gene_set_name[-half_str_len_threshold - 3 :]
                )

            if new_gene_set_name in new_gene_set_names:
                logger.warning(
                    "%s already existing in the new names, adding a suffix.",
                    new_gene_set_name,
                )
                for i in range(10):
                    new_gene_set_name = (
                        new_gene_set_name[:-3] + f"{i} " + new_gene_set_name[-3:]
                    )
                    if new_gene_set_name not in new_gene_set_names:
                        break
            new_gene_set_names.add(new_gene_set_name)
            gene_set.name = new_gene_set_name


def load_pathways(
    collections: Union[str, List[str]],
    genes: List[str],
    fraction_available: Union[float, List[float]] = 0.2,
    min_gene_count: Union[int, List[int]] = 15,
    max_gene_count: Union[int, List[int]] = -1,
    keep: List[str] = None,
    remove: List[str] = None,
):
    """Load pathways from the existing msigdb."""

    if isinstance(collections, str):
        collections = [collections]
    n_collections = len(collections)

    if isinstance(fraction_available, float):
        fraction_available = [fraction_available for _ in range(n_collections)]

    if isinstance(min_gene_count, int):
        min_gene_count = [min_gene_count for _ in range(n_collections)]

    if isinstance(max_gene_count, int):
        max_gene_count = [max_gene_count for _ in range(n_collections)]

    if keep is None:
        keep = []

    if remove is None:
        remove = []

    # load msigdb files located at ./msigdb (.gmt extension)
    msigdb = MolecularSignaturesDatabase(config.MSIGDB_DIR, version="7.5.1")

    all_gene_sets = tuple()
    size_dfs = []

    for i, c in enumerate(collections):
        logger.info(
            "Loading collection %s with at least %2.1f%% "
            "of genes available and at least %s genes",
            c,
            fraction_available[i] * 100,
            min_gene_count[i],
        )

        gene_sets = msigdb.load(c, id_type="symbols").gene_sets

        if len(genes) > 0:
            gene_sets = (
                Pathways(gene_sets)
                .subset(
                    genes,
                    fraction_available[i],
                    min_gene_count[i],
                    max_gene_count[i],
                    keep,
                )
                .gene_sets
            )

        size_dfs.append(
            pd.DataFrame(
                {
                    "Name": [gs.name for gs in gene_sets],
                    "Size": [len(gs.genes) for gs in gene_sets],
                    "Collection": c,
                }
            )
        )
        all_gene_sets += gene_sets

        logger.info(
            "Loaded %s pathways from collection %s with median size of %s genes",
            len(gene_sets),
            c,
            np.median([len(gs.genes) for gs in gene_sets]),
        )

    pathways = Pathways(gene_sets=all_gene_sets)

    for k, name in pathways.find_redundant().items():
        remove += name[1:]

    if len(remove) > 0:
        logger.info("Removing following redundant pathways:\n%s", ", ".join(remove))

    pathways = pathways.remove(remove)
    logger.info(
        "Loaded in total %s pathways with median size of %s genes",
        len(pathways),
        np.median([len(gs.genes) for gs in pathways.gene_sets]),
    )
    return pathways, size_dfs

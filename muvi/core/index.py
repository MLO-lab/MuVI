import numpy as np


def _normalize_index(indexer, index, as_idx=True):
    # work with ints, convert at the end
    # if single str, get idx and put to list
    # TODO: can be an issue if any of the indices is named 'all'..
    if isinstance(indexer, str):
        if indexer == "all":
            indexer = range(len(index))
        else:
            indexer = [index.get_loc(indexer)]
    # if single integer, put to list
    if isinstance(indexer, (np.integer, int)):
        indexer = [indexer]
    # work with np array
    indexer = np.array(indexer)
    # if mask, get indices where True
    if issubclass(indexer.dtype.type, np.bool_):
        indexer = np.where(indexer)[0]
    # if str, get indices where names match
    if issubclass(indexer.dtype.type, (np.str_)):
        indexer = index.get_indexer(indexer)
    if as_idx:
        if issubclass(indexer.dtype.type, (np.integer)):
            return indexer
        else:
            raise IndexError(f"Invalid index `{indexer}`")
    return index[indexer]

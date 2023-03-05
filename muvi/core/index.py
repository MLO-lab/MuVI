import numpy as np


def _normalize_index(indexer, index, as_idx=True):
    # work with ints, convert at the end
    # if single str, get idx and put to list
    # TODO: can be an issue if any of the indices is named 'all'..
    if indexer is None:
        raise IndexError("None index.")
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
    # if empty
    if len(indexer) == 0:
        raise IndexError(f"Empty index, `{indexer}`.")
    # if mask, get indices where True
    if isinstance(indexer[0], (bool, np.bool_)):
        indexer = np.where(indexer)[0]
    # if all False from previous boolean mask
    if len(indexer) == 0:
        raise IndexError(f"Empty index, `{indexer}`.")
    # note empty, get first element
    # reason: dtype of str was not working for pd.Index
    # if str, get indices where names match
    if isinstance(indexer[0], (str, np.str_)):
        indexer = index.get_indexer(indexer)
    if isinstance(indexer[0], (int, np.integer)):
        if as_idx:
            return indexer
        return index[indexer]
    raise IndexError(f"Invalid index, `{indexer}`.")

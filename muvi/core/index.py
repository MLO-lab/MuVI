import logging

import numpy as np


logger = logging.getLogger(__name__)


def _normalize_index(indexer, index, as_idx=True):
    # work with ints, convert at the end
    # if single str, get idx and put to list
    # TODO: can be an issue if any of the indices is named 'all'..
    if indexer is None:
        raise IndexError("None index.")
    if isinstance(indexer, str):
        indexer = range(len(index)) if indexer == "all" else [index.get_loc(indexer)]
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
    # if str, get indices where names match
    if isinstance(indexer[0], (str, np.str_)):
        _indexer = index.get_indexer(indexer)
        bad_idx = [idx for _idx, idx in zip(_indexer, indexer) if _idx == -1]
        if len(bad_idx) > 0:
            logger.warning(f"Invalid index, `{bad_idx}`, removing...")
        indexer = [_idx for _idx in _indexer if _idx != -1]
    # if all bad indices
    if len(indexer) == 0:
        raise IndexError(f"Empty index, `{indexer}`.")
    if isinstance(indexer[0], (int, np.integer)):
        if as_idx:
            return indexer
        return index[indexer]
    raise IndexError(f"Invalid index, `{indexer}`.")

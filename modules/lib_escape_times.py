import numpy as np
import networkx as nx

def get_ErdosRenyi_connected_adjmatrix(N,P):
    while True:
        G = nx.fast_gnp_random_graph(N, P)
        if nx.is_connected(G):
            break
        #else:
        #    print("Graph disconnected. Regenerating...")
    A = nx.to_numpy_array(G, dtype=np.float64)
    return A,G


def _exists(X):
    return not(type(X) is type(None))

def find_first(cond, axis=None, no_axis_return_as_array=True, return_complete_index=True, not_found=None):
    """
    Find the first True element(s) in a boolean array.

    This function locates the first occurrence(s) of True values in a boolean
    array `cond`, either globally (flattened) or independently along a given
    axis.

    Parameters
    ----------
    cond : array_like of bool
        Boolean array indicating the condition to be tested.
    axis : int or None, optional
        Axis along which to search for the first True value.
        - If None (default), the array is treated as flattened and the first
          True element in row-major (C) order is returned.
        - If an integer, the search is performed independently for each index
          along that axis.
    no_axis_return_as_array : bool, optional
        Only relevant when `axis` is None.
        - If True (default), return the result as a one-element list.
        - If False, return the index as a scalar integer.
    return_complete_index : bool, optional
        Only relevant when `axis` is not None.
        - If True (default), return the full index tuple into `cond`.
        - If False, return only the index/indices orthogonal to `axis`
          (i.e., the remaining dimensions).
    not_found : optional
        value to return when cond was not met

    Returns
    -------
    res : list or scalar
        If `axis` is None:
            - A one-element list containing the flattened index of the first
              True value in `cond`, or an integer if
              `no_axis_return_as_array=False`.
            - If no True values are found, returns an empty list (or an empty
              array-equivalent).
        If `axis` is not None:
            - A list of length `cond.shape[axis]`.
            - Each element is either:
                * a tuple containing the index of the first True value along
                  that axis, or
                * an empty array if no True value exists for that axis index.

    Notes
    -----
    - The search order is row-major (C order), as determined by
      `np.argwhere`.
    - For `axis` searches, the *first* occurrence along the remaining
      dimensions is returned.
    - Empty results are represented by zero-length NumPy arrays.
    - Mixing tuple indices and empty-array placeholders is intentional and
      allows direct use of valid results for NumPy indexing.

    Examples
    --------
    >>> a = np.array([[0.1, 0.9, 0.2],
    ...               [0.95, 0.3, 0.4]])

    First True in flattened array:
    >>> find_first(a > 0.9)
    [3]

    First True per column:
    >>> find_first(a > 0.9, axis=1)
    [(1, 0), array([], dtype=int64), array([], dtype=int64)]

    Return only orthogonal indices:
    >>> find_first(a > 0.9, axis=1, return_complete_index=False)
    [1, array([], dtype=int64), array([], dtype=int64)]
    """
    N         = cond.shape[axis] if _exists(axis)      else 1
    not_found = not_found        if _exists(not_found) else np.zeros((0,),dtype=np.int64)
    res       = [ not_found for _ in range(N) ]
    found     = [ False     for _ in range(N) ]
    if not (_exists(axis) or no_axis_return_as_array):
        # axis is NOT given
        # AND
        # do NOT return as array
        res=res[0]
    ind = np.argwhere(cond)
    if ind.size > 0:
        if _exists(axis):
            for k in ind:
                j = k[axis]
                if not found[j]: #(type(res[j]) is np.ndarray) and (res[j].size == 0):
                    found[j] = True
                    if return_complete_index:
                        res[j] = tuple(k)
                    else:
                        restmp = [ i for d,i in enumerate(k) if d!=axis ]
                        if len(restmp)>1:
                            res[j] = tuple(restmp)
                        else:
                            res[j] = restmp[0]
        else: # flattened array
            res = [np.ravel_multi_index(ind[0],dims=cond.shape)] # returns first index
            if not no_axis_return_as_array:
                res = res[0]
    return res

def calc_escape_time(Z,Z_amp_threshold=1.0,axis=1,dt=1.0):
    return np.array([ (k*dt if k.size>0 else np.nan) for k in find_first(np.abs(Z)>Z_amp_threshold,axis=axis,return_complete_index=False)],dtype=np.float64)

def calc_min_escape_time(Z,Z_amp_threshold=1.0,axis=1,dt=1.0):
    return dt*np.nanmin(find_first(np.abs(Z)>Z_amp_threshold,axis=axis,return_complete_index=False,not_found=np.nan))

def find_ROI_min_escape_time(Z,Z_amp_threshold=1.0,axis=1):
    return np.nanargmin(find_first(np.abs(Z)>Z_amp_threshold,axis=axis,return_complete_index=False,not_found=np.nan))

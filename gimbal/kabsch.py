r"""
Module for aligning similar molecular geometries using the Kabsch algorithm.

For a reference and test set of three dimensional vectors, the Kabsch
algorithm determines the optimal rotation matrix to minimize the RMSD
between each vector pair.

After translating the vector sets of centroids, the covariance matrix **A**
is calculated by :math:`\mathbf{A} = \mathbf{P}^T \mathbf{Q}`. Then, using
singular value decomposition, :math:`\mathbf{V S W}^T = \mathbf{A}`. The
scaling component **S** is discarded. The handedness of the coordinate
system is determined by :math:`d = \mathrm{sign}(\det(\mathbf{W V}^T))`. The
rotation matrix is then found by

.. math::

    \mathbf{U} = \mathbf{W} \begin{bmatrix} 1 & 0 & 0 \\
                                            0 & 1 & 0 \\
                                            0 & 0 & d \end{bmatrix} \mathbf{V}^T

(See en.wikipedia.org/wiki/Kabsch_algorithm)

The best match for multiple references can be found by the minimum RMSD.
Sets of equivalent vectors (atoms) can be permuted as well.
"""
import itertools
import numpy as np
from scipy import linalg
import gimbal.displace as disp


def rmsd(test, ref, wgt=None):
    """Returns the root mean squared deviation of a test geometry with
    respect to a reference.

    Weights can be provided (e.g. atomic masses) with the optional
    variable wgt.

    Parameters
    ----------
    test : (N, 3) array_like
        The cartesian test geometry.
    ref : (N, 3) array_like
        The cartesian reference geometry.
    wgt : (N,) array_like, optional
        The atomic weights for computing the RMSD. If None (default),
        all weights are unity.

    Returns
    -------
    float
        The root mean squared deviation between test and ref.
    """
    if wgt is None:
        return np.sqrt(np.sum((test - ref) ** 2) / (3 * np.size(test)))
    else:
        return np.sqrt(np.sum(wgt[:,np.newaxis] * (test - ref) ** 2) /
                       (3 * np.sum(wgt) * np.size(test)))


def kabsch(test, ref, wgt=None, refl=True):
    """Returns the Kabsch rotational matrix to map a test geometry onto
    a reference.

    If weights are provided, they are used to weight the test vectors
    before forming the covariance matrix. This minimizes the weighted
    RMSD between the two geometries. If refl=True, improper rotations
    are also permitted.

    Parameters
    ----------
    test : (N, 3) array_like
        The cartesian test geometry.
    ref : (N, 3) array_like
        The cartesian reference geometry.
    wgt : (N,) array_like, optional
        The atomic weights for computing the RMSD. If None (default),
        all weights are unity.
    refl : bool, optional
        Specifies if reflections (improper rotations) are permitted.
        Default is True.

    Returns
    -------
    (3, 3) ndarray
        The rotational matrix that optimally maps test to ref.
    """
    if wgt is None:
        cov = test.T.dot(ref)
    else:
        wgt = np.array(wgt)
        cov = (wgt[:,np.newaxis] * test).T.dot(ref)
    rot1, scale, rot2 = linalg.svd(cov)
    if not refl:
        rot1[:,-1] *= linalg.det(rot1) * linalg.det(rot2)
    return rot1.dot(rot2)


def map_onto(elem, test, ref, wgt=None, ind=None, cent=None):
    """Returns the optimal mapping of a test geometry onto a reference
    using the Kabsch algorithm.

    The centre of mass of both test and ref are subtracted by default.
    If an index or list of indices is provided for cent, only the centre
    of mass of the provided atoms is subtracted. Alternatively, ind can
    be provided to only map a subset of atoms.

    Parameters
    ----------
    elem : (N,) array_like
        The atomic symbols.
    test : (N, 3) array_like
        The cartesian test geometry.
    ref : (N, 3) array_like
        The cartesian reference geometry.
    wgt : (N,) array_like, optional
        The atomic weights for computing the RMSD. If None (default),
        all weights are unity.
    ind : int or array_like, optional
        Indices of test to be mapped onto ref. If None (default), all
        atoms are used.
    cent : int or array_like, optional
        Indices used to specify the centre of mass as the origin of
        rotation. If None (default), the centre of mass is calculated
        for all atoms.

    Returns
    -------
    (N, 3) ndarray
        The test geometry optimally mapped onto the ref geometry.
    """
    if cent is None:
        cm_test = disp.get_centremass(elem, test)
        cm_ref = disp.get_centremass(elem, ref)
    else:
        cm_test = disp.get_centremass(elem[cent], test[cent])
        cm_ref = disp.get_centremass(elem[cent], ref[cent])

    new_test = test - cm_test
    new_ref = ref - cm_ref
    if ind is None:
        return new_test.dot(kabsch(new_test, new_ref, wgt=wgt)) + cm_ref
    else:
        return new_test.dot(kabsch(new_test[ind], new_ref[ind],
                                   wgt=wgt)) + cm_ref


def opt_permute(elem, test, ref, plist=None, equiv=None, wgt=None, ind=None,
                cent=None):
    """Determines optimal permutation of test geometry indices for
    mapping onto reference.

    Parameters
    ----------
    elem : (N,) array_like
        The atomic symbols.
    test : (N, 3) array_like
        The cartesian test geometry.
    ref : (N, 3) array_like
        The cartesian reference geometry.
    plist : list, optional
        A list of sets of atomic positions that can be permuted,
        parsed by :func:`_permute_elmnt`.
    equiv : array_like, optional
        A list of sets of indices that are symmetrically equivalent
        and interchangeable, parsed by :func:`_permute_group`.
    wgt : (N,) array_like, optional
        The atomic weights for computing the RMSD. If None (default),
        all weights are unity.
    ind : int or array_like, optional
        Indices of test to be mapped onto ref. If None (default), all
        atoms are used.
    cent : int or array_like, optional
        Indices used to specify the centre of mass as the origin of
        rotation. If None (default), the centre of mass is calculated

    Returns
    -------
    geom : (N, 3) ndarray
        The test geometry optimally mapped onto the ref geometry
        including atomic permutations.
    err : float
        The root mean squared deviation between test and ref.
    """
    if wgt is not None:
        wgt = np.array(wgt)
    kwargs = dict(wgt=wgt, ind=ind, cent=cent)
    if plist is None and equiv is None:
        geom = map_onto(elem, test, ref, **kwargs)
        if ind is None:
            return geom, rmsd(test, ref, wgt=wgt)
        else:
            return geom, rmsd(test[ind], ref[ind], wgt=wgt)
    else:
        eqs = _permute_group(equiv)
        prs = _permute_elmnt(plist)

        min_geom = test
        min_err = 1e10
        for i in eqs:
            for j in prs:
                xyz = np.copy(test)
                xyz[eqs[0]] = xyz[i]
                xyz[prs[0]] = xyz[j]
                geom = map_onto(elem, xyz, ref, **kwargs)
                err = rmsd(geom, ref, wgt=wgt)
                if err < min_err:
                    min_geom = geom
                    min_err = err

        return min_geom, min_err


def opt_ref(elem, test, reflist, **kwargs):
    """Determines optimal reference geometry for a given test geometry.

    Parameters
    ----------
    elem : (N,) array_like
        The atomic symbols.
    test : (N, 3) array_like
        The cartesian test geometry.
    reflist : (M, N, 3) array_like
        The cartesian reference geometries.
    kwargs : dict, optional
        Additional keyword arguments used in :func:`opt_permute`.

    Returns
    -------
    geom : (N, 3) ndarray
        The test geometry optimally mapped onto the ref geometry with
        the minimum RMSD.
    ind : int
        The index of the optimal ref geometry.
    """
    nrefs = len(reflist)
    geoms = np.empty((nrefs,) + test.shape)
    err = np.empty(nrefs)
    for i in range(nrefs):
        geoms[i], err[i] = opt_permute(elem, test, reflist[i], **kwargs)

    optref = np.argmin(err)
    return geoms[optref], optref


def opt_multi(elem, testlist, reflist, **kwargs):
    """Determines the optimal geometries of a set of test geometries
    against a set of reference geometries.

    Parameters
    ----------
    elem : (N,) array_like
        The atomic symbols.
    testlist : (L, N, 3) array_like
        The cartesian test geometries.
    reflist : (M, N, 3) array_like
        The cartesian reference geometries.
    kwargs : dict, optional
        Additional keyword arguments used in :func:`opt_permute`.

    Returns
    -------
    geomlist : (M,...) list
        The test geometries optimally mapped onto each ref geometry with
        the minimum RMSD. The L test geometries are sorted into to the
        M reference geometries.
    """
    geomlist = [[] for i in range(len(reflist))]
    for test in testlist:
        geom, ind = opt_ref(elem, test, reflist, **kwargs)
        geomlist[ind].append(geom)

    return geomlist


def _tuple2list(tupl):
    """Iteratively converts nested tuple to nested list.

    Parameters
    ----------
    tupl : tuple
        The tuple to be converted.

    Returns
    -------
    list
        The converted list of lists.
    """
    return list((_tuple2list(x) if isinstance(x, tuple) else x for x in tupl))


def _permute_elmnt(plist):
    """Generates an array of permutations of a list of lists of permutable
    indices and a list of permutable groups of indices.

    Parameters
    ----------
    plist : list
        A list of lists of indices that can be permuted.

    Returns
    -------
    list
        A list of each possible permutation of the indices.

    Examples
    --------
    >>> print(_permute_elmnt([0, 2]))
    [[0, 2], [2, 0]]
    >>> print(_permute_elmnt([[0, 2], [1, 4]]))
    [[0, 2, 1, 4], [0, 2, 4, 1], [2, 0, 1, 4], [2, 0, 4, 1]]
    """
    if plist is None:
        return [0]
    elif isinstance(plist[0], int):
        plist = [plist]

    multi_perms = [itertools.permutations(i) for i in plist]
    multi_perms = _tuple2list(itertools.product(*multi_perms))
    return [[item for sublist in i for item in sublist] for i in multi_perms]


def _permute_group(plist):
    """Generates an array of permutations of groups of indices.

    Parameters
    ----------
    plist : list
        A list of lists of indices that are symmetry equivalent.

    Returns
    -------
    list
        A list of each possible permutation of the indices.

    Examples
    --------
    >>> print(_permute_group([0, 2]))
    [[0, 2]]
    >>> print(_permute_group([[0, 2], [1, 4]]))
    [[0, 2, 1, 4], [1, 4, 0, 2]]
    """
    if plist is None:
        return [0]
    elif isinstance(plist[0], int):
        plist = [plist]

    group_perms = _tuple2list(itertools.permutations(plist))
    return [[item for sublist in i for item in sublist] for i in group_perms]

"""
Module for aligning similar molecular geometries using the Kabsch algorithm.

For a reference and test set of three dimensional vectors, the Kabsch
algorithm determines the optimal rotation matrix to minimize the RMSD
between each vector pair.

After translating the vector sets of centroids, the covariance matrix A
is calculated by A = P^T Q. Then, using singular value decomposition,
V S W^T = A. The scaling component S is discarded. The handedness of the
coordinate system is determined by d = sgn(det(W V^T)). The rotation
matrix is then found by
       _       _
      |  1 0 0  |
U = W |  0 1 0  | V^T
      |_ 0 0 d _|

(See en.wikipedia.org/wiki/Kabsch_algorithm)

The best match for multiple references can be found by the minimum RMSD.
Sets of equivalent vectors (atoms) can be permuted as well.
"""
import itertools
import numpy as np
from scipy import linalg
import geomtools.displace as disp


def rmsd(test, ref, wgt=None):
    """Returns the root mean squared deviation of a test geometry with
    respect to a reference.

    Weights can be provided (e.g. atomic masses) with the optional
    variable wgt.
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
    """
    if wgt is None:
        cov = test.T.dot(ref)
    else:
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
    """
    if cent is None:
        new_test = disp.centre_mass(elem, test)
        new_ref = disp.centre_mass(elem, ref)
    else:
        new_test = disp.centre_mass(elem[cent], test[cent])
        new_ref = disp.centre_mass(elem[cent], ref[cent])
    if ind is None:
        return new_test.dot(kabsch(new_test, new_ref, wgt=wgt))
    else:
        return new_test.dot(kabsch(new_test[ind], new_ref[ind], wgt=wgt))


def opt_permute(elem, test, ref, plist=None, equiv=None, wgt=None, ind=None,
                cent=None):
    """Determines optimal permutation of test geometry indices for
    mapping onto reference."""
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
    """Determines optimal reference geometry for a given test geometry."""
    nrefs = len(reflist)
    geoms = np.empty((nrefs,) + test.shape)
    err = np.empty(nrefs)
    for i in range(nrefs):
        geoms[i], err[i] = opt_permute(elem, test, reflist[i], **kwargs)

    optref = np.argmin(err)
    return geoms[optref], optref


def opt_multi(elem, testlist, reflist, **kwargs):
    """Determines the optimal geometries of a set of test geometries
    against a set of reference geometries."""
    geomlist = [[] for i in range(len(reflist))]
    for test in testlist:
        geom, ind = opt_ref(elem, test, reflist, **kwargs)
        geomlist[ind].append(geom)

    return geomlist


def _tuple2list(tupl):
    """Iteratively converts nested tuple to nested list."""
    return list((_tuple2list(x) if isinstance(x, tuple) else x for x in tupl))


def _permute_elmnt(plist):
    """Generates an array of permutations of a list of lists of permutable
    indices and a list of permutable groups of indices."""
    if plist is None:
        return [0]
    elif isinstance(plist[0], int):
        plist = [plist]

    multi_perms = [itertools.permutations(i) for i in plist]
    multi_perms = _tuple2list(itertools.product(*multi_perms))
    return [[item for sublist in i for item in sublist] for i in multi_perms]


def _permute_group(plist):
    """Generates an array of permutations of groups of indices."""
    if plist is None:
        return [0]
    elif isinstance(plist[0], int):
        plist = [plist]

    group_perms = _tuple2list(itertools.permutations(plist))
    return [[item for sublist in i for item in sublist] for i in group_perms]

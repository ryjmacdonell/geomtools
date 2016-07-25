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

(Source: en.wikipedia.org/wiki/Kabsch_algorithm)

The best match for multiple references can be found by the minimum RMSD.
Sets of equivalent vectors (atoms) can be permuted as well. In cases that
are independent of chirality, the vectors may also be inverted.
"""
import itertools
import numpy as np
from scipy import linalg


def tuple2list(tupl):
    """Iteratively converts nested tuple to nested list."""
    return list((tuple2list(x) if isinstance(x, tuple) else x for x in tupl))


def permute(plist):
    """Generates an array of permutations of a list of lists of permutable
    indices."""
    if plist is None:
        return 0, [0]
    elif not isinstance(plist[0], list):
        plist = [plist]

    unperm = [item for sublist in plist for item in sublist]
    single_perms = [list(itertools.permutations(i)) for i in plist]
    prod_perms = tuple2list(itertools.product(*single_perms))
    final_perms = [[item for sublist in i for item in sublist]
                   for i in prod_perms]
    return unperm, final_perms


def rmsd(test, ref):
    """Returns the root mean squared deviation of a test geometry with
    respect to a reference."""
    if test.shape != ref.shape:
        raise ValueError('operands could not be broadcast together with '
                         'shapes {}, {}'.format(test.shape, ref.shape))
    return np.sqrt(np.sum((test - ref) ** 2) / np.size(test))


def kabsch(test, ref):
    """Returns the Kabsch rotational matrix to map a test geometry onto
    a reference."""
    cov = test.T.dot(ref)
    rot1, scale, rot2 = linalg.svd(cov)
    rot1[:,-1] *= np.sign(linalg.det(rot1) * linalg.det(rot2))
    return rot1.dot(rot2)


def map_onto(test, ref):
    """Returns the optimal mapping of a test geometry onto a reference."""
    return test.dot(kabsch(test, ref))


def opt_permute(test, ref, plist=None, invert=True):
    """Determines optimal permutation of test geometry indices for
    mapping onto reference."""
    ind0, perms = permute(plist)
    geoms = np.empty((2 * len(perms) if invert else len(perms),
                      *test.shape))

    for i, ind in enumerate(perms):
        j = 2 * i if invert else i
        xyz = np.copy(test)
        xyz[ind0] = xyz[ind]
        geoms[j] = map_onto(xyz, ref)
        if invert:
            geoms[j+1] = map_onto(-xyz, ref)

    err = np.array([rmsd(xyz, ref) for xyz in geoms])
    return geoms[np.argmin(err)], np.min(err)


def opt_ref(test, reflist, plist=None, invert=True):
    """Determines optimal reference geometry for a given test geometry."""
    nrefs = len(reflist)
    geoms = np.empty((nrefs, *test.shape))
    err = np.empty(nrefs)
    for i in range(nrefs):
        geoms[i], err[i] = opt_permute(test, reflist[i], plist, invert)

    optref = np.argmin(err)
    return geoms[optref], optref


def opt_multi(testlist, reflist, plist=None, invert=True):
    """Determines the optimal geometries of a set of test geometries
    against a set of reference geometries."""
    geomlist = [[] for i in len(reflist)]
    for test in testlist:
        geom, ind = opt_ref(test, reflist, plist, invert)
        geomlist[ind].append(geom)

    return geomlist

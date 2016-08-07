"""
Module for determining bonded atoms and natural coordinates.

The distance between N elements (atoms) in a set of cartesian coordinates
can be determined by taking the norm of the outer difference of the
coordinate array. An N x N adjacency matrix can then be formed by comparing
the distance matrix to an upper (and lower) threshold.

For an adjacency matrix A, the elements connected by k links (bonds) is
given by the matrix A^k. In molecular geometries, this can be used to find
all sets of bonds (k = 1), angles (k = 2) and dihedral angles (k = 3).

Small rings can be measured from the eigenvectors of A. For example, if
B = eig(A), the number of three-membered rings is given by
sum_ij B_ij^3 / 6.
"""
import numpy as np
from scipy import linalg
import geomtools.constants as con


def minor(arr, i, j):
    """Returns the minor of an array.

    Given indices i, j, the minor of the matrix is defined as the original
    matrix excluding row i and column j.
    """
    rows = np.array(range(i) + range(i + 1, arr.shape[0]))[:, np.newaxis]
    cols = np.array(range(j) + range(j + 1, arr.shape[1]))
    return arr[rows, cols]


def build_adjmat(elem, xyz, error=0.56):
    """Returns an adjacency matrix from a set of atoms.

    At present, thresholds are set to the Jmol defaults of covalent
    radius + 0.56 and covalent radius - 0.91.
    """
    rad = con.get_covrad(elem)
    upthresh = np.add.outer(rad, rad) + error
    lothresh = upthresh - 0.35 - 2*error

    xyz_diff = xyz.T[:,:,np.newaxis] - xyz.T[:,np.newaxis,:]
    blength = np.sqrt(np.sum(xyz_diff ** 2, axis=0))

    bonded = (blength < upthresh) & (blength > lothresh)
    return bonded.astype(int)


def k_power(mat, k):
    """Returns the kth power of a square matrix.

    The elements (A^k)_ij of the kth power of an adjacency matrix
    represent the number of k-length paths from element i to element j,
    including repetitions.
    """
    new_mat = np.copy(mat)
    for i in range(k-1):
        new_mat = new_mat.dot(mat)

    return new_mat


def len_k_path(adjmat, k):
    """Returns the number of paths of length k from an adjacency matrix.

    The returned matrix excludes all elements from l = k - 1 to 1 in order
    to avoid double counting bonds."""
    new_adjmat = np.copy(adjmat)
    sum_adjmat = np.zeros_like(adjmat)
    for i in range(k-1):
        if i % 2 == 0:
            # even
            pass
        else:
            #odd
            pass
        sum_adjmat += new_adjmat
        new_adjmat = new_adjmat.dot(adjmat)

    # subtract double counted bonds?
    return new_adjmat


def num_k_loops(adjmat, k):
    """Returns the number of loops of length k.

    Only works for 3-loops and 4-loops at the moment.
    TODO: Generalize this
    """
    if k < 3:
        raise ValueError('Loops must have 3 or more elements.')

    eigs = linalg.eigh(adjmat)[0]
    if k == 3:
        loop = np.sum(eigs ** 3) / 6
        total = int(round(loop3))
    elif k == 4:
        adj2 = k_power(adjmat, 2)
        loop = (np.sum(eigs ** 4) - 2 * np.sum(adj2) + np.sum(adjmat)) / 8

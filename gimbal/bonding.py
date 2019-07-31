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
import gimbal.constants as con


def build_adjmat(elem, xyz, error=0.56, lothresh=0.4):
    """Returns an adjacency matrix from a set of atoms.

    At present, thresholds are set to the Rasmol defaults of covalent
    radius + 0.56 and covalent radius - 0.91.

    Parameters
    ----------
    elem : (N,) array_like
        List of atomic symbols.
    xyz : (N, 3) array_like
        Array of cartesian atomic positions.
    error : float, optional
        The allowable error for the upper bound of bond lengths in
        Angstroms. Default is 0.56 based on JMol.
    lothresh : float, optional
        The lower bound for all bond lengths. Default is 0.4 based on JMol.

    Returns
    -------
    (N, N) ndarray
        The adjacency matrix, an N x N array of ones at off-diagonal
        elements where two atoms are bonded, zeros otherwise.
    """
    rad = con.get_covrad(elem)
    upthresh = np.add.outer(rad, rad) + error

    xyz_diff = xyz.T[:,:,np.newaxis] - xyz.T[:,np.newaxis,:]
    blength = np.sqrt(np.sum(xyz_diff**2, axis=0))

    bonded = (blength < upthresh) & (blength > lothresh)
    return bonded.astype(int)


def power(mat, k):
    """Returns the kth power of a square matrix.

    The elements (A^k)_ij of the kth power of an adjacency matrix
    represent the number of k-length paths from element i to element j,
    including repetitions.

    Parameters
    ----------
    mat : (N, N) array_like
        The input square matrix, A.
    k : int
        The power of the matrix.

    Returns
    -------
    (N, N) ndarray
        The input matrix to the power k, A^k.
    """
    return np.linalg.matrix_power(mat, k)


def path_len(adjmat, k):
    """Returns the matrix of paths of length k from an adjacency matrix.

    Ideally, all elements should be unity unless loops are present. Loops
    are not fully accounted for at the moment. They should lead to
    nonzero diagonal elements.

    Parameters
    ----------
    adjmat : (N, N) array_like
        The adjacency matrix.
    k : int
        The desired path length.

    Returns
    -------
    (N, N) ndarray
        The matrix of elements connected by k bonds.
    """
    new_mat = power(adjmat, k)
    new_mat -= np.diag(np.diag(new_mat))
    return new_mat


def num_neighbours(adjmat, k):
    """Returns the number of atoms k atoms away from each atom.

    Parameters
    ----------
    adjmat : (N, N) array_like
        The adjacency matrix.
    k : int
        The desired path length

    Returns
    -------
    (N,) ndarray
        A vector of length N with the number of path of length k for
        each atom.
    """
    return np.sum(path_len(adjmat, k), axis=0)


def num_loops(adjmat, k):
    """Returns the number of loops of length k.

    Only works for 3-loops and 4-loops at the moment.

    Parameters
    ----------
    adjmat : (N, N) array_like
        The adjacency matrix.
    k : int
        The desired number of atoms in the loop.

    Returns
    -------
    int
        The number of loops with k atoms.
    """
    if k < 3:
        raise ValueError('Loops must have 3 or more elements.')

    eigs = linalg.eigh(adjmat)[0]
    if k == 3:
        loop = np.sum(eigs ** 3) / 6
    elif k == 4:
        adj2 = power(adjmat, 2)
        loop = (np.sum(eigs ** 4) - 2 * np.sum(adj2) + np.sum(adjmat)) / 8
    else:
        raise ValueError('Loops of more than 4 elements not supported.')

    return int(round(loop))

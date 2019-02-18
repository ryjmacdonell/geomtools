"""
Tests for the bonding module.
"""
import pytest
import numpy as np
import gimbal.bonding as bonding


h2 = (['H', 'H'],
      np.array([[ 0.000000,  0.000000,  0.370000],
                [ 0.000000,  0.000000, -0.370000]]))
ch4 = (['C', 'H', 'H', 'H', 'H'],
       np.array([[ 0.000000,  0.000000,  0.000000],
                 [ 0.629118,  0.629118,  0.629118],
                 [-0.629118, -0.629118,  0.629118],
                 [ 0.629118, -0.629118, -0.629118],
                 [-0.629118,  0.629118, -0.629118]]))
c4 = (['C', 'C', 'C', 'C'],
      np.array([[ 0.000000,  1.071142,  0.147626],
                [ 0.000000, -1.071142,  0.147626],
                [-1.071142,  0.000000, -0.147626],
                [ 1.071142,  0.000000, -0.147626]]))
p4 = (['P', 'P', 'P', 'P'],
      np.array([[-0.885000, -0.511000, -0.361300],
                [ 0.885000, -0.511000, -0.361300],
                [ 0.000000,  1.021900, -0.361300],
                [ 0.000000,  0.000000,  1.083900]]))


def test_build_adjmat_default():
    amat = bonding.build_adjmat(*h2)
    assert np.all(amat == np.array([[0, 1], [1, 0]]))


def test_build_adjmat_error():
    amat = bonding.build_adjmat(*h2, error=0.)
    assert np.all(amat == 0)


def test_build_adjmat_lothresh():
    amat = bonding.build_adjmat(*h2, lothresh=0.8)
    assert np.all(amat == 0)


def test_power():
    mat = np.array([[1, -1], [1, 1]])
    prod = np.array([[-2, -2], [2, -2]])
    assert np.all(bonding.power(mat, 3) == prod)


def test_path_len():
    amat = bonding.build_adjmat(*ch4)
    soln = np.ones(5, dtype=int) - np.eye(5, dtype=int)
    soln[0,:] = 0
    soln[:,0] = 0
    assert np.all(bonding.path_len(amat, 2) == soln)


def test_num_neighbours():
    amat = bonding.build_adjmat(*ch4)
    soln = np.array([0, 3, 3, 3, 3])
    assert np.all(bonding.num_neighbours(amat, 2) == soln)


def test_num_loops_3():
    amat = bonding.build_adjmat(*p4)
    assert bonding.num_loops(amat, 3) == 4


def test_num_loops_4():
    amat = bonding.build_adjmat(*c4)
    assert bonding.num_loops(amat, 4) == 1


def test_num_loops_less_than_3():
    amat = bonding.build_adjmat(*ch4)
    with pytest.raises(ValueError, match=r'Loops must have 3 .*'):
        bonding.num_loops(amat, 2)


def test_num_loops_more_than_3():
    amat = bonding.build_adjmat(*ch4)
    with pytest.raises(ValueError, match=r'Loops of more than 4 .*'):
        bonding.num_loops(amat, 5)

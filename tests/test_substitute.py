"""
Tests for the substitute module.
"""
import pytest
import numpy as np
import gimbal.substitute as substitute


lib = substitute.SubLib()
c2h4 = (['C', 'C', 'H', 'H', 'H', 'H'],
        np.array([[ 0.00000000,  0.00000000,  0.66748000],
                  [ 0.00000000,  0.00000000, -0.66748000],
                  [ 0.00000000,  0.92283200,  1.23769500],
                  [ 0.00000000, -0.92283200,  1.23769500],
                  [ 0.00000000,  0.92283200, -1.23769500],
                  [ 0.00000000, -0.92283200, -1.23769500]]))
c2h3f = (['C', 'C', 'F', 'H', 'H', 'H'],
         np.array([[ 0.00000000,  0.00000000,  0.66748000],
                   [ 0.00000000,  0.00000000, -0.66748000],
                   [ 0.00000000,  1.15695604,  1.38235951],
                   [ 0.00000000, -0.92283200,  1.23769500],
                   [ 0.00000000,  0.92283200, -1.23769500],
                   [ 0.00000000, -0.92283200, -1.23769500]]))
c2h3me = (['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
          np.array([[ 0.00000000,  0.00000000,  0.66748000],
                    [ 0.00000000,  0.00000000, -0.66748000],
                    [ 0.00000000,  1.22501228,  1.42441125],
                    [ 0.88600000,  1.81433276,  1.18787083],
                    [-0.88600000,  1.81433276,  1.18787083],
                    [ 0.00000000,  1.00799073,  2.49284919],
                    [ 0.00000000, -0.92283200,  1.23769500],
                    [ 0.00000000,  0.92283200, -1.23769500],
                    [ 0.00000000, -0.92283200, -1.23769500]]))
c2h3me_ax = (['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
             np.array([[ 0.00000000,  0.00000000,  0.66748000],
                       [ 0.00000000,  0.00000000, -0.66748000],
                       [ 0.00000000,  1.22501228,  1.42441125],
                       [-0.42313789,  0.22682954,  1.31113487],
                       [ 0.59992678,  1.24989422,  2.33419954],
                       [-0.81613949,  1.9409233 ,  1.52463972],
                       [ 0.00000000, -0.92283200,  1.23769500],
                       [ 0.00000000,  0.92283200, -1.23769500],
                       [ 0.00000000, -0.92283200, -1.23769500]]))


def test_import_sub_methyl():
    elem, xyz = substitute.import_sub('ch3')
    assert np.all(elem == lib.elem['me'])
    assert np.allclose(xyz, lib.xyz['me'])


def test_import_sub_ethyl():
    elem, xyz = substitute.import_sub('ch2ch3')
    assert np.all(elem == lib.elem['et'])
    assert np.allclose(xyz, lib.xyz['et'])


def test_subst_fluoro_default():
    elem, xyz, vec = substitute.subst(c2h4[0], c2h4[1], 'f', 2)
    assert np.all(elem == c2h3f[0])
    assert np.allclose(xyz, c2h3f[1])
    assert np.allclose(vec, np.zeros_like(xyz))


def test_subst_fluoro_multisub():
    elem, xyz, vec = substitute.subst(c2h4[0], c2h4[1], 'f', [2, 3])
    assert np.all(elem == np.hstack((c2h3f[0][:3], c2h3f[0][4:])))
    assert np.allclose(xyz, np.vstack((c2h3f[1][:3], c2h3f[1][4:])))


def test_subst_fluoro_vec():
    vec = np.ones_like(c2h4[1])
    elem, xyz, vec = substitute.subst(c2h4[0], c2h4[1], 'f', 2, vec=vec)
    soln = np.copy(vec)
    soln[2] = 0.
    assert np.all(elem == c2h3f[0])
    assert np.allclose(xyz, c2h3f[1])
    assert np.allclose(vec, soln)


def test_subst_fluoro_ibond():
    elem, xyz, vec = substitute.subst(c2h4[0], c2h4[1], 'f', 2, ibond=0)
    assert np.all(elem == c2h3f[0])
    assert np.allclose(xyz, c2h3f[1])


def test_subst_methyl_plane_ind():
    elem, xyz, vec = substitute.subst(c2h4[0], c2h4[1], 'ch3', 2, pl=1)
    assert np.all(elem == c2h3me[0])
    assert np.allclose(xyz, c2h3me[1])


def test_subst_methyl_plane_vec():
    elem, xyz, vec = substitute.subst(c2h4[0], c2h4[1], 'ch3', 2, pl=[1, 1, 1])
    assert np.all(elem == c2h3me_ax[0])
    assert np.allclose(xyz, c2h3me_ax[1])


def test_subst_ibond_equals_ipos():
    with pytest.raises(ValueError, match=r'sub and bond indices .*'):
        elem, xyz, vec = substitute.subst(c2h4[0], c2h4[1], 'f', 2, ibond=2)


def test_subst_pl_equals_ipos():
    with pytest.raises(ValueError, match=r'plane and sub indices .*'):
        elem, xyz, vec = substitute.subst(c2h4[0], c2h4[1], 'f', 2, pl=2)


def test_subst_pl_equals_ibond():
    with pytest.raises(ValueError, match=r'plane and bond indices .*'):
        elem, xyz, vec = substitute.subst(c2h4[0], c2h4[1], 'f', 2, ibond=3,
                                          pl=3)

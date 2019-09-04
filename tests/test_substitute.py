"""
Tests for the substitute module.
"""
import pytest
import numpy as np
import gimbal.substitute as substitute
from examples import Geometries as eg


def test_add_subs_fails():
    lib = substitute.SubLib()
    with pytest.raises(ValueError, match=r'Number of inds != number of .*'):
        elem, xyz = lib.add_subs(['me', 'me', 'me'], inds=[1, 2])


def test_import_sub_methyl():
    lib = substitute.SubLib()
    elem, xyz = substitute.import_sub('ch3')
    assert np.all(elem == lib.elem['me'])
    assert np.allclose(xyz, lib.xyz['me'])


def test_import_sub_ethyl():
    lib = substitute.SubLib()
    elem, xyz = substitute.import_sub('ch2ch3')
    assert np.all(elem == lib.elem['et'])
    assert np.allclose(xyz, lib.xyz['et'])


def test_subst_fluoro_default():
    elem, xyz, vec = substitute.subst(eg.c2h4[0], eg.c2h4[1], 'f', 2)
    assert np.all(elem == eg.c2h3f[0])
    assert np.allclose(xyz, eg.c2h3f[1])
    assert vec is None


def test_subst_fluoro_multisub():
    elem, xyz, vec = substitute.subst(eg.c2h4[0], eg.c2h4[1], 'f', [2, 3])
    assert np.all(elem == np.hstack((eg.c2h3f[0][:3], eg.c2h3f[0][4:])))
    assert np.allclose(xyz, np.vstack((eg.c2h3f[1][:3], eg.c2h3f[1][4:])))


def test_subst_fluoro_with_fluoro():
    inp_xyz = np.copy(eg.c2h3f[1])
    inp_xyz[2] += np.ones(3)
    elem, xyz, vec = substitute.subst(eg.c2h3f[0], inp_xyz, 'f', 2)
    assert np.all(elem == eg.c2h3f[0])
    assert np.allclose(xyz, inp_xyz)


def test_subst_fluoro_vec():
    vec = np.ones_like(eg.c2h4[1])
    elem, xyz, vec = substitute.subst(eg.c2h4[0], eg.c2h4[1], 'f', 2, vec=vec)
    soln = np.copy(vec)
    soln[2] = 0.
    assert np.all(elem == eg.c2h3f[0])
    assert np.allclose(xyz, eg.c2h3f[1])
    assert np.allclose(vec, soln)


def test_subst_fluoro_ibond():
    elem, xyz, vec = substitute.subst(eg.c2h4[0], eg.c2h4[1], 'f', 2, ibond=0)
    assert np.all(elem == eg.c2h3f[0])
    assert np.allclose(xyz, eg.c2h3f[1])


def test_subst_methyl_plane_ind():
    elem, xyz, vec = substitute.subst(eg.c2h4[0], eg.c2h4[1], 'ch3', 2, pl=1)
    assert np.all(elem == eg.c2h3me[0])
    assert np.allclose(xyz, eg.c2h3me[1])


def test_subst_methyl_plane_vec():
    elem, xyz, vec = substitute.subst(eg.c2h4[0], eg.c2h4[1], 'ch3', 2,
                                      pl=[1, 1, 1])
    assert np.all(elem == eg.c2h3me_ax[0])
    assert np.allclose(xyz, eg.c2h3me_ax[1])


def test_subst_ibond_equals_ipos():
    with pytest.raises(ValueError, match=r'sub and bond indices .*'):
        elem, xyz, vec = substitute.subst(eg.c2h4[0], eg.c2h4[1], 'f', 2,
                                          ibond=2)


def test_subst_pl_equals_ipos():
    with pytest.raises(ValueError, match=r'plane and sub indices .*'):
        elem, xyz, vec = substitute.subst(eg.c2h4[0], eg.c2h4[1], 'f', 2, pl=2)


def test_subst_pl_equals_ibond():
    with pytest.raises(ValueError, match=r'plane and bond indices .*'):
        elem, xyz, vec = substitute.subst(eg.c2h4[0], eg.c2h4[1], 'f', 2,
                                          ibond=3, pl=3)

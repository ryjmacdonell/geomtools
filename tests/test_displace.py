"""
Tests for the displace module.
"""
import pytest
import numpy as np
import gimbal.displace as displace
from examples import Geometries as eg


def test_VectorParser_explicit():
    vp = displace.VectorParser()
    assert np.allclose(vp('[1., 2., 3.]'), [1, 2, 3])


def test_VectorParser_vector():
    vp = displace.VectorParser()
    assert np.allclose(vp([1, 2, 3]), [1, 2, 3])


def test_VectorParser_cross_unit():
    vp = displace.VectorParser()
    ax = np.array([[6, -3, 2], [1, 1, 1], [2, 2, 0]])
    ax2 = np.array([1, 2, 3])
    assert np.allclose(vp(ax, unit=True), ax2 / np.linalg.norm(ax2))


def test_VectorParser_xyz_getter():
    xyz = eg.c2h4[1]
    vp = displace.VectorParser(xyz)
    assert np.allclose(vp.xyz, xyz)


def test_VectorParser_xyz_unary():
    xyz = eg.c2h4[1]
    vp = displace.VectorParser(xyz)
    assert np.allclose(vp('-1'), -xyz[1])


def test_VectorParser_xyz_subtract():
    xyz = eg.c2h4[1]
    vp = displace.VectorParser(xyz)
    assert np.allclose(vp('3 - 1'), xyz[3] - xyz[1])


def test_VectorParser_xyz_cross():
    xyz = eg.c2h4[1]
    vp = displace.VectorParser(xyz)
    soln = np.cross(xyz[3] - xyz[1], xyz[4] - xyz[1])
    assert np.allclose(vp('(3 - 1) x (4 - 1)'), soln)


def test_VectorParser_xyz_proj():
    xyz = eg.c2h4[1]
    vp = displace.VectorParser(xyz)
    soln = np.dot(xyz[3], xyz[1]) / np.dot(xyz[1], xyz[1]) * xyz[1]
    assert np.allclose(vp('(3 o 1) / (1 o 1) * 1'), soln)


def test_VectorParser_xyz_mixed():
    xyz = eg.c2h4[1]
    vp = displace.VectorParser(xyz)
    soln = [1.2, xyz[3,0]**2, np.dot(xyz[3], xyz[2]-xyz[1])]
    assert np.allclose(vp('[1.2, 3[0]^2., 3 o (2 - 1)]'), soln)


def test_VectorParser_unrec_axis():
    vp = displace.VectorParser()
    with pytest.raises(ValueError, match=r'Axis specification not recognized'):
        rot = vp([1, 1, 1, 1])


def test_translate_ax_all():
    new_xyz = displace.translate(eg.c2h4[1], np.sqrt(3), [1, 1, 1])
    diff_xyz = new_xyz - eg.c2h4[1]
    assert np.allclose(diff_xyz, np.ones((6, 3)))


def test_translate_x_carbon():
    new_xyz = displace.translate(eg.c2h4[1], 1., 'X', ind=[0,1])
    diff_xyz = new_xyz - eg.c2h4[1]
    soln = np.zeros((6, 3))
    soln[:2,0] = 1.
    assert np.allclose(diff_xyz, soln)


def test_translate_x_bohr():
    new_xyz = displace.translate(eg.c2h4[1], 2., '-X', units='bohr')
    diff_xyz = new_xyz - eg.c2h4[1]
    soln = np.zeros((6, 3))
    soln[:,0] = -1.05835442
    assert np.allclose(diff_xyz, soln)


def test_rotmat_zero():
    rot = displace.rotmat(0., [0, 0, 1])
    assert np.allclose(rot, np.eye(3))


def test_rotmat_invert():
    rot = displace.rotmat(np.pi, [0, 0, 1], det=-1)
    assert np.allclose(rot, -np.eye(3))


def test_rotmat_degrees():
    rot = displace.rotmat(90., [0, 1, 0], units='deg')
    soln = np.flip(np.eye(3), axis=0)
    soln[0,2] = -1.
    assert np.allclose(rot, soln)


def test_rotmat_det_error():
    with pytest.raises(ValueError, match=r'Determinant of a rotational .*'):
        rot = displace.rotmat(0., [0, 0, 1], det=2.)


def test_angax_identity():
    ang, ax, d = displace.angax(np.eye(3))
    assert np.isclose(ang, 0.)
    assert np.allclose(ax, [0, 0, 1])
    assert np.isclose(d, 1.)


def test_angax_invert():
    ang, ax, d = displace.angax(-np.eye(3))
    assert np.isclose(ang, np.pi)
    assert np.allclose(ax, [0, 0, 1])
    assert np.isclose(d, -1.)


def test_angax_pi():
    rot = -np.eye(3)
    rot[1,1] = 1.
    ang, ax, d = displace.angax(rot)
    assert np.isclose(ang, np.pi)
    assert np.allclose(ax, [0, 1, 0])
    assert np.isclose(d, 1.)


def test_angax_degrees():
    u = np.ones(3) / np.sqrt(3)
    rot = np.array([[0, u[2], -u[1]], [-u[2], 0, u[0]], [u[1], -u[0], 0]])
    rot += np.outer(u, u)
    ang, ax, d = displace.angax(rot, units='deg')
    assert np.isclose(ang, 90.)
    assert np.allclose(ax, u)
    assert np.isclose(d, 1.)


def test_angax_det_error():
    with pytest.raises(ValueError, match=r'Determinant of a rotational .*'):
        ang, ax, d = displace.angax(2*np.eye(3))


def test_rotate_degrees():
    xyz = eg.c2h4[1]
    new_xyz = displace.rotate(xyz, 90., 'Z', units='deg')
    soln = np.array([-xyz[:,1], -xyz[:,0], xyz[:,2]]).T
    assert np.allclose(new_xyz, soln)


def test_rotate_invert():
    new_xyz = displace.rotate(eg.c2h4[1], np.pi, '-Z', det=-1)
    assert np.allclose(new_xyz, -eg.c2h4[1])


def test_rotate_reflect():
    soln = np.copy(eg.c2h4[1])
    new_xyz = displace.rotate(soln, 0., 'XxZ', det=-1)
    soln[:,1] *= -1
    assert np.allclose(new_xyz, soln)


def test_rotate_carbons():
    soln = np.copy(eg.c2h4[1])
    new_xyz = displace.rotate(soln, np.pi, 'X', ind=[0,1])
    soln[[0, 1]] = soln[[1, 0]]
    assert np.allclose(new_xyz, soln)


def test_rotate_origin():
    xyz = np.copy(eg.c2h4[1])
    new_xyz = displace.rotate(xyz, np.pi/2, 'Y', origin=xyz[0])
    soln = np.array([xyz[:,2] - xyz[0,2], xyz[:,1], xyz[0,2]*np.ones(6)]).T
    assert np.allclose(new_xyz, soln)


def test_align_pos_default():
    xyz = np.ones((1, 3))
    new_xyz = displace.align_pos(xyz, -np.ones(3), np.zeros(3))
    assert np.allclose(new_xyz, 2*np.ones(3))


def test_align_pos_ind():
    xyz = np.array([np.ones(3), np.zeros(3)])
    new_xyz = displace.align_pos(xyz, xyz[0], xyz[1], ind=0)
    assert np.allclose(new_xyz[0], new_xyz[1])


def test_align_axis_default():
    xyz = np.copy(eg.c2h4[1])
    new_xyz = displace.align_axis(xyz, xyz[0]-xyz[1], 'Y')
    soln = np.array([xyz[:,0], xyz[:,2], -xyz[:,1]]).T
    assert np.allclose(new_xyz, soln)


def test_align_axis_same():
    ax1 = np.array([1., -2., 0.])
    ax2 = 2*ax1
    xyz = np.copy(eg.c2h4[1])
    new_xyz = displace.align_axis(xyz, ax1, ax2)
    assert np.allclose(new_xyz, xyz)


def test_align_axis_pi_y():
    ax = np.array([0., -2., 0.])
    soln = np.copy(eg.c2h4[1])
    new_xyz = displace.align_axis(soln, ax, 'Y')
    soln[:,1] *= -1
    assert np.allclose(new_xyz, soln)


def test_align_axis_pi_z():
    ax = np.array([0., 0., -2.])
    soln = np.copy(eg.c2h4[1])
    new_xyz = displace.align_axis(soln, ax, 'Z')
    soln[:,2] *= -1
    assert np.allclose(new_xyz, soln)


def test_align_axis_ind():
    soln = np.copy(eg.c2h4[1])
    new_xyz = displace.align_axis(soln, 'Z', '-Y', ind=[0,1])
    soln[[0,1],1] = -soln[[0,1],2]
    soln[[0,1],2] = 0.
    assert np.allclose(new_xyz, soln)


def test_align_axis_origin():
    xyz = np.copy(eg.c2h4[1])
    new_xyz = displace.align_axis(xyz, 'Z', 'X', origin=xyz[0])
    soln = np.array([xyz[:,2] - xyz[0,2], xyz[:,1], xyz[0,2]*np.ones(6)]).T
    assert np.allclose(new_xyz, soln)


def test_get_centremass_molecule():
    xyz = eg.c2h4[1] + np.ones(3)
    cm = displace.get_centremass(eg.c2h4[0], xyz)
    assert np.allclose(cm, np.ones(3))


def test_get_centremass_atom():
    cm = displace.get_centremass(eg.c2h4[0][0], eg.c2h4[1][0])
    assert np.allclose(cm, eg.c2h4[1][0])


def test_centre_mass():
    xyz = eg.c2h4[1] + np.ones(3)
    new_xyz = displace.centre_mass(eg.c2h4[0], xyz)
    diff_xyz = new_xyz - xyz
    assert np.allclose(diff_xyz, -np.ones((6, 3)))

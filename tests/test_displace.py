"""
Tests for the displace module.
"""
import pytest
import numpy as np
import gimbal.displace as displace


c2h4 = (['C', 'C', 'H', 'H', 'H', 'H'],
        np.array([[ 0.000000,  0.000000,  0.667480],
                  [ 0.000000,  0.000000, -0.667480],
                  [ 0.000000,  0.922832,  1.237695],
                  [ 0.000000, -0.922832,  1.237695],
                  [ 0.000000,  0.922832, -1.237695],
                  [ 0.000000, -0.922832, -1.237695]]))


def test_translate_ax_all():
    new_xyz = displace.translate(c2h4[1], np.sqrt(3), [1, 1, 1])
    diff_xyz = new_xyz - c2h4[1]
    assert np.allclose(diff_xyz, np.ones((6, 3)))


def test_translate_x_carbon():
    new_xyz = displace.translate(c2h4[1], 1., 'x', ind=[0,1])
    diff_xyz = new_xyz - c2h4[1]
    soln = np.zeros((6, 3))
    soln[:2,0] = 1.
    assert np.allclose(diff_xyz, soln)


def test_translate_x_bohr():
    new_xyz = displace.translate(c2h4[1], 2., '-x', units='bohr')
    diff_xyz = new_xyz - c2h4[1]
    soln = np.zeros((6, 3))
    soln[:,0] = -1.05835442
    assert np.allclose(diff_xyz, soln)


def test_rotmat_zero():
    rot = displace.rotmat(0., 'z')
    assert np.allclose(rot, np.eye(3))


def test_rotmat_invert():
    rot = displace.rotmat(np.pi, 'z', det=-1)
    assert np.allclose(rot, -np.eye(3))


def test_rotmat_degrees():
    rot = displace.rotmat(90., 'y', units='deg')
    soln = np.flip(np.eye(3), axis=0)
    soln[0,2] = -1.
    assert np.allclose(rot, soln)


def test_rotmat_plane_axis():
    rot = displace.rotmat(np.pi/2, [[0, 1, -1], [0, 0, 0], [-1, 1, 0]])
    u = np.ones(3) / np.sqrt(3)
    soln = np.array([[0, u[2], -u[1]], [-u[2], 0, u[0]], [u[1], -u[0], 0]])
    soln += np.outer(u, u)
    assert np.allclose(rot, soln)


def test_rotmat_unrec_axis():
    with pytest.raises(ValueError, match=r'Axis specification not recognized'):
        rot = displace.rotmat(0., [1, 1])


def test_rotmat_det_error():
    with pytest.raises(ValueError, match=r'Determinant of a rotational .*'):
        rot = displace.rotmat(0., 'z', det=2.)


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
    xyz = c2h4[1]
    new_xyz = displace.rotate(xyz, 90., 'z', units='deg')
    soln = np.array([-xyz[:,1], -xyz[:,0], xyz[:,2]]).T
    assert np.allclose(new_xyz, soln)


def test_rotate_invert():
    new_xyz = displace.rotate(c2h4[1], np.pi, '-z', det=-1)
    assert np.allclose(new_xyz, -c2h4[1])


def test_rotate_reflect():
    soln = np.copy(c2h4[1])
    new_xyz = displace.rotate(soln, 0., 'xz', det=-1)
    soln[:,1] *= -1
    assert np.allclose(new_xyz, soln)


def test_rotate_carbons():
    soln = np.copy(c2h4[1])
    new_xyz = displace.rotate(soln, np.pi, 'x', ind=[0,1])
    soln[[0, 1]] = soln[[1, 0]]
    assert np.allclose(new_xyz, soln)


def test_rotate_origin():
    xyz = np.copy(c2h4[1])
    new_xyz = displace.rotate(xyz, np.pi/2, 'y', origin=xyz[0])
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
    xyz = np.copy(c2h4[1])
    new_xyz = displace.align_axis(xyz, xyz[0]-xyz[1], 'y')
    soln = np.array([xyz[:,0], xyz[:,2], -xyz[:,1]]).T
    assert np.allclose(new_xyz, soln)


def test_align_axis_ind():
    soln = np.copy(c2h4[1])
    new_xyz = displace.align_axis(soln, 'z', '-y', ind=[0,1])
    soln[[0,1],1] = -soln[[0,1],2]
    soln[[0,1],2] = 0.
    assert np.allclose(new_xyz, soln)


def test_align_axis_origin():
    xyz = np.copy(c2h4[1])
    new_xyz = displace.align_axis(xyz, 'z', 'x', origin=xyz[0])
    soln = np.array([xyz[:,2] - xyz[0,2], xyz[:,1], xyz[0,2]*np.ones(6)]).T
    assert np.allclose(new_xyz, soln)


def test_get_centremass_molecule():
    xyz = c2h4[1] + np.ones(3)
    cm = displace.get_centremass(c2h4[0], xyz)
    assert np.allclose(cm, np.ones(3))


def test_get_centremass_atom():
    cm = displace.get_centremass(c2h4[0][0], c2h4[1][0])
    assert np.allclose(cm, c2h4[1][0])


def test_centre_mass():
    xyz = c2h4[1] + np.ones(3)
    new_xyz = displace.centre_mass(c2h4[0], xyz)
    diff_xyz = new_xyz - xyz
    assert np.allclose(diff_xyz, -np.ones((6, 3)))

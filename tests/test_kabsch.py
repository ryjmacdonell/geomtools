"""
Tests for the kabsch module.
"""
import numpy as np
import gimbal.kabsch as kabsch


elem = ['C', 'H', 'H', 'F', 'F']
xyz1 = np.array([[ 0.000000,  0.000000,  0.000000],
                 [ 0.000000,  0.000000,  1.000000],
                 [ 0.000000,  1.000000,  0.000000],
                 [ 1.000000,  0.000000,  0.000000],
                 [-1.000000,  0.000000,  0.000000]])
xyz2 = np.array([[ 0.000000,  0.000000,  0.000000],
                 [ 1.020000, -1.020000,  0.000000],
                 [ 1.020000,  1.020000,  0.000000],
                 [-1.020000,  1.020000,  0.000000],
                 [ 0.000000,  0.000000,  1.020000]])
xyz3 = np.array([[ 0.000000,  0.000000,  0.000000],
                 [ 0.000000,  0.000000, -1.010000],
                 [ 0.000000,  0.990000,  0.000000],
                 [ 1.000000,  0.000000,  0.000000],
                 [-1.000000,  0.000000,  0.000000]])
xyz4 = np.array([[ 0.000000,  0.000000,  0.000000],
                 [ 1.000000,  0.000000,  0.000000],
                 [ 0.000000,  1.000000,  0.000000],
                 [ 0.000000,  0.000000,  0.980000],
                 [-1.020000,  0.000000,  0.000000]])
xyz5 = np.array([[ 0.000000,  0.000000,  0.000000],
                 [ 0.000000, -1.000000,  0.000000],
                 [ 0.000000,  0.000000, -1.000000],
                 [ 0.000000,  1.000000,  0.000000],
                 [ 1.000000, -1.000000,  0.000000]])


def test_kabsch_default():
    rot = kabsch.kabsch(xyz3, xyz1)
    soln = np.eye(3)
    soln[2,2] = -1
    assert np.allclose(rot, soln)


def test_kabsch_no_reflect():
    rot = kabsch.kabsch(xyz3, xyz1, refl=False)
    soln = -np.eye(3)
    soln[0,0] = 1
    assert np.allclose(rot, soln)


def test_map_onto_default():
    new_xyz3 = kabsch.map_onto(elem, xyz3, xyz1)
    assert np.all(np.abs(new_xyz3 - xyz1) < 1e-2)


def test_map_onto_ind_cent():
    new_xyz4 = kabsch.map_onto(elem, xyz4, xyz1, ind=[0,1,2], cent=0)
    assert np.allclose(new_xyz4[[0,1,2]], xyz1[[0,1,2]])


def test_opt_permute_default():
    new_xyz3, rmsd = kabsch.opt_permute(elem, xyz3, xyz1)
    assert np.all(np.abs(new_xyz3 - xyz1) < 1e-2)
    assert rmsd < 3e-1


def test_opt_permute_plist_single():
    new_xyz3, rmsd = kabsch.opt_permute(elem, xyz3, xyz1, plist=[1,2])
    assert np.all(np.abs(new_xyz3 - xyz1) < 1e-2)
    assert rmsd < 3e-3


def test_opt_permute_plist_multiple():
    new_xyz4, rmsd = kabsch.opt_permute(elem, xyz4, xyz2, plist=[[1,2],[3,4]],
                                        cent=0)
    soln = xyz2 / 1.02
    soln[:4] *= np.sqrt(2)/2
    soln[3] *= 1.02
    soln[4] *= 0.98
    assert np.allclose(new_xyz4, soln)


def test_opt_permute_equiv():
    new_xyz5, rmsd = kabsch.opt_permute(elem, xyz5, xyz1, equiv=[[1,3],[2,4]],
                                        cent=0)
    soln = np.zeros((5, 3))
    soln[1,2] = 1.
    soln[2,:2] = np.sqrt(2)/2
    soln[3,0] = np.sqrt(2)
    soln[4,:2] = -np.sqrt(2)/2
    assert np.allclose(new_xyz5, soln)


def test_opt_permute_weight():
    mwgts = [12., 1., 1., 19., 19.]
    new_xyz3, rmsd = kabsch.opt_permute(elem, xyz3, xyz1, wgt=mwgts)
    assert np.all(np.abs(new_xyz3 - xyz1) < 1e-2)
    assert rmsd < 5e-2


def test_opt_permute_ind():
    new_xyz5, rmsd = kabsch.opt_permute(elem, xyz5, xyz1, ind=[0,1,2], cent=0)
    assert np.allclose(new_xyz5[:3], xyz1[:3])
    assert rmsd < 4e-1


def test_opt_ref_default():
    new_xyz5, ind = kabsch.opt_ref(elem, xyz5, [xyz1, xyz2])
    assert kabsch.rmsd(new_xyz5, xyz2) < kabsch.rmsd(new_xyz5, xyz1)
    assert ind == 1


def test_opt_multi_default():
    match1, match2 = kabsch.opt_multi(elem, [xyz3, xyz4, xyz5], [xyz1, xyz2])
    assert kabsch.rmsd(match1[0], xyz1) < kabsch.rmsd(match1[0], xyz2)
    assert kabsch.rmsd(match1[1], xyz1) < kabsch.rmsd(match1[1], xyz2)
    assert kabsch.rmsd(match2[0], xyz2) < kabsch.rmsd(match2[0], xyz1)

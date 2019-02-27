"""
Tests for the kabsch module.
"""
import numpy as np
import gimbal.kabsch as kabsch
from examples import Geometries as eg


def test_kabsch_default():
    rot = kabsch.kabsch(eg.ch2f2[3], eg.ch2f2[1])
    soln = np.eye(3)
    soln[2,2] = -1
    assert np.allclose(rot, soln)


def test_kabsch_no_reflect():
    rot = kabsch.kabsch(eg.ch2f2[3], eg.ch2f2[1], refl=False)
    soln = -np.eye(3)
    soln[0,0] = 1
    assert np.allclose(rot, soln)


def test_map_onto_default():
    new_xyz3 = kabsch.map_onto(eg.ch2f2[0], eg.ch2f2[3], eg.ch2f2[1])
    assert np.all(np.abs(new_xyz3 - eg.ch2f2[1]) < 1e-2)


def test_map_onto_ind_cent():
    new_xyz4 = kabsch.map_onto(eg.ch2f2[0], eg.ch2f2[4], eg.ch2f2[1],
                               ind=[0,1,2], cent=0)
    assert np.allclose(new_xyz4[[0,1,2]], eg.ch2f2[1][[0,1,2]])


def test_opt_permute_default():
    new_xyz3, rmsd = kabsch.opt_permute(eg.ch2f2[0], eg.ch2f2[3], eg.ch2f2[1])
    assert np.all(np.abs(new_xyz3 - eg.ch2f2[1]) < 1e-2)
    assert rmsd < 3e-1


def test_opt_permute_plist_single():
    new_xyz3, rmsd = kabsch.opt_permute(eg.ch2f2[0], eg.ch2f2[3], eg.ch2f2[1],
                                        plist=[1,2])
    assert np.all(np.abs(new_xyz3 - eg.ch2f2[1]) < 1e-2)
    assert rmsd < 3e-3


def test_opt_permute_plist_multiple():
    new_xyz4, rmsd = kabsch.opt_permute(eg.ch2f2[0], eg.ch2f2[4], eg.ch2f2[2],
                                        plist=[[1,2],[3,4]], cent=0)
    soln = eg.ch2f2[2] / 1.02
    soln[:4] *= np.sqrt(2)/2
    soln[3] *= 1.02
    soln[4] *= 0.98
    assert np.allclose(new_xyz4, soln)


def test_opt_permute_equiv_single():
    new_xyz3, rmsd = kabsch.opt_permute(eg.ch2f2[0], eg.ch2f2[3], eg.ch2f2[1],
                                        equiv=[1,2])
    assert np.all(np.abs(new_xyz3 - eg.ch2f2[1]) < 1e-2)
    assert rmsd < 3e-3


def test_opt_permute_equiv_multiple():
    new_xyz5, rmsd = kabsch.opt_permute(eg.ch2f2[0], eg.ch2f2[5], eg.ch2f2[1],
                                        equiv=[[1,3],[2,4]], cent=0)
    soln = np.zeros((5, 3))
    soln[1,2] = 1.
    soln[2,:2] = np.sqrt(2)/2
    soln[3,0] = np.sqrt(2)
    soln[4,:2] = -np.sqrt(2)/2
    assert np.allclose(new_xyz5, soln)


def test_opt_permute_weight():
    mwgts = [12., 1., 1., 19., 19.]
    new_xyz3, rmsd = kabsch.opt_permute(eg.ch2f2[0], eg.ch2f2[3], eg.ch2f2[1],
                                        wgt=mwgts)
    assert np.all(np.abs(new_xyz3 - eg.ch2f2[1]) < 1e-2)
    assert rmsd < 5e-2


def test_opt_permute_ind():
    new_xyz5, rmsd = kabsch.opt_permute(eg.ch2f2[0], eg.ch2f2[5], eg.ch2f2[1],
                                        ind=[0,1,2], cent=0)
    assert np.allclose(new_xyz5[:3], eg.ch2f2[1][:3])
    assert rmsd < 4e-1


def test_opt_ref_default():
    new_xyz5, ind = kabsch.opt_ref(eg.ch2f2[0], eg.ch2f2[5],
                                   [eg.ch2f2[1], eg.ch2f2[2]])
    assert kabsch.rmsd(new_xyz5, eg.ch2f2[2]) < kabsch.rmsd(new_xyz5, eg.ch2f2[1])
    assert ind == 1


def test_opt_multi_default():
    match1, match2 = kabsch.opt_multi(eg.ch2f2[0], [eg.ch2f2[3], eg.ch2f2[4], eg.ch2f2[5]],
                                      [eg.ch2f2[1], eg.ch2f2[2]])
    assert kabsch.rmsd(match1[0], eg.ch2f2[1]) < kabsch.rmsd(match1[0], eg.ch2f2[2])
    assert kabsch.rmsd(match1[1], eg.ch2f2[1]) < kabsch.rmsd(match1[1], eg.ch2f2[2])
    assert kabsch.rmsd(match2[0], eg.ch2f2[2]) < kabsch.rmsd(match2[0], eg.ch2f2[1])

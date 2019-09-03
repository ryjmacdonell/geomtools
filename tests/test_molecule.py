"""
Tests for the molecule module.
"""
import pytest
import numpy as np
import gimbal.molecule as molecule
from examples import Geometries as eg
from examples import Files as ef


def test_empty_BaseMolecule():
    mol = molecule.BaseMolecule()
    assert len(mol.elem) == 0
    assert len(mol.xyz) == 0
    assert len(mol.vec) == 0
    assert mol.comment == ''
    assert str(mol) == '[]'
    assert repr(mol) == 'BaseMolecule(\'\', [])'


def test_single_atom_vec_BaseMolecule():
    mol = molecule.BaseMolecule(*eg.he, vec=np.ones(3))
    zf = 3*'{:14.8f}'.format(0)
    of = 3*'{:14.8f}'.format(1)
    ze = ','.join(3*['{:16.8e}'.format(0)])
    oe = ','.join(3*['{:16.8e}'.format(1)])
    assert mol.natm == 1
    assert np.all(mol.elem == ['He'])
    assert np.allclose(mol.xyz, np.zeros((1, 3)))
    assert str(mol) == '[[He' + zf + '\n    ' + of + ']]'
    assert repr(mol) == ('BaseMolecule(\'\',\n [[He,' + ze +
                         ',\n      ' + oe + ']])')


def test_two_atom_comment_BaseMolecule():
    mol = molecule.BaseMolecule(*eg.h2, comment='comment')
    assert mol.natm == 2
    assert np.all(mol.elem == eg.h2[0])
    assert np.allclose(mol.xyz, eg.h2[1])
    assert 'comment' in str(mol)


def test_eleven_atom_BaseMolecule():
    mol = molecule.BaseMolecule(*eg.c3h8)
    assert mol.natm == 11
    assert np.all(mol.elem == eg.c3h8[0])
    assert np.allclose(mol.xyz, eg.c3h8[1])
    assert '...' in str(mol)
    assert '...' in repr(mol)


def test_BaseMolecule_xyz_not_3d():
    with pytest.raises(ValueError, match=r'Molecular geometry must be 3.*'):
        mol = molecule.BaseMolecule('H', np.zeros(2))


def test_BaseMolecule_vec_not_3d():
    with pytest.raises(ValueError, match=r'Molecular vector must be 3.*'):
        mol = molecule.BaseMolecule('H', np.zeros(3), vec=np.ones(2))


def test_BaseMolecule_len_elem_not_equal_len_xyz():
    with pytest.raises(ValueError, match=r'Number of element labels .*'):
        mol = molecule.BaseMolecule('H', np.zeros((2, 3)))


def test_BaseMolecule_xyz_vec_different_shape():
    with pytest.raises(ValueError, match=r'Cartesian geometry and vector .*'):
        mol = molecule.BaseMolecule(['H', 'H'], np.zeros((2, 3)),
                                    np.ones((1, 3)))


def test_BaseMolecule_copy():
    mol1 = molecule.BaseMolecule(*eg.c2h4)
    mol2 = mol1.copy()
    assert np.all(mol1.elem == eg.c2h4[0])
    assert np.all(mol2.elem == eg.c2h4[0])
    assert np.allclose(mol1.xyz, eg.c2h4[1])
    assert np.allclose(mol2.xyz, eg.c2h4[1])


def test_BaseMolecule_save():
    mol = molecule.BaseMolecule(*eg.he)
    mol.elem = np.array(['H'])
    mol.xyz = np.ones((1, 3))
    mol.comment = 'comment'
    mol.save()
    assert mol.saved
    assert np.all(mol.save_elem == ['H'])
    assert np.allclose(mol.save_xyz, np.ones((1, 3)))
    assert mol.save_comment == 'comment'


def test_BaseMolecule_revert():
    mol = molecule.BaseMolecule(*eg.he)
    mol.elem = np.array(['H'])
    mol.xyz = np.ones((1, 3))
    mol.comment = 'comment'
    mol.revert()
    assert mol.saved
    assert np.all(mol.elem == eg.he[0])
    assert np.allclose(mol.xyz, eg.he[1])
    assert mol.comment == ''


def test_BaseMolecule_add_atoms_single():
    mol = molecule.BaseMolecule(*eg.he)
    mol.add_atoms('H', np.ones(3))
    soln_elem = np.hstack((eg.he[0], 'H'))
    soln_xyz = np.vstack((eg.he[1], np.ones(3)))
    assert np.all(mol.elem == soln_elem)
    assert np.allclose(mol.xyz, soln_xyz)
    assert not mol.saved


def test_BaseMolecule_add_atoms_multiple():
    mol = molecule.BaseMolecule(*eg.he)
    mol.add_atoms(*eg.ch4)
    soln_elem = np.hstack((eg.he[0], eg.ch4[0]))
    soln_xyz = np.vstack((eg.he[1], eg.ch4[1]))
    assert np.all(mol.elem == soln_elem)
    assert np.allclose(mol.xyz, soln_xyz)
    assert not mol.saved


def test_BaseMolecule_add_atoms_new_vec():
    mol = molecule.BaseMolecule(*eg.he)
    mol.add_atoms('H', np.ones(3), new_vec=2*np.ones(3))
    soln_elem = np.hstack((eg.he[0], 'H'))
    soln_xyz = np.vstack((eg.he[1], np.ones(3)))
    soln_vec = np.vstack((np.zeros(3), 2*np.ones(3)))
    assert np.all(mol.elem == soln_elem)
    assert np.allclose(mol.xyz, soln_xyz)
    assert np.allclose(mol.vec, soln_vec)


def test_BaseMolecule_rm_atoms_single():
    mol = molecule.BaseMolecule(*eg.ch4)
    mol.rm_atoms(0)
    assert np.all(mol.elem == eg.ch4[0][1:])
    assert np.allclose(mol.xyz, eg.ch4[1][1:])
    assert not mol.saved


def test_BaseMolecule_rm_atoms_multiple():
    mol = molecule.BaseMolecule(*eg.ch4)
    mol.rm_atoms((1, 2, 3, 4))
    assert np.all(mol.elem == eg.ch4[0][0])
    assert np.allclose(mol.xyz, eg.ch4[1][0])
    assert not mol.saved


def test_BaseMolecule_rearrange_all():
    mol = molecule.BaseMolecule(*eg.ch4)
    ind = [4, 2, 3, 0, 1]
    mol.rearrange(ind)
    assert np.all(mol.elem == eg.ch4[0][ind])
    assert np.allclose(mol.xyz, eg.ch4[1][ind])
    assert not mol.saved


def test_BaseMolecule_rearrange_old_ind():
    mol = molecule.BaseMolecule(*eg.ch4)
    ninds = [1, 2]
    oinds = [3, 4]
    mol.rearrange(ninds, old_ind=oinds)
    soln_elem = eg.ch4[0][[0, 3, 4, 1, 2]]
    soln_xyz = eg.ch4[1][[0, 3, 4, 1, 2]]
    assert np.all(mol.elem == soln_elem)
    assert np.allclose(mol.xyz, soln_xyz)
    assert not mol.saved


def test_BaseMolecule_rearrange_fails():
    mol = molecule.BaseMolecule(*eg.ch4)
    with pytest.raises(IndexError, match=r'Old and new indices must be .*'):
        mol.rearrange([1, 3])


def test_empty_Molecule():
    mol = molecule.Molecule()
    assert len(mol.elem) == 0
    assert len(mol.xyz) == 0
    assert len(mol.vec) == 0
    assert mol.comment == ''
    assert str(mol) == '[]'
    assert repr(mol) == 'Molecule(\'\', [])'


def test_single_atom_vec_Molecule():
    mol = molecule.Molecule(*eg.he, vec=np.ones(3))
    zf = 3*'{:14.8f}'.format(0)
    of = 3*'{:14.8f}'.format(1)
    ze = ','.join(3*['{:16.8e}'.format(0)])
    oe = ','.join(3*['{:16.8e}'.format(1)])
    assert mol.natm == 1
    assert np.all(mol.elem == ['XM', 'He'])
    assert np.allclose(mol.xyz, np.zeros((2, 3)))
    assert str(mol) == '[[He' + zf + '\n    ' + of + ']]'
    assert repr(mol) == ('Molecule(\'\',\n [[XM,' + ze + ',\n      ' + ze +
                         '],\n  [He,' + ze + ',\n      ' + oe + ']])')


def test_eleven_atom_Molecule():
    mol = molecule.Molecule(*eg.c3h8)
    assert mol.natm == 11
    assert np.all(mol.elem == np.hstack(('XM', eg.c3h8[0])))
    assert np.allclose(mol.xyz, np.vstack((np.zeros(3), eg.c3h8[1])), atol=1e-7)
    assert '...' in str(mol)
    assert '...' in repr(mol)


def test_Molecule_add_Molecule():
    mol1 = molecule.Molecule(*eg.he)
    mol2 = molecule.Molecule(*eg.c2h4)
    bund = mol1 + mol2
    assert np.all(bund.molecules[0].elem[1:] == eg.he[0])
    assert np.all(bund.molecules[0].xyz[1:] == eg.he[1])
    assert np.all(bund.molecules[1].elem[1:] == eg.c2h4[0])
    assert np.all(bund.molecules[1].xyz[1:] == eg.c2h4[1])


def test_Molecule_add_MoleculeBundle():
    mol = molecule.Molecule(*eg.he)
    bund1 = molecule.MoleculeBundle(molecule.Molecule(*eg.c2h4))
    bund = mol + bund1
    assert np.all(bund.molecules[0].elem[1:] == eg.he[0])
    assert np.all(bund.molecules[0].xyz[1:] == eg.he[1])
    assert np.all(bund.molecules[1].elem[1:] == eg.c2h4[0])
    assert np.all(bund.molecules[1].xyz[1:] == eg.c2h4[1])


def test_Molecule_copy():
    mol1 = molecule.Molecule(*eg.c2h4)
    mol2 = mol1.copy()
    assert np.all(mol1.elem[1:] == eg.c2h4[0])
    assert np.all(mol2.elem[1:] == eg.c2h4[0])
    assert np.allclose(mol1.xyz[1:], eg.c2h4[1])
    assert np.allclose(mol2.xyz[1:], eg.c2h4[1])


def test_Molecule_rearrange_all():
    mol = molecule.Molecule(*eg.ch4)
    ind = [5, 3, 4, 1, 2]
    oind = [i-1 for i in ind]
    mol.rearrange(ind)
    assert np.all(mol.elem[1:] == eg.ch4[0][oind])
    assert np.allclose(mol.xyz[1:], eg.ch4[1][oind])
    assert not mol.saved


def test_Molecule_read_filename(tmpdir):
    f = tmpdir.join('tmp.xyz')
    f.write(ef.xyz_novec)
    mol = molecule.Molecule()
    mol.read(str(f.realpath()))
    assert np.all(mol.elem[1:] == eg.ch4[0])
    assert np.allclose(mol.xyz[1:], eg.ch4[1])
    assert not mol.saved


def test_Molecule_read_openfile_vec(tmpdir):
    f = ef.tmpf(tmpdir, ef.xyz_vec)
    mol = molecule.Molecule()
    mol.read(f, hasvec=True)
    assert np.all(mol.elem[1:] == eg.ch4[0])
    assert np.allclose(mol.xyz[1:], eg.ch4[1])
    assert not mol.saved


def test_Molecule_write_filename(tmpdir):
    f = tmpdir.join('tmp.xyz')
    mol = molecule.Molecule(*eg.ch4, comment='comment line')
    mol.write(str(f.realpath()))
    assert f.read() == ef.xyz_novec


def test_Molecule_write_openfile(tmpdir):
    f = tmpdir.join('tmp.xyz')
    mol = molecule.Molecule(*eg.ch4, comment='comment line')
    mol.write(f.open(mode='w'))
    assert f.read() == ef.xyz_novec


def test_Molecule_get_mass():
    mol = molecule.Molecule(*eg.ch4)
    mass = mol.get_mass()
    soln = [0.00000000] + [12.00000000] + 4*[1.00782504]
    assert np.allclose(mass, soln)


def test_Molecule_get_formula_single_atom():
    mol = molecule.Molecule(*eg.he)
    form = mol.get_formula()
    assert form == 'He'


def test_Molecule_get_formula_multiple_atoms():
    mol = molecule.Molecule(*eg.c2h4)
    form = mol.get_formula()
    assert form == 'C2H4'


def test_Molecule_measure():
    mol = molecule.Molecule(*eg.c2h4_ms)
    oop = mol.measure('oop', 2, 4, 3, 1)
    assert np.isclose(oop, -np.pi/4)


def test_Molecule_centre_mass():
    mol = molecule.Molecule(*eg.ch4)
    mol.xyz[1] += 2
    mol.centre_mass()
    mass = mol.get_mass()
    cm = np.sum(mass[:,np.newaxis] * mol.xyz, axis=0) / np.sum(mass)
    assert np.allclose(mol.xyz[0], np.zeros(3))
    assert np.allclose(cm, np.zeros(3))


def test_Molecule_translate():
    mol = molecule.Molecule(*eg.he)
    mol.translate(1, [1, 1, 1])
    assert np.allclose(mol.xyz, np.ones(3)/np.sqrt(3))


def test_Molecule_rotate_no_vec():
    mol = molecule.Molecule(*eg.c2h4)
    mol.rotate(np.pi/2, 'Z')
    soln = np.array([-eg.c2h4[1][:,1], -eg.c2h4[1][:,0], eg.c2h4[1][:,2]]).T
    assert np.allclose(mol.xyz[1:], soln)


def test_Molecule_rotate_vec():
    mol = molecule.Molecule(*eg.c2h4, vec=np.ones((6, 3)))
    mol.rotate(np.pi/2, 'Z')
    soln = np.ones_like(eg.c2h4[1])
    soln[:,0] = -1
    assert np.allclose(mol.vec[1:], soln)


def test_Molecule_match_to_ref():
    mol5 = molecule.Molecule(eg.ch2f2[0], eg.ch2f2[5])
    mol1 = molecule.Molecule(eg.ch2f2[0], eg.ch2f2[1])
    mol2 = molecule.Molecule(eg.ch2f2[0], eg.ch2f2[2])
    bund = molecule.MoleculeBundle([mol1, mol2])
    ind = mol5.match_to_ref(bund)
    assert np.all(np.abs(mol5.xyz - mol2.xyz) < 1)
    assert not np.all(np.abs(mol5.xyz - mol1.xyz) < 1)
    assert ind == 1


def test_Molecule_subst():
    mol = molecule.Molecule(*eg.c2h4)
    mol.subst('ch3', 3, pl=2)
    assert np.all(mol.elem[1:] == eg.c2h3me[0])
    assert np.allclose(mol.xyz[1:], eg.c2h3me[1])


def test_Molecule_subst_vec():
    mol = molecule.Molecule(*eg.c2h4)
    mol.print_vec = True
    mol.vec = np.ones_like(mol.xyz)
    mol.subst('ch3', 3, pl=2)
    newvec = np.vstack((np.ones((2, 3)), np.zeros((4, 3)), np.ones((3, 3))))
    assert np.all(mol.elem[1:] == eg.c2h3me[0])
    assert np.allclose(mol.xyz[1:], eg.c2h3me[1])
    assert np.allclose(mol.vec[1:], newvec)


def test_empty_MoleculeBundle():
    bund = molecule.MoleculeBundle()
    assert len(bund.molecules) == 0
    assert bund.nmol == 0
    assert str(bund) == '[]'
    assert repr(bund) == 'MoleculeBundle()'


def test_single_atom_MoleculeBundle():
    mol = molecule.Molecule(*eg.he)
    bund = molecule.MoleculeBundle(mol)
    zf = 3*'{:14.8f}'.format(0)
    ze = 3*',{:16.8e}'.format(0)
    assert bund.nmol == 1
    assert str(bund) == '[[[He' + zf + ']]]'
    assert repr(bund) == ('MoleculeBundle(\n Molecule(\'\',\n  [[XM' + ze +
                          '],\n   [He' + ze + ']]))')


def test_seven_molecule_MoleculeBundle():
    mol = molecule.Molecule(*eg.he)
    bund = molecule.MoleculeBundle(7*[mol])
    assert bund.nmol == 7
    assert '...' in str(bund)
    assert '...' in repr(bund)


def test_MoleculeBundle_add_MoleculeBundle():
    mol1 = molecule.Molecule(*eg.he)
    mol2 = molecule.Molecule(*eg.ch4)
    mol3 = molecule.Molecule(*eg.c2h4)
    bund1 = molecule.MoleculeBundle(mol1)
    bund23 = molecule.MoleculeBundle([mol2, mol3])
    bund = bund1 + bund23
    assert np.all(bund.molecules[0].elem[1:] == eg.he[0])
    assert np.all(bund.molecules[0].xyz[1:] == eg.he[1])
    assert np.all(bund.molecules[1].elem[1:] == eg.ch4[0])
    assert np.all(bund.molecules[1].xyz[1:] == eg.ch4[1])
    assert np.all(bund.molecules[2].elem[1:] == eg.c2h4[0])
    assert np.all(bund.molecules[2].xyz[1:] == eg.c2h4[1])


def test_MoleculeBundle_add_wrong_type():
    bund1 = molecule.MoleculeBundle(molecule.Molecule(*eg.he))
    with pytest.raises(TypeError, match=r'Addition not supported for types .*'):
        bund = bund1 + np.ones(3)


def test_MoleculeBundle_iadd_Molecule():
    mol1 = molecule.Molecule(*eg.he)
    mol2 = molecule.Molecule(*eg.ch4)
    mol3 = molecule.Molecule(*eg.c2h4)
    bund = molecule.MoleculeBundle([mol1, mol2])
    bund += mol3
    assert np.all(bund.molecules[0].elem[1:] == eg.he[0])
    assert np.all(bund.molecules[0].xyz[1:] == eg.he[1])
    assert np.all(bund.molecules[1].elem[1:] == eg.ch4[0])
    assert np.all(bund.molecules[1].xyz[1:] == eg.ch4[1])
    assert np.all(bund.molecules[2].elem[1:] == eg.c2h4[0])
    assert np.all(bund.molecules[2].xyz[1:] == eg.c2h4[1])


def test_MoleculeBundle_wrong_type():
    with pytest.raises(TypeError, match=r'Elements of molecule bundle must .*'):
        bund = molecule.MoleculeBundle(eg.he)


def test_MoleculeBundle_copy():
    mol1 = molecule.Molecule(*eg.he)
    mol2 = molecule.Molecule(*eg.ch4)
    bund1 = molecule.MoleculeBundle([mol1, mol2])
    bund2 = bund1.copy()
    assert np.all(bund1.molecules[0].elem == bund2.molecules[0].elem)
    assert np.all(bund1.molecules[1].elem == bund2.molecules[1].elem)
    assert np.allclose(bund1.molecules[0].xyz, bund2.molecules[0].xyz)
    assert np.allclose(bund1.molecules[1].xyz, bund2.molecules[1].xyz)


def test_MoleculeBundle_save():
    mol = molecule.Molecule(*eg.he)
    bund = molecule.MoleculeBundle([mol, mol])
    bund.molecules = [mol]
    bund.molecules[0].xyz += 1
    bund.molecules[0].elem[1] = 'H'
    bund.save()
    assert bund.saved
    assert bund.molecules[0].saved
    assert len(bund.save_molecules) == 1


def test_MoleculeBundle_revert():
    mol = molecule.Molecule(*eg.he)
    bund = molecule.MoleculeBundle([mol, mol])
    bund.molecules = [mol]
    bund.molecules[0].xyz += 1
    bund.molecules[0].elem[1] = 'H'
    bund.revert()
    assert bund.saved
    assert bund.molecules[0].saved
    assert len(bund.save_molecules) == 2
    assert bund.molecules[1].elem[1] == 'He'


def test_MoleculeBundle_rearrange_all():
    mol1 = molecule.Molecule('He', np.zeros(3))
    mol2 = molecule.Molecule('Ne', np.ones(3))
    mol3 = molecule.Molecule('Ar', -np.ones(3))
    bund = molecule.MoleculeBundle([mol1, mol2, mol3])
    bund.rearrange([2, 0, 1])
    assert bund.molecules[0] == mol3
    assert bund.molecules[1] == mol1
    assert bund.molecules[2] == mol2
    assert not bund.saved


def test_MoleculeBundle_rearrange_old_ind():
    mol1 = molecule.Molecule('He', np.zeros(3))
    mol2 = molecule.Molecule('Ne', np.ones(3))
    mol3 = molecule.Molecule('Ar', -np.ones(3))
    bund = molecule.MoleculeBundle([mol1, mol2, mol3])
    bund.rearrange(1, old_ind=0)
    assert bund.molecules[0] == mol2
    assert bund.molecules[1] == mol1
    assert bund.molecules[2] == mol3
    assert not bund.saved


def test_MoleculeBundle_add_molecules():
    mol1 = molecule.Molecule('He', np.zeros(3))
    mol2 = molecule.Molecule('Ne', np.ones(3))
    mol3 = molecule.Molecule('Ar', -np.ones(3))
    bund = molecule.MoleculeBundle(mol1)
    bund.add_molecules([mol2, mol3])
    assert bund.molecules[0] == mol1
    assert bund.molecules[1] == mol2
    assert bund.molecules[2] == mol3
    assert not bund.saved


def test_MoleculeBundle_rm_molecules():
    mol1 = molecule.Molecule('He', np.zeros(3))
    mol2 = molecule.Molecule('Ne', np.ones(3))
    mol3 = molecule.Molecule('Ar', -np.ones(3))
    bund = molecule.MoleculeBundle([mol1, mol2, mol3])
    bund.rm_molecules(1)
    assert bund.nmol == 2
    assert bund.molecules[0] == mol1
    assert bund.molecules[1] == mol3
    assert not bund.saved


def test_MoleculeBundle_read_filename(tmpdir):
    f = tmpdir.join('tmp.xyz')
    f.write(ef.xyz_novec + ef.zmtvar_nocom)
    bund = molecule.MoleculeBundle()
    bund.read(str(f.realpath()))
    assert np.all(bund.molecules[0].elem[1:] == eg.ch4[0])
    assert np.all(bund.molecules[1].elem[1:] == eg.ch4[0])
    assert np.allclose(bund.molecules[0].xyz[1:], eg.ch4[1])
    assert np.allclose(bund.molecules[1].xyz[1:], eg.ch4_zmt[1])
    assert not bund.saved


def test_MoleculeBundle_read_openfile(tmpdir):
    f = ef.tmpf(tmpdir, ef.xyz_novec + ef.zmtvar_nocom)
    bund = molecule.MoleculeBundle()
    bund.read(f)
    assert np.all(bund.molecules[0].elem[1:] == eg.ch4[0])
    assert np.all(bund.molecules[1].elem[1:] == eg.ch4[0])
    assert np.allclose(bund.molecules[0].xyz[1:], eg.ch4[1])
    assert np.allclose(bund.molecules[1].xyz[1:], eg.ch4_zmt[1])
    assert not bund.saved


def test_MoleculeBundle_write_filename(tmpdir):
    f = tmpdir.join('tmp.xyz')
    mol = molecule.Molecule(*eg.ch4, comment='comment line')
    bund = molecule.MoleculeBundle([mol, mol])
    bund.write(str(f.realpath()))
    assert f.read() == 2*ef.xyz_novec


def test_MoleculeBundle_write_openfile(tmpdir):
    f = tmpdir.join('tmp.xyz')
    mol = molecule.Molecule(*eg.ch4, comment='comment line')
    bund = molecule.MoleculeBundle([mol, mol])
    bund.write(f.open(mode='w'))
    assert f.read() == 2*ef.xyz_novec


def test_MoleculeBundle_write_vec(tmpdir):
    f = tmpdir.join('tmp.xyz')
    vec = np.ones((5, 3))
    mol = molecule.Molecule(*eg.ch4, vec=vec, comment='comment line')
    bund = molecule.MoleculeBundle([mol, mol])
    bund.write(f.open(mode='w'))
    assert f.read() == 2*ef.xyz_vec


def test_MoleculeBundle_measure():
    mol1 = molecule.Molecule(eg.ch2f2[0], eg.ch2f2[1])
    mol2 = molecule.Molecule(eg.ch2f2[0], eg.ch2f2[2])
    bund = molecule.MoleculeBundle([mol1, mol2])
    stre = bund.measure('stre', 2, 3)
    assert np.allclose(stre, [np.sqrt(2), 2.04])


def test_MoleculeBundle_match_to_ref():
    mol1 = molecule.Molecule(eg.ch2f2[0], eg.ch2f2[1])
    mol2 = molecule.Molecule(eg.ch2f2[0], eg.ch2f2[2])
    mol3 = molecule.Molecule(eg.ch2f2[0], eg.ch2f2[3])
    mol4 = molecule.Molecule(eg.ch2f2[0], eg.ch2f2[4])
    mol5 = molecule.Molecule(eg.ch2f2[0], eg.ch2f2[5])
    bund1 = molecule.MoleculeBundle([mol3, mol4, mol5])
    bund2 = molecule.MoleculeBundle([mol1, mol2])
    inds = bund1.match_to_ref(bund2)
    assert np.allclose(inds, [0, 0, 1])


def test_import_molecule(tmpdir):
    f = tmpdir.join('tmp.xyz')
    f.write(ef.xyz_novec)
    mol = molecule.import_molecule(str(f.realpath()))
    assert np.all(mol.elem[1:] == eg.ch4[0])
    assert np.allclose(mol.xyz[1:], eg.ch4[1])


def test_import_bundle_single_file(tmpdir):
    f = tmpdir.join('tmp.xyz')
    f.write(ef.xyz_novec + ef.zmtvar_nocom)
    bund = molecule.import_bundle(str(f.realpath()))
    assert np.all(bund.molecules[0].elem[1:] == eg.ch4[0])
    assert np.all(bund.molecules[1].elem[1:] == eg.ch4[0])
    assert np.allclose(bund.molecules[0].xyz[1:], eg.ch4[1])
    assert np.allclose(bund.molecules[1].xyz[1:], eg.ch4_zmt[1])


def test_import_bundle_multiple_files(tmpdir):
    f1 = tmpdir.join('tmp1.xyz')
    f2 = tmpdir.join('tmp2.xyz')
    f1.write(ef.zmtvar_nocom + ef.zmtvar_nocom)
    f2.write(ef.xyz_novec)
    bund = molecule.import_bundle([str(f1.realpath()), str(f2.realpath())])
    assert np.all(bund.molecules[0].elem[1:] == eg.ch4[0])
    assert np.all(bund.molecules[1].elem[1:] == eg.ch4[0])
    assert np.all(bund.molecules[2].elem[1:] == eg.ch4[0])
    assert np.allclose(bund.molecules[0].xyz[1:], eg.ch4_zmt[1])
    assert np.allclose(bund.molecules[1].xyz[1:], eg.ch4_zmt[1])
    assert np.allclose(bund.molecules[2].xyz[1:], eg.ch4[1])

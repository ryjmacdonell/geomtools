"""
Tests for the molecule module.
"""
import pytest
import numpy as np
import gimbal.molecule as molecule


he = ('He', np.zeros(3))
c2h4 = (['C', 'C', 'H', 'H', 'H', 'H'],
        np.array([[ 0.000000,  0.000000,  0.667480],
                  [ 0.000000,  0.000000, -0.667480],
                  [ 0.000000,  0.922832,  1.237695],
                  [ 0.000000, -0.922832,  1.237695],
                  [ 0.000000,  0.922832, -1.237695],
                  [ 0.000000, -0.922832, -1.237695]]))
c3h8 = (['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
        np.array([[ 0.000000,  0.000000,  0.578619],
                  [ 0.000000,  1.266857, -0.269285],
                  [ 0.000000, -1.266857, -0.269285],
                  [-0.876898,  0.000000,  1.235614],
                  [ 0.876898,  0.000000,  1.235614],
                  [ 0.000000,  2.166150,  0.352968],
                  [ 0.000000, -2.166150,  0.352968],
                  [ 0.883619,  1.304234, -0.913505],
                  [-0.883619,  1.304234, -0.913505],
                  [-0.883619, -1.304234, -0.913505],
                  [ 0.883619, -1.304234, -0.913505]]))


def test_empty_BaseMolecule():
    mol = molecule.BaseMolecule()
    assert len(mol.elem) == 0
    assert len(mol.xyz) == 0
    assert len(mol.vec) == 0
    assert mol.comment == ''
    assert str(mol) == '[]'
    assert repr(mol) == 'BaseMolecule(\'\', [])'


def test_single_atom_BaseMolecule():
    mol = molecule.BaseMolecule(*he)
    zf = 3*'{:14.8f}'.format(0)
    ze = 3*',{:16.8e}'.format(0)
    assert mol.natm == 1
    assert np.all(mol.elem == ['He'])
    assert np.allclose(mol.xyz, np.zeros((1, 3)))
    assert str(mol) == '[[He' + zf + ']]'
    assert repr(mol) == 'BaseMolecule(\'\',\n [[He' + ze + ']])'


def test_eleven_atom_BaseMolecule():
    mol = molecule.BaseMolecule(*c3h8)
    assert mol.natm == 11
    assert np.all(mol.elem == c3h8[0])
    assert np.allclose(mol.xyz, c3h8[1])
    assert '...' in str(mol)
    assert '...' in repr(mol)


def test_BaseMolecule_xyz_not_3d():
    pass


def test_BaseMolecule_vec_not_3d():
    pass


def test_BaseMolecule_len_elem_not_equal_len_xyz():
    pass


def test_BaseMolecule_xyz_vec_different_shape():
    pass


def test_BaseMolecule_copy():
    mol1 = molecule.BaseMolecule(*c2h4)
    mol2 = mol1.copy()
    assert np.all(mol1.elem == c2h4[0])
    assert np.all(mol2.elem == c2h4[0])
    assert np.allclose(mol1.xyz, c2h4[1])
    assert np.allclose(mol2.xyz, c2h4[1])


def test_BaseMolecule_save():
    pass


def test_BaseMolecule_revert():
    pass


def test_BaseMolecule_set_geom():
    pass


def test_BaseMolecule_set_vec():
    pass


def test_BaseMolecule_set_comment():
    pass


def test_BaseMolecule_add_atoms_single():
    pass


def test_BaseMolecule_add_atoms_multiple():
    pass


def test_BaseMolecule_add_atoms_new_vec():
    pass


def test_BaseMolecule_rm_atoms_single():
    pass


def test_BaseMolecule_rm_atoms_multiple():
    pass


def test_BaseMolecule_rearrange_all():
    pass


def test_BaseMolecule_rearrange_old_ind():
    pass


def test_BaseMolecule_rearrange_fails():
    pass


def test_empty_Molecule():
    mol = molecule.Molecule()
    assert len(mol.elem) == 0
    assert len(mol.xyz) == 0
    assert len(mol.vec) == 0
    assert mol.comment == ''
    assert str(mol) == '[]'
    assert repr(mol) == 'Molecule(\'\', [])'


def test_single_atom_Molecule():
    mol = molecule.Molecule(*he)
    zf = 3*'{:14.8f}'.format(0)
    ze = 3*',{:16.8e}'.format(0)
    assert mol.natm == 1
    assert np.all(mol.elem == ['XM', 'He'])
    assert np.allclose(mol.xyz, np.zeros((2, 3)))
    assert str(mol) == '[[He' + zf + ']]'
    assert repr(mol) == ('Molecule(\'\',\n [[XM' + ze + '],\n' +
                         '  [He' + ze + ']])')


def test_eleven_atom_Molecule():
    mol = molecule.Molecule(*c3h8)
    assert mol.natm == 11
    assert np.all(mol.elem == np.hstack(('XM', c3h8[0])))
    assert np.allclose(mol.xyz, np.vstack((np.zeros(3), c3h8[1])), atol=1e-7)
    assert '...' in str(mol)
    assert '...' in repr(mol)


def test_Molecule_addition():
    mol1 = molecule.Molecule(*he)
    mol2 = molecule.Molecule(*c2h4)
    bund = mol1 + mol2
    assert np.all(bund.molecules[0].elem[1:] == he[0])
    assert np.all(bund.molecules[0].xyz[1:] == he[1])
    assert np.all(bund.molecules[1].elem[1:] == c2h4[0])
    assert np.all(bund.molecules[1].xyz[1:] == c2h4[1])


def test_Molecule_copy():
    mol1 = molecule.Molecule(*c2h4)
    mol2 = mol1.copy()
    assert np.all(mol1.elem[1:] == c2h4[0])
    assert np.all(mol2.elem[1:] == c2h4[0])
    assert np.allclose(mol1.xyz[1:], c2h4[1])
    assert np.allclose(mol2.xyz[1:], c2h4[1])


def test_Molecule_set_vec():
    pass


def test_Molecule_read_filename():
    pass


def test_Molecule_read_openfile():
    pass


def test_Molecule_write_filename():
    pass


def test_Molecule_write_openfile():
    pass


def test_Molecule_get_natm():
    pass


def test_Molecule_get_elem():
    pass


def test_Molecule_get_xyz():
    pass


def test_Molecule_get_vec():
    pass


def test_Molecule_get_comment():
    pass


def test_Molecule_get_mass():
    pass


def test_Molecule_get_formula_single_atom():
    pass


def test_Molecule_get_formula_multiple_atoms():
    pass


def test_Molecule_measure():
    pass


def test_Molecule_centre_mass():
    pass


def test_Molecule_translate():
    pass


def test_Molecule_rotate_no_vec():
    pass


def test_Molecule_rotate_vec():
    pass


def test_Molecule_match_to_ref():
    pass


def test_Molecule_subst():
    pass


def test_empty_MoleculeBundle():
    bund = molecule.MoleculeBundle()
    assert len(bund.molecules) == 0
    assert bund.nmol == 0
    assert str(bund) == '[]'
    assert repr(bund) == 'MoleculeBundle()'


def test_single_atom_MoleculeBundle():
    mol = molecule.Molecule(*he)
    bund = molecule.MoleculeBundle(mol)
    zf = 3*'{:14.8f}'.format(0)
    ze = 3*',{:16.8e}'.format(0)
    assert bund.nmol == 1
    assert str(bund) == '[[[He' + zf + ']]]'
    assert repr(bund) == ('MoleculeBundle(\n Molecule(\'\',\n  [[XM' + ze +
                          '],\n   [He' + ze + ']]))')


def test_seven_molecule_MoleculeBundle():
    mol = molecule.Molecule(*he)
    bund = molecule.MoleculeBundle(7*[mol])
    assert bund.nmol == 7
    assert '...' in str(bund)
    assert '...' in repr(bund)


def test_MoleculeBundle_addition():
    pass


def test_MoleculeBundle_iaddition():
    pass


def test_MoleculeBundle_copy():
    pass


def test_MoleculeBundle_save():
    pass


def test_MoleculeBundle_revert():
    pass


def test_MoleculeBundle_rearrange_all():
    pass


def test_MoleculeBundle_rearrange_old_ind():
    pass


def test_MoleculeBundle_add_molecules_single():
    pass


def test_MoleculeBundle_add_molecules_multiple():
    pass


def test_MoleculeBundle_rm_molecules_single():
    pass


def test_MoleculeBundle_rm_molecules_multiple():
    pass


def test_MoleculeBundle_read():
    pass


def test_MoleculeBundle_write_filename():
    pass


def test_MoleculeBundle_write_openfile():
    pass


def test_MoleculeBundle_get_nmol():
    pass


def test_MoleculeBundle_get_molecules():
    pass


def test_MoleculeBundle_measure():
    pass


def test_MoleculeBundle_match_to_ref():
    pass


def test_import_molecule():
    pass


def test_import_bundle_single_file():
    pass


def test_import_bundle_multiple_files():
    pass

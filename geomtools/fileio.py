"""
File input/output functions for molecular geometry files.

Can support XYZ, COLUMBUS and ZMAT formats. Input and output both require
an open file to support multiple geometries.
TODO: Add custom formats.
"""
import numpy as np
import geomtools.constants as con
import geomtools.displace as displace


def read_xyz(infile):
    """Reads input file in XYZ format."""
    natm = int(infile.readline())
    infile.readline()
    data = np.array([line.split() for line in infile.readlines()])
    elem = data[:natm, 0]
    xyz = data[:natm, 1:].astype(float)
    return natm, elem, xyz


def read_col(infile):
    """Reads input file in COLUMBUS format."""
    data = np.array([line.split() for line in infile.readlines()])
    natm = len(data)
    elem = data[:, 0]
    xyz = data[:, 2:-1].astype(float) * con.conv('bohr','ang')
    return natm, elem, xyz


def read_zmat(infile):
    """Reads input file in ZMAT format.

    ZMatrix files are in the format:
    A
    B 1 R
    C indR R indA A
    D indR R indA A indT T
    E indR R indA A indT T
    ...
    Where A, B, C, D, E are atomic labels, indR, indA, indT are reference
    atom indices, R are bond lengths, A are bond angles and T are dihedral
    angles. For example, E is a distance R from atom indR with an
    E-indR-indA angle of A and an E-indR-indA-indT dihedral angle of T.
    TODO: Add alternative format, where R, A, T are variables given below
    the matrix (separated by one space). This can be distinguished by
    whether R, A, T are can be converted to float.
    """
    data = [line.split() for line in infile.readlines()]
    natm = len(data) # In alternative format, ends at first blank line
    elem = np.array([line[0] for line in data])

    xyz = np.zeros((natm, 3))
    for i in range(natm):
        if i == 0:
            # leave first molecule at origin
            continue
        elif i == 1:
            # move along z-axis by R
            xyz = displace.translate(xyz, 1, float(data[1][2]), [0, 0, 1])
        elif i == 2:
            indR = int(data[2][1]) - 1
            indA = int(data[2][3]) - 1
            xyz[2] = xyz[indR]
            # move from indR towards indA by R
            xyz = displace.translate(xyz, 2, float(data[2][2]),
                                     xyz[indA]-xyz[indR])
            # rotate into xz-plane by A
            xyz = displace.rotate(xyz, 2, float(data[2][4]), [0, 1, 0],
                                  origin=xyz[indR], units='deg')
        else:
            indR = int(data[i][1]) - 1
            indA = int(data[i][3]) - 1
            indT = int(data[i][5]) - 1
            xyz[i] = xyz[indR]
            # move from indR towards indA by R
            xyz = displace.translate(xyz, i, float(data[i][2]),
                                     xyz[indA]-xyz[indR])
            # rotate about (indT-indA)x(indR-indA) by A
            xyz = displace.rotate(xyz, i, float(data[i][4]),
                                  np.cross(xyz[indT]-xyz[indA],
                                           xyz[indR]-xyz[indA]),
                                  origin=xyz[indR], units='deg')
            # rotate about indR-indA by T
            xyz = displace.rotate(xyz, i, float(data[i][6]),
                                  xyz[indR]-xyz[indA],
                                  origin=xyz[indR], units='deg')

    xyz = displace.centre_mass(elem, xyz)
    return natm, elem, xyz


def write_xyz(outfile, natm, elem, xyz, comment=''):
    """Writes geometry to an output file in XYZ format."""
    outfile.write(' {}\n{}\n'.format(natm, comment))
    for atm, pos in zip(elem, xyz):
        outfile.write('{:4s}{:12.6f}{:12.6f}{:12.6f}\n'.format(atm, *pos))


def write_col(outfile, natm, elem, xyz, comment=''):
    """Writes geometry to an output file in COLUMBUS format."""
    if comment != '':
        outfile.write(comment + '\n')
    for atm, pos in zip(elem, xyz * con.conv('ang','bohr')):
        outfile.write(' {:<2}{:7.1f}{:14.8f}{:14.8f}{:14.8f}{:14.8f}'
                      '\n'.format(atm, con.get_num(atm), *pos,
                                  con.get_mass(atm)))


def write_zmat(outfile, natm, elem, xyz, comment=''):
    """Writes geometry to an output file in ZMAT format.

    TODO: At present, each atom uses the previous atoms in order as
    references. This could be made 'smarter' using the bonding module.
    """
    if comment != '':
        outfile.write(comment + '\n')
    for i in range(natm):
        if i == 0:
            # first element has just the symbol
            outfile.write('{:<2}\n'.format(elem[0]))
        elif i == 1:
            # second element has symbol, index, bond length
            outfile.write('{:<2}{:3d}{:12.6f}'
                          '\n'.format(elem[1], 1, displace.stre(xyz, [0,1])))
        elif i == 2:
            # third element has symbol, index, bond length, index, bond angle
            outfile.write('{:<2}{:3d}{:12.6f}{:3d}{:12.6f}'
                          '\n'.format(elem[2], 2, displace.stre(xyz, [1,2]),
                                      1, displace.bend(xyz, [0,1,2],
                                                       units='deg')))
        else:
            # all other elements have symbol, index, bond length, index,
            # bond angle, index, dihedral angle
            outfile.write('{:<2}{:3d}{:12.6f}{:3d}{:12.6f}{:3d}{:12.6f}'
                          '\n'.format(elem[i], i, displace.stre(xyz, [i-1,i]),
                                      i-1, displace.bend(xyz, [i-2,i-1,i],
                                                         units='deg'),
                                      i-2, displace.tors(xyz, [i-3,i-2,i-1,i],
                                                         units='deg')))


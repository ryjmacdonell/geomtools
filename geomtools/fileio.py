"""
File input/output functions for molecular geometry files.

Can support XYZ, COLUMBUS and ZMAT formats. Input and output both require
an open file to support multiple geometries.
TODO: Finish ZMAT functions. Add custom formats.
"""
import sys
import numpy as np
import constants as con
import displace
import molecule


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
    xyz = data[:, 2:-1].astype(float) * con.a0                        
    return natm, elem, xyz


def read_zmat(infile):                                                 
    """Reads input file in ZMAT format."""                                  
    pass # this might require importing the displacement module


def write_xyz(outfile, natm, elem, xyz, comment=''):                                   
    """Writes geometry to an output file in XYZ format."""                  
    outfile.write(' {}\n{}\n'.format(natm, comment))                   
                                                                            
    for a, pos in zip(elem, xyz):                                 
        outfile.write('{:4s}{:12.6f}{:12.6f}{:12.6f}\n'.format(a, *pos))    


def write_col(outfile, natm, elem, xyz, comment=''):                                   
    """Writes geometry to an output file in COLUMBUS format."""             
    if comment != '':                                                       
        outfile.write(comment + '\n')                               
                                                                            
    for a, pos in zip(elem, xyz / con.a0):                       
        outfile.write(' {:<2}{:7.1f}{:14.8f}{:14.8f}{:14.8f}{:14.8f}'       
                      '\n'.format(a, con.get_num(a), *pos, con.get_mass(a)))


def write_zmat(outfile, natm, elem, xyz, comment=''):                                  
    """Writes geometry to an output file in ZMAT format.
    
    This could be made 'smarter' using the bonding module."""                 
    if comment != '':                                                           
        outfile.write(comment + '\n')

    for i in range(natm):
        if i == 0:
            # first element has just the symbol
            outfile.write('{:<2}\n'.format(elem[0]))
        elif i == 1:
            # second element has symbol, index, bond length
            outfile.write('{:<2}{:3d}{:12.6f}'
                          '\n'.format(elem[1], 1, molecule.stre(xyz, [0,1])))
        elif i == 2:
            # third element has symbol, index, bond length, index, bond angle
            outfile.write('{:<2}{:3d}{:12.6f}{:3d}{:12.6f}'
                          '\n'.format(elem[2], 2, molecule.stre(xyz, [1,2]),
                                      1, molecule.bend(xyz, [0,1,2],
                                                       units='deg')))
        else:
            # all other elements have symbol, index, bond length, index,
            # bond angle, index, dihedral angle
            outfile.write('{:<2}{:3d}{:12.6f}{:3d}{:12.6f}{:3d}{:12.6f}'
                          '\n'.format(elem[i], i, molecule.stre(xyz, [i-1,i]),
                                      i-1, molecule.bend(xyz, [i-2,i-1,i], 
                                                         units='deg'), 
                                      i-2, molecule.tors(xyz, [i-3,i-2,i-1,i], 
                                                         units='deg')))


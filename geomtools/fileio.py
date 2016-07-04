"""
File input/output functions for molecular geometry files.

Can support XYZ, COLUMBUS and ZMAT formats. Input and output both require
an open file to support multiple geometries.
TODO: Finish ZMAT functions. Add custom formats.
"""
import sys
import numpy as np
import displace


# Global constants
a0 = 0.52917721092                                                 


def get_num(elem):
    """Returns atomic number from atomic symbol."""
    num = {'X':0, 'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8,
           'F':9, 'Ne':10, 'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16,
           'Cl':17, 'Ar':18}               
    return num[elem]


def get_mass(elem):
    """Returns atomic mass from atomic symbol."""
    mass = {'X':0.00000000, 'H':1.00782504, 'He':4.00260325, 'Li':7.01600450,
            'Be':9.01218250, 'B':11.00930530, 'C':12.00000000, 'N':14.00307401,
            'O':15.99491464, 'F':18.99840325, 'Ne':19.99243910, 
            'Na':22.98976970, 'Mg':23.98504500, 'Al':26.98154130, 
            'Si':27.97692840, 'P':30.97376340, 'S':31.97207180, 
            'Cl':34.96885273, 'Ar':39.96238310}
    return mass[elem]


def read_xyz(infile):                                                  
    """Reads input file in XYZ format."""                                   
    natm = int(infile.readline())                                     
    infile.readline()                                                      
    data = np.array([line.split() for line in infile.readlines()])         
                                                                            
    elem = data[:self.natm, 0]                                         
    xyz = data[:self.natm, 1:].astype(float)                           
    return natm, elem, xyz
                                                                            

def read_col(infile):                                                  
    """Reads input file in COLUMBUS format."""                              
    data = np.array([line.split() for line in infile.readlines()])         
                                                                            
    natm = len(data)                                                   
    elem = data[:, 0]                                                  
    xyz = data[:, 2:-1].astype(float) * self.a0                        
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
    global a0
    if comment != '':                                                       
        outfile.write('{}\n'.format(comment))                               
                                                                            
    for a, pos in zip(elem, xyz / a0):                       
        outfile.write(' {:<2}{:7.1f}{:14.8f}{:14.8f}{:14.8f}{:14.8f}'       
                      '\n'.format(a, get_num(a), *pos, get_mass(a)))


def write_zmat(outfile, natm, elem, xyz, comment=''):                                  
    """Writes geometry to an output file in ZMAT format."""                 
    pass # this is relatively easy

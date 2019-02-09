#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2018 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#
from psi4 import core
from psi4 import driver
#from psi4.driver import driver
from psi4.driver.p4util import *

# Necessary conversion factor
au_to_j = qcel.constants.get('atomic unit of mass')
amu_to_j = qcel.constants.get('atomic mass constant')
au_to_amu = au_to_j / amu_to_j

def calc_wfn_overlap(ciwfn_m, S, ciwfn_p):
    """ calculate the overlap between wfn_m and wfn_p"""    
    # Assume RHF for now
    Ca_m = ciwfn_m.Ca().to_array()
    Ca_p = ciwfn_p.Ca().to_array()
    mo_overlap = Ca_m.T @ S @ Ca_p
    #print(mo_overlap)
    mo_overlap = np.diagonal(mo_overlap)
    
    #orbitals = ciwfn_m.get_orbitals("ALL").to_array()
    #print(Ca_m)
    #print(orbitals)
    #print(Ca_m-orbitals)
    dvec_m = ciwfn_m.D_vector()
    dvec_p = ciwfn_p.D_vector()
    
    na = ciwfn_m.nalpha()
    nb = ciwfn_m.nbeta()

    block = 0

    dvec_m.init_io_files(True)
    block_coeffs_m = np.array(ciwfn_m.get_coeffs(dvec_m,0,block))
    block_occs_m = np.array(ciwfn_m.get_occs(dvec_m,0,block)).reshape((len(block_coeffs_m),na+nb))
    dvec_m.close_io_files(True)

    dvec_p.init_io_files(True)
    block_coeffs_p = np.array(ciwfn_p.get_coeffs(dvec_p,0,block))
    block_occs_p = np.array(ciwfn_p.get_occs(dvec_p,0,block)).reshape((len(block_coeffs_p),na+nb))
    dvec_p.close_io_files(True)

    total_overlap = 0.0
    coeff_check = 0.0

    while len(block_coeffs_m) > 0:

        #print('MINUS BLOCK {:d} has length {:d} / {:d}'.format(block,len(block_coeffs_m),len(block_occs_m)))
        #print('PLUS  BLOCK {:d} has length {:d} / {:d}'.format(block,len(block_coeffs_p),len(block_occs_p)))
        
        #print(np.amin(block_coeffs_m - block_coeffs_p))
        #print(np.amax(block_coeffs_m - block_coeffs_p))

        #print(np.amax(block_coeffs_m))
        #print(np.amax(block_coeffs_p))

        #print(np.amin(block_coeffs_m))
        #print(np.amin(block_coeffs_p))

        block += 1

        for det in range(len(block_coeffs_m)):

            det_overlap = block_coeffs_m[det]*block_coeffs_p[det]
            if abs(det_overlap) < 1.0e-12:
                continue
            coeff_check += block_coeffs_m[det]*block_coeffs_p[det]
            for ei in range(na+nb):
                orb = block_occs_m[det, ei]
                det_overlap *= mo_overlap[orb]
                if orb != block_occs_p[det,ei]:
                    print('ERROR in det {:d} of block {:d}'.format((det,block-1)))
                    print('  minus ', block_occs_m[det])
                    print('  plus ', block_occs_p[det])
            
            total_overlap += det_overlap

        dvec_m.init_io_files(True)
        block_coeffs_m = np.array(ciwfn_m.get_coeffs(dvec_m,0,block))
        block_occs_m = np.array(ciwfn_m.get_occs(dvec_m,0,block)).reshape((len(block_coeffs_m),na+nb))
        dvec_m.close_io_files(True)

        dvec_p.init_io_files(True)
        block_coeffs_p = np.array(ciwfn_p.get_coeffs(dvec_p,0,block))
        block_occs_p = np.array(ciwfn_p.get_occs(dvec_p,0,block)).reshape((len(block_coeffs_p),na+nb))
        dvec_p.close_io_files(True)

    print('checking coefficient normalization: ', coeff_check) 
    return total_overlap

def calc_wfn_overlap_scf(wfn_m, S, wfn_p):
    """ calculate the overlap between wfn_m and wfn_p"""    
    # Assume RHF for now
    Ca_m = wfn_m.Ca().to_array()
    Ca_p = wfn_p.Ca().to_array()
    mo_overlap = Ca_m.T @ S @ Ca_p
    wfn_overlap = 1.0
    for e in range(wfn_m.nalpha()):
        wfn_overlap *= mo_overlap[e,e]
    return wfn_overlap ** 2

def calc_dboc(wfn):
    """
        Main driver for managing Raman Optical activity computations with
        CC response theory.

        Uses distributed finite differences approach -->
            1. Sets up a database to keep track of running/finished/waiting
                computations.
            2. Generates separate input files for displaced geometries.
            3. When all displacements are run, collects the necessary information
                from each displaced computation, and computes final result.
    """

    # Find better ways of ensuring these conditions in the future
    if wfn.name() == 'CIWavefunction':
        print('Doing CI Calculation')
    elif wfn.name() in ['RHF, ROHF, UHF']:
        print('Doing HF Calculation')
    else:
        print('Cannot do DBOC for {:s} wavefunction'.format(wfn.name()))
   
    if wfn.molecule().units() != 'Bohr':
        print('Error: Must be in units of Bohr')
    if not wfn.molecule().schoenflies_symbol() != 'C1':
        print('Error: Must be in C1 symmetry')
    if not wfn.molecule().com_fixed():
        print('Error: COM must be fixed')
    if not wfn.molecule().orientation_fixed():
        print('Error: Orientation must be fixed')

    
    # Finite Difference displacement
    dR = 5.0e-4
    print("Performing DBOC with dR: ", dR, " Bohr")

    # Define these for conveniece
    mol = wfn.molecule()
    geom = mol.geometry()
    init_coords = geom.to_array()
    
    e_dboc = 0.0
    mints = core.MintsHelper(wfn)
    
    # generate a minus (-) and plus (+) displacement per nuclear coord
    for ai, atom in enumerate(init_coords):
        for dim in range(3):
            
            # finite difference minus displacement
            geom_m = np.array(init_coords)
            geom_m[ai,dim] -= dR
            mol.set_geometry(core.Matrix.from_array(geom_m))
            
            # get wavefunction AND basis set
            e_m, ref_wfn_m = driver.energy('scf',return_wfn=True)
            ciwfn_m = core.detci(ref_wfn_m)
            basis_m = ciwfn_m.basisset()
    
            # finite difference plus displacement
            geom_p = np.array(init_coords)
            geom_p[ai,dim] += dR
            mol.set_geometry(core.Matrix.from_array(geom_p))
            
            # get wavefunction AND basis set
            e_m, ref_wfn_p = driver.energy('scf',return_wfn=True)
            ciwfn_p = core.detci(ref_wfn_p)
            basis_p = ciwfn_p.basisset()
            
            # The overlap matrix between Psi_m and Psi_p
            S_pm = mints.ao_overlap(basis_m,basis_p).to_array()
            overlap_pm = calc_wfn_overlap(ciwfn_m, S_pm, ciwfn_p)
            
            print("1-S", 1 - overlap_pm)
            e_dboc_comp = (1-overlap_pm) / mol.mass(ai) 
            e_dboc += e_dboc_comp

            ciwfn_m.cleanup_ci()
            ciwfn_m.cleanup_dpd()
            ciwfn_p.cleanup_ci()
            ciwfn_p.cleanup_dpd()
    
    e_dboc /= (2*dR)**2
    e_dboc *= au_to_amu

    print("Total E_DBOC: ", e_dboc, "Eh")
    print("Total E_DBOC: ", e_dboc*219474.631371, "cm^-1")

    print('Resetting Coordinates to:')
    print(init_coords)
    mol.set_geometry(core.Matrix.from_array(init_coords))

    return e_dboc

def calc_dboc_scf(wfn):
    """
        Main driver for managing Raman Optical activity computations with
        CC response theory.

        Uses distributed finite differences approach -->
            1. Sets up a database to keep track of running/finished/waiting
                computations.
            2. Generates separate input files for displaced geometries.
            3. When all displacements are run, collects the necessary information
                from each displaced computation, and computes final result.
    """

    # Find better ways of ensuring these conditions in the future
    if wfn.name() == 'CIWavefunction':
        print('Doing CI Calculation')
    elif wfn.name() in ['RHF, ROHF, UHF']:
        print('Doing HF Calculation')
    else:
        print('Cannot do DBOC for {:s} wavefunction'.format(wfn.name()))
   
    if wfn.molecule().units() != 'Bohr':
        print('Error: Must be in units of Bohr')
    if not wfn.molecule().schoenflies_symbol() != 'C1':
        print('Error: Must be in C1 symmetry')
    if not wfn.molecule().com_fixed():
        print('Error: COM must be fixed')
    if not wfn.molecule().orientation_fixed():
        print('Error: Orientation must be fixed')

    
    # Finite Difference displacement
    dR = 5.0e-4
    print("Performing DBOC with dR: ", dR, " Bohr")

    # Define these for conveniece
    mol = wfn.molecule()
    geom = mol.geometry()
    init_coords = geom.to_array()
    
    e_dboc = 0.0
    mints = core.MintsHelper(wfn)
    
    # generate a minus (-) and plus (+) displacement per nuclear coord
    for ai, atom in enumerate(init_coords):
        for dim in range(3):
            
            # finite difference minus displacement
            geom_m = np.array(init_coords)
            geom_m[ai,dim] -= dR
            mol.set_geometry(core.Matrix.from_array(geom_m))
            
            # get wavefunction AND basis set
            e_m, wfn_m = driver.energy('scf',return_wfn=True)
            basis_m = wfn_m.basisset()
    
            # finite difference plus displacement
            geom_p = np.array(init_coords)
            geom_p[ai,dim] += dR
            mol.set_geometry(core.Matrix.from_array(geom_p))
            
            # get wavefunction AND basis set
            e_p, wfn_p = driver.energy('scf',return_wfn=True)
            basis_p = wfn_p.basisset()
            
            # The overlap matrix between Psi_m and Psi_p
            S_pm = mints.ao_overlap(basis_m,basis_p).to_array()
            overlap_pm = calc_wfn_overlap_scf(wfn_m, S_pm, wfn_p)
            
            #print("1-S", 1 - overlap_pm)
            e_dboc_comp = (1-overlap_pm) / mol.mass(ai) 
            e_dboc += e_dboc_comp
    
    e_dboc /= (2*dR)**2
    e_dboc *= au_to_amu

    print("Total E_DBOC: ", e_dboc, "Eh")
    print("Total E_DBOC: ", e_dboc*219474.631371, "cm^-1")

    print('Resetting Coordinates to:')
    print(init_coords)
    mol.set_geometry(core.Matrix.from_array(init_coords))

    return e_dboc

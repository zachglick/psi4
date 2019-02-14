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

from numpy import linalg

# Necessary conversion factor
au_to_j = qcel.constants.get('atomic unit of mass')
amu_to_j = qcel.constants.get('atomic mass constant')
au_to_amu = au_to_j / amu_to_j


class CIVect:
    
    def __init__(self, scf_wfn):

        self.scf_wfn = scf_wfn


def compare_orbitals(Ca_m, Ca_p):
    
    num = abs(Ca_p - Ca_m)
    num[num < 1.0e-10] = 0.0

    den = (abs(Ca_p) + abs(Ca_m))/2

    norm = num/den
    norm[num < 1.0e-4] = 0.0
    
    np.set_printoptions(suppress=True)
    print('(+) orbitals: ')
    print(np.round(Ca_p,3))
    print('(-) orbitals: ')
    print(np.round(Ca_m,3))
    print('The normalized difference between the (+) and (-) orbitals (should be small)')
    print(np.round(norm,4))
    np.set_printoptions(suppress=False)

def reorder_rows(S_mo):
    """
    S_mo is a MO x MO matrix describing the overlap between the (-) and (+) rows
    """
    signs = np.zeros(len(S_mo))
    order = np.zeros(len(S_mo), dtype=int)
    for i, row in enumerate(S_mo):
        ind = np.argmax(abs(row))
        if row[ind] < 0:
            signs[ind] = -1
        else:
            signs[ind] = 1
        order[ind] = i
    return signs, order

    

def read_dets(ciwfn):

    block = 0
    dvec = ciwfn.D_vector()
    dvec.init_io_files(True)
    
    coeffs = []
    occs = []
    #read coeffs/occs a block at a time
    while(True):

        coeffs_block = ciwfn.get_coeffs(dvec,0,block)
        occs_block = ciwfn.get_occs(dvec,0,block)

        if len(coeffs_block) is 0:
            break

        coeffs.extend(coeffs_block)
        occs.extend(occs_block)

        block += 1

    dvec.close_io_files(True)

    coeffs = np.array(coeffs)
    occs = np.array(occs).reshape((len(coeffs),ciwfn.nalpha()+ciwfn.nbeta()))
    
    #print("Pyt First coefficient: {:.20f}".format(coeffs[0]))
    return coeffs, occs
def calc_det(mat):

    if len(mat) == 1:
        return mat[0,0]
    elif len(mat) == 2:
        return mat[0,0]*mat[1,1]-mat[0,1]*mat[1,0]

def get_det(S_mo, occ_m, occ_p, na, nb):

    a_m = occ_m[:na]
    a_p = occ_p[:na]
    b_m = occ_m[na:]
    b_p = occ_p[na:]

    a_mat = S_mo[a_m][:,a_p]
    b_mat = S_mo[b_m][:,b_p]

    a_det = np.linalg.det(a_mat)
    b_det = np.linalg.det(b_mat)
    #a_det = calc_det(a_mat)
    #b_det = calc_det(b_mat)

    total_det = a_det*b_det

    str_m = ' '.join([str(elem) + "a" for elem in a_m] + [str(elem) + "b" for elem in b_m])
    str_p = ' '.join([str(elem) + "a" for elem in a_p] + [str(elem) + "b" for elem in b_p])

    #print('Calculating determinant between occupations:')
    #print(' (-) {:s}'.format(str_m))
    #print(' (+) {:s}'.format(str_p))
    #print('Alpha orbital submatrix:')
    #print(a_mat)
    #print('Beta orbital submatrix:')
    #print(b_mat)
    #print('|A|*|B| = {:.10f} * {:.10f} = {:.10f}'.format(a_det,b_det,total_det))

    return total_det


def calc_wfn_overlap(coeffs_m, occs_m, S_mo, coeffs_p, occs_p, na, nb):
    """ calculate the overlap between wfn_m and wfn_p"""    

    ndet = len(coeffs_m)

    S_det = np.zeros((ndet,ndet))

    for im, occ_m in enumerate(occs_m):
        for ip, occ_p in enumerate(occs_p):
            S_det[im,ip] = get_det(S_mo, occ_m, occ_p, na, nb)


    #print(S_det)

    total_overlap = 0.0

    for im, cm in enumerate(coeffs_m):
        for ip, cp in enumerate(coeffs_p):
            total_overlap += cm*S_det[im,ip]*cp

    return total_overlap

def calc_dboc(wfn):

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

    # Define these for convenience
    mol = wfn.molecule()
    geom = mol.geometry()
    init_coords = geom.to_array()
    
    e_dboc = 0.0
    mints = core.MintsHelper(wfn)
    
    # generate a minus (-) and plus (+) displacement per nuclear coord
    for ai, atom in enumerate(init_coords):
        for dim in range(3):

            # finite difference for the (-) displacement
            geom_m = np.array(init_coords)
            geom_m[ai,dim] -= dR
            mol.set_geometry(core.Matrix.from_array(geom_m))
            print('computing (-) displacement for atom {:d} in dimension {:d}:'.format(ai,dim))
            print(mol.geometry().to_array())
            
            # Get a CI wavefunction AND the basis / MOs / CI determinants
            e_m, ref_wfn_m = driver.energy('scf',return_wfn=True)
            ciwfn_m = core.detci(ref_wfn_m)
            basis_m = ciwfn_m.basisset()
            Ca_m = ciwfn_m.Ca().to_array()
            coeffs_m, occs_m = read_dets(ciwfn_m)

            # Get alpha and beta electron counts (assume RHF for now)
            na = ciwfn_m.nalpha()
            nb = ciwfn_m.nbeta()
    
            # Finite difference for the (+) displacement
            geom_p = np.array(init_coords)
            geom_p[ai,dim] += dR
            mol.set_geometry(core.Matrix.from_array(geom_p))
            print('computing (+) displacement for atom {:d} in dimension {:d}:'.format(ai,dim))
            print(mol.geometry().to_array())
            
            # Get a CI wavefunction AND the basis / MOs / CI determinants
            e_p, ref_wfn_p = driver.energy('scf',return_wfn=True)
            ciwfn_p = core.detci(ref_wfn_p)
            basis_p = ciwfn_p.basisset()
            Ca_p = ciwfn_p.Ca().to_array()
            coeffs_p, occs_p = read_dets(ciwfn_p)

            # Make sure the reference determinant coefficients are in phase sign-wise
            if coeffs_m[0] * coeffs_p[0] < 0:
                coeffs_p = -1 * coeffs_p
           
            # Switch to string block order (shouldn't affect anything)
            #new_order = np.argsort(10*occs_m[:,0]+occs_m[:,1])
            #occs_m = occs_m[new_order]
            #coeffs_m = coeffs_m[new_order]
            #occs_p = occs_p[new_order]
            #coeffs_p = coeffs_p[new_order]
            
            # Print coefficients and occupations
            #print(np.round(np.hstack((np.array([coeffs_m]).T,occs_m)),3))
            #print(np.round(np.hstack((np.array([coeffs_p]).T,occs_p)),3))
            #print(np.hstack((np.array([coeffs_m]).T,np.array([coeffs_p]).T,occs_m)))
            print('CI coefficients: [ (-), (+), and difference]')
            np.set_printoptions(suppress=True)
            print(np.round(np.hstack((np.array([coeffs_m]).T,np.array([coeffs_p]).T,np.array([coeffs_p-coeffs_m]).T)),10))
            np.set_printoptions(suppress=False)

            # Make sure the orbitals are in the same order
            for row, orbs_m in enumerate(occs_m):
                orbs_p = occs_p[row]
                for col, orb_m in enumerate(orbs_m):
                    orb_p = orbs_p[col]
                    if orb_p != orb_m:
                        print('!!! ERROR in row {:d} of vector !!!'.format(row))
                        print(orbs_m)
                        print(orbs_p)
            if ciwfn_p.nalpha() != na or ciwfn_p.nbeta() != nb:
                print('!!! ERROR Different numbers of alpha and beta electrons!!!')
            print('eps')
            print(ciwfn_m.epsilon_a().to_array())
            print(ciwfn_p.epsilon_a().to_array())
            
            # See how similar the orbitals are between the two wavefunctions
            compare_orbitals(Ca_m, Ca_p)

            # Get the overlap (in terms of both AOS and MOS) between (-) and (+)
            S_ao = mints.ao_overlap(basis_m,basis_p).to_array()
            S_mo = Ca_m.T @ S_ao @ Ca_p 
            
            print('MO Overlap between (-) and (+):')
            print(np.round(S_mo,3))

            # reorder the MOS to get maximum overlap
            signs, order = reorder_rows(S_mo)
            print(signs)
            print(order)

            #for i, sign in enumerate(signs):
            #    if sign < 0:
            #       S_mo[i] = -1*S_mo[i] 
            #S_mo = S_mo[order]
            #signs, order = reorder_rows(S_mo)
            #print(signs)
            #print(order)

            #print('AO Overlap between (-) and (+):')
            #print(np.round(S_ao,3))
            print('MO Overlap between (-) and (+):')
            print(np.round(S_mo,3))

            overlap_mp = calc_wfn_overlap(coeffs_m, occs_m, S_mo, coeffs_p, occs_p, na, nb)
            
            print("1-S", 1 - overlap_mp)
            e_dboc_comp = (1-overlap_mp) / mol.mass(ai) 
            e_dboc += e_dboc_comp

            ciwfn_m.cleanup_ci()
            ciwfn_m.cleanup_dpd()

            ciwfn_p.cleanup_ci()
            ciwfn_p.cleanup_dpd()
    
    e_dboc /= (2*dR)**2
    e_dboc *= au_to_amu

    print("Total E_DBOC: ", e_dboc, "Eh")
    print("Total E_DBOC: ", e_dboc * 219474.631371, "cm^-1")

    mol.set_geometry(core.Matrix.from_array(init_coords))

    return e_dboc

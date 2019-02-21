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
mass_e = qcel.constants.get('electron mass in u')

class CIVect:
    
    def __init__(self, scf_wfn):

        # basic wfn properties
        self.scf_wfn = scf_wfn
        self.na = self.scf_wfn.nalpha()
        self.nb = self.scf_wfn.nbeta()
        self.Ca = self.scf_wfn.Ca().to_array()
        self.basis = self.scf_wfn.basisset()
        
        # do CI and store the results
        self.ci_wfn = core.detci(self.scf_wfn)
        self.read_CIDets(0)
        self.ci_wfn.cleanup_ci()
        self.ci_wfn.cleanup_dpd()
    
    # read the blocked CI coefficients / alpha-beta occupations from disk
    def read_CIDets(self, root):

        block = 0
        dvec = self.ci_wfn.D_vector()
        dvec.init_io_files(True)
        
        coeffs = []
        occs = []

        # read coeffs/occs a block at a time
        while(True):

            coeffs_block = self.ci_wfn.get_coeffs(dvec,root,block)
            occs_block = self.ci_wfn.get_occs(dvec,root,block)

            if len(coeffs_block) is 0:
                break

            coeffs.extend(coeffs_block)
            occs.extend(occs_block)
            block += 1

        dvec.close_io_files(True)

        self.ndet = len(coeffs)

        self.coeffs = np.array(coeffs)
        self.occs_a = np.array(occs).reshape( (self.ndet,self.na+self.nb) )[:,:self.na]
        self.occs_b = np.array(occs).reshape( (self.ndet,self.na+self.nb) )[:,self.na:]



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
            signs[i] = -1
        else:
            signs[i] = 1
        order[ind] = i
    return signs, order

def calc_wfn_overlap(vec_m, S_mo, vec_p):
    """ calculate the overlap between wfn_m and wfn_p"""    

    #S_mo = ( S_mo.T + S_mo ) / 2.0

    ndet = vec_m.ndet
    na = vec_m.na
    nb = vec_m.nb

    #print('S_mo')
    #print(S_mo)

    # Overlap of the (-) and (+) CI wfns
    S_det = np.zeros((ndet,ndet))

    for im in range(ndet):
        a_m = vec_m.occs_a[im]
        b_m = vec_m.occs_b[im]
        for ip in range(ndet):
            a_p = vec_p.occs_a[ip]
            b_p = vec_p.occs_b[ip]
            
            # Alpha and Beta 'submatrices' of S_det
            a_mat = S_mo[a_m][:,a_p]
            b_mat = S_mo[b_m][:,b_p]

            a_det = np.linalg.det(a_mat)
            b_det = np.linalg.det(b_mat)
            S_det[im,ip] = a_det * b_det

            #print('Calculating determinant between occupations:')
            #str_m = ' '.join([str(elem) + "a" for elem in a_m] + [str(elem) + "b" for elem in b_m])
            #str_p = ' '.join([str(elem) + "a" for elem in a_p] + [str(elem) + "b" for elem in b_p])
            #print(' (-) {:s}'.format(str_m))
            #print(' (+) {:s}'.format(str_p))
            #print('Alpha orbital submatrix:')
            #print(a_mat)
            #print('Beta orbital submatrix:')
            #print(b_mat)
            #print('|A|*|B| = {:.10f} * {:.10f} = {:.10f}'.format(a_det,b_det,total_det))

    #print(S_det)
    #for i in [0,7,11,15]:
    #to_flip = list(range(19,100)) + [0]
    #to_flip = list(range(0,19))
    #print(to_flip)
    #to_flip = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    #for i in range(ndet):
    #    aorb = vec_m.occs_a[i][0]
    #    borb = vec_m.occs_b[i][0]
    #    if aorb <= borb:
    #        S_det[i] = -1*S_det[i]
    #        S_det[:,i] = -1*S_det[:,i]
    #print(S_det)



    #S_sign = np.zeros((ndet,ndet), dtype=int)
    #C_sign = np.zeros((ndet,ndet), dtype=int)
    #CSC_sign = np.zeros((ndet,ndet), dtype=int)

    #for im in range(ndet):
    #    for ip in range(ndet):
    #        
    #        THRESH = 1.0e-15

    #        if abs(S_det[im,ip]) < THRESH:
    #            continue
    #        elif S_det[im,ip] < 0.0:
    #            S_sign[im,ip] = -1
    #        else:
    #            S_sign[im,ip] = 1
    #        
    #        CC = vec_m.coeffs[im] * vec_p.coeffs[ip]

    #        if abs(CC) < THRESH:
    #            continue
    #        elif CC < 0.0:
    #            C_sign[im,ip] = -1
    #        else:
    #            C_sign[im,ip] = 1


    #        CSC_sign[im,ip] = S_sign[im,ip]*C_sign[im,ip]
    #print(S_sign)
    #print(C_sign)
    #print(CSC_sign)

    total_overlap = 0.0

    #print('S_det')
    #print(S_det)
    for im, cm in enumerate(vec_m.coeffs):
        for ip, cp in enumerate(vec_p.coeffs):
            S_det[im,ip] = cm*S_det[im,ip]*cp
            total_overlap += S_det[im,ip]

    #print('CS_detC')
    #S_det = ( S_det.T + S_det ) / 2.0
    #S_det[abs(S_det) < 10e-15] = 0.0
    #np.set_printoptions(precision=0)
    #print(S_det)
    #np.set_printoptions(precision=8)


    return total_overlap

def calc_dboc(wfn):

    # DBOC only under specific input conditions (for now)
    if wfn.name() != 'RHF':
        raise Exception('Cannot do DBOC for {:s} wavefunction'.format(wfn.name()))
    if wfn.molecule().units() != 'Bohr':
        raise Exception('Must be in units of Bohr')
    if not wfn.molecule().schoenflies_symbol() != 'C1':
        raise Exception('Must be in C1 symmetry')
    if not wfn.molecule().com_fixed():
        raise Exception('COM must be fixed')
    if not wfn.molecule().orientation_fixed():
        raise Exception('Orientation must be fixed')
    
    # Finite Difference displacement
    dR = 5.0e-4
    print("Performing DBOC with dR: ", dR, " a.u.")

    # Define these for convenience
    mints = core.MintsHelper(wfn)
    mol = wfn.molecule()
    init_geom = mol.geometry().to_array()
    
    e_dboc = 0.0
    
    # generate a minus (-) and plus (+) displacement per nuclear coord
    for ai, atom in enumerate(init_geom):

        # The mass of the nucleus
        mass_nuc = mol.mass(ai) - mass_e * mol.true_atomic_number(ai)

        for dim in range(3):
            
            print("displacement {:d} of {:d}".format(3*ai+dim+1,3*len(init_geom)))

            # finite difference for the (-) displacement
            geom_m = np.array(init_geom)
            geom_m[ai,dim] -= dR
            mol.set_geometry(core.Matrix.from_array(geom_m))
           
            # make a CIVect
            e_m, ref_wfn_m = driver.energy('scf',return_wfn=True)
            vec_m = CIVect(ref_wfn_m)
    

            # Finite difference for the (+) displacement
            geom_p = np.array(init_geom)
            geom_p[ai,dim] += dR
            mol.set_geometry(core.Matrix.from_array(geom_p))
            
            # make a CIVect
            e_p, ref_wfn_p = driver.energy('scf',return_wfn=True)
            vec_p = CIVect(ref_wfn_p)


            # Get the overlap (in both AO and MO space) between (-) and (+)
            S_ao = mints.ao_overlap(vec_m.basis,vec_p.basis).to_array()
            S_mo = vec_m.Ca.T @ S_ao @ vec_p.Ca 
            
            #print('AO Overlap between (-) and (+):')
            #print(np.round(S_ao,7))
            #print('MO Overlap between (-) and (+):')
            #print(np.round(S_mo,7))

            ## reorder the MOS of (-) to get maximum overlap
            #signs, order = reorder_rows(S_mo)
            #print('before')
            #print('signs ' + str(signs))
            #print('order' + str(order))

            #for i, sign in enumerate(signs):
            #    if sign < 0:
            #        vec_m.Ca[:,i] = -1*vec_m.Ca[:,i] 
            #vec_m.Ca = vec_m.Ca.T[order].T

            #ref_wfn_m.set_Ca(core.Matrix.from_array(vec_m.Ca))
            #vec_m = CIVect(ref_wfn_m)

            #S_mo = vec_m.Ca.T @ S_ao @ vec_p.Ca 

            #signs, order = reorder_rows(S_mo)
            #print('after')
            #print('signs ' + str(signs))
            #print('order' + str(order))

            # Make sure the reference determinant coefficients are in phase sign-wise
            if vec_m.coeffs[0] * vec_p.coeffs[0] < 0:
                print('Reversed the (-) CI coefficients')
                vec_m.coeffs = -1 * vec_m.coeffs

            # Make sure the CI determinants are in the same order
            if not ( np.array_equal(vec_m.occs_a,vec_p.occs_a) and np.array_equal(vec_m.occs_b,vec_p.occs_b) ):
                raise Exception("CI Determinants in different order")
            
            print('CI Coefficients of (-) and (+)')
            print(np.hstack((np.array([vec_m.coeffs]).T,np.array([vec_p.coeffs]).T,vec_m.occs_a,vec_m.occs_b)))

            overlap_mp = calc_wfn_overlap(vec_m, S_mo, vec_p)
            
            print("1 - <(+)|(-)>", 1 - overlap_mp )
            e_dboc_comp = (1-overlap_mp) / mass_nuc
            e_dboc += e_dboc_comp

            # See how similar the orbitals are between the two wavefunctions
            #compare_orbitals(vec_m.Ca, vec_p.Ca)

            # Print coefficients and occupations
            #print(np.round(np.hstack((np.array([coeffs_m]).T,occs_m)),3))
            #print(np.round(np.hstack((np.array([coeffs_p]).T,occs_p)),3))

            #print('CI coefficients: [ (-), (+), and difference]')
            #np.set_printoptions(suppress=True)
            #print(np.round(np.hstack((np.array([coeffs_m]).T,np.array([coeffs_p]).T,np.array([coeffs_p-coeffs_m]).T)),10))
            #np.set_printoptions(suppress=False)
    
    e_dboc /= (2*dR)**2
    e_dboc *= au_to_amu

    print("Total E_DBOC: ", e_dboc, "Eh")
    print("Total E_DBOC: ", e_dboc * 219474.631371, "cm^-1")

    mol.set_geometry(core.Matrix.from_array(init_geom))

    return e_dboc

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

# Helper class to collect DETCI info from C++
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

    def read_CIDets(self, root):

        block = 0
        dvec = self.ci_wfn.D_vector()
        dvec.init_io_files(True)
        
        coeffs = []
        str_inds = []
        occs_a = []
        occs_b = []

        # read coeffs/occs a block at a time
        while(True):
            
            # the coefficients of each determinant in this block
            coeffs_block = self.ci_wfn.get_coeffs(dvec,root,block)
            if len(coeffs_block) is 0:
                break
            # for each determinant, the alpha and beta string index that describes it
            str_inds_block = self.ci_wfn.get_string_inds(dvec,root,block)
            # for all alpha and beta strings in this block, the orbital occupations defining those strings
            occs_block = self.ci_wfn.get_string_occs(dvec,root,block)
            # first two indices give number of alpha strings, number of beta strings
            nstr_a_block = occs_block[0]
            nstr_b_block = occs_block[1]
            # trim the alpha/beta string count off
            occs_block = occs_block[2:]
            # each string is read as a 'string index' and na or nb 'orbital indices'
            len_a_block = nstr_a_block * (self.na + 1)
            len_b_block = nstr_b_block * (self.nb + 1)
            assert len(occs_block) == (len_a_block + len_b_block)
            occs_a_block = occs_block[:len_a_block]
            occs_b_block = occs_block[len_a_block:]

            coeffs.extend(coeffs_block)
            str_inds.extend(str_inds_block)
            occs_a.extend(occs_a_block)
            occs_b.extend(occs_b_block)

            block += 1

        dvec.close_io_files(True)

        self.ndet = len(coeffs)
        self.coeffs = np.array(coeffs)

        # some of the strings recorded in occs_a and occs_b may be duplicates
        # get the max alpha and beta string index, which must be the number of strings - 1

        occs_a = np.array(occs_a).reshape( (int(len(occs_a)/(self.na + 1)), self.na + 1 ) )
        occs_b = np.array(occs_b).reshape( (int(len(occs_b)/(self.nb + 1)), self.nb + 1 ) )

        self.nstr_a = np.max(occs_a[:,0])+1
        self.nstr_b = np.max(occs_b[:,0])+1
        
        self.coeffs = np.zeros((self.nstr_a, self.nstr_b))
        self.occs_a = np.zeros((self.nstr_a,self.na),dtype=int)
        self.occs_b = np.zeros((self.nstr_b,self.nb),dtype=int)

        for occ in occs_a:
            ind = occ[0]
            self.occs_a[ind] = occ[1:]

        for occ in occs_b:
            ind = occ[0]
            self.occs_b[ind] = occ[1:]

        for ind, coeff in enumerate(coeffs):
            ind_a, ind_b = str_inds[2*ind], str_inds[2*ind+1]
            self.coeffs[ind_a, ind_b] = coeff

def align_orbs(C_tar, C_ref, S_ao, degens):
    S_mo = C_tar.T @ S_ao @ C_ref
    C_tar = C_tar.T

    # make the largest value of each row positive
    for i in range(len(S_mo)) :
        if abs(min(S_mo[i])) > max(S_mo[i]):
            print('  Flipping phase of orbital {}'.format(i))
            S_mo[i]  = -1*S_mo[i]
            C_tar[i] = -1*C_tar[i]

    # put the largest values on the diagonals
    for i in range(len(S_mo)):
        row = S_mo[i]
        ind = np.argmax(row)
        if ind != i:
            print('  Swapping orbitals {} and {}'.format(i,ind))
            S_mo[[i, ind]] = S_mo[[ind, i]]
            C_tar[[i, ind]] = C_tar[[ind, i]]
    
    # rotate each degenerate pair of orbitals to max overlap with reference
    for pair in degens:
        i, j = pair[0], pair[1]
        print('  Rotating orbitals {} and {}'.format(i,j))
        adj = S_mo[i,i] + S_mo[j,j]
        opp = S_mo[i,j] - S_mo[j,i]
        hyp = np.sqrt(adj**2 + opp**2)

        cos = adj/hyp
        sin= opp/hyp

        arr_i = np.array(S_mo[i])
        arr_j = np.array(S_mo[j])
        Crow_i = np.array(C_tar[i])
        Crow_j = np.array(C_tar[j])

        S_mo[i] = cos*arr_i - sin*arr_j
        S_mo[j] = cos*arr_j + sin*arr_i
        C_tar[i] = cos*Crow_i - sin*Crow_j
        C_tar[j] = cos*Crow_j + sin*Crow_i

    return C_tar.T

def calc_wfn_overlap(vec_m, S_mo, vec_p):
    """ calculate the overlap between wfn_m and wfn_p"""    

    # Overlap of the (-) and (+) CI wfns
    S_a = np.zeros((vec_m.nstr_a, vec_p.nstr_a))
    for im in range(vec_m.nstr_a):
        for ip in range(vec_p.nstr_a):
            occs_m = vec_m.occs_a[im]
            occs_p = vec_p.occs_a[ip]
            det = S_mo[occs_m][:,occs_p]
            S_a[im,ip] = np.linalg.det(det)

    S_b = np.zeros((vec_m.nstr_b, vec_p.nstr_b))
    for im in range(vec_m.nstr_b):
        for ip in range(vec_p.nstr_b):
            occs_m = vec_m.occs_b[im]
            occs_p = vec_p.occs_b[ip]
            det = S_mo[occs_m][:,occs_p]
            S_b[im,ip] = np.linalg.det(det)

    #total_overlap = 0.0

    #for ia in range(vec_m.nstr_a):
    #    for ib in range(vec_m.nstr_b):
    #        coeff_i = vec_m.coeffs[ia,ib]

    #        for ja in range(vec_p.nstr_a):
    #            for jb in range(vec_p.nstr_b):
    #                coeff_j = vec_p.coeffs[ja,jb]

    #                s_a = S_a[ia,ja]
    #                s_b = S_b[ib,jb]
    #                total_overlap += coeff_i * s_a * s_b * coeff_j

    mat_mb_pa_1 = vec_m.coeffs.T @ S_a
    mat_mb_pa_2 = S_b @ vec_m.coeffs.T

    #debug_mat(mat_mb_pa_1)
    #debug_mat(mat_mb_pa_2)

    mat = np.multiply(mat_mb_pa_1,mat_mb_pa_2)
    return np.sum(mat)

# write matrix to file for debugging
def debug_mat(mat):
    with open('DEBUG','a+') as f:
        f.write(np.array_str(np.round(mat,10), max_line_width=1000000, precision=5, suppress_small=False))
        f.write('\n')

# write string to file for debugging
def debug_str(string):
    with open('DEBUG','a+') as f:
        f.write(string)
        f.write('\n')

def calc_dboc(wfn):

    # DBOC only under specific input conditions
    # Find a cleaner way to enforce this later
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
    
    # perform SCF at the original geometry
    # save the location of the orbitals, we'll use them as a guess throughout
    e_c, ref_wfn_c = driver.energy('scf',return_wfn=True)
    C_c = ref_wfn_c.Ca().to_array()
    psi_scratch = core.IOManager.shared_object().get_default_path()
    fname = os.path.split(os.path.abspath(core.get_writer_file_prefix(wfn.molecule().name())))[1]
    fname = os.path.join(psi_scratch, fname + ".180.npy")

    # finite difference displacement distance
    dR = 5.0e-4
    print("Performing DBOC with dR: ", dR, " a.u.")
    with open('DEBUG','w+') as f:
        f.write("Performing DBOC with dR: {:f}  a.u.\n".format(dR))

    # defined for convenience
    mints = core.MintsHelper(wfn)
    mol = wfn.molecule()
    init_geom = mol.geometry().to_array()
    
    e_dboc = 0.0
    
    # generate a minus (-) and plus (+) displacement per nuclear coord
    # calculate overlap of (-) and (+) to get dboc contribution of that coord
    for ai, atom in enumerate(init_geom):

        # The mass of the nucleus
        mass_nuc = mol.mass(ai) - mass_e * mol.true_atomic_number(ai)

        for dim in range(3):
            
            debug_str("\n\n\n~~~~~~~~~~~~~~~~~~~~nuclear coordinate {:d} of {:d}~~~~~~~~~~~~~~~~~~~~\n".format(3*ai+dim+1,3*len(init_geom)))
            print("\n\nnuclear coordinate {:d} of {:d}\n".format(3*ai+dim+1,3*len(init_geom)))

            # Create the (-) displacement and get SCF orbitals
            geom_m = np.array(init_geom)
            geom_m[ai,dim] -= dR
            mol.set_geometry(core.Matrix.from_array(geom_m))
            ref_wfn_c.to_file(fname)
            e_m, ref_wfn_m = driver.energy('scf',return_wfn=True)
            C_m = ref_wfn_m.Ca().to_array()

            # Create the (+) displacement and get SCF orbitals
            geom_p = np.array(init_geom)
            geom_p[ai,dim] += dR
            mol.set_geometry(core.Matrix.from_array(geom_p))
            ref_wfn_c.to_file(fname)
            e_p, ref_wfn_p = driver.energy('scf',return_wfn=True)
            C_p = ref_wfn_p.Ca().to_array()

            # Get the overlap (in both AO and MO space) between (-) and (+) orbitals
            S_ao_mp = mints.ao_overlap(ref_wfn_m.basisset(),ref_wfn_p.basisset())
            S_ao_mc = mints.ao_overlap(ref_wfn_m.basisset(),ref_wfn_c.basisset())
            S_ao_pc = mints.ao_overlap(ref_wfn_p.basisset(),ref_wfn_c.basisset())
            
            # look at orbital energies and record degenerate pairs
            eps_m = ref_wfn_m.epsilon_a().to_array()
            eps_p = ref_wfn_p.epsilon_a().to_array()
            degens = []
            for i in range(len(eps_m)-1):
                if abs(eps_m[i] - eps_m[i+1]) > 1.0e-5 :
                    continue
                elif abs(eps_p[i] - eps_p[i+1]) > 1.0e-5 :
                    print('THIS SHOULD NEVER HAPPEN')
                    continue
                degens.append((i,i+1,))


            # Get the overlap (in both AO and MO space) between (-) and (+) orbitals
            #S_mo = C_m.T @ S_ao_mp.np @ C_p 
            #debug_str('\nMO Overlap between (-) and (+):\n')
            #debug_mat(S_mo)
            #debug_str('\nDiagonal:\n')
            #debug_mat(S_mo.diagonal())

            # Align orbitals to maximize overlap
            print('Aligning (-)')
            C_m = align_orbs(C_m, C_c, S_ao_mc, degens)
            print('Aligning (+)')
            C_p = align_orbs(C_p, C_c, S_ao_pc, degens)
            print('Aligning (-) with (+)')
            C_m = align_orbs(C_m, C_p, S_ao_mp, degens)

            ref_wfn_m.set_Ca(core.Matrix.from_array(C_m))
            ref_wfn_p.set_Ca(core.Matrix.from_array(C_p))

            # Do DETCI with SCF wavefunctions
            vec_m = CIVect(ref_wfn_m)
            vec_p = CIVect(ref_wfn_p)

            # Make sure the reference determinant coefficients are in phase
            if vec_m.coeffs[0,0] * vec_p.coeffs[0,0] < 0:
                #print('Reversed the (-) CI coefficients')
                vec_m.coeffs = -1 * vec_m.coeffs

            # Make sure the CI determinants are in the same order
            if not ( np.array_equal(vec_m.occs_a,vec_p.occs_a) and np.array_equal(vec_m.occs_b,vec_p.occs_b) ):
                raise Exception("CI Determinants in different order")
            
            # Get the overlap between (-) and (+)
            S_mo = C_m.T @ S_ao_mp.np @ C_p 
            overlap_mp = calc_wfn_overlap(vec_m, S_mo, vec_p)
            
            # Contribution of overlap to E_DBOC
            print("1 - <(+)|(-)>", 1 - overlap_mp )
            e_dboc_comp = (1-overlap_mp) / mass_nuc
            e_dboc += e_dboc_comp
    
    e_dboc /= (2*dR)**2
    e_dboc *= au_to_amu

    print("Total E_DBOC: ", e_dboc, "Eh")
    print("Total E_DBOC: ", e_dboc * 219474.631371, "cm^-1")

    mol.set_geometry(core.Matrix.from_array(init_geom))

    return e_dboc

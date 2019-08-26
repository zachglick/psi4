/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2019 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

// usapt.h, to be included in dftsapt.h if needed.

#ifndef ASAPT_H
#define ASAPT_H

#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/typedefs.h"
#include "psi4/libqt/qt.h"
#include <map>
#include <array>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace psi {

class JK;
class DFHelper;
class Options;
class PSIO;
class AtomicDensity;

namespace sapt {

/*- Open-shell generalization of SAPT0
    Will be the basis for DFTSAPT
    Will need to create better class hierarchy -*/

class ASAPT0 {
   private:
    Options& options_;

   protected:
    // SAPT type (until properly subclassed)
    std::string type_;

    // The dimer wavefunction (will stow results to the Python layer)
    SharedWavefunction d_;

    // Print flag
    int print_;
    // Debug flag
    int debug_;
    // Bench flag
    int bench_;

    bool sSAPT0_scale_ = false;
    bool exch_scale_ = true;   
    bool ind_scale_ = true;     
    bool ind_resp_ = true;      
    bool sep_core_ = true;       
    bool full_orbital_;
    bool multipole_;   
    int multipole_order_;


    // CPKS maximum iterations
    int cpks_maxiter_;
    // CPKS convergence threshold
    double cpks_delta_;
    // Memory in doubles
    size_t memory_;

    // Energies table
    std::map<std::string, double> energies_;

    // Dimer primary basis set
    std::shared_ptr<BasisSet> primary_;
    // Monomer A primary basis set
    std::shared_ptr<BasisSet> primary_A_;
    // Monomer B primary basis set
    std::shared_ptr<BasisSet> primary_B_;
    // Dimer -RI or -MP2FIT auxiliary basis set
    std::shared_ptr<BasisSet> mp2fit_;
    // Dimer -JKFIT auxiliary basis set
    std::shared_ptr<BasisSet> jkfit_;

    // Dimer SCF energy
    double E_dimer_;
    // Monomer A SCF energy
    double E_monomer_A_;
    // Monomer B SCF energy
    double E_monomer_B_;

    // Dimer geometry
    std::shared_ptr<Molecule> dimer_;
    // Monomer A geometry
    std::shared_ptr<Molecule> monomer_A_;
    // Monomer B geometry
    std::shared_ptr<Molecule> monomer_B_;

    // Dimer dipole field
    std::array<double, 3> dimer_field_;
    // Monomer A dipole field
    std::array<double, 3> monomer_A_field_;
    // Monomer B dipole field
    std::array<double, 3> monomer_B_field_;


    // Monomer A C matrix (full occ), alpha spin
    std::shared_ptr<Matrix> Cocc_A_;
    // Monomer B C matrix (full occ), alpha spin
    std::shared_ptr<Matrix> Cocc_B_;
    // Monomer A C matrix (full vir), alpha spin
    std::shared_ptr<Matrix> Cvir_A_;
    // Monomer B C matrix (full vir), alpha spin
    std::shared_ptr<Matrix> Cvir_B_;

    // Monomer A eps vector (full occ), alpha spin
    std::shared_ptr<Vector> eps_occ_A_;
    // Monomer B eps vector (full occ), alpha spin
    std::shared_ptr<Vector> eps_occ_B_;
    // Monomer A eps vector (full vir), alpha spin
    std::shared_ptr<Vector> eps_vir_A_;
    // Monomer B eps vector (full vir), alpha spin
    std::shared_ptr<Vector> eps_vir_B_;

    // Monomer A C matrix (active occ), alpha spin
    std::shared_ptr<Matrix> Caocc_A_;
    // Monomer B C matrix (active occ), alpha spin
    std::shared_ptr<Matrix> Caocc_B_;
    // Monomer A C matrix (active vir), alpha spin
    std::shared_ptr<Matrix> Cavir_A_;
    // Monomer B C matrix (active vir), alpha spin
    std::shared_ptr<Matrix> Cavir_B_;

    // Monomer A C matrix (frozen occ), alpha spin
    std::shared_ptr<Matrix> Cfocc_A_;
    // Monomer B C matrix (frozen occ), alpha spin
    std::shared_ptr<Matrix> Cfocc_B_;
    // Monomer A C matrix (frozen vir), alpha spin
    std::shared_ptr<Matrix> Cfvir_A_;
    // Monomer B C matrix (frozen vir), alpha spin
    std::shared_ptr<Matrix> Cfvir_B_;

    // Monomer A eps vector (active occ), alpha spin
    std::shared_ptr<Vector> eps_aocc_A_;
    // Monomer B eps vector (active occ), alpha spin
    std::shared_ptr<Vector> eps_aocc_B_;
    // Monomer A eps vector (active vir), alpha spin
    std::shared_ptr<Vector> eps_avir_A_;
    // Monomer B eps vector (active vir), alpha spin
    std::shared_ptr<Vector> eps_avir_B_;

    // Monomer A eps vector (frozen occ), alpha spin
    std::shared_ptr<Vector> eps_focc_A_;
    // Monomer B eps vector (frozen occ), alpha spin
    std::shared_ptr<Vector> eps_focc_B_;
    // Monomer A eps vector (frozen vir), alpha spin
    std::shared_ptr<Vector> eps_fvir_A_;
    // Monomer B eps vector (frozen vir), alpha spin
    std::shared_ptr<Vector> eps_fvir_B_;
      
    // Shared matrices (Fock-like)
    std::map<std::string, std::shared_ptr<Matrix> > vars_;




    // => Electronic Response Localization <= //
    
    // Localized occupied orbitals of monomer A (n x a)
    std::shared_ptr<Matrix> Locc_A_;
    // Localized occupied orbitals of monomer B (n x b)
    std::shared_ptr<Matrix> Locc_B_;
    // Localization transformation for monomer A (a x \bar a)
    std::shared_ptr<Matrix> Uocc_A_;
    // Localization transformation for monomer B (b x \bar b)
    std::shared_ptr<Matrix> Uocc_B_;

    // Local occupied orbital atomic population of monomer A (A x a)
    std::shared_ptr<Matrix> Q_A_;
    // Local occupied orbital atomic population of monomer B (B x b)
    std::shared_ptr<Matrix> Q_B_;
    // Molecular occupied orbital atomic assignment of monomer A (A x a)
    std::shared_ptr<Matrix> R_A_;
    // Molecular occupied orbital atomic assignment of monomer B (B x b)
    std::shared_ptr<Matrix> R_B_;





    // ISA partition for monomer A
    std::shared_ptr<AtomicDensity> atomic_A_;
    // ISA partition for monomer B
    std::shared_ptr<AtomicDensity> atomic_B_;






    // Print author/sizing/spec info
    virtual void print_header() const;
    // Obligatory
    virtual void print_trailer();



    // Compute the atomic density partition
    void atomize();
    // Compute L and U according to Localizer algorithm and tolerances
    void localize(); 
    // Compute Q and R
    void populate();
    // Compute QAC and VAB via quadrature
    void ps();
    // Compute Aar, WBar, and RAC via DF
    void df();
    std::shared_ptr<DFHelper> dfh_;
    
    // Compute Elst
    void elst();
    // Compute Exch
    void exch();
    // Compute Ind
    void ind();
    // Compute Disp
    void disp();
    // Analyze results
    void analyze();

    // Special cl/ncl-Elst with multipoles
    void elst_multipole();
    
    // Special orbital-basis Elst
    void elst_orbital();
    // Special orbital-basis Ind
    void ind_orbital();


    // Final matrices
    std::shared_ptr<Matrix> Elst_AB;
    std::shared_ptr<Matrix> Exch_ab;
    std::shared_ptr<Matrix> IndAB_aB;
    std::shared_ptr<Matrix> IndBA_Ab;
    std::shared_ptr<Matrix> Disp_ab;


    // Hartree-Fock-like terms (Elst, Exch, Ind)
    virtual void fock_terms();
    ////// MP2-like terms (Disp)
    ////virtual void mp2_terms();

    // => Helper Methods <= //

    // Build the AO-basis dimer overlap matrix
    std::shared_ptr<Matrix> build_S(std::shared_ptr<BasisSet> basis);
    // Build the potential integral matrix
    std::shared_ptr<Matrix> build_V(std::shared_ptr<BasisSet> basis);

    // Build the alpha and beta S_ij matrices in the dimer occupied space
    std::shared_ptr<Matrix> build_Sij(std::shared_ptr<Matrix> S);
    // Build the S^\infty expansion in the dimer occupied space
    std::shared_ptr<Matrix> build_Sij_n(std::shared_ptr<Matrix> Sij);
    // Build the Cbar matrices from S^\infty
    std::map<std::string, std::shared_ptr<Matrix> > build_Cbar(std::shared_ptr<Matrix> S);

    // Compute the CPKS solution
    std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix> > compute_x(std::shared_ptr<JK> jk,
                                                                           std::shared_ptr<Matrix> w_B,
                                                                           std::shared_ptr<Matrix> w_A);

    // Build the ExchInd20 potential in the monomer A ov space
    std::shared_ptr<Matrix> build_exch_ind_pot(std::map<std::string, std::shared_ptr<Matrix> >& vars);
    // Build the Ind20 potential in the monomer A ov space
    std::shared_ptr<Matrix> build_ind_pot(std::map<std::string, std::shared_ptr<Matrix> >& vars);

    void initialize(SharedWavefunction mA, SharedWavefunction mB);

   public:
    // Constructor, call this with 3 converged SCF jobs (dimer, monomer A, monomer B)
    ASAPT0(SharedWavefunction d, SharedWavefunction mA, SharedWavefunction mB, Options& options,
           std::shared_ptr<PSIO> psio);
    virtual ~ASAPT0();

    // Compute the ASAPT0 analysis
    virtual double compute_energy();

    //    void fd(int nA, int nB, double** CAp, double** Sp, int nso, int no, double** CBp, double** Cp,
    //    std::shared_ptr<Matrix> Sa);
};
class CPKS_ASAPT0 {
    friend class ASAPT0;

   protected:
    // => Global Data <= //

    // Convergence tolerance
    double delta_;
    // Maximum allowed iterations
    int maxiter_;
    // JK Object 
    std::shared_ptr<JK> jk_;
    
    // => Monomer A Problem <= //

    // Perturbation applied to A
    std::shared_ptr<Matrix> w_A_;
    // Response of A
    std::shared_ptr<Matrix> x_A_;
    // Active occ orbital coefficients of A
    std::shared_ptr<Matrix> Cocc_A_;
    // Active vir orbital coefficients of A
    std::shared_ptr<Matrix> Cvir_A_;
    // Active occ orbital eigenvalues of A
    std::shared_ptr<Vector> eps_occ_A_;
    // Active vir orbital eigenvalues of A
    std::shared_ptr<Vector> eps_vir_A_;

    // => Monomer B Problem <= //

    // Perturbation applied to B
    std::shared_ptr<Matrix> w_B_;
    // Response of B
    std::shared_ptr<Matrix> x_B_;
    // Active occ orbital coefficients of B
    std::shared_ptr<Matrix> Cocc_B_;
    // Active vir orbital coefficients of B
    std::shared_ptr<Matrix> Cvir_B_;
    // Active occ orbital eigenvalues of B
    std::shared_ptr<Vector> eps_occ_B_;
    // Active vir orbital eigenvalues of B
    std::shared_ptr<Vector> eps_vir_B_;

    // Form the s = Ab product for the provided vectors b (may or may not need more iterations)
    std::map<std::string, std::shared_ptr<Matrix> > product(std::map<std::string, std::shared_ptr<Matrix> > b);
    // Apply the denominator from r into z
    void preconditioner(std::shared_ptr<Matrix> r,
                        std::shared_ptr<Matrix> z,
                        std::shared_ptr<Vector> o,
                        std::shared_ptr<Vector> v);

   public:
    CPKS_ASAPT0();
    virtual ~CPKS_ASAPT0();

    void compute_cpks();
};
}  // namespace sapt
}  // namespace psi

#endif

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

#include "asapt0.h"
#include "atomic.h"

#include <ctime>

#include "psi4/physconst.h"

#include "psi4/lib3index/dfhelper.h"
#include "psi4/lib3index/3index.h"
#include "psi4/libfock/jk.h"
#include "psi4/libfock/cubature.h"
#include "psi4/libfock/points.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/local.h"
#include "psi4/libmints/electrostatic.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"

namespace psi {
namespace sapt {

// TODO: ROHF orbitals for coupled induction
// TODO: SAPT charge transfer energy
ASAPT0::ASAPT0(SharedWavefunction d, SharedWavefunction mA, SharedWavefunction mB, Options& options,
               std::shared_ptr<PSIO> psio)
    : options_(options) {

    // store some general options
    print_ = options.get_int("PRINT");
    debug_ = options.get_int("DEBUG");
    bench_ = options.get_int("BENCH");

    // store some SAPT specific options
    //sSAPT0_scale_    = options.get_bool("ASAPT_sSAPT0_SCALE");
    //exch_scale_      = options.get_bool("ASAPT_EXCH_SCALE");
    //ind_scale_       = options.get_bool("ASAPT_IND_SCALE");
    //ind_resp_        = options.get_bool("ASAPT_IND_RESPONSE");
    //sep_core_        = options.get_bool("ASAPT_SEPARATE_CORE");
    //full_orbital_    = options.get_bool("ASAPT_FULL_ORBITAL");
    //multipole_       = options.get_bool("ASAPT_MULTIPOLE");
    //multipole_order_ = options.get_int("ASAPT_MULTIPOLE_ORDER");
    
    // TODO: Use general memory manager?
    memory_ = (unsigned long int)(Process::environment.get_memory() * options.get_double("SAPT_MEM_FACTOR") * 0.125);

    // should this be cpks or cphf? 
    cpks_maxiter_ = options.get_int("MAXITER");
    cpks_delta_ = options.get_double("D_CONVERGENCE");

    dimer_     = d->molecule();
    monomer_A_ = mA->molecule();
    monomer_B_ = mB->molecule();

    E_dimer_     = d->energy();
    E_monomer_A_ = mA->energy();
    E_monomer_B_ = mB->energy();

    primary_   = d->basisset();
    primary_A_ = mA->basisset();
    primary_B_ = mB->basisset();

    dimer_field_ = d->get_dipole_field_strength();
    monomer_A_field_ = mA->get_dipole_field_strength();
    monomer_B_field_ = mB->get_dipole_field_strength();

    if (primary_->nbf() != primary_A_->nbf() || primary_->nbf() != primary_B_->nbf()) {
        throw PSIEXCEPTION("Monomer-centered bases not allowed in ASAPT0");
    }

    mp2fit_ = d->get_basisset("DF_BASIS_SAPT");
    jkfit_ = d->get_basisset("DF_BASIS_SCF");
    initialize(mA, mB);
    d_ = d;

}

void ASAPT0::initialize(SharedWavefunction mA, SharedWavefunction mB) {
    type_ = "ASAPT0";

    Cocc_A_ = mA->Ca_subset("AO", "OCC");
    Cvir_A_ = mA->Ca_subset("AO", "VIR");
    eps_occ_A_ = mA->epsilon_a_subset("AO", "OCC");
    eps_vir_A_ = mA->epsilon_a_subset("AO", "VIR");

    Cfocc_A_ = mA->Ca_subset("AO", "FROZEN_OCC");
    Caocc_A_ = mA->Ca_subset("AO", "ACTIVE_OCC");
    Cavir_A_ = mA->Ca_subset("AO", "ACTIVE_VIR");
    Cfvir_A_ = mA->Ca_subset("AO", "FROZEN_VIR");

    eps_focc_A_ = mA->epsilon_a_subset("AO", "FROZEN_OCC");
    eps_aocc_A_ = mA->epsilon_a_subset("AO", "ACTIVE_OCC");
    eps_avir_A_ = mA->epsilon_a_subset("AO", "ACTIVE_VIR");
    eps_fvir_A_ = mA->epsilon_a_subset("AO", "FROZEN_VIR");

    Cocc_B_ = mB->Ca_subset("AO", "OCC");
    Cvir_B_ = mB->Ca_subset("AO", "VIR");
    eps_occ_B_ = mB->epsilon_a_subset("AO", "OCC");
    eps_vir_B_ = mB->epsilon_a_subset("AO", "VIR");

    Cfocc_B_ = mB->Ca_subset("AO", "FROZEN_OCC");
    Caocc_B_ = mB->Ca_subset("AO", "ACTIVE_OCC");
    Cavir_B_ = mB->Ca_subset("AO", "ACTIVE_VIR");
    Cfvir_B_ = mB->Ca_subset("AO", "FROZEN_VIR");

    eps_focc_B_ = mB->epsilon_a_subset("AO", "FROZEN_OCC");
    eps_aocc_B_ = mB->epsilon_a_subset("AO", "ACTIVE_OCC");
    eps_avir_B_ = mB->epsilon_a_subset("AO", "ACTIVE_VIR");
    eps_fvir_B_ = mB->epsilon_a_subset("AO", "FROZEN_VIR");

}

ASAPT0::~ASAPT0() {
}

double ASAPT0::compute_energy() {
    energies_["HF"] = E_dimer_ - E_monomer_A_ - E_monomer_B_; // TODO: get dHF loaded correctly
    print_header();



    timer_on("ASAPT: Fock");
    fock_terms();
    timer_off("ASAPT: Fock");

    timer_on("ASAPT: Atom");
    atomize();
    timer_off("ASAPT: Atom");

    timer_on("ASAPT: Local");
    localize();
    timer_off("ASAPT: Local");

    timer_on("ASAPT: Pop");
    populate();
    timer_off("ASAPT: Pop");

    timer_on("ASAPT: PS");
    ps();
    timer_off("ASAPT: PS");
    
    timer_on("ASAPT: DF");
    df();
    timer_off("ASAPT: DF");

    timer_on("ASAPT: Elst");
    elst();
    timer_off("ASAPT: Elst");
    
    timer_on("ASAPT: Exch");
    exch();
    timer_off("ASAPT: Exch");

    timer_on("ASAPT: Ind");
    ind();
    timer_off("ASAPT: Ind");

    timer_on("ASAPT: Disp");
    disp();
    timer_off("ASAPT: Disp");

    //if (multipole_) {
    //    
    //    timer_on("ASAPT: Mult Elst");
    //    elst_multipole();
    //    timer_off("ASAPT: Mult Elst");

    //}

    //if (full_orbital_) {

    //    timer_on("ASAPT: Elst Orbs");
    //    elst_orbital();
    //    timer_off("ASAPT: Elst Orbs");

    //    timer_on("ASAPT: Ind Orbs");
    //    ind_orbital();
    //    timer_off("ASAPT: Ind Orbs");
    //}

    timer_on("ASAPT: Anal");
    analyze();
    timer_off("ASAPT: Anal");

    print_trailer();
    return 0.0;
}

void ASAPT0::print_header() const{
}


void ASAPT0::print_trailer() {
    Process::environment.globals["ASAPT0 EXCH ENERGY"] = 0.0;
    Process::environment.globals["ASAPT0 ELST ENERGY"] = 0.0;
    Process::environment.globals["ASAPT0 IND ENERGY"] = 0.0;
    Process::environment.globals["ASAPT0 DISP ENERGY"] = 0.0;
    Process::environment.globals["ASAPT0 TOTAL ENERGY"] = 0.0;
    Process::environment.globals["ASAPT0 ENERGY"] = 0.0;
    Process::environment.globals["CURRENT ENERGY"] = 0.0;
}

void ASAPT0::analyze() {
    auto Exch_AB  = linalg::triplet(Q_A_,Exch_ab,Q_B_,false,false,true);
    auto IndAB_AB = linalg::doublet(Q_A_,IndAB_aB,false,false);
    auto IndBA_AB = linalg::doublet(IndBA_Ab,Q_B_,false,true);
    auto Disp_AB  = linalg::triplet(Q_A_,Disp_ab,Q_B_,false,false,true);

    d_->set_array_variable("Elst_AB", Elst_AB);
    d_->set_array_variable("Exch_AB", Exch_AB);
    d_->set_array_variable("IndAB_AB", IndAB_AB);
    d_->set_array_variable("IndBA_AB", IndBA_AB);
    d_->set_array_variable("Disp_AB", Disp_AB);

    // Atomic population is a vector
    auto NAvec = atomic_A_->N();
    auto NBvec = atomic_B_->N();

    // We can only attach matrices to the wavefunction
    auto NA = std::make_shared<Matrix>("Pop_A", NAvec->dim(), 1);
    auto NB = std::make_shared<Matrix>("Pop_B", NBvec->dim(), 1);

    // Copy the vectors to 1D matrices
    for(int iA=0; iA < NAvec->dim(); ++iA) {
        NA->set(iA, 0, NAvec->get(iA));
    }

    for(int iB=0; iB < NBvec->dim(); ++iB) {
        NB->set(iB, 0, NBvec->get(iB));
    }

    d_->set_array_variable("Pop_A", NA);
    d_->set_array_variable("Pop_B", NB);


    auto Elst_A  = Elst_AB->collapse(1);
    auto Elst_B  = Elst_AB->collapse(0);
    auto Exch_A  = Exch_AB->collapse(1);
    auto Exch_B  = Exch_AB->collapse(0);
    auto Exch_a  = Exch_ab->collapse(1);
    auto Exch_b  = Exch_ab->collapse(0);
    auto IndAB_A = IndAB_AB->collapse(1);
    auto IndAB_B = IndAB_AB->collapse(0);
    auto IndAB_a = IndAB_aB->collapse(1);
    auto IndBA_A = IndBA_AB->collapse(1);
    auto IndBA_B = IndBA_AB->collapse(0);
    auto IndBA_b = IndBA_Ab->collapse(0);
    auto Disp_A  = Disp_AB->collapse(1);
    auto Disp_B  = Disp_AB->collapse(0);
    auto Disp_a  = Disp_ab->collapse(1);
    auto Disp_b  = Disp_ab->collapse(0);

    // Print Order 2
    outfile->Printf("Elst_AB\n");
    Elst_AB->print_out();

    outfile->Printf("Exch_AB\n");
    Exch_AB->print_out();

    outfile->Printf("IndAB_AB\n");
    IndAB_AB->print_out();

    outfile->Printf("IndBA_AB\n");
    IndBA_AB->print_out();

    outfile->Printf("Disp_AB\n");
    Disp_AB->print_out();

    // Print Order 1
    outfile->Printf("Elst_A\n");
    Elst_A->print_out();

    outfile->Printf("Elst_B\n");
    Elst_B->print_out();

    outfile->Printf("Exch_A\n");
    Exch_A->print_out();

    outfile->Printf("Exch_B\n");
    Exch_B->print_out();

    outfile->Printf("IndAB_A\n");
    IndAB_A->print_out();

    outfile->Printf("IndAB_B\n");
    IndAB_B->print_out();

    outfile->Printf("IndBA_A\n");
    IndBA_A->print_out();

    outfile->Printf("IndBA_B\n");
    IndBA_B->print_out();

    outfile->Printf("Disp_A\n");
    Disp_A->print_out();

    outfile->Printf("Disp_B\n");
    Disp_B->print_out();

    // Print Order 0
    outfile->Printf("Elst Total: \n");
    auto Elst_sum = Elst_A->collapse(0);
    Elst_sum->print_out();

    outfile->Printf("Exch Total: \n");
    auto Exch_sum = Exch_A->collapse(0);
    Exch_sum->print_out();

    outfile->Printf("IndAB Total: \n");
    auto IndAB_sum = IndAB_A->collapse(0);
    IndAB_sum->print_out();

    outfile->Printf("IndBA Total: \n");
    auto IndBA_sum = IndBA_A->collapse(0);
    IndBA_sum->print_out();

    outfile->Printf("Disp Total: \n");
    auto Disp_sum = Disp_A->collapse(0);
    Disp_sum->print_out();

}





void ASAPT0::atomize()
{
    outfile->Printf(" ATOMIZATION:\n\n");
    
    atomic_A_ = AtomicDensity::build("STOCKHOLDER", primary_A_, Process::environment.options);
    atomic_A_->set_name("A");
    atomic_A_->compute(linalg::doublet(Cocc_A_,Cocc_A_,false,true));
    atomic_A_->compute_charges(1.0);

    atomic_B_ = AtomicDensity::build("STOCKHOLDER", primary_B_, Process::environment.options);
    atomic_B_->set_name("B");
    atomic_B_->compute(linalg::doublet(Cocc_B_,Cocc_B_,false,true));
    atomic_B_->compute_charges(1.0);
}




void ASAPT0::localize()
{
    outfile->Printf(" LOCALIZATION:\n\n");

    if (sep_core_) {

        outfile->Printf("  Local Core Orbitals for Monomer A:\n\n");
        std::shared_ptr<Localizer> localfA = Localizer::build("PIPEK_MEZEY", primary_, Cfocc_A_, Process::environment.options);
        localfA->localize();
        std::shared_ptr<Matrix> Lfocc_A = localfA->L();
        std::shared_ptr<Matrix> Ufocc_A = localfA->U();

        int nfA = eps_focc_A_->dimpi()[0];
        std::shared_ptr<Matrix> FcfA(new Matrix("FcfA", nfA, nfA));
        FcfA->set_diagonal(eps_focc_A_); 
        std::shared_ptr<Matrix> FlfA = localfA->fock_update(FcfA);
        FlfA->set_name("FlfA");    
        //FcfA->print();
        //FlfA->print();

        outfile->Printf("  Local Valence Orbitals for Monomer A:\n\n");
        std::shared_ptr<Localizer> localaA = Localizer::build("PIPEK_MEZEY", primary_, Caocc_A_, Process::environment.options);
        localaA->localize();
        std::shared_ptr<Matrix> Laocc_A = localaA->L();
        std::shared_ptr<Matrix> Uaocc_A = localaA->U();

        int naA = eps_aocc_A_->dimpi()[0];
        std::shared_ptr<Matrix> FcaA(new Matrix("FcaA", naA, naA));
        FcaA->set_diagonal(eps_aocc_A_); 
        std::shared_ptr<Matrix> FlaA = localaA->fock_update(FcaA);
        FlaA->set_name("FlaA");    
        //FcaA->print();
        //FlaA->print();

        std::vector<std::shared_ptr<Matrix> > LAlist;
        LAlist.push_back(Lfocc_A);
        LAlist.push_back(Laocc_A);
        Locc_A_ = linalg::horzcat(LAlist); 

        Uocc_A_ = std::shared_ptr<Matrix>(new Matrix("Uocc A", nfA + naA, nfA + naA));
        double** Uocc_Ap = Uocc_A_->pointer();
        double** Ufocc_Ap = Ufocc_A->pointer();
        double** Uaocc_Ap = Uaocc_A->pointer();
        for (int i = 0; i < nfA; i++) {
            ::memcpy(&Uocc_Ap[i][0],&Ufocc_Ap[i][0],sizeof(double)*nfA);
        }
        for (int i = 0; i < naA; i++) {
            ::memcpy(&Uocc_Ap[i+nfA][nfA],&Uaocc_Ap[i][0],sizeof(double)*naA);
        }

        outfile->Printf("  Local Core Orbitals for Monomer B:\n\n");
        std::shared_ptr<Localizer> localfB = Localizer::build("PIPEK_MEZEY", primary_, Cfocc_B_, Process::environment.options);
        localfB->localize();
        std::shared_ptr<Matrix> Lfocc_B = localfB->L();
        std::shared_ptr<Matrix> Ufocc_B = localfB->U();

        int nfB = eps_focc_B_->dimpi()[0];
        std::shared_ptr<Matrix> FcfB(new Matrix("FcfB", nfB, nfB));
        FcfB->set_diagonal(eps_focc_B_); 
        std::shared_ptr<Matrix> FlfB = localfB->fock_update(FcfB);
        FlfB->set_name("FlfB");    
        //FcfB->print();
        //FlfB->print();

        outfile->Printf("  Local Valence Orbitals for Monomer B:\n\n");
        std::shared_ptr<Localizer> localaB = Localizer::build("PIPEK_MEZEY", primary_, Caocc_B_, Process::environment.options);
        localaB->localize();
        std::shared_ptr<Matrix> Laocc_B = localaB->L();
        std::shared_ptr<Matrix> Uaocc_B = localaB->U();

        int naB = eps_aocc_B_->dimpi()[0];
        std::shared_ptr<Matrix> FcaB(new Matrix("FcaA", naB, naB));
        FcaB->set_diagonal(eps_aocc_B_); 
        std::shared_ptr<Matrix> FlaB = localaB->fock_update(FcaB);
        FlaB->set_name("FlaB");    
        //FcaB->print();
        //FlaB->print();

        std::vector<std::shared_ptr<Matrix> > LBlist;
        LBlist.push_back(Lfocc_B);
        LBlist.push_back(Laocc_B);
        Locc_B_ = linalg::horzcat(LBlist); 

        Uocc_B_ = std::shared_ptr<Matrix>(new Matrix("Uocc B", nfB + naB, nfB + naB));
        double** Uocc_Bp = Uocc_B_->pointer();
        double** Ufocc_Bp = Ufocc_B->pointer();
        double** Uaocc_Bp = Uaocc_B->pointer();
        for (int i = 0; i < nfB; i++) {
            ::memcpy(&Uocc_Bp[i][0],&Ufocc_Bp[i][0],sizeof(double)*nfB);
        }
        for (int i = 0; i < naB; i++) {
            ::memcpy(&Uocc_Bp[i+nfB][nfB],&Uaocc_Bp[i][0],sizeof(double)*naB);
        }

    } else {

        outfile->Printf("  Local Orbitals for Monomer A:\n\n");
        std::shared_ptr<Localizer> localA = Localizer::build("PIPEK_MEZEY", primary_, Cocc_A_, Process::environment.options);
        localA->localize();
        Locc_A_ = localA->L();
        Uocc_A_ = localA->U();

        int na = eps_occ_A_->dimpi()[0];
        std::shared_ptr<Matrix> FcA(new Matrix("FcA", na, na));
        FcA->set_diagonal(eps_occ_A_); 
        std::shared_ptr<Matrix> FlA = localA->fock_update(FcA);
        FlA->set_name("FlA");    

        //FcA->print();
        //FlA->print();

        outfile->Printf("  Local Orbitals for Monomer B:\n\n");
        std::shared_ptr<Localizer> localB = Localizer::build("PIPEK_MEZEY", primary_, Cocc_B_, Process::environment.options);
        localB->localize();
        Locc_B_ = localB->L();
        Uocc_B_ = localB->U();

        int nb = eps_occ_B_->dimpi()[0];
        std::shared_ptr<Matrix> FcB(new Matrix("FcB", nb, nb));
        FcB->set_diagonal(eps_occ_B_); 
        std::shared_ptr<Matrix> FlB = localB->fock_update(FcB);
        FlB->set_name("FlB");    

        //FcB->print();
        //FlB->print();

    }

}



void ASAPT0::populate()
{
    outfile->Printf("  POPULATION:\n\n");

    // => Sizing <= //

    int na = Cocc_A_->colspi()[0];
    int nb = Cocc_B_->colspi()[0];

    int nA = 0;
    std::vector<int> cA;
    for (int A = 0; A < monomer_A_->natom(); A++) {
        if (monomer_A_->Z(A) != 0.0) {
            nA++;
            cA.push_back(A);
        }
    }

    int nB = 0;
    std::vector<int> cB;
    for (int B = 0; B < monomer_B_->natom(); B++) {
        if (monomer_B_->Z(B) != 0.0) {
            nB++;
            cB.push_back(B);
        }
    }

    // => Targets <= //

    std::shared_ptr<Matrix> Q2A(new Matrix("QA (A x a)", nA, na));
    std::shared_ptr<Matrix> Q2B(new Matrix("QB (B x b)", nB, nb));
    double** Q2Ap = Q2A->pointer();
    double** Q2Bp = Q2B->pointer();

    // => Molecular Grid <= //
    
    std::shared_ptr<MolecularGrid> grid = atomic_A_->grid();
    int max_points = grid->max_points();
    double* xp = grid->x();
    double* yp = grid->y();
    double* zp = grid->z();
    double* wp = grid->w();
        
    // => Temps <= //
    
    std::shared_ptr<Matrix> WA(new Matrix("WA", nA, max_points));
    std::shared_ptr<Matrix> WB(new Matrix("WB", nB, max_points));
    double** WAp = WA->pointer();
    double** WBp = WB->pointer();
    
    // => Local orbital collocation computer <= //

    std::shared_ptr<UKSFunctions> points = std::shared_ptr<UKSFunctions>(new UKSFunctions(primary_,grid->max_points(),grid->max_functions()));
    points->set_ansatz(0);
    points->set_Cs(Locc_A_,Locc_B_);
    double** psiap = points->orbital_value("PSI_A")->pointer();
    double** psibp = points->orbital_value("PSI_B")->pointer();

    // => Master Loop <= //

    const std::vector<std::shared_ptr<BlockOPoints> >& blocks = grid->blocks();
    size_t offset = 0L;
    for (int ind = 0; ind < blocks.size(); ind++) {
        int npoints = blocks[ind]->npoints();
        points->compute_orbitals(blocks[ind]);
        for (int a = 0; a < na; a++) {
            for (int P = 0; P < npoints; P++) {
                psiap[a][P] *= psiap[a][P];
            }
        }
        for (int b = 0; b < nb; b++) {
            for (int P = 0; P < npoints; P++) {
                psibp[b][P] *= psibp[b][P];
            }
        }
        atomic_A_->compute_weights(npoints,&xp[offset],&yp[offset],&zp[offset],WAp,&wp[offset]);
        atomic_B_->compute_weights(npoints,&xp[offset],&yp[offset],&zp[offset],WBp,&wp[offset]);
        C_DGEMM('N','T',nA,na,npoints,1.0,WAp[0],max_points,psiap[0],max_points,1.0,Q2Ap[0],na); 
        C_DGEMM('N','T',nB,nb,npoints,1.0,WBp[0],max_points,psibp[0],max_points,1.0,Q2Bp[0],nb); 
        offset += npoints;
    }


    outfile->Printf("    Grid-based orbital charges.\n\n");

    outfile->Printf("    Monomer A Grid Errors (Orbital Normalizations):\n");
    for (int a = 0; a < na; a++) {
        double val = 0.0;
        for (int A = 0; A < nA; A++) {
            val += Q2Ap[A][a]; 
        } 
        C_DSCAL(nA,1.0/val,&Q2Ap[0][a],na);
        outfile->Printf("    %4d: %11.3E\n", a+1, fabs(1.0 - val));
    }
    outfile->Printf("\n");

    outfile->Printf("    Monomer B Grid Errors (Orbital Normalizations):\n");
    for (int b = 0; b < nb; b++) {
        double val = 0.0;
        for (int B = 0; B < nB; B++) {
            val += Q2Bp[B][b]; 
        } 
        C_DSCAL(nB,1.0/val,&Q2Bp[0][b],nb);
        outfile->Printf("    %4d: %11.3E\n", b+1, fabs(1.0 - val));
    }
    outfile->Printf("\n");

    outfile->Printf("    Orbital charges renormalized.\n\n");

    // => Globals <= //

    Q_A_ = Q2A; 
    Q_B_ = Q2B; 
    
    R_A_ = linalg::doublet(Q_A_,Uocc_A_,false,true);
    R_B_ = linalg::doublet(Q_B_,Uocc_B_,false,true);

    // The ASAPT visualization and analysis container
    //vis_ = std::shared_ptr<ASAPTVis>(new ASAPTVis(primary_,monomer_A_,monomer_B_,Locc_A_,Locc_B_,Q_A_,Q_B_,atomic_A_,atomic_B_));
}



void ASAPT0::ps()
{
    outfile->Printf("  ATOMIC PSEUDOSPECTRAL:\n\n");
    
    // ==> Sizing <== //

    int nn  = primary_->nbf();

    int na = Cocc_A_->colspi()[0];
    int nb = Cocc_B_->colspi()[0];

    int nA = 0;
    std::vector<int> cA;
    for (int A = 0; A < monomer_A_->natom(); A++) {
        if (monomer_A_->Z(A) != 0.0) {
            nA++;
            cA.push_back(A);
        }
    }

    int nB = 0;
    std::vector<int> cB;
    for (int B = 0; B < monomer_B_->natom(); B++) {
        if (monomer_B_->Z(B) != 0.0) {
            nB++;
            cB.push_back(B);
        }
    }

    int nr = Cvir_A_->colspi()[0];
    int ns = Cvir_B_->colspi()[0];

    // ==> Spectral space to project onto <== //

    int nQ = jkfit_->nbf();

    // ==> Auxiliary basis representation of atomic ESP (unfitted) <== //
    
    // Grid Definition
    std::shared_ptr<Vector> x = atomic_A_->x();
    std::shared_ptr<Vector> y = atomic_A_->y();
    std::shared_ptr<Vector> z = atomic_A_->z();
    std::shared_ptr<Vector> w = atomic_A_->w();
    std::shared_ptr<Vector> rhoa = atomic_A_->rho();
    std::shared_ptr<Vector> rhob = atomic_B_->rho();
    int nP = x->dimpi()[0];
    int max_points = atomic_A_->grid()->max_points();
    double* xp = x->pointer();
    double* yp = y->pointer();
    double* zp = z->pointer();
    double* wp = w->pointer();
    double* rhoap = rhoa->pointer();
    double* rhobp = rhob->pointer();

    // Grid charges (blocked. Nuke 'em Rico!)
    std::shared_ptr<Matrix> QAP(new Matrix("Q_A_P", nA, max_points));
    std::shared_ptr<Matrix> QBQ(new Matrix("Q_B_Q", nB, max_points));
    double** QAPp = QAP->pointer();
    double** QBQp = QBQ->pointer();

    // Targets
    std::shared_ptr<Matrix> QAC(new Matrix("Q_A^C", nA, nQ));
    std::shared_ptr<Matrix> QBD(new Matrix("Q_B^D", nB, nQ));
    double** QACp = QAC->pointer();
    double** QBDp = QBD->pointer();

    std::shared_ptr<Matrix> VAB(new Matrix("V_A^B", nA, nB));
    std::shared_ptr<Matrix> VBA(new Matrix("V_B^A", nB, nA));
    double** VABp = VAB->pointer();
    double** VBAp = VBA->pointer();

    // Threading 
    int nthreads = 0;
    #ifdef _OPENMP
        nthreads = omp_get_max_threads();
    #endif

    // Integral computers and thread-safe targets
    std::shared_ptr<IntegralFactory> Vfact = std::make_shared<IntegralFactory>(jkfit_,BasisSet::zero_ao_basis_set(), jkfit_,BasisSet::zero_ao_basis_set());
    std::vector<std::shared_ptr<Matrix> > ZxyzT;
    std::vector<std::shared_ptr<Matrix> > VtempT;
    std::vector<std::shared_ptr<Matrix> > QACT;
    std::vector<std::shared_ptr<Matrix> > QBDT;
    std::vector<std::shared_ptr<PotentialInt> > VintT;
    for (int thread = 0; thread < nthreads; thread++) {
        ZxyzT.push_back(std::shared_ptr<Matrix>(new Matrix("Zxyz",1,4)));
        VtempT.push_back(std::shared_ptr<Matrix>(new Matrix("Vtemp",nQ,1)));
        QACT.push_back(std::shared_ptr<Matrix>(new Matrix("QACT",nA,nQ)));
        QBDT.push_back(std::shared_ptr<Matrix>(new Matrix("QBDT",nB,nQ)));
        VintT.push_back(std::shared_ptr<PotentialInt>(static_cast<PotentialInt*>(Vfact->ao_potential())));
        VintT[thread]->set_charge_field(ZxyzT[thread]);
    }

    std::shared_ptr<Matrix> Zxyz(new Matrix("Zxyz",1,4));
    double** Zxyzp = Zxyz->pointer();
    std::shared_ptr<PotentialInt> Vint(static_cast<PotentialInt*>(Vfact->ao_potential()));
    Vint->set_charge_field(Zxyz);
    std::shared_ptr<Matrix> Vtemp(new Matrix("Vtemp",nQ,1));
    double** Vtempp = Vtemp->pointer();

    // Master loop 
    for (int offset = 0; offset < nP; offset += max_points) {
        int npoints = (offset + max_points >= nP ? nP - offset : max_points);
        atomic_A_->compute_weights(npoints,&xp[offset],&yp[offset],&zp[offset],QAPp,&rhoap[offset]);
        atomic_B_->compute_weights(npoints,&xp[offset],&yp[offset],&zp[offset],QBQp,&rhobp[offset]);

        #pragma omp parallel for schedule(dynamic)
        for (int P = 0; P < npoints; P++) {

            // Thread info
            int thread = 0;
            #ifdef _OPENMP
                thread = omp_get_thread_num();
            #endif

            // Pointers
            double** ZxyzTp = ZxyzT[thread]->pointer();
            double** VtempTp = VtempT[thread]->pointer();
            double** QACTp = QACT[thread]->pointer();       
            double** QBDTp = QBDT[thread]->pointer();       
 
            // Absolute point indexing
            int Pabs = P + offset;
            
            // => Q_A^P and Q_B^Q <= //

            VtempT[thread]->zero();
            ZxyzTp[0][0] = 1.0;
            ZxyzTp[0][1] = xp[Pabs];
            ZxyzTp[0][2] = yp[Pabs];
            ZxyzTp[0][3] = zp[Pabs]; 
            VintT[thread]->compute(VtempT[thread]);

            // Potential integrals add a spurious minus sign
            C_DGER(nA,nQ,-wp[Pabs],&QAPp[0][P],max_points,VtempTp[0],1,QACTp[0],nQ);
            C_DGER(nB,nQ,-wp[Pabs],&QBQp[0][P],max_points,VtempTp[0],1,QBDTp[0],nQ);

        }

        for (int P = 0; P < npoints; P++) {
        
            // Absolute point indexing
            int Pabs = P + offset;
            
            // => V_A^B <= //

            for (int B = 0; B < nB; B++) {
                double Z  = monomer_B_->Z(cB[B]);
                double xc = monomer_B_->x(cB[B]);
                double yc = monomer_B_->y(cB[B]);
                double zc = monomer_B_->z(cB[B]);
                double R = sqrt((xp[Pabs] - xc) * (xp[Pabs] - xc) + 
                                (yp[Pabs] - yc) * (yp[Pabs] - yc) + 
                                (zp[Pabs] - zc) * (zp[Pabs] - zc));
                if (R < 1.0E-12) continue;
                double Q = Z * wp[Pabs] / R; 
                for (int A = 0; A < nA; A++) {
                    VABp[A][B] += -1.0 * Q * QAPp[A][P]; 
                }
            }
            
            // => V_B^A <= //

            for (int A = 0; A < nA; A++) {
                double Z  = monomer_A_->Z(cA[A]);
                double xc = monomer_A_->x(cA[A]);
                double yc = monomer_A_->y(cA[A]);
                double zc = monomer_A_->z(cA[A]);
                double R = sqrt((xp[Pabs] - xc) * (xp[Pabs] - xc) + 
                                (yp[Pabs] - yc) * (yp[Pabs] - yc) + 
                                (zp[Pabs] - zc) * (zp[Pabs] - zc));
                if (R < 1.0E-12) continue;
                double Q = Z * wp[Pabs] / R; 
                for (int B = 0; B < nB; B++) {
                    VBAp[B][A] += -1.0 * Q * QBQp[B][P]; 
                }
            }

        }
    }

    for (int thread = 0; thread < nthreads; thread++) {
        QAC->add(QACT[thread]);
        QBD->add(QBDT[thread]);
    }

    vars_["QAC"] = QAC;
    vars_["QBD"] = QBD;
    vars_["VAB"] = VAB;   
    vars_["VBA"] = VBA;   
}


void ASAPT0::df()
{
    outfile->Printf("  ATOMIC DENSITY FITTING:\n\n");
    
    // ==> Sizing <== //

    int nn  = primary_->nbf();

    int na = Cocc_A_->colspi()[0];
    int nb = Cocc_B_->colspi()[0];

    int nA = 0;
    std::vector<int> cA;
    for (int A = 0; A < monomer_A_->natom(); A++) {
        if (monomer_A_->Z(A) != 0.0) {
            nA++;
            cA.push_back(A);
        }
    }

    int nB = 0;
    std::vector<int> cB;
    for (int B = 0; B < monomer_B_->natom(); B++) {
        if (monomer_B_->Z(B) != 0.0) {
            nB++;
            cB.push_back(B);
        }
    }

    int nr = Cvir_A_->colspi()[0];
    int ns = Cvir_B_->colspi()[0];

    // ==> DF ERI Setup (JKFIT Type, in Full Basis) <== //

    int nQ = jkfit_->nbf();

    std::vector<SharedMatrix> Cstack_vec;
    Cstack_vec.push_back(Cocc_A_);
    Cstack_vec.push_back(Cvir_A_);
    Cstack_vec.push_back(Cocc_B_);
    Cstack_vec.push_back(Cvir_B_);

    size_t doubles = Process::environment.get_memory() * 0.8 / sizeof(double);
    //size_t max_MO = 0;
    //for (auto& mat : Cstack_vec) max_MO = std::max(max_MO, (size_t)mat->ncol());

    // Build DFHelper
    dfh_ = std::make_shared<DFHelper>(primary_, jkfit_);
    dfh_->set_memory(doubles);
    dfh_->set_method("DIRECT_iaQ");
    dfh_->set_nthreads(1); // TODO
    //dfh_->set_metric_pow(-1.0); // WHAT POWER !?!?!?
    dfh_->initialize();
    // Should I freeze MO Core?

    // Define spaces
    dfh_->add_space("a", Cstack_vec[0]);
    dfh_->add_space("r", Cstack_vec[1]);
    dfh_->add_space("b", Cstack_vec[2]);
    dfh_->add_space("s", Cstack_vec[3]);

    // add transformations
    dfh_->add_transformation("Aar", "a", "r"/* ,"Qpq"*/);
    dfh_->add_transformation("Abs", "b", "s"/* ,"Qpq"*/);

    // transform
    dfh_->transform();
    dfh_->print_header();
    //df.reset();

    auto metric = std::make_shared<FittingMetric>(jkfit_, true);
    metric->form_eig_inverse(options_.get_double("DF_FITTING_CONDITION")); // this supposedly forms J^{-1/2}
    SharedMatrix J_mhalf = metric->get_metric();

    //////std::map<std::string, std::shared_ptr<Tensor> >& ints = df->ints();
    //////std::shared_ptr<Tensor> AarT = ints["Aar"];
    //////std::shared_ptr<Tensor> AbsT = ints["Abs"];

   
    // ==> Symmetrically-Fitted Atomic ESPs <== //

    std::shared_ptr<Matrix> QAC = vars_["QAC"];
    std::shared_ptr<Matrix> QBD = vars_["QBD"];
    vars_.erase("QAC");
    vars_.erase("QBD");

    std::shared_ptr<Matrix> RAC = linalg::doublet(QAC, J_mhalf);
    std::shared_ptr<Matrix> RBD = linalg::doublet(QBD, J_mhalf);
    double** RACp = RAC->pointer();
    double** RBDp = RBD->pointer();

    vars_["RAC"] = RAC;
    vars_["RBD"] = RBD;


    // ==> Setup Atomic Electrostatic Fields for Induction <== //

    dfh_->add_disk_tensor("WAbs_nuc", std::make_tuple(nA, nb, ns));
    dfh_->add_disk_tensor("WBar_nuc", std::make_tuple(nB, na, nr));

    // => Nuclear Part (PITA) <= //
    
    auto Zxyz2 = std::make_shared<Matrix>("Zxyz", 1, 4);
    double** Zxyz2p = Zxyz2->pointer();
    auto Vfact2 = std::make_shared<IntegralFactory>(primary_);
    std::shared_ptr<PotentialInt> Vint2(static_cast<PotentialInt*>(Vfact2->ao_potential()));
    Vint2->set_charge_field(Zxyz2);
    auto Vtemp2 = std::make_shared<Matrix>("Vtemp2", nn, nn);

    for (int A = 0; A < nA; A++) {
        Vtemp2->zero();
        Zxyz2p[0][0] = monomer_A_->Z(cA[A]);
        Zxyz2p[0][1] = monomer_A_->x(cA[A]);
        Zxyz2p[0][2] = monomer_A_->y(cA[A]);
        Zxyz2p[0][3] = monomer_A_->z(cA[A]); 
        Vint2->compute(Vtemp2);
        std::shared_ptr<Matrix> Vbs = linalg::triplet(Cocc_B_, Vtemp2, Cvir_B_, true, false, false);
        dfh_->write_disk_tensor("WAbs_nuc", Vbs, {A, A+1});
        //dfh_->write_disk_tensor("WAbs", Vbs, {A, A+1});
    }

    for (int B = 0; B < nB; B++) {
        Vtemp2->zero();
        Zxyz2p[0][0] = monomer_B_->Z(cB[B]);
        Zxyz2p[0][1] = monomer_B_->x(cB[B]);
        Zxyz2p[0][2] = monomer_B_->y(cB[B]);
        Zxyz2p[0][3] = monomer_B_->z(cB[B]); 
        Vint2->compute(Vtemp2);
        std::shared_ptr<Matrix> Var = linalg::triplet(Cocc_A_, Vtemp2, Cvir_A_, true, false, false);
        dfh_->write_disk_tensor("WBar_nuc", Var, {B, B+1});
        //dfh_->write_disk_tensor("WBar", Var, {B, B+1});
    }

    // => Electronic Part (Massive PITA) <= //

    dfh_->add_disk_tensor("WAbs", std::make_tuple(nA, nb, ns));
    dfh_->add_disk_tensor("WBar", std::make_tuple(nB, na, nr));
    std::shared_ptr<Matrix> TsQ(new Matrix("TsQ",ns,nQ));
    std::shared_ptr<Matrix> T1As(new Matrix("T1As",nA,ns));
    std::shared_ptr<Matrix> T2As(new Matrix("T2As",1,ns));
    double** TsQp = TsQ->pointer(); 
    double** T1Asp = T1As->pointer(); 
    double** T2Asp = T2As->pointer(); 
    for (size_t b = 0; b < nb; b++) {
        dfh_->fill_tensor("Abs", TsQ, {b, b + 1});
        C_DGEMM('N', 'T', nA, ns, nQ, 1.0, RACp[0], nQ, TsQp[0], nQ, 0.0, T1Asp[0], ns); // ZLG 2.0 -> 1.0
        for (size_t A = 0; A < nA; A++) {
            dfh_->fill_tensor("WAbs_nuc", T2As, {A, A + 1}, {b, b + 1});
            C_DAXPY(ns, 1.0, T1Asp[A], 1, T2Asp[0], 1);  // ZLG 1.0 -> 2.0
            dfh_->write_disk_tensor("WAbs", T2As, {A, A + 1}, {b, b + 1});
            //dfh_->write_disk_tensor("WAbs", T1As, {A, A + 1}, {b, b + 1});
        }
    }

    std::shared_ptr<Matrix> TrQ(new Matrix("TrQ",nr,nQ));
    std::shared_ptr<Matrix> T1Br(new Matrix("T1Br",nB,nr));
    std::shared_ptr<Matrix> T2Br(new Matrix("T2Br",1,nr));
    double** TrQp = TrQ->pointer(); 
    double** T1Brp = T1Br->pointer(); 
    double** T2Brp = T2Br->pointer(); 
    for (size_t a = 0; a < na; a++) {
        dfh_->fill_tensor("Aar", TrQ, {a, a + 1});
        C_DGEMM('N', 'T', nB, nr, nQ, 1.0, RBDp[0], nQ, TrQp[0], nQ, 0.0, T1Brp[0], nr); // ZLG 2.0 -> 1.0
        for (size_t B = 0; B < nB; B++) {
            dfh_->fill_tensor("WBar_nuc", T2Br, {B, B + 1}, {a, a + 1});
            C_DAXPY(nr , 1.0, T1Brp[B], 1, T2Brp[0], 1); 
            dfh_->write_disk_tensor("WBar", T2Br, {B, B + 1}, {a, a + 1});
            //dfh_->write_disk_tensor("WBar", T1Br, {B, B + 1}, {a, a + 1});
        }
    }

}


void ASAPT0::elst()
{
    outfile->Printf("  ELECTROSTATICS:\n\n");

    // ==> Sizing <== //

    int nn  = primary_->nbf();

    int na = Cocc_A_->colspi()[0];
    int nb = Cocc_B_->colspi()[0];

    int nA = 0;
    std::vector<int> cA;
    for (int A = 0; A < monomer_A_->natom(); A++) {
        if (monomer_A_->Z(A) != 0.0) {
            nA++;
            cA.push_back(A);
        }
    }

    int nB = 0;
    std::vector<int> cB;
    for (int B = 0; B < monomer_B_->natom(); B++) {
        if (monomer_B_->Z(B) != 0.0) {
            nB++;
            cB.push_back(B);
        }
    }

    int nr = Cvir_A_->colspi()[0];
    int ns = Cvir_B_->colspi()[0];

    // ==> Dependencies <== //
    
    std::shared_ptr<Matrix> RAC = vars_["RAC"];
    std::shared_ptr<Matrix> RBD = vars_["RBD"];
    std::shared_ptr<Matrix> VAB = vars_["VAB"];
    std::shared_ptr<Matrix> VBA = vars_["VBA"];
    vars_.erase("RAC");
    vars_.erase("RBD");
    vars_.erase("VAB");
    vars_.erase("VBA");

    double** VABp = VAB->pointer();
    double** VBAp = VBA->pointer();

    // ==> Elst <== //

    double Elst10 = 0.0;
    std::vector<double> Elst10_terms;
    Elst10_terms.resize(4);

    std::shared_ptr<Matrix> Elst_atoms(new Matrix("Elst (A x B)", nA, nB));
    double** Elst_atomsp = Elst_atoms->pointer();

    // => A <-> B <= //
     
    for (int A = 0; A < nA; A++) {
        for (int B = 0; B < nB; B++) {
            double val = monomer_A_->Z(cA[A]) * monomer_B_->Z(cB[B]) / (monomer_A_->xyz(cA[A]).distance(monomer_B_->xyz(cB[B])));
            Elst10_terms[3] += val;
            Elst_atomsp[A][B] += val;
        }
    }

    // => a <-> b <= //

    std::shared_ptr<Matrix> Elst10_3 = linalg::doublet(RAC,RBD,false,true);
    double** Elst10_3p = Elst10_3->pointer();
    for (int A = 0; A < nA; A++) {
        for (int B = 0; B < nB; B++) {
            double val = /*4.0 **/ Elst10_3p[A][B];
            Elst10_terms[2] += val;
            Elst_atomsp[A][B] += val;
        }
    }

    // => A <-> b <= //

    for (int A = 0; A < nA; A++) {
        for (int B = 0; B < nB; B++) {
            double val = VBAp[B][A];
            Elst10_terms[1] += val;
            Elst_atomsp[A][B] += val;
        }
    }

    // => a <-> B <= //

    for (int B = 0; B < nB; B++) {
        for (int A = 0; A < nA; A++) {
            double val = VABp[A][B];
            Elst10_terms[0] += val;
            Elst_atomsp[A][B] += val;
        }
    }

    for (int k = 0; k < Elst10_terms.size(); k++) {
        Elst10 += Elst10_terms[k];
    }
    //if (debug_) {
        for (int k = 0; k < Elst10_terms.size(); k++) {
            outfile->Printf("    Elst10,r (%1d)        = %18.12lf H\n",k+1,Elst10_terms[k]);
        }
    //}
    energies_["Elst10,r"] = Elst10;
    outfile->Printf("    Elst10,r            = %18.12lf H\n",Elst10);
    outfile->Printf("\n");

    // Grid error analysis 
    outfile->Printf("  ==> Grid Errors <==\n\n");
    outfile->Printf("    True Elst10,r       = %18.12lf H\n",energies_["Elst10,r"]);
    outfile->Printf("    Grid Elst10,r       = %18.12lf H\n",Elst10);
    outfile->Printf("    Grid Elst10,r Error = %18.12lf H\n",Elst10 - energies_["Elst10,r"]);
    outfile->Printf("    Grid Elst10,r Rel   = %18.3E -\n",(Elst10 - energies_["Elst10,r"]) / energies_["Elst10,r"]);
    outfile->Printf("\n");

    //vis_->vars()["Elst_AB"] = Elst_atoms;
    Elst_AB = Elst_atoms;
    dfh_->clear_spaces();
}






void ASAPT0::exch()
{
    outfile->Printf("  EXCHANGE:\n\n");

    // ==> Sizing <== //

    int nn = primary_->nbf();
    int na = Cocc_A_->colspi()[0];
    int nb = Cocc_B_->colspi()[0];
    //int nA = mol->natom();
    //int nB = mol->natom();
    int nr = Cvir_A_->colspi()[0];
    int ns = Cvir_B_->colspi()[0];
    int nQ = jkfit_->nbf();

    // ==> Stack Variables <== //

    std::shared_ptr<Matrix> S   = vars_["S"];
    std::shared_ptr<Matrix> V_A = vars_["V_A"];
    std::shared_ptr<Matrix> J_A = vars_["J_A"];
    std::shared_ptr<Matrix> V_B = vars_["V_B"];
    std::shared_ptr<Matrix> J_B = vars_["J_B"];

    outfile->Printf("Monomer A orbitals: %d occ, %d vir, %f loc\n", na, nr, Locc_A_->colspi()[0]);
    outfile->Printf("Monomer B orbitals: %d occ, %d vir, %f loc\n", nb, ns, Locc_B_->colspi()[0]);

    // ==> DF ERI Setup (JKFIT Type, in Full Basis) <== //

    std::vector<std::shared_ptr<Matrix> > Cs;
    Cs.push_back(Locc_A_);
    Cs.push_back(Cvir_A_);
    Cs.push_back(Locc_B_);
    Cs.push_back(Cvir_B_);

    size_t doubles = Process::environment.get_memory() * 0.8 / sizeof(double);
    //size_t max_MO = 0;
    //for (auto& mat : Cstack_vec) max_MO = std::max(max_MO, (size_t)mat->ncol());

    // Build DFHelper
    //dfh_->clear_all();
    //dfh_ = std::make_shared<DFHelper>(primary_, jkfit_);
    //dfh_->set_memory(doubles);
    //dfh_->set_method("DIRECT_iaQ");
    //dfh_->set_nthreads(1); // TODO
    //dfh_->set_metric_pow(-0.5);

    dfh_->initialize();

    // Define spaces
    dfh_->add_space("a", Cs[0]);
    dfh_->add_space("r", Cs[1]);
    dfh_->add_space("b", Cs[2]);
    dfh_->add_space("s", Cs[3]);

    // add transformations
    dfh_->add_transformation("Aar", "a", "r");
    dfh_->add_transformation("Abs", "b", "s");

    // transform
    dfh_->transform();
    dfh_->print_header();

    // ==> Electrostatic Potentials <== //

    std::shared_ptr<Matrix> W_A(J_A->clone());
    W_A->copy(J_A);
    W_A->scale(2.0);
    W_A->add(V_A);

    std::shared_ptr<Matrix> W_B(J_B->clone());
    W_B->copy(J_B);
    W_B->scale(2.0);
    W_B->add(V_B);

    std::shared_ptr<Matrix> WAbs = linalg::triplet(Locc_B_, W_A, Cvir_B_, true, false, false);
    std::shared_ptr<Matrix> WBar = linalg::triplet(Locc_A_, W_B, Cvir_A_, true, false, false);
    double** WBarp = WBar->pointer();
    double** WAbsp = WAbs->pointer();

    W_A.reset();
    W_B.reset();

    // ==> Exchange S^2 Computation <== //

    std::shared_ptr<Matrix> Sab = linalg::triplet(Locc_A_, S, Locc_B_, true, false, false);
    std::shared_ptr<Matrix> Sba = linalg::triplet(Locc_B_, S, Locc_A_, true, false, false);
    std::shared_ptr<Matrix> Sas = linalg::triplet(Locc_A_, S, Cvir_B_, true, false, false);
    std::shared_ptr<Matrix> Sbr = linalg::triplet(Locc_B_, S, Cvir_A_, true, false, false);
    double** Sabp = Sab->pointer();
    double** Sbap = Sba->pointer();
    double** Sasp = Sas->pointer();
    double** Sbrp = Sbr->pointer();

    auto WBab = std::make_shared<Matrix>("WBab", na, nb);
    double** WBabp = WBab->pointer();
    auto WAba = std::make_shared<Matrix>("WAba", nb, na);
    double** WAbap = WAba->pointer();

    C_DGEMM('N', 'T', na, nb, nr, 1.0, WBarp[0], nr, Sbrp[0], nr, 0.0, WBabp[0], nb);
    C_DGEMM('N', 'T', nb, na, ns, 1.0, WAbsp[0], ns, Sasp[0], ns, 0.0, WAbap[0], na);

    auto E_exch1 = std::make_shared<Matrix>("E_exch [a <x- b]", na, nb);
    double** E_exch1p = E_exch1->pointer();
    auto E_exch2 = std::make_shared<Matrix>("E_exch [a -x> b]", na, nb);
    double** E_exch2p = E_exch2->pointer();

    for (int a = 0; a < na; a++) {
        for (int b = 0; b < nb; b++) {
            E_exch1p[a][b] -= 2.0 * Sabp[a][b] * WBabp[a][b];
            E_exch2p[a][b] -= 2.0 * Sbap[b][a] * WAbap[b][a];
        }
    }

    // E_exch1->print_out();
    // E_exch2->print_out();

    auto TrQ = std::make_shared<Matrix>("TrQ", nr, nQ);
    double** TrQp = TrQ->pointer();
    auto TsQ = std::make_shared<Matrix>("TsQ", ns, nQ);
    double** TsQp = TsQ->pointer();
    auto TbQ = std::make_shared<Matrix>("TbQ", nb, nQ);
    double** TbQp = TbQ->pointer();
    auto TaQ = std::make_shared<Matrix>("TaQ", na, nQ);
    double** TaQp = TaQ->pointer();

    // ZLG start here

    dfh_->add_disk_tensor("Bab", std::make_tuple(na, nb, nQ));

    for (size_t a = 0; a < na; a++) {
        dfh_->fill_tensor("Aar", TrQ, {a, a + 1});
        C_DGEMM('N', 'N', nb, nQ, nr, 1.0, Sbrp[0], nr, TrQp[0], nQ, 0.0, TbQp[0], nQ);
        dfh_->write_disk_tensor("Bab", TbQ, {a, a + 1});
    }

    dfh_->add_disk_tensor("Bba", std::make_tuple(nb, na, nQ));

    for (size_t b = 0; b < nb; b++) {
        dfh_->fill_tensor("Abs", TsQ, {b, b + 1});
        C_DGEMM('N', 'N', na, nQ, ns, 1.0, Sasp[0], ns, TsQp[0], nQ, 0.0, TaQp[0], nQ);
        dfh_->write_disk_tensor("Bba", TaQ, {b, b + 1});
    }

    auto E_exch3 = std::make_shared<Matrix>("E_exch [a <x-x> b]", na, nb);
    double** E_exch3p = E_exch3->pointer();

    for (size_t a = 0; a < na; a++) {
        dfh_->fill_tensor("Bab", TbQ, {a, a + 1});
        for (size_t b = 0; b < nb; b++) {
            dfh_->fill_tensor("Bba", TaQ, {b, b + 1}, {a, a + 1});
            E_exch3p[a][b] -= 2.0 * C_DDOT(nQ, TbQp[b], 1, TaQp[0], 1);
        }
    }

    E_exch3->print_out();

    dfh_->clear_spaces();
    //ZLG: End of stuff I have to mess with
      
    auto E_exch = std::make_shared<Matrix>("E_exch (a x b)", na, nb);
    double** E_exchp = E_exch->pointer();

    double Exch10_2 = 0.0;
    std::vector<double> Exch10_2_terms;
    Exch10_2_terms.resize(3);

    for (int a = 0; a < na; a++) {
        for (int b = 0; b < nb; b++) {
            E_exchp[a][b] = E_exch1p[a][b] +
                            E_exch2p[a][b] +
                            E_exch3p[a][b];
            Exch10_2_terms[0] += E_exch1p[a][b];
            Exch10_2_terms[1] += E_exch2p[a][b];
            Exch10_2_terms[2] += E_exch3p[a][b];
        }
    }

    for (int k = 0; k < Exch10_2_terms.size(); k++) {
        Exch10_2 += Exch10_2_terms[k];
    }
    //if (debug_) {
        for (int k = 0; k < Exch10_2_terms.size(); k++) {
            outfile->Printf("    Exch10(S^2) (%1d)     = %18.12lf H\n",k+1,Exch10_2_terms[k]);
        }
    //}
    energies_["Exch10(S^2)"] = Exch10_2;
    outfile->Printf("    Exch10(S^2)         = %18.12lf [Eh]\n", Exch10_2);
    outfile->Printf("\n");

    // => Exchange scaling <= //

    if(exch_scale_ && sSAPT0_scale_) {
        throw PSIEXCEPTION("Exchange scaling and sSAPT0 both selected\n\n");
    }
    else if (exch_scale_) {
        double scale = energies_["Exch10"] / energies_["Exch10(S^2)"];
        E_exch->scale(scale);
        outfile->Printf("    Scaling ASAPT Exchange by %11.3E to match S^\\infty\n\n", scale);
    } else if (sSAPT0_scale_) {
        double sSAPT0_scale_factor_ = energies_["Exch10"] / energies_["Exch10(S^2)"];
        sSAPT0_scale_factor_ = pow(sSAPT0_scale_factor_,3.0);
        outfile->Printf("    Scaling ASAPT Exch-Ind and Exch-Disp by %11.3E \n\n", sSAPT0_scale_factor_);
    }
    
    //vis_->vars()["Exch_ab"] = E_exch;
    Exch_ab = E_exch;
}


void ASAPT0::ind()
{
    outfile->Printf( "  INDUCTION:\n\n");

    // => Sizing <= //

    int nn = primary_->nbf();

    int na = Cocc_A_->colspi()[0];
    int nb = Cocc_B_->colspi()[0];
    int nA = 0;
    std::vector<int> cA;
    for (int A = 0; A < monomer_A_->natom(); A++) {
        if (monomer_A_->Z(A) != 0.0) {
            nA++;
            cA.push_back(A);
        }
    }

    int nB = 0;
    std::vector<int> cB;
    for (int B = 0; B < monomer_B_->natom(); B++) {
        if (monomer_B_->Z(B) != 0.0) {
            nB++;
            cB.push_back(B);
        }
    }

    int nr = Cvir_A_->colspi()[0];
    int ns = Cvir_B_->colspi()[0];

    // ==> Stack Variables <== //

    double* eap = eps_occ_A_->pointer();
    double* ebp = eps_occ_B_->pointer();
    double* erp = eps_vir_A_->pointer();
    double* esp = eps_vir_B_->pointer();

    std::shared_ptr<Matrix> S = vars_["S"];
    std::shared_ptr<Matrix> D_A = vars_["D_A"];
    std::shared_ptr<Matrix> V_A = vars_["V_A"];
    std::shared_ptr<Matrix> J_A = vars_["J_A"];
    std::shared_ptr<Matrix> K_A = vars_["K_A"];
    std::shared_ptr<Matrix> D_B = vars_["D_B"];
    std::shared_ptr<Matrix> V_B = vars_["V_B"];
    std::shared_ptr<Matrix> J_B = vars_["J_B"];
    std::shared_ptr<Matrix> K_B = vars_["K_B"];
    std::shared_ptr<Matrix> J_O = vars_["J_O"];
    std::shared_ptr<Matrix> K_O = vars_["K_O"];
    std::shared_ptr<Matrix> J_P_A = vars_["J_P_A"];
    std::shared_ptr<Matrix> J_P_B = vars_["J_P_B"];

    // ==> MO Amplitudes/Sources (by source atom) <== //

    auto xA = std::make_shared<Matrix>("xA", na, nr);
    auto xB = std::make_shared<Matrix>("xB", nb, ns);
    double** xAp = xA->pointer();
    double** xBp = xB->pointer();

    auto wB = std::make_shared<Matrix>("wB", na, nr);
    auto wA = std::make_shared<Matrix>("wA", nb, ns);
    double** wBp = wB->pointer();
    double** wAp = wA->pointer();

    // ==> Generalized ESP (Flat and Exchange) <== //

    std::map<std::string, std::shared_ptr<Matrix> > mapA;
    mapA["Cocc_A"] = Locc_A_;
    mapA["Cvir_A"] = Cvir_A_;
    mapA["Cocc_B"] = Locc_B_;
    mapA["Cvir_B"] = Cvir_B_;
    mapA["S"] = S;
    mapA["D_A"] = D_A;
    mapA["V_A"] = V_A;
    mapA["J_A"] = J_A;
    mapA["K_A"] = K_A;
    mapA["D_B"] = D_B;
    mapA["V_B"] = V_B;
    mapA["J_B"] = J_B;
    mapA["K_B"] = K_B;
    mapA["J_O"] = J_O;
    mapA["K_O"] = K_O;
    mapA["J_P"] = J_P_A;

    std::shared_ptr<Matrix> wBT = build_ind_pot(mapA);
    std::shared_ptr<Matrix> uBT = build_exch_ind_pot(mapA);
    double** wBTp = wBT->pointer();
    double** uBTp = uBT->pointer();

    K_O->transpose_this();

    std::map<std::string, std::shared_ptr<Matrix> > mapB;
    mapB["Cocc_A"] = Locc_B_;
    mapB["Cvir_A"] = Cvir_B_;
    mapB["Cocc_B"] = Locc_A_;
    mapB["Cvir_B"] = Cvir_A_;
    mapB["S"] = S;
    mapB["D_A"] = D_B;
    mapB["V_A"] = V_B;
    mapB["J_A"] = J_B;
    mapB["K_A"] = K_B;
    mapB["D_B"] = D_A;
    mapB["V_B"] = V_A;
    mapB["J_B"] = J_A;
    mapB["K_B"] = K_A;
    mapB["J_O"] = J_O;
    mapB["K_O"] = K_O;
    mapB["J_P"] = J_P_B;

    std::shared_ptr<Matrix> wAT = build_ind_pot(mapB);
    std::shared_ptr<Matrix> uAT = build_exch_ind_pot(mapB);
    double** wATp = wAT->pointer();
    double** uATp = uAT->pointer();

    K_O->transpose_this();

    // ==> Uncoupled Targets <== //

    auto Ind20u_AB_terms = std::make_shared<Matrix>("Ind20 [A<-B] (a x B)", na, nB);
    auto Ind20u_BA_terms = std::make_shared<Matrix>("Ind20 [B<-A] (A x b)", nA, nb);
    double** Ind20u_AB_termsp = Ind20u_AB_terms->pointer();
    double** Ind20u_BA_termsp = Ind20u_BA_terms->pointer();

    double Ind20u_AB = 0.0;
    double Ind20u_BA = 0.0;

    auto ExchInd20u_AB_terms = std::make_shared<Matrix>("ExchInd20 [A<-B] (a x B)", na, nB);
    auto ExchInd20u_BA_terms = std::make_shared<Matrix>("ExchInd20 [B<-A] (A x b)", nA, nb);
    double** ExchInd20u_AB_termsp = ExchInd20u_AB_terms->pointer();
    double** ExchInd20u_BA_termsp = ExchInd20u_BA_terms->pointer();

    double ExchInd20u_AB = 0.0;
    double ExchInd20u_BA = 0.0;

    auto Indu_AB_terms = std::make_shared<Matrix>("Ind [A<-B] (a x B)", na, nB);
    auto Indu_BA_terms = std::make_shared<Matrix>("Ind [B<-A] (A x b)", nA, nb);
    double** Indu_AB_termsp = Indu_AB_terms->pointer();
    double** Indu_BA_termsp = Indu_BA_terms->pointer();

    double Indu_AB = 0.0;
    double Indu_BA = 0.0;

    // ==> A <- B Uncoupled <== //

    
    for (size_t B = 0; B < nB; B++) {

        // ESP
        dfh_->fill_tensor("WBar", wBp[0], {B, B+1});
        
        // Uncoupled amplitude
        for (int a = 0; a < na; a++) {
            for (int r = 0; r < nr; r++) {
                xAp[a][r] = wBp[a][r] / (eap[a] - erp[r]);
            }
        }

        // Backtransform the amplitude to LO
        xA->scale(-1.0); // ZLG: Debug
        std::shared_ptr<Matrix> x2A = linalg::doublet(Uocc_A_, xA, true, false);
        double** x2Ap = x2A->pointer();

        // Zip up the Ind20 contributions
        for (int a = 0; a < na; a++) {
            double Jval = 2.0 * C_DDOT(nr,x2Ap[a], 1, wBTp[a], 1);
            double Kval = 2.0 * C_DDOT(nr,x2Ap[a], 1, uBTp[a], 1);
            Ind20u_AB_termsp[a][B] = Jval;
            Ind20u_AB += Jval;
            ExchInd20u_AB_termsp[a][B] = Kval;
            ExchInd20u_AB += Kval;
            Indu_AB_termsp[a][B] = Jval + Kval;
            Indu_AB += Jval + Kval;
        }

    } 

    // ==> B <- A Uncoupled <== //

    for (size_t A = 0; A < nA; A++) {

        // ESP
        //fread(wAp[0],sizeof(double),nb*ns,WAbsf); 
        dfh_->fill_tensor("WAbs", wAp[0], {A, A+1});
        
        // Uncoupled amplitude
        for (int b = 0; b < nb; b++) {
            for (int s = 0; s < ns; s++) {
                xBp[b][s] = wAp[b][s] / (ebp[b] - esp[s]);
            }
        }

        // Backtransform the amplitude to LO
        xB->scale(-1.0); // ZLG: Debug
        std::shared_ptr<Matrix> x2B = linalg::doublet(Uocc_B_, xB, true, false);
        double** x2Bp = x2B->pointer();

        // Zip up the Ind20 contributions
        for (int b = 0; b < nb; b++) {
            double Jval = 2.0 * C_DDOT(ns,x2Bp[b], 1, wATp[b], 1);
            double Kval = 2.0 * C_DDOT(ns,x2Bp[b], 1, uATp[b], 1);
            Ind20u_BA_termsp[A][b] = Jval;
            Ind20u_BA += Jval;
            ExchInd20u_BA_termsp[A][b] = Kval;
            ExchInd20u_BA += Kval;
            Indu_BA_termsp[A][b] = Jval + Kval;
            Indu_BA += Jval + Kval;
        }
    }
    dfh_->clear_spaces();

    double Ind20u = Ind20u_AB + Ind20u_BA;
    outfile->Printf("    Ind20,u (A<-B)      = %18.12lf [Eh]\n", Ind20u_AB);
    outfile->Printf("    Ind20,u (B<-A)      = %18.12lf [Eh]\n", Ind20u_BA);
    outfile->Printf("    Ind20,u             = %18.12lf [Eh]\n", Ind20u);

    double ExchInd20u = ExchInd20u_AB + ExchInd20u_BA;
    outfile->Printf("    Exch-Ind20,u (A<-B) = %18.12lf [Eh]\n", ExchInd20u_AB);
    outfile->Printf("    Exch-Ind20,u (B<-A) = %18.12lf [Eh]\n", ExchInd20u_BA);
    outfile->Printf("    Exch-Ind20,u        = %18.12lf [Eh]\n", ExchInd20u);
    outfile->Printf("\n");

    double Ind = Ind20u + ExchInd20u;
    std::shared_ptr<Matrix> Ind_AB_terms = Indu_AB_terms;
    std::shared_ptr<Matrix> Ind_BA_terms = Indu_BA_terms;

    ind_resp_ = true;
    // No coupled induction for now
    if (ind_resp_) {

        outfile->Printf( "  COUPLED INDUCTION (You asked for it!):\n\n");

        // ==> Coupled Targets <== //

        std::shared_ptr<Matrix> Ind20r_AB_terms(new Matrix("Ind20 [A<-B] (a x B)", na, nB));
        std::shared_ptr<Matrix> Ind20r_BA_terms(new Matrix("Ind20 [B<-A] (A x b)", nA, nb));
        double** Ind20r_AB_termsp = Ind20r_AB_terms->pointer();
        double** Ind20r_BA_termsp = Ind20r_BA_terms->pointer();

        double Ind20r_AB = 0.0; 
        double Ind20r_BA = 0.0;

        std::shared_ptr<Matrix> ExchInd20r_AB_terms(new Matrix("ExchInd20 [A<-B] (a x B)", na, nB));
        std::shared_ptr<Matrix> ExchInd20r_BA_terms(new Matrix("ExchInd20 [B<-A] (A x b)", nA, nb));
        double** ExchInd20r_AB_termsp = ExchInd20r_AB_terms->pointer();
        double** ExchInd20r_BA_termsp = ExchInd20r_BA_terms->pointer();

        double ExchInd20r_AB = 0.0; 
        double ExchInd20r_BA = 0.0;

        std::shared_ptr<Matrix> Indr_AB_terms(new Matrix("Ind [A<-B] (a x B)", na, nB));
        std::shared_ptr<Matrix> Indr_BA_terms(new Matrix("Ind [B<-A] (A x b)", nA, nb));
        double** Indr_AB_termsp = Indr_AB_terms->pointer();
        double** Indr_BA_termsp = Indr_BA_terms->pointer();

        double Indr_AB = 0.0; 
        double Indr_BA = 0.0;

        // => JK Object <= //

        //std::shared_ptr<JK> jk = JK::build_JK();


        // TODO: Account for 2-index overhead in memory
        int nso = Cocc_A_->nrow();
        long int jk_memory = (long int)memory_;
        jk_memory -= 24 * nso * nso;
        jk_memory -=  4 * na * nso;
        jk_memory -=  4 * nb * nso;
        if (jk_memory < 0L) {
            throw PSIEXCEPTION("Too little static memory for ASAPT::induction");
        }
        auto jk = JK::build_JK(primary_, jkfit_, options_, false, jk_memory);
        jk->set_memory((unsigned long int )jk_memory);
        jk->set_do_J(true);
        jk->set_do_K(true);
        jk->initialize();
        jk->print_header();

        // ==> Master Loop over perturbing atoms <== //
    
        int nC = std::max(nA,nB);

        //fseek(WBarf,0L,SEEK_SET);
        //fseek(WAbsf,0L,SEEK_SET);
        
        for (int C = 0; C < nC; C++) {
            
            //if (C < nB) fread(wBp[0],sizeof(double),na*nr,WBarf); 
            if (C < nB) dfh_->fill_tensor("WBar", wB, {C, C+1});

            //if (C < nA) fread(wAp[0],sizeof(double),nb*ns,WAbsf); 
            if (C < nA) dfh_->fill_tensor("WAbs", wA, {C, C+1});

            outfile->Printf("    Responses for (A <- Atom B = %3d) and (B <- Atom A = %3d)\n\n",
                    (C < nB ? C : nB - 1), (C < nA ? C : nA - 1));

            std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix> > x_sol = compute_x(jk,wB,wA);
            xA = x_sol.first;
            xB = x_sol.second;
            xA->scale(-1.0);
            xB->scale(-1.0);

            if (C < nB) {
                // Backtransform the amplitude to LO
                std::shared_ptr<Matrix> x2A = linalg::doublet(Uocc_A_,xA,true,false);
                double** x2Ap = x2A->pointer();

                // Zip up the Ind20 contributions
                for (int a = 0; a < na; a++) {
                    double Jval = 2.0 * C_DDOT(nr,x2Ap[a],1,wBTp[a],1);
                    double Kval = 2.0 * C_DDOT(nr,x2Ap[a],1,uBTp[a],1);
                    Ind20r_AB_termsp[a][C] = Jval;
                    Ind20r_AB += Jval;
                    ExchInd20r_AB_termsp[a][C] = Kval;
                    ExchInd20r_AB += Kval;
                    Indr_AB_termsp[a][C] = Jval + Kval;
                    Indr_AB += Jval + Kval;
                }
            }

            if (C < nA) { 
                // Backtransform the amplitude to LO
                std::shared_ptr<Matrix> x2B = linalg::doublet(Uocc_B_,xB,true,false);
                double** x2Bp = x2B->pointer();

                // Zip up the Ind20 contributions
                for (int b = 0; b < nb; b++) {
                    double Jval = 2.0 * C_DDOT(ns,x2Bp[b],1,wATp[b],1);
                    double Kval = 2.0 * C_DDOT(ns,x2Bp[b],1,uATp[b],1);
                    Ind20r_BA_termsp[C][b] = Jval;
                    Ind20r_BA += Jval;
                    ExchInd20r_BA_termsp[C][b] = Kval;
                    ExchInd20r_BA += Kval;
                    Indr_BA_termsp[C][b] = Jval + Kval;
                    Indr_BA += Jval + Kval;
                }
            }
        }

        double Ind20r = Ind20r_AB + Ind20r_BA;
        outfile->Printf("    Ind20,r (A<-B)      = %18.12lf H\n",Ind20r_AB);
        outfile->Printf("    Ind20,r (B<-A)      = %18.12lf H\n",Ind20r_BA);
        outfile->Printf("    Ind20,r             = %18.12lf H\n",Ind20r);

        double ExchInd20r = ExchInd20r_AB + ExchInd20r_BA;
        outfile->Printf("    Exch-Ind20,r (A<-B) = %18.12lf H\n",ExchInd20r_AB);
        outfile->Printf("    Exch-Ind20,r (B<-A) = %18.12lf H\n",ExchInd20r_BA);
        outfile->Printf("    Exch-Ind20,r        = %18.12lf H\n",ExchInd20r);
        outfile->Printf("\n");

        Ind = Ind20r + ExchInd20r;
        Ind_AB_terms = Indr_AB_terms;
        Ind_BA_terms = Indr_BA_terms;
    }

    // => Induction scaling <= //

    if (ind_scale_) {
        double dHF = 0.0;
        if (energies_["HF"] != 0.0) {
            dHF = energies_["HF"] - energies_["Elst10,r"] - energies_["Exch10"] - energies_["Ind20,r"] - energies_["Exch-Ind20,r"];
        } 
        outfile->Printf("Delta Hartree Fock term %f \n", dHF);

        double IndHF = energies_["Ind20,r"] + energies_["Exch-Ind20,r"] + dHF;
        double IndSAPT0 = energies_["Ind20,r"] + energies_["Exch-Ind20,r"];

        double Sdelta = IndHF / IndSAPT0;
        double SrAB = (ind_resp_ ? 1.0 : (energies_["Ind20,r (A<-B)"] + energies_["Exch-Ind20,r (A<-B)"]) / (energies_["Ind20,u (A<-B)"] + energies_["Exch-Ind20,u (A<-B)"]));
        double SrBA = (ind_resp_ ? 1.0 : (energies_["Ind20,r (B<-A)"] + energies_["Exch-Ind20,r (B<-A)"]) / (energies_["Ind20,u (B<-A)"] + energies_["Exch-Ind20,u (B<-A)"]));

        outfile->Printf( "    Scaling for delta HF        = %11.3E\n", Sdelta);
        outfile->Printf( "    Scaling for response (A<-B) = %11.3E\n", SrAB);
        outfile->Printf( "    Scaling for response (B<-A) = %11.3E\n", SrBA);
        outfile->Printf( "    Scaling for total (A<-B)    = %11.3E\n", Sdelta * SrAB);
        outfile->Printf( "    Scaling for total (B<-A)    = %11.3E\n", Sdelta * SrBA);
        outfile->Printf( "\n");

        Ind_AB_terms->scale(Sdelta * SrAB);
        Ind_BA_terms->scale(Sdelta * SrBA);
    }
    
    //vis_->vars()["IndAB_aB"] = Ind_AB_terms;
    //vis_->vars()["IndBA_Ab"] = Ind_BA_terms;
    IndAB_aB = Ind_AB_terms;
    IndBA_Ab = Ind_BA_terms;
}

void ASAPT0::disp()
{
    outfile->Printf( "  DISPERSION:\n\n");

    // => Sizing <= //

    int nn = primary_->nbf();

    int naa = Caocc_A_->colspi()[0];
    int nab = Caocc_B_->colspi()[0];

    int na  = Locc_A_->colspi()[0]; 
    int nb  = Locc_B_->colspi()[0]; 

    int nfa = na - naa;
    int nfb = nb - nab;

    int nA = 0;
    std::vector<int> cA;
    for (int A = 0; A < monomer_A_->natom(); A++) {
        if (monomer_A_->Z(A) != 0.0) {
            nA++;
            cA.push_back(A);
        }
    }

    int nB = 0;
    std::vector<int> cB;
    for (int B = 0; B < monomer_B_->natom(); B++) {
        if (monomer_B_->Z(B) != 0.0) {
            nB++;
            cB.push_back(B);
        }
    }

    int nr = Cavir_A_->colspi()[0];
    int ns = Cavir_B_->colspi()[0];
    int nQ = mp2fit_->nbf();
    size_t naQ = naa * (size_t) nQ;
    size_t nbQ = nab * (size_t) nQ;

    int nT = 1;
    #ifdef _OPENMP
        nT = omp_get_max_threads();
    #endif

    // => Stashed Variables <= //

    std::shared_ptr<Matrix> S   = vars_["S"];
    std::shared_ptr<Matrix> U_A = vars_["U_A"];
    std::shared_ptr<Matrix> L_A = vars_["L_A"];
    std::shared_ptr<Matrix> D_A = vars_["D_A"];
    std::shared_ptr<Matrix> P_A = vars_["P_A"];
    std::shared_ptr<Matrix> V_A = vars_["V_A"];
    std::shared_ptr<Matrix> J_A = vars_["J_A"];
    std::shared_ptr<Matrix> K_A = vars_["K_A"];
    std::shared_ptr<Matrix> U_B = vars_["U_B"];
    std::shared_ptr<Matrix> L_B = vars_["L_B"];
    std::shared_ptr<Matrix> D_B = vars_["D_B"];
    std::shared_ptr<Matrix> P_B = vars_["P_B"];
    std::shared_ptr<Matrix> V_B = vars_["V_B"];
    std::shared_ptr<Matrix> J_B = vars_["J_B"];
    std::shared_ptr<Matrix> K_B = vars_["K_B"];
    std::shared_ptr<Matrix> K_O = vars_["K_O"];

    std::shared_ptr<Matrix> Q2A = Q_A_;
    std::shared_ptr<Matrix> Q2B = Q_B_;
    double** Q2Ap = Q2A->pointer();
    double** Q2Bp = Q2B->pointer();

    // => Auxiliary C matrices <= //

    std::shared_ptr<Matrix> Cr1 = linalg::triplet(D_B,S,Cavir_A_);
    Cr1->scale(-1.0);
    Cr1->add(Cavir_A_);
    std::shared_ptr<Matrix> Cs1 = linalg::triplet(D_A,S,Cavir_B_);
    Cs1->scale(-1.0);
    Cs1->add(Cavir_B_);
    std::shared_ptr<Matrix> Ca2 = linalg::triplet(D_B,S,Caocc_A_);
    std::shared_ptr<Matrix> Cb2 = linalg::triplet(D_A,S,Caocc_B_);
    std::shared_ptr<Matrix> Cr3 = linalg::triplet(D_B,S,Cavir_A_);
    std::shared_ptr<Matrix> CrX = linalg::triplet(linalg::triplet(D_A,S,D_B),S,Cavir_A_);
    Cr3->subtract(CrX);
    Cr3->scale(2.0);
    std::shared_ptr<Matrix> Cs3 = linalg::triplet(D_A,S,Cavir_B_);
    std::shared_ptr<Matrix> CsX = linalg::triplet(linalg::triplet(D_B,S,D_A),S,Cavir_B_);
    Cs3->subtract(CsX);
    Cs3->scale(2.0);
    std::shared_ptr<Matrix> Ca4 = linalg::triplet(linalg::triplet(D_A,S,D_B),S,Caocc_A_);
    Ca4->scale(-2.0);
    std::shared_ptr<Matrix> Cb4 = linalg::triplet(linalg::triplet(D_B,S,D_A),S,Caocc_B_);
    Cb4->scale(-2.0);

    // => Auxiliary V matrices <= //

    std::shared_ptr<Matrix> Jbr = linalg::triplet(Caocc_B_,J_A,Cavir_A_,true,false,false);
    Jbr->scale(2.0);
    std::shared_ptr<Matrix> Kbr = linalg::triplet(Caocc_B_,K_A,Cavir_A_,true,false,false);
    Kbr->scale(-1.0);

    std::shared_ptr<Matrix> Jas = linalg::triplet(Caocc_A_,J_B,Cavir_B_,true,false,false);
    Jas->scale(2.0);
    std::shared_ptr<Matrix> Kas = linalg::triplet(Caocc_A_,K_B,Cavir_B_,true,false,false);
    Kas->scale(-1.0);

    std::shared_ptr<Matrix> KOas = linalg::triplet(Caocc_A_,K_O,Cavir_B_,true,false,false);
    KOas->scale(1.0);
    std::shared_ptr<Matrix> KObr = linalg::triplet(Caocc_B_,K_O,Cavir_A_,true,true,false);
    KObr->scale(1.0);

    std::shared_ptr<Matrix> JBas = linalg::triplet(linalg::triplet(Caocc_A_,S,D_B,true,false,false),J_A,Cavir_B_);
    JBas->scale(-2.0);
    std::shared_ptr<Matrix> JAbr = linalg::triplet(linalg::triplet(Caocc_B_,S,D_A,true,false,false),J_B,Cavir_A_);
    JAbr->scale(-2.0);

    std::shared_ptr<Matrix> Jbs = linalg::triplet(Caocc_B_,J_A,Cavir_B_,true,false,false);
    Jbs->scale(4.0);
    std::shared_ptr<Matrix> Jar = linalg::triplet(Caocc_A_,J_B,Cavir_A_,true,false,false);
    Jar->scale(4.0);

    std::shared_ptr<Matrix> JAas = linalg::triplet(linalg::triplet(Caocc_A_,J_B,D_A,true,false,false),S,Cavir_B_);
    JAas->scale(-2.0);
    std::shared_ptr<Matrix> JBbr = linalg::triplet(linalg::triplet(Caocc_B_,J_A,D_B,true,false,false),S,Cavir_A_);
    JBbr->scale(-2.0);

    // Get your signs right Hesselmann!
    std::shared_ptr<Matrix> Vbs = linalg::triplet(Caocc_B_,V_A,Cavir_B_,true,false,false);
    Vbs->scale(2.0);
    std::shared_ptr<Matrix> Var = linalg::triplet(Caocc_A_,V_B,Cavir_A_,true,false,false);
    Var->scale(2.0);
    std::shared_ptr<Matrix> VBas = linalg::triplet(linalg::triplet(Caocc_A_,S,D_B,true,false,false),V_A,Cavir_B_);
    VBas->scale(-1.0);
    std::shared_ptr<Matrix> VAbr = linalg::triplet(linalg::triplet(Caocc_B_,S,D_A,true,false,false),V_B,Cavir_A_);
    VAbr->scale(-1.0);
    std::shared_ptr<Matrix> VRas = linalg::triplet(linalg::triplet(Caocc_A_,V_B,P_A,true,false,false),S,Cavir_B_);
    VRas->scale(1.0);
    std::shared_ptr<Matrix> VSbr = linalg::triplet(linalg::triplet(Caocc_B_,V_A,P_B,true,false,false),S,Cavir_A_);
    VSbr->scale(1.0);

    std::shared_ptr<Matrix> Sas = linalg::triplet(Caocc_A_,S,Cavir_B_,true,false,false);
    std::shared_ptr<Matrix> Sbr = linalg::triplet(Caocc_B_,S,Cavir_A_,true,false,false);

    std::shared_ptr<Matrix> Qbr(Jbr->clone());
    Qbr->zero();
    Qbr->add(Jbr);
    Qbr->add(Kbr);
    Qbr->add(KObr);
    Qbr->add(JAbr);
    Qbr->add(JBbr);
    Qbr->add(VAbr);
    Qbr->add(VSbr);

    std::shared_ptr<Matrix> Qas(Jas->clone());
    Qas->zero();
    Qas->add(Jas);
    Qas->add(Kas);
    Qas->add(KOas);
    Qas->add(JAas);
    Qas->add(JBas);
    Qas->add(VBas);
    Qas->add(VRas);

    std::shared_ptr<Matrix> SBar = linalg::triplet(linalg::triplet(Caocc_A_,S,D_B,true,false,false),S,Cavir_A_);
    std::shared_ptr<Matrix> SAbs = linalg::triplet(linalg::triplet(Caocc_B_,S,D_A,true,false,false),S,Cavir_B_);

    std::shared_ptr<Matrix> Qar(Jar->clone());
    Qar->zero();
    Qar->add(Jar);
    Qar->add(Var);

    std::shared_ptr<Matrix> Qbs(Jbs->clone());
    Qbs->zero();
    Qbs->add(Jbs);
    Qbs->add(Vbs);

    Jbr.reset();
    Kbr.reset();
    Jas.reset();
    Kas.reset();
    KOas.reset();
    KObr.reset();
    JBas.reset();
    JAbr.reset();
    Jbs.reset();
    Jar.reset();
    JAas.reset();
    JBbr.reset();
    Vbs.reset();
    Var.reset();
    VBas.reset();
    VAbr.reset();
    VRas.reset();
    VSbr.reset();

    S.reset();
    L_A.reset();
    D_A.reset();
    P_A.reset();
    V_A.reset();
    J_A.reset();
    K_A.reset();
    L_B.reset();
    D_B.reset();
    P_B.reset();
    V_B.reset();
    J_B.reset();
    K_B.reset();
    K_O.reset();

    vars_.clear(); // ZLG: fix this
    //if (!full_orbital_) {
    //    vars_.clear();
    //}

    // => Memory <= //

    // => Integrals from the THCE <= //

    //std::shared_ptr<DFERI> df = DFERI::build(primary_,mp2fit_,Process::environment.options);
    //df->clear();

    std::vector<std::shared_ptr<Matrix> > Cs;
    Cs.push_back(Caocc_A_);
    Cs.push_back(Cavir_A_);
    Cs.push_back(Caocc_B_);
    Cs.push_back(Cavir_B_);
    Cs.push_back(Cr1);
    Cs.push_back(Cs1);
    Cs.push_back(Ca2);
    Cs.push_back(Cb2);
    Cs.push_back(Cr3);
    Cs.push_back(Cs3);
    Cs.push_back(Ca4);
    Cs.push_back(Cb4);
    //std::shared_ptr<Matrix> Call = linalg::horzcat(Cs);
    //Cs.clear();

    //df->set_C(Call);
    //df->set_memory(memory_ - Call->nrow() * Call->ncol());

    dfh_->clear_all();
    dfh_ = std::make_shared<DFHelper>(primary_, mp2fit_);
    size_t doubles = Process::environment.get_memory() * 0.5 / sizeof(double);
    dfh_->set_memory(doubles);
    dfh_->set_method("DIRECT_iaQ");
    dfh_->set_nthreads(1); // TODO
    //dfh_->set_metric_pow(-1.0); // WHAT POWER !?!?!?
    dfh_->initialize();

    int offset = 0;
    dfh_->add_space("a", Cs[0]);
    dfh_->add_space("r", Cs[1]);
    dfh_->add_space("b", Cs[2]);
    dfh_->add_space("s", Cs[3]);
    dfh_->add_space("r1", Cs[4]);
    dfh_->add_space("s1", Cs[5]);
    dfh_->add_space("a2", Cs[6]);
    dfh_->add_space("b2", Cs[7]);
    dfh_->add_space("r3", Cs[8]);
    dfh_->add_space("s3", Cs[9]);
    dfh_->add_space("a4", Cs[10]);
    dfh_->add_space("b4", Cs[11]);

    // Disk stuff is all transposed for ab exposure, but transforms down to a or b first for speed

    dfh_->add_transformation("Aar", "r" ,  "a" );
    dfh_->add_transformation("Abs", "s" ,  "b" );
    dfh_->add_transformation("Bas", "s1",  "a" );
    dfh_->add_transformation("Bbr", "r1",  "b" );
    dfh_->add_transformation("Cas", "s" ,  "a2");
    dfh_->add_transformation("Cbr", "r" ,  "b2");
    dfh_->add_transformation("Dar", "r3",  "a" );
    dfh_->add_transformation("Dbs", "s3",  "b" );
    dfh_->add_transformation("Ear", "r" ,  "a4");
    dfh_->add_transformation("Ebs", "s" ,  "b4");
    dfh_->add_disk_tensor("Far", std::make_tuple(nr, naa, nQ));
    dfh_->add_disk_tensor("Fbs", std::make_tuple(ns, nab, nQ));

    Cr1.reset();
    Cs1.reset();
    Ca2.reset();
    Cb2.reset();
    Cr3.reset();
    Cs3.reset();
    Ca4.reset();
    Cb4.reset();
    //Call.reset();

    //df->print_header();
    //df->compute();
    dfh_->transform();
    auto Aar_s = dfh_->get_tensor_shape("Aar");
    auto Abs_s = dfh_->get_tensor_shape("Abs");
    auto Bas_s = dfh_->get_tensor_shape("Bas");
    auto Bbr_s = dfh_->get_tensor_shape("Bbr");
    auto Cas_s = dfh_->get_tensor_shape("Cas");
    auto Cbr_s = dfh_->get_tensor_shape("Cbr");
    auto Dar_s = dfh_->get_tensor_shape("Dar");
    auto Dbs_s = dfh_->get_tensor_shape("Dbs");
    auto Ear_s = dfh_->get_tensor_shape("Ear");
    auto Ebs_s = dfh_->get_tensor_shape("Ebs");

    outfile->Printf("Aar : %d  %d  %d \n", std::get<0>(Aar_s), std::get<1>(Aar_s), std::get<2>(Aar_s));
    outfile->Printf("Abs : %d  %d  %d \n", std::get<0>(Abs_s), std::get<1>(Abs_s), std::get<2>(Abs_s));
    outfile->Printf("Bas : %d  %d  %d \n", std::get<0>(Bas_s), std::get<1>(Bas_s), std::get<2>(Bas_s));
    outfile->Printf("Bbr : %d  %d  %d \n", std::get<0>(Bbr_s), std::get<1>(Bbr_s), std::get<2>(Bbr_s));
    outfile->Printf("Cas : %d  %d  %d \n", std::get<0>(Cas_s), std::get<1>(Cas_s), std::get<2>(Cas_s));
    outfile->Printf("Cbr : %d  %d  %d \n", std::get<0>(Cbr_s), std::get<1>(Cbr_s), std::get<2>(Cbr_s));
    outfile->Printf("Dar : %d  %d  %d \n", std::get<0>(Dar_s), std::get<1>(Dar_s), std::get<2>(Dar_s));
    outfile->Printf("Dbs : %d  %d  %d \n", std::get<0>(Dbs_s), std::get<1>(Dbs_s), std::get<2>(Dbs_s));
    outfile->Printf("Ear : %d  %d  %d \n", std::get<0>(Ear_s), std::get<1>(Ear_s), std::get<2>(Ear_s));
    outfile->Printf("Ebs : %d  %d  %d \n", std::get<0>(Ebs_s), std::get<1>(Ebs_s), std::get<2>(Ebs_s));


    outfile->Printf(" na : %d   nfa : %d  naa : %d \n", na, nfa, naa);
    outfile->Printf(" nb : %d   nfb : %d  nab : %d \n", nb, nfb, nab);
    outfile->Printf(" nr : %d \n", nr);
    outfile->Printf(" ns : %d \n", ns);
    outfile->Printf(" nQ : %d \n", nQ);
    outfile->Printf(" nA : %d \n", nA);
    outfile->Printf(" nB : %d \n", nB);

    //std::map<std::string, std::shared_ptr<Tensor> >& ints = df->ints();

    //std::shared_ptr<Tensor> AarT = ints["Aar"];
    //std::shared_ptr<Tensor> AbsT = ints["Abs"];
    //std::shared_ptr<Tensor> BasT = ints["Bas"];
    //std::shared_ptr<Tensor> BbrT = ints["Bbr"];
    //std::shared_ptr<Tensor> CasT = ints["Cas"];
    //std::shared_ptr<Tensor> CbrT = ints["Cbr"];
    //std::shared_ptr<Tensor> DarT = ints["Dar"];
    //std::shared_ptr<Tensor> DbsT = ints["Dbs"];
    //std::shared_ptr<Tensor> EarT = ints["Ear"];
    //std::shared_ptr<Tensor> EbsT = ints["Ebs"];

    //df.reset();

    // => Blocking <= //

    long int overhead = 0L;
    overhead += 5L * nT * na * nb;
    overhead += 2L * na * ns + 2L * nb * nr + 2L * na * nr + 2L * nb * ns;
    long int rem = memory_ - overhead;

    if (rem < 0L) {
        throw PSIEXCEPTION("Too little static memory for DFTSAPT::mp2_terms");
    }

    long int cost_r = 2L * naa * nQ + 2L * nab * nQ;
    long int max_r = rem / (2L * cost_r);
    long int max_s = max_r;
    max_r = (max_r > nr ? nr : max_r);
    max_s = (max_s > ns ? ns : max_s);
    if (max_s < 1L || max_s < 1L) {
        throw PSIEXCEPTION("Too little dynamic memory for DFTSAPT::mp2_terms");
    }
    max_r = 1;
    max_s = 1;

    outfile->Printf(" max_r : %d \n", max_r);
    outfile->Printf(" max_s : %d \n", max_s);

    // => Tensor Slices <= //

    auto Aar = std::make_shared<Matrix>("Aar",max_r*naa,nQ);
    auto Abs = std::make_shared<Matrix>("Abs",max_s*nab,nQ);
    auto Bas = std::make_shared<Matrix>("Bas",max_s*naa,nQ);
    auto Bbr = std::make_shared<Matrix>("Bbr",max_r*nab,nQ);
    auto Cas = std::make_shared<Matrix>("Cas",max_s*naa,nQ);
    auto Cbr = std::make_shared<Matrix>("Cbr",max_r*nab,nQ);
    auto Dar = std::make_shared<Matrix>("Dar",max_r*naa,nQ);
    auto Dbs = std::make_shared<Matrix>("Dbs",max_s*nab,nQ);

    // => Thread Work Arrays <= //

    std::vector<std::shared_ptr<Matrix> > Tab;
    std::vector<std::shared_ptr<Matrix> > Vab;
    std::vector<std::shared_ptr<Matrix> > T2ab;
    std::vector<std::shared_ptr<Matrix> > V2ab;
    std::vector<std::shared_ptr<Matrix> > Iab;
    for (int t = 0; t < nT; t++) {
        Tab.push_back(std::make_shared<Matrix>("Tab",naa,nab));
        Vab.push_back(std::make_shared<Matrix>("Vab",naa,nab));
        T2ab.push_back(std::make_shared<Matrix>("T2ab",na,nb));
        V2ab.push_back(std::make_shared<Matrix>("V2ab",na,nb));
        Iab.push_back(std::make_shared<Matrix>("Iab",naa,nb));
    }

    // => Pointers <= //

    double** Aarp = Aar->pointer();
    double** Absp = Abs->pointer();
    double** Basp = Bas->pointer();
    double** Bbrp = Bbr->pointer();
    double** Casp = Cas->pointer();
    double** Cbrp = Cbr->pointer();
    double** Darp = Dar->pointer();
    double** Dbsp = Dbs->pointer();

    double** Sasp = Sas->pointer();
    double** Sbrp = Sbr->pointer();
    double** SBarp = SBar->pointer();
    double** SAbsp = SAbs->pointer();

    double** Qasp = Qas->pointer();
    double** Qbrp = Qbr->pointer();
    double** Qarp = Qar->pointer();
    double** Qbsp = Qbs->pointer();

    double*  eap  = eps_aocc_A_->pointer();
    double*  ebp  = eps_aocc_B_->pointer();
    double*  erp  = eps_avir_A_->pointer();
    double*  esp  = eps_avir_B_->pointer();

    // => File Pointers <= //

    //FILE* Aarf = AarT->file_pointer();
    //FILE* Absf = AbsT->file_pointer();
    //FILE* Basf = BasT->file_pointer();
    //FILE* Bbrf = BbrT->file_pointer();
    //FILE* Casf = CasT->file_pointer();
    //FILE* Cbrf = CbrT->file_pointer();
    //FILE* Darf = DarT->file_pointer();
    //FILE* Dbsf = DbsT->file_pointer();
    //FILE* Earf = EarT->file_pointer();
    //FILE* Ebsf = EbsT->file_pointer();

    // => Slice D + E -> F <= //

    for (int ir = 0; ir < nr; ir += 1) {
        dfh_->fill_tensor("Dar", Darp[0], {ir, ir + 1});
        dfh_->fill_tensor("Ear", Aarp[0], {ir, ir + 1});
        C_DAXPY(naQ,1.0,Aarp[0],1,Darp[0],1);
        dfh_->write_disk_tensor("Far", Dar, {ir, ir + 1});
    }

    for (int is = 0; is < ns; is += 1) {
        dfh_->fill_tensor("Dbs", Dbs, {is, is+1});
        dfh_->fill_tensor("Ebs", Abs, {is, is+1});
        C_DAXPY(nbQ,1.0,Absp[0],1,Dbsp[0],1);
        dfh_->write_disk_tensor("Fbs", Dbs, {is, is + 1});
    }

    // => Targets <= //

    double Disp20 = 0.0;
    double ExchDisp20 = 0.0;

    // => Local Targets <= //

    std::vector<std::shared_ptr<Matrix> > E_disp20_threads;
    std::vector<std::shared_ptr<Matrix> > E_exch_disp20_threads;
    for (int t = 0; t < nT; t++) {
        E_disp20_threads.push_back(std::shared_ptr<Matrix>(new Matrix("E_disp20",na,nb)));
        E_exch_disp20_threads.push_back(std::shared_ptr<Matrix>(new Matrix("E_exch_disp20",na,nb)));
    }

    // => MO => LO Transform <= //

    double** UAp = Uocc_A_->pointer();
    double** UBp = Uocc_B_->pointer();

    // ==> Master Loop <== //

    for (int ir = 0; ir < nr; ir += 1) {

        dfh_->fill_tensor("Aar", Aar, {ir, ir + 1});
        dfh_->fill_tensor("Bbr", Bbr, {ir, ir + 1});
        dfh_->fill_tensor("Cbr", Cbr, {ir, ir + 1});
        dfh_->fill_tensor("Far", Dar, {ir, ir + 1});

        for (int is = 0; is < ns; is += max_s) {

            dfh_->fill_tensor("Abs", Abs, {is, is+1});
            dfh_->fill_tensor("Bas", Bas, {is, is+1});
            dfh_->fill_tensor("Cas", Cas, {is, is+1});
            dfh_->fill_tensor("Fbs", Dbs, {is, is+1});

            long int nrs = 1;

            for (long int rs = 0L; rs < nrs; rs++) {
                int r = rs / 1;
                int s = rs % 1;

                int thread = 0;
                #ifdef _OPENMP
                    thread = omp_get_thread_num();
                #endif

                double** E_disp20Tp = E_disp20_threads[thread]->pointer();
                double** E_exch_disp20Tp = E_exch_disp20_threads[thread]->pointer();
                
                double** Tabp  = Tab[thread]->pointer();
                double** Vabp  = Vab[thread]->pointer();
                double** T2abp = T2ab[thread]->pointer();
                double** V2abp = V2ab[thread]->pointer();
                double** Iabp  = Iab[thread]->pointer();

                // => Amplitudes, Disp20 <= //

                C_DGEMM('N','T',naa,nab,nQ,1.0,Aarp[(r)*naa],nQ,Absp[(s)*nab],nQ,0.0,Vabp[0],nab);
                for (int a = 0; a < naa; a++) {
                    for (int b = 0; b < nab; b++) {
                        Tabp[a][b] = Vabp[a][b] / (eap[a] + ebp[b] - erp[r + ir] - esp[s + is]);
                    }
                }

                C_DGEMM('N','N',naa,nb,nab,1.0,Tabp[0],nab,UBp[nfb],nb,0.0,Iabp[0],nb);
                C_DGEMM('T','N',na,nb,naa,1.0,UAp[nfa],na,Iabp[0],nb,0.0,T2abp[0],nb);
                C_DGEMM('N','N',naa,nb,nab,1.0,Vabp[0],nab,UBp[nfb],nb,0.0,Iabp[0],nb);
                C_DGEMM('T','N',na,nb,naa,1.0,UAp[nfa],na,Iabp[0],nb,0.0,V2abp[0],nb);

                for (int a = 0; a < na; a++) {
                    for (int b = 0; b < nb; b++) {
                        E_disp20Tp[a][b] += 4.0 * T2abp[a][b] * V2abp[a][b];
                        Disp20 += 4.0 * T2abp[a][b] * V2abp[a][b];
                    }
                }

                // => Exch-Disp20 <= //

                // > Q1-Q3 < //

                C_DGEMM('N','T',naa,nab,nQ,1.0,Basp[(s)*naa],nQ,Bbrp[(r)*nab],nQ,0.0,Vabp[0],nab);
                C_DGEMM('N','T',naa,nab,nQ,1.0,Casp[(s)*naa],nQ,Cbrp[(r)*nab],nQ,1.0,Vabp[0],nab);
                C_DGEMM('N','T',naa,nab,nQ,1.0,Aarp[(r)*naa],nQ,Dbsp[(s)*nab],nQ,1.0,Vabp[0],nab);
                C_DGEMM('N','T',naa,nab,nQ,1.0,Darp[(r)*naa],nQ,Absp[(s)*nab],nQ,1.0,Vabp[0],nab);

                // > V,J,K < //

                C_DGER(naa,nab,1.0,&Sasp[0][s + is], ns,&Qbrp[0][r + ir], nr,Vabp[0],nab);
                C_DGER(naa,nab,1.0,&Qasp[0][s + is], ns,&Sbrp[0][r + ir], nr,Vabp[0],nab);
                C_DGER(naa,nab,1.0,&Qarp[0][r + ir], nr,&SAbsp[0][s + is],ns,Vabp[0],nab);
                C_DGER(naa,nab,1.0,&SBarp[0][r + ir],nr,&Qbsp[0][s + is], ns,Vabp[0],nab);

                C_DGEMM('N','N',naa,nb,nab,1.0,Vabp[0],nab,UBp[nfb],nb,0.0,Iabp[0],nb);
                C_DGEMM('T','N',na,nb,naa,1.0,UAp[nfa],na,Iabp[0],nb,0.0,V2abp[0],nb);

                for (int a = 0; a < na; a++) {
                    for (int b = 0; b < nb; b++) {
                        E_exch_disp20Tp[a][b] -= 2.0 * T2abp[a][b] * V2abp[a][b];
                        ExchDisp20 -= 2.0 * T2abp[a][b] * V2abp[a][b];
                    }
                }
            }
        }
    }

    std::shared_ptr<Matrix> E_disp20(new Matrix("E_disp20", na, nb));
    std::shared_ptr<Matrix> E_exch_disp20(new Matrix("E_exch_disp20", na, nb));
    double** E_disp20p = E_disp20->pointer();
    double** E_exch_disp20p = E_exch_disp20->pointer();

    for (int t = 0; t < nT; t++) {
        E_disp20->add(E_disp20_threads[t]);
        E_exch_disp20->add(E_exch_disp20_threads[t]);
    }

    std::shared_ptr<Matrix> E_disp(new Matrix("E_disp (a x b)", na, nb));
    double** E_dispp = E_disp->pointer();

    for (int a = 0; a < na; a++) {
        for (int b = 0; b < nb; b++) {
            E_dispp[a][b] = E_disp20p[a][b] +
                            E_exch_disp20p[a][b];
        }
    }

    //E_disp20->print();
    //E_exch_disp20->print();
    //E_disp->print();

    energies_["Disp20"] = Disp20;
    energies_["Exch-Disp20"] = ExchDisp20;
    outfile->Printf("    Disp20              = %18.12lf H\n",Disp20);
    outfile->Printf("    Exch-Disp20         = %18.12lf H\n",ExchDisp20);
    outfile->Printf("\n");

    //vis_->vars()["Disp_ab"] = E_disp;
    Disp_ab = E_disp;
}



void ASAPT0::fock_terms() {
    outfile->Printf("  SCF TERMS:\n\n");

    // ==> Setup <== //

    // => Compute the D matrices <= //

    std::shared_ptr<Matrix> D_A = linalg::doublet(Cocc_A_, Cocc_A_, false, true);
    std::shared_ptr<Matrix> D_B = linalg::doublet(Cocc_B_, Cocc_B_, false, true);

    // => Compute the P matrices <= //

    std::shared_ptr<Matrix> P_A = linalg::doublet(Cvir_A_, Cvir_A_, false, true);
    std::shared_ptr<Matrix> P_B = linalg::doublet(Cvir_B_, Cvir_B_, false, true);

    // => Compute the S matrix <= //

    std::shared_ptr<Matrix> S = build_S(primary_);

    // => Compute the V matrices <= //

    std::shared_ptr<Matrix> V_A = build_V(primary_A_);
    std::shared_ptr<Matrix> V_B = build_V(primary_B_);

    // => JK Object <= //

    // TODO: Recompute exactly how much memory is needed
    size_t nA = Cocc_A_->ncol();
    size_t nB = Cocc_B_->ncol();
    size_t nso = Cocc_A_->nrow();
    long int jk_memory = (long int)memory_;
    jk_memory -= 24 * nso * nso;
    // Not sure why it should be 4, just taken from DFTSAPT code.
    jk_memory -= 4 * nA * nso;
    jk_memory -= 4 * nB * nso;
    if (jk_memory < 0L) {
        throw PSIEXCEPTION("Too little static memory for ASAPT0::fock_terms");
    }

    std::shared_ptr<JK> jk = JK::build_JK(primary_, jkfit_, options_, false, (size_t)jk_memory);

    jk->set_memory((size_t)jk_memory);

    // ==> Generalized Fock Source Terms [Elst/Exch] <== //

    // => Steric Interaction Density Terms (T) <= //
    std::shared_ptr<Matrix> Sij = build_Sij(S);
    std::shared_ptr<Matrix> Sij_n = build_Sij_n(Sij);
    Sij_n->set_name("Sij^inf (MO)");

    /* Build all of these matrices at once.
       I like that C_T_BA is actually CB T[BA] , so my naming convention
       is reversed compared to the DFTSAPT code.
       And AB quantities are not needed so we never compute them. */
    std::map<std::string, std::shared_ptr<Matrix> > Cbar_n = build_Cbar(Sij_n);
    std::shared_ptr<Matrix> C_T_A_n = Cbar_n["C_T_A"];
    std::shared_ptr<Matrix> C_T_B_n = Cbar_n["C_T_B"];
    std::shared_ptr<Matrix> C_T_BA_n = Cbar_n["C_T_BA"];
    std::shared_ptr<Matrix> C_T_AB_n = Cbar_n["C_T_AB"];

    Sij.reset();
    Sij_n.reset();

    std::shared_ptr<Matrix> C_AS = linalg::triplet(P_B,S,Cocc_A_);


    //////    Create matrices needed for the S^{2} formula as well

    ////std::shared_ptr<Matrix> S_CA = linalg::doublet(S, Cocc_A_);
    ////std::shared_ptr<Matrix> C_AS = linalg::doublet(P_B, S_CA);
    //////    Next one for DM-based formula (debug purposes for now)
    ////std::shared_ptr<Matrix> C_AB = linalg::doublet(D_B, S_CA);
    ////S_CA = linalg::doublet(S, Cocc_A_);
    //////    Next one for DM-based formula (debug purposes for now)
    ////std::shared_ptr<Matrix> C_AB = linalg::doublet(D_B, S_CA);
    ////S_CA.reset();

    // => Load the JK Object <= //

    std::vector<SharedMatrix>& Cl = jk->C_left();
    std::vector<SharedMatrix>& Cr = jk->C_right();
    Cl.clear();
    Cr.clear();

    // J/K[P^A]
    Cl.push_back(Cocc_A_);
    Cr.push_back(Cocc_A_);
    // J/K[T^A, S^\infty]
    Cl.push_back(Cocc_A_);
    Cr.push_back(C_T_A_n);
    // J/K[T^AB, S^\infty]
    Cl.push_back(Cocc_A_);
    Cr.push_back(C_T_AB_n);
    // J/K[S_as]
    Cl.push_back(Cocc_A_);
    Cr.push_back(C_AS);
    // J/K[P^B]
    Cl.push_back(Cocc_B_);
    Cr.push_back(Cocc_B_);

    // => Initialize the JK object <= //

    jk->set_do_J(true);
    jk->set_do_K(true);
    jk->initialize();
    jk->print_header();

    // => Compute the JK matrices <= //

    jk->compute();

    // => Unload the JK Object <= //

    const std::vector<SharedMatrix>& J = jk->J();
    const std::vector<SharedMatrix>& K = jk->K();

    SharedMatrix J_A      = J[0];
    SharedMatrix J_T_A_n  = J[1];
    SharedMatrix J_T_AB_n = J[2];
    SharedMatrix J_AS     = J[3];
    SharedMatrix J_B      = J[4];

    SharedMatrix K_A      = K[0];
    SharedMatrix K_T_A_n  = K[1];
    SharedMatrix K_T_AB_n = K[2];
    SharedMatrix K_AS     = K[3];
    SharedMatrix K_B      = K[4];

    // ==> Electrostatic Terms <== //

    // Classical physics (watch for cancellation!)

    double Enuc = 0.0;
    Enuc += dimer_->nuclear_repulsion_energy(dimer_field_);
    Enuc -= monomer_A_->nuclear_repulsion_energy(monomer_A_field_);
    Enuc -= monomer_B_->nuclear_repulsion_energy(monomer_B_field_);

    double Elst10 = 0.0;
    std::vector<double> Elst10_terms;
    Elst10_terms.resize(4);
    Elst10_terms[0] += 2.0 * D_A->vector_dot(V_B);
    Elst10_terms[1] += 2.0 * D_B->vector_dot(V_A);
    Elst10_terms[2] += 4.0 * D_B->vector_dot(J_A);
    Elst10_terms[3] += 1.0 * Enuc;
    for (int k = 0; k < Elst10_terms.size(); k++) {
        Elst10 += Elst10_terms[k];
    }
    //if (debug_) {
        for (int k = 0; k < Elst10_terms.size(); k++) {
            outfile->Printf("    Elst10,r (%1d)        = %18.12lf H\n",k+1,Elst10_terms[k]);
        }
    //}
    energies_["Elst10,r"] = Elst10;
    outfile->Printf("    Elst10,r            = %18.12lf H\n",Elst10);
    // ==> Exchange Terms (S^\infty) <== //

    // => Compute the T matrices <= //

    std::shared_ptr<Matrix> T_A_n  = linalg::doublet(Cocc_A_, C_T_A_n, false, true);
    std::shared_ptr<Matrix> T_B_n  = linalg::doublet(Cocc_B_, C_T_B_n, false, true);
    std::shared_ptr<Matrix> T_BA_n = linalg::doublet(Cocc_B_, C_T_BA_n, false, true);
    std::shared_ptr<Matrix> T_AB_n = linalg::doublet(Cocc_A_, C_T_AB_n, false, true);

    C_T_A_n.reset();
    C_T_B_n.reset();
    C_T_BA_n.reset();
    C_T_AB_n.reset();

    double Exch10_n = 0.0;
    std::vector<double> Exch10_n_terms;
    Exch10_n_terms.resize(9);
    Exch10_n_terms[0] -= 2.0 * D_A->vector_dot(K_B);
    Exch10_n_terms[1] += 2.0 * T_A_n->vector_dot(V_B);
    Exch10_n_terms[1] += 4.0 * T_A_n->vector_dot(J_B);
    Exch10_n_terms[1] -= 2.0 * T_A_n->vector_dot(K_B);
    Exch10_n_terms[2] += 2.0 * T_B_n->vector_dot(V_A);
    Exch10_n_terms[2] += 4.0 * T_B_n->vector_dot(J_A);
    Exch10_n_terms[2] -= 2.0 * T_B_n->vector_dot(K_A);
    Exch10_n_terms[3] += 2.0 * T_AB_n->vector_dot(V_A);
    Exch10_n_terms[3] += 4.0 * T_AB_n->vector_dot(J_A);
    Exch10_n_terms[3] -= 2.0 * T_AB_n->vector_dot(K_A);
    Exch10_n_terms[4] += 2.0 * T_AB_n->vector_dot(V_B);
    Exch10_n_terms[4] += 4.0 * T_AB_n->vector_dot(J_B);
    Exch10_n_terms[4] -= 2.0 * T_AB_n->vector_dot(K_B);
    Exch10_n_terms[5] += 4.0 * T_B_n->vector_dot(J_T_AB_n);
    Exch10_n_terms[5] -= 2.0 * T_B_n->vector_dot(K_T_AB_n);
    Exch10_n_terms[6] += 4.0 * T_A_n->vector_dot(J_T_AB_n);
    Exch10_n_terms[6] -= 2.0 * T_A_n->vector_dot(K_T_AB_n);
    Exch10_n_terms[7] += 4.0 * T_B_n->vector_dot(J_T_A_n);
    Exch10_n_terms[7] -= 2.0 * T_B_n->vector_dot(K_T_A_n);
    Exch10_n_terms[8] += 4.0 * T_AB_n->vector_dot(J_T_AB_n);
    Exch10_n_terms[8] -= 2.0 * T_AB_n->vector_dot(K_T_AB_n);
    for (int k = 0; k < Exch10_n_terms.size(); k++) {
        Exch10_n += Exch10_n_terms[k];
    }
    //if (debug_) {
        for (int k = 0; k < Exch10_n_terms.size(); k++) {
            outfile->Printf("    Exch10 (%1d)          = %18.12lf H\n",k+1,Exch10_n_terms[k]);
        }
    //}
    energies_["Exch10"] = Exch10_n;
    outfile->Printf("    Exch10              = %18.12lf H\n",Exch10_n);

    T_A_n.reset();
    T_B_n.reset();
    T_BA_n.reset();
    T_AB_n.reset();
    J_T_A_n.reset();
    J_T_AB_n.reset();
    K_T_A_n.reset();
    K_T_AB_n.reset();

    // ==> Exchange Terms (S^2) <== //

    double Exch10_2 = 0.0;
    std::vector<double> Exch10_2_terms;
    Exch10_2_terms.resize(3);
    Exch10_2_terms[0] -= 2.0 * linalg::triplet(linalg::triplet(D_A,S,D_B),S,P_A)->vector_dot(V_B);
    Exch10_2_terms[0] -= 4.0 * linalg::triplet(linalg::triplet(D_A,S,D_B),S,P_A)->vector_dot(J_B);
    Exch10_2_terms[1] -= 2.0 * linalg::triplet(linalg::triplet(D_B,S,D_A),S,P_B)->vector_dot(V_A);
    Exch10_2_terms[1] -= 4.0 * linalg::triplet(linalg::triplet(D_B,S,D_A),S,P_B)->vector_dot(J_A);
    Exch10_2_terms[2] -= 2.0 * linalg::triplet(P_A,S,D_B)->vector_dot(K_AS);
    for (int k = 0; k < Exch10_2_terms.size(); k++) {
        Exch10_2 += Exch10_2_terms[k];
    }
    //if (debug_) {
        for (int k = 0; k < Exch10_2_terms.size(); k++) {
            outfile->Printf("    Exch10(S^2) (%1d)     = %18.12lf H\n",k+1,Exch10_2_terms[k]);
        }
    //}
    energies_["Exch10(S^2)"] = Exch10_2;
    outfile->Printf("    Exch10(S^2)         = %18.12lf H\n",Exch10_2);
    outfile->Printf( "\n");

    // Clear up some memory
    J_AS.reset();
    K_AS.reset();


    // ==> Uncorrelated Second-Order Response Terms [Induction] <== //

    // => ExchInd perturbations <= //

    std::shared_ptr<Matrix> C_O_A = linalg::triplet(D_B,S,Cocc_A_);
    std::shared_ptr<Matrix> C_P_A = linalg::triplet(linalg::triplet(D_B,S,D_A),S,Cocc_B_);
    std::shared_ptr<Matrix> C_P_B = linalg::triplet(linalg::triplet(D_A,S,D_B),S,Cocc_A_);

    Cl.clear();
    Cr.clear();

    // J/K[O]
    Cl.push_back(Cocc_A_);
    Cr.push_back(C_O_A);
    // J/K[P_B]
    Cl.push_back(Cocc_A_);
    Cr.push_back(C_P_B);    
    // J/K[P_B]
    Cl.push_back(Cocc_B_);
    Cr.push_back(C_P_A);

    // => Compute the JK matrices <= //

    jk->compute();

    // => Unload the JK Object <= //

    std::shared_ptr<Matrix> J_O      = J[0];
    std::shared_ptr<Matrix> J_P_B    = J[1];
    std::shared_ptr<Matrix> J_P_A    = J[2];

    std::shared_ptr<Matrix> K_O      = K[0];
    std::shared_ptr<Matrix> K_P_B    = K[1];
    std::shared_ptr<Matrix> K_P_A    = K[2];

    // ==> Generalized ESP (Flat and Exchange) <== //

    std::map<std::string, std::shared_ptr<Matrix> > mapA;
    mapA["Cocc_A"] = Cocc_A_;
    mapA["Cvir_A"] = Cvir_A_;
    mapA["S"] = S;
    mapA["D_A"] = D_A;
    mapA["V_A"] = V_A;
    mapA["J_A"] = J_A;
    mapA["K_A"] = K_A;
    mapA["D_B"] = D_B;
    mapA["V_B"] = V_B;
    mapA["J_B"] = J_B;
    mapA["K_B"] = K_B;
    mapA["J_O"] = J_O;
    mapA["K_O"] = K_O;
    mapA["J_P"] = J_P_A; 

    std::shared_ptr<Matrix> wB = build_ind_pot(mapA);
    std::shared_ptr<Matrix> uB = build_exch_ind_pot(mapA);

    K_O->transpose_this();

    std::map<std::string, std::shared_ptr<Matrix> > mapB;
    mapB["Cocc_A"] = Cocc_B_;
    mapB["Cvir_A"] = Cvir_B_;
    mapB["S"] = S;
    mapB["D_A"] = D_B;
    mapB["V_A"] = V_B;
    mapB["J_A"] = J_B;
    mapB["K_A"] = K_B;
    mapB["D_B"] = D_A;
    mapB["V_B"] = V_A;
    mapB["J_B"] = J_A;
    mapB["K_B"] = K_A;
    mapB["J_O"] = J_O;
    mapB["K_O"] = K_O;
    mapB["J_P"] = J_P_B; 

    std::shared_ptr<Matrix> wA = build_ind_pot(mapB);
    std::shared_ptr<Matrix> uA = build_exch_ind_pot(mapB);

    K_O->transpose_this();

    // ==> Uncoupled Induction <== //

    std::shared_ptr<Matrix> xuA(wB->clone());
    std::shared_ptr<Matrix> xuB(wA->clone());

    {
        int na = eps_occ_A_->dimpi()[0];
        int nb = eps_occ_B_->dimpi()[0];
        int nr = eps_vir_A_->dimpi()[0];
        int ns = eps_vir_B_->dimpi()[0];

        double** xuAp = xuA->pointer();
        double** xuBp = xuB->pointer();
        double** wAp = wA->pointer();
        double** wBp = wB->pointer();
        double*  eap = eps_occ_A_->pointer();
        double*  erp = eps_vir_A_->pointer();
        double*  ebp = eps_occ_B_->pointer();
        double*  esp = eps_vir_B_->pointer();

        for (int a = 0; a < na; a++) {
            for (int r = 0; r < nr; r++) {
                xuAp[a][r] = wBp[a][r] / (eap[a] - erp[r]);
            }
        }

        for (int b = 0; b < nb; b++) {
            for (int s = 0; s < ns; s++) {
                xuBp[b][s] = wAp[b][s] / (ebp[b] - esp[s]);
            }
        }
    }

    // ==> Induction <== //

    double Ind20u_AB = 2.0 * xuA->vector_dot(wB);
    double Ind20u_BA = 2.0 * xuB->vector_dot(wA);
    double Ind20u = Ind20u_AB + Ind20u_BA;
    energies_["Ind20,u (A<-B)"] = Ind20u_AB;
    energies_["Ind20,u (B<-A)"] = Ind20u_BA;
    energies_["Ind20,u"] = Ind20u;
    outfile->Printf("    Ind20,u (A<-B)      = %18.12lf H\n",Ind20u_AB);
    outfile->Printf("    Ind20,u (B<-A)      = %18.12lf H\n",Ind20u_BA);
    outfile->Printf("    Ind20,u             = %18.12lf H\n",Ind20u);

    // => Exchange-Induction <= //

    double ExchInd20u_AB = 2.0 * xuA->vector_dot(uB);
    double ExchInd20u_BA = 2.0 * xuB->vector_dot(uA);
    double ExchInd20u = ExchInd20u_AB + ExchInd20u_BA;
    outfile->Printf("    Exch-Ind20,u (A<-B) = %18.12lf H\n",ExchInd20u_AB);
    outfile->Printf("    Exch-Ind20,u (B<-A) = %18.12lf H\n",ExchInd20u_BA);
    outfile->Printf("    Exch-Ind20,u        = %18.12lf H\n",ExchInd20u);
    outfile->Printf("\n");

    energies_["Exch-Ind20,u (A<-B)"] = ExchInd20u_AB;
    energies_["Exch-Ind20,u (B<-A)"] = ExchInd20u_BA;
    energies_["Exch-Ind20,u"] = ExchInd20u_AB + ExchInd20u_BA;

    // => Coupled Induction <= //

    // Compute CPKS
    timer_on("SAPT0: CPHF");
    std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix> > x_sol = compute_x(jk,wB,wA);
    std::shared_ptr<Matrix> xA = x_sol.first;
    std::shared_ptr<Matrix> xB = x_sol.second;
    timer_off("SAPT0: CPHF");

    // Backward in Ed's convention
    xA->scale(-1.0);
    xB->scale(-1.0);

    // => Induction <= //

    double Ind20r_AB = 2.0 * xA->vector_dot(wB);
    double Ind20r_BA = 2.0 * xB->vector_dot(wA);
    double Ind20r = Ind20r_AB + Ind20r_BA;
    energies_["Ind20,r (A<-B)"] = Ind20r_AB;
    energies_["Ind20,r (B<-A)"] = Ind20r_BA;
    energies_["Ind20,r"] = Ind20r;
    outfile->Printf("    Ind20,r (A<-B)      = %18.12lf H\n",Ind20r_AB);
    outfile->Printf("    Ind20,r (B<-A)      = %18.12lf H\n",Ind20r_BA);
    outfile->Printf("    Ind20,r             = %18.12lf H\n",Ind20r);

    // => Exchange-Induction <= //

    double ExchInd20r_AB = 2.0 * xA->vector_dot(uB);
    double ExchInd20r_BA = 2.0 * xB->vector_dot(uA);
    double ExchInd20r = ExchInd20r_AB + ExchInd20r_BA;
    outfile->Printf("    Exch-Ind20,r (A<-B) = %18.12lf H\n",ExchInd20r_AB);
    outfile->Printf("    Exch-Ind20,r (B<-A) = %18.12lf H\n",ExchInd20r_BA);
    outfile->Printf("    Exch-Ind20,r        = %18.12lf H\n",ExchInd20r);
    outfile->Printf("\n");

    energies_["Exch-Ind20,r (A<-B)"] = ExchInd20r_AB;
    energies_["Exch-Ind20,r (B<-A)"] = ExchInd20r_BA;
    energies_["Exch-Ind20,r"] = ExchInd20r_AB + ExchInd20r_BA;

    vars_["S"]   = S;
    vars_["D_A"] = D_A;
    vars_["P_A"] = P_A;
    vars_["V_A"] = V_A;
    vars_["J_A"] = J_A;
    vars_["K_A"] = K_A;
    vars_["D_B"] = D_B;
    vars_["P_B"] = P_B;
    vars_["V_B"] = V_B;
    vars_["J_B"] = J_B;
    vars_["K_B"] = K_B;
    vars_["J_O"] = J_O;
    vars_["K_O"] = K_O;
    vars_["J_P_A"] = J_P_A;
    vars_["J_P_B"] = J_P_B;
}




std::shared_ptr<Matrix> ASAPT0::build_S(std::shared_ptr<BasisSet> basis) {
    auto factory = std::make_shared<IntegralFactory>(basis);
    std::shared_ptr<OneBodyAOInt> Sint(factory->ao_overlap());
    auto S = std::make_shared<Matrix>("S (AO)", basis->nbf(), basis->nbf());
    Sint->compute(S);
    return S;
}
std::shared_ptr<Matrix> ASAPT0::build_V(std::shared_ptr<BasisSet> basis) {
    auto factory = std::make_shared<IntegralFactory>(basis);
    std::shared_ptr<OneBodyAOInt> Sint(factory->ao_potential());
    auto S = std::make_shared<Matrix>("V (AO)", basis->nbf(), basis->nbf());
    Sint->compute(S);
    return S;
}
std::shared_ptr<Matrix> ASAPT0::build_Sij(std::shared_ptr<Matrix> S) {
    int nso = Cocc_A_->nrow();
    int nocc_A = Cocc_A_->ncol();
    int nocc_B = Cocc_B_->ncol();
    int nocc = nocc_A + nocc_B;

    auto Sij = std::make_shared<Matrix>("Sija (MO)", nocc, nocc);
    auto T = std::make_shared<Matrix>("T", nso, nocc_B);

    double** Sp = S->pointer();
    double** Tp = T->pointer();
    double** Sijp = Sij->pointer();
    double** CAp = Cocc_A_->pointer();
    double** CBp = Cocc_B_->pointer();

    C_DGEMM('N', 'N', nso, nocc_B, nso, 1.0, Sp[0], nso, CBp[0], nocc_B, 0.0, Tp[0], nocc_B);
    C_DGEMM('T', 'N', nocc_A, nocc_B, nso, 1.0, CAp[0], nocc_A, Tp[0], nocc_B, 0.0, &Sijp[0][nocc_A], nocc);

    Sij->copy_upper_to_lower();

    return Sij;
}
  
// TOMODIF - spin
std::shared_ptr<Matrix> ASAPT0::build_Sij_n(std::shared_ptr<Matrix> Sij) {
    int nocc = Sij->nrow();

    auto Sij2 = std::make_shared<Matrix>("Sij^inf (MO)", nocc, nocc);

    double** Sijp = Sij->pointer();
    double** Sij2p = Sij2->pointer();

    Sij2->copy(Sij);
    for (int i = 0; i < nocc; i++) {
        Sij2p[i][i] = 1.0;
    }

    int info;

    info = C_DPOTRF('L', nocc, Sij2p[0], nocc);
    if (info) {
        throw PSIEXCEPTION("Sij DPOTRF failed. How far up the steric wall are you?");
    }

    info = C_DPOTRI('L', nocc, Sij2p[0], nocc);
    if (info) {
        throw PSIEXCEPTION("Sij DPOTRI failed. How far up the steric wall are you?");
    }

    Sij2->copy_upper_to_lower();

    for (int i = 0; i < nocc; i++) {
        Sij2p[i][i] -= 1.0;
    }

    return Sij2;
}

std::map<std::string, std::shared_ptr<Matrix> > ASAPT0::build_Cbar(std::shared_ptr<Matrix> S)
{
    std::map<std::string, std::shared_ptr<Matrix> > Cbar;

    int nso = Cocc_A_->nrow();
    int nA = Cocc_A_->ncol();
    int nB = Cocc_B_->ncol();
    int no = nA + nB;

    double** Sp = S->pointer();
    double** CAp = Cocc_A_->pointer();
    double** CBp = Cocc_B_->pointer();
    double** Cp;

    Cbar["C_T_A"] = std::shared_ptr<Matrix>(new Matrix("C_T_A", nso, nA));
    Cp = Cbar["C_T_A"]->pointer();
    C_DGEMM('N','N',nso,nA,nA,1.0,CAp[0],nA,&Sp[0][0],no,0.0,Cp[0],nA);

    Cbar["C_T_B"] = std::shared_ptr<Matrix>(new Matrix("C_T_B", nso, nB));
    Cp = Cbar["C_T_B"]->pointer();
    C_DGEMM('N','N',nso,nB,nB,1.0,CBp[0],nB,&Sp[nA][nA],no,0.0,Cp[0],nB);

    Cbar["C_T_BA"] = std::shared_ptr<Matrix>(new Matrix("C_T_BA", nso, nB));
    Cp = Cbar["C_T_BA"]->pointer();
    C_DGEMM('N','N',nso,nB,nA,1.0,CAp[0],nA,&Sp[0][nA],no,0.0,Cp[0],nB);

    Cbar["C_T_AB"] = std::shared_ptr<Matrix>(new Matrix("C_T_AB", nso, nA));
    Cp = Cbar["C_T_AB"]->pointer();
    C_DGEMM('N','N',nso,nA,nB,1.0,CBp[0],nB,&Sp[nA][0],no,0.0,Cp[0],nA);

    return Cbar;
}

std::shared_ptr<Matrix> ASAPT0::build_ind_pot(std::map<std::string, std::shared_ptr<Matrix> >& vars)
{
    std::shared_ptr<Matrix> Ca = vars["Cocc_A"];
    std::shared_ptr<Matrix> Cr = vars["Cvir_A"];
    std::shared_ptr<Matrix> V_B = vars["V_B"];
    std::shared_ptr<Matrix> J_B = vars["J_B"];

    std::shared_ptr<Matrix> W(V_B->clone());
    W->copy(J_B);
    W->scale(2.0);
    W->add(V_B);

    return linalg::triplet(Ca,W,Cr,true,false,false);
}
std::shared_ptr<Matrix> ASAPT0::build_exch_ind_pot(std::map<std::string, std::shared_ptr<Matrix> >& vars)
{
    std::shared_ptr<Matrix> Ca = vars["Cocc_A"];
    std::shared_ptr<Matrix> Cr = vars["Cvir_A"];

    std::shared_ptr<Matrix> S = vars["S"];

    std::shared_ptr<Matrix> D_A = vars["D_A"];
    std::shared_ptr<Matrix> J_A = vars["J_A"];
    std::shared_ptr<Matrix> K_A = vars["K_A"];
    std::shared_ptr<Matrix> V_A = vars["V_A"];
    std::shared_ptr<Matrix> D_B = vars["D_B"];
    std::shared_ptr<Matrix> J_B = vars["J_B"];
    std::shared_ptr<Matrix> K_B = vars["K_B"];
    std::shared_ptr<Matrix> V_B = vars["V_B"];

    std::shared_ptr<Matrix> J_O = vars["J_O"]; // J[D^A S D^B]
    std::shared_ptr<Matrix> K_O = vars["K_O"]; // K[D^A S D^B]
    std::shared_ptr<Matrix> J_P = vars["J_P"]; // J[D^B S D^A S D^B]

    std::shared_ptr<Matrix> W(K_B->clone());
    std::shared_ptr<Matrix> T;

    // 1
    W->copy(K_B);
    W->scale(-1.0);

    // 2
    T = linalg::triplet(S,D_B,J_A);
    T->scale(-2.0);
    W->add(T);    

    // 3
    T->copy(K_O);
    T->scale(1.0);
    W->add(T);
    
    // 4
    T->copy(J_O);
    T->scale(-2.0);
    W->add(T);

    // 5
    T = linalg::triplet(S,D_B,K_A);
    T->scale(1.0);
    W->add(T);

    // 6
    T = linalg::triplet(J_B,D_B,S);
    T->scale(-2.0);
    W->add(T);

    // 7
    T = linalg::triplet(K_B,D_B,S);
    T->scale(1.0);
    W->add(T);

    // 8
    T = linalg::triplet(linalg::triplet(S,D_B,J_A),D_B,S);
    T->scale(2.0);
    W->add(T);

    // 9
    T = linalg::triplet(linalg::triplet(J_B,D_A,S),D_B,S);
    T->scale(2.0);
    W->add(T);

    // 10
    T = linalg::triplet(K_O,D_B,S);
    T->scale(-1.0);
    W->add(T);

    // 11
    T->copy(J_P);
    T->scale(2.0);
    W->add(T);
    
    // 12
    T = linalg::triplet(linalg::triplet(S,D_B,S),D_A,J_B);
    T->scale(2.0);
    W->add(T);

    // 13
    T = linalg::triplet(S,D_B,K_O,false,false,true);
    T->scale(-1.0);
    W->add(T);

    // 14
    T = linalg::triplet(S,D_B,V_A);
    T->scale(-1.0);
    W->add(T);    
    
    // 15
    T = linalg::triplet(V_B,D_B,S);
    T->scale(-1.0);
    W->add(T);
    
    // 16
    T = linalg::triplet(linalg::triplet(S,D_B,V_A),D_B,S);
    T->scale(1.0);
    W->add(T);

    // 17
    T = linalg::triplet(linalg::triplet(V_B,D_A,S),D_B,S);
    T->scale(1.0);
    W->add(T);

    // 18
    T = linalg::triplet(linalg::triplet(S,D_B,S),D_A,V_B);
    T->scale(1.0);
    W->add(T);

    return linalg::triplet(Ca,W,Cr,true,false,false);
}



std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix> > ASAPT0::compute_x(std::shared_ptr<JK> jk, std::shared_ptr<Matrix> w_B, std::shared_ptr<Matrix> w_A)
{
    auto cpks = std::make_shared<CPKS_ASAPT0>();

    // Effective constructor
    cpks->delta_ = cpks_delta_;
    cpks->maxiter_ = cpks_maxiter_;
    cpks->jk_ = jk;

    cpks->w_A_ = w_B; // Reversal of convention
    cpks->Cocc_A_ = Cocc_A_;
    cpks->Cvir_A_ = Cvir_A_;
    cpks->eps_occ_A_ = eps_occ_A_;
    cpks->eps_vir_A_ = eps_vir_A_;

    cpks->w_B_ = w_A; // Reversal of convention
    cpks->Cocc_B_ = Cocc_B_;
    cpks->Cvir_B_ = Cvir_B_;
    cpks->eps_occ_B_ = eps_occ_B_;
    cpks->eps_vir_B_ = eps_vir_B_;

    // Gogo CPKS
    cpks->compute_cpks();

    // Unpack
    std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix> > x_sol = make_pair(cpks->x_A_,cpks->x_B_);

    return x_sol;
}

CPKS_ASAPT0::CPKS_ASAPT0()
{
}
CPKS_ASAPT0::~CPKS_ASAPT0()
{
}
void CPKS_ASAPT0::compute_cpks()
{
    // Allocate
    x_A_ = std::shared_ptr<Matrix>(w_A_->clone());
    x_B_ = std::shared_ptr<Matrix>(w_B_->clone());
    x_A_->zero();
    x_B_->zero();

    std::shared_ptr<Matrix> r_A(w_A_->clone());
    std::shared_ptr<Matrix> z_A(w_A_->clone());
    std::shared_ptr<Matrix> p_A(w_A_->clone());
    std::shared_ptr<Matrix> r_B(w_B_->clone());
    std::shared_ptr<Matrix> z_B(w_B_->clone());
    std::shared_ptr<Matrix> p_B(w_B_->clone());

    // Initialize (x_0 = 0)
    r_A->copy(w_A_);
    r_B->copy(w_B_);

    preconditioner(r_A,z_A,eps_occ_A_,eps_vir_A_);
    preconditioner(r_B,z_B,eps_occ_B_,eps_vir_B_);

    // Uncoupled value
    //fprintf(outfile, "(A<-B): %24.16E\n", -2.0 * z_A->vector_dot(w_A_));
    //fprintf(outfile, "(B<-A): %24.16E\n", -2.0 * z_B->vector_dot(w_B_));

    p_A->copy(z_A);
    p_B->copy(z_B);

    double zr_old_A = z_A->vector_dot(r_A);
    double zr_old_B = z_B->vector_dot(r_B);

    double r2A = 1.0;
    double r2B = 1.0;

    double b2A = sqrt(w_A_->vector_dot(w_A_));
    double b2B = sqrt(w_B_->vector_dot(w_B_));

    outfile->Printf("  ==> CPKS Iterations <==\n\n");

    outfile->Printf("    Maxiter     = %11d\n", maxiter_);
    outfile->Printf("    Convergence = %11.3E\n", delta_);
    outfile->Printf("\n");

    time_t start;
    time_t stop;

    start = time(NULL);

    outfile->Printf("    -----------------------------------------\n");
    outfile->Printf("    %-4s %11s  %11s  %10s\n", "Iter", "Monomer A", "Monomer B", "Time [s]");
    outfile->Printf("    -----------------------------------------\n");

    int iter;
    for (iter = 0; iter < maxiter_; iter++) {

        std::map<std::string, std::shared_ptr<Matrix> > b;
        if (r2A > delta_) {
            b["A"] = p_A;
        }
        if (r2B > delta_) {
            b["B"] = p_B;
        }

        std::map<std::string, std::shared_ptr<Matrix> > s =
            product(b);

        if (r2A > delta_) {
            std::shared_ptr<Matrix> s_A = s["A"];
            double alpha = r_A->vector_dot(z_A) / p_A->vector_dot(s_A);
            if (alpha < 0.0) {
                throw PSIEXCEPTION("Monomer A: A Matrix is not SPD");
            }
            int no = x_A_->nrow();
            int nv = x_A_->ncol();
            double** xp = x_A_->pointer();
            double** rp = r_A->pointer();
            double** pp = p_A->pointer();
            double** sp = s_A->pointer();
            C_DAXPY(no*nv, alpha,pp[0],1,xp[0],1);
            C_DAXPY(no*nv,-alpha,sp[0],1,rp[0],1);
            r2A = sqrt(C_DDOT(no*nv,rp[0],1,rp[0],1)) / b2A;
        }

        if (r2B > delta_) {
            std::shared_ptr<Matrix> s_B = s["B"];
            double alpha = r_B->vector_dot(z_B) / p_B->vector_dot(s_B);
            if (alpha < 0.0) {
                throw PSIEXCEPTION("Monomer B: A Matrix is not SPD");
            }
            int no = x_B_->nrow();
            int nv = x_B_->ncol();
            double** xp = x_B_->pointer();
            double** rp = r_B->pointer();
            double** pp = p_B->pointer();
            double** sp = s_B->pointer();
            C_DAXPY(no*nv, alpha,pp[0],1,xp[0],1);
            C_DAXPY(no*nv,-alpha,sp[0],1,rp[0],1);
            r2B = sqrt(C_DDOT(no*nv,rp[0],1,rp[0],1)) / b2B;
        }

        stop = time(NULL);
        outfile->Printf("    %-4d %11.3E%1s %11.3E%1s %10ld\n", iter+1,
            r2A, (r2A < delta_ ? "*" : " "),
            r2B, (r2B < delta_ ? "*" : " "),
            stop-start
            );

        if (r2A <= delta_ && r2B <= delta_) {
            break;
        }

        if (r2A > delta_) {
            preconditioner(r_A,z_A,eps_occ_A_,eps_vir_A_);
            double zr_new = z_A->vector_dot(r_A);
            double beta = zr_new / zr_old_A;
            zr_old_A = zr_new;
            int no = x_A_->nrow();
            int nv = x_A_->ncol();
            double** pp = p_A->pointer();
            double** zp = z_A->pointer();
            C_DSCAL(no*nv,beta,pp[0],1);
            C_DAXPY(no*nv,1.0,zp[0],1,pp[0],1);
        }

        if (r2B > delta_) {
            preconditioner(r_B,z_B,eps_occ_B_,eps_vir_B_);
            double zr_new = z_B->vector_dot(r_B);
            double beta = zr_new / zr_old_B;
            zr_old_B = zr_new;
            int no = x_B_->nrow();
            int nv = x_B_->ncol();
            double** pp = p_B->pointer();
            double** zp = z_B->pointer();
            C_DSCAL(no*nv,beta,pp[0],1);
            C_DAXPY(no*nv,1.0,zp[0],1,pp[0],1);
        }
    }

    outfile->Printf("    -----------------------------------------\n");
    outfile->Printf("\n");

    if (iter == maxiter_)
        throw PSIEXCEPTION("CPKS did not converge.");
}
void CPKS_ASAPT0::preconditioner(std::shared_ptr<Matrix> r,
                               std::shared_ptr<Matrix> z,
                               std::shared_ptr<Vector> o,
                               std::shared_ptr<Vector> v)
{
    int no = o->dim();
    int nv = v->dim();

    double** rp = r->pointer();
    double** zp = z->pointer();

    double* op = o->pointer();
    double* vp = v->pointer();

    for (int i = 0; i < no; i++) {
        for (int a = 0; a < nv; a++) {
            zp[i][a] = rp[i][a] / (vp[a] - op[i]);
        }
    }
}
std::map<std::string, std::shared_ptr<Matrix> > CPKS_ASAPT0::product(std::map<std::string, std::shared_ptr<Matrix> > b)
{
    std::map<std::string, std::shared_ptr<Matrix> > s;

    bool do_A = b.count("A");
    bool do_B = b.count("B");

    std::vector<SharedMatrix>& Cl = jk_->C_left();
    std::vector<SharedMatrix>& Cr = jk_->C_right();
    Cl.clear();
    Cr.clear();

    if (do_A) {
        Cl.push_back(Cocc_A_);
        int no = b["A"]->nrow();
        int nv = b["A"]->ncol();
        int nso = Cvir_A_->nrow();
        double** Cp = Cvir_A_->pointer();
        double** bp = b["A"]->pointer();
        std::shared_ptr<Matrix> T(new Matrix("T",nso,no));
        double** Tp = T->pointer();
        C_DGEMM('N','T',nso,no,nv,1.0,Cp[0],nv,bp[0],nv,0.0,Tp[0],no);
        Cr.push_back(T);
    }

    if (do_B) {
        Cl.push_back(Cocc_B_);
        int no = b["B"]->nrow();
        int nv = b["B"]->ncol();
        int nso = Cvir_B_->nrow();
        double** Cp = Cvir_B_->pointer();
        double** bp = b["B"]->pointer();
        std::shared_ptr<Matrix> T(new Matrix("T",nso,no));
        double** Tp = T->pointer();
        C_DGEMM('N','T',nso,no,nv,1.0,Cp[0],nv,bp[0],nv,0.0,Tp[0],no);
        Cr.push_back(T);
    }

    jk_->compute();

    const std::vector<SharedMatrix>& J = jk_->J();
    const std::vector<SharedMatrix>& K = jk_->K();

    int indA = 0;
    int indB = (do_A ? 1 : 0);

    if (do_A) {
        std::shared_ptr<Matrix> Jv = J[indA];
        std::shared_ptr<Matrix> Kv = K[indA];
        Jv->scale(4.0);
        Jv->subtract(Kv);
        Jv->subtract(Kv->transpose());

        int no = b["A"]->nrow();
        int nv = b["A"]->ncol();
        int nso = Cvir_A_->nrow();
        std::shared_ptr<Matrix> T(new Matrix("T", no, nso));
        s["A"] = std::shared_ptr<Matrix>(new Matrix("S", no, nv));
        double** Cop = Cocc_A_->pointer();
        double** Cvp = Cvir_A_->pointer();
        double** Jp = Jv->pointer();
        double** Tp = T->pointer();
        double** Sp = s["A"]->pointer();
        C_DGEMM('T','N',no,nso,nso,1.0,Cop[0],no,Jp[0],nso,0.0,Tp[0],nso);
        C_DGEMM('N','N',no,nv,nso,1.0,Tp[0],nso,Cvp[0],nv,0.0,Sp[0],nv);

        double** bp = b["A"]->pointer();
        double* op = eps_occ_A_->pointer();
        double* vp = eps_vir_A_->pointer();
        for (int i = 0; i < no; i++) {
            for (int a = 0; a < nv; a++) {
                Sp[i][a] += bp[i][a] * (vp[a] - op[i]);
            }
        }
    }

    if (do_B) {
        std::shared_ptr<Matrix> Jv = J[indB];
        std::shared_ptr<Matrix> Kv = K[indB];
        Jv->scale(4.0);
        Jv->subtract(Kv);
        Jv->subtract(Kv->transpose());

        int no = b["B"]->nrow();
        int nv = b["B"]->ncol();
        int nso = Cvir_B_->nrow();
        std::shared_ptr<Matrix> T(new Matrix("T", no, nso));
        s["B"] = std::shared_ptr<Matrix>(new Matrix("S", no, nv));
        double** Cop = Cocc_B_->pointer();
        double** Cvp = Cvir_B_->pointer();
        double** Jp = Jv->pointer();
        double** Tp = T->pointer();
        double** Sp = s["B"]->pointer();
        C_DGEMM('T','N',no,nso,nso,1.0,Cop[0],no,Jp[0],nso,0.0,Tp[0],nso);
        C_DGEMM('N','N',no,nv,nso,1.0,Tp[0],nso,Cvp[0],nv,0.0,Sp[0],nv);

        double** bp = b["B"]->pointer();
        double* op = eps_occ_B_->pointer();
        double* vp = eps_vir_B_->pointer();
        for (int i = 0; i < no; i++) {
            for (int a = 0; a < nv; a++) {
                Sp[i][a] += bp[i][a] * (vp[a] - op[i]);
            }
        }
    }

    return s;
}







}  // namespace sapt
}  // namespace psi

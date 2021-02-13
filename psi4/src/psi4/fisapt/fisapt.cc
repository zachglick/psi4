/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2021 The Psi4 Developers.
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

#include "fisapt.h"

#include <algorithm>
#include <ctime>
#include <functional>
#include <set>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "psi4/psi4-dec.h"
#include "psi4/physconst.h"

#include "psi4/lib3index/dfhelper.h"
#include "psi4/lib3index/3index.h"
#include "psi4/libcubeprop/csg.h"
#include "psi4/libdiis/diismanager.h"
#include "psi4/libfock/jk.h"
#include "psi4/libfock/cubature.h"
#include "psi4/libfock/points.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/extern.h"
#include "psi4/libmints/factory.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/orthog.h"
#include "psi4/libmints/potential.h"
#include "psi4/libmints/vector.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libqt/qt.h"

#include "local2.h"

namespace psi {

namespace fisapt {

FISAPT::FISAPT(SharedWavefunction scf) : options_(Process::environment.options), reference_(scf) { common_init(); }
FISAPT::FISAPT(SharedWavefunction scf, Options& options) : options_(options), reference_(scf) { common_init(); }
FISAPT::~FISAPT() {}

void FISAPT::common_init() {
    reference_->set_module("fisapt");

    primary_ = reference_->basisset();
    doubles_ = Process::environment.get_memory() / sizeof(double) * options_.get_double("FISAPT_MEM_SAFETY_FACTOR");

    matrices_["Cfocc"] = reference_->Ca_subset("AO", "FROZEN_OCC");

    vectors_["eps_all"] = reference_->epsilon_a_subset("AO", "ALL");

    matrices_["Call"] = reference_->Ca_subset("AO", "ALL");
    matrices_["Cocc"] = reference_->Ca_subset("AO", "OCC");
    matrices_["Cvir"] = reference_->Ca_subset("AO", "VIR");

    vectors_["eps_occ"] = reference_->epsilon_a_subset("AO", "OCC");
    vectors_["eps_vir"] = reference_->epsilon_a_subset("AO", "VIR");

    matrices_["Caocc"] = reference_->Ca_subset("AO", "ACTIVE_OCC");
    matrices_["Cavir"] = reference_->Ca_subset("AO", "ACTIVE_VIR");
    matrices_["Cfvir"] = reference_->Ca_subset("AO", "FROZEN_VIR");

    vectors_["eps_focc"] = reference_->epsilon_a_subset("AO", "FROZEN_OCC");
    vectors_["eps_aocc"] = reference_->epsilon_a_subset("AO", "ACTIVE_OCC");
    vectors_["eps_avir"] = reference_->epsilon_a_subset("AO", "ACTIVE_VIR");
    vectors_["eps_fvir"] = reference_->epsilon_a_subset("AO", "FROZEN_VIR");
}

void FISAPT::print_header() {
    outfile->Printf("\t --------------------------------------------\n");
    outfile->Printf("\t                    FISAPT0                  \n");
    outfile->Printf("\t                  Rob Parrish                \n");
    outfile->Printf("\t --------------------------------------------\n");
    outfile->Printf("\n");

    outfile->Printf("    Do F-SAPT = %11s\n", options_.get_bool("FISAPT_DO_FSAPT") ? "Yes" : "No");
    outfile->Printf("    Do Plot   = %11s\n", options_.get_bool("FISAPT_DO_PLOT") ? "Yes" : "No");
    outfile->Printf("    Memory    = %11.3f [GiB]\n", (doubles_ * 8) / (1024.0 * 1024.0 * 1024.0));
    outfile->Printf("\n");
}

void FISAPT::localize() {
    outfile->Printf("  ==> Localization (IBO) <==\n\n");

    std::shared_ptr<Matrix> Focc =
        std::make_shared<Matrix>("Focc", vectors_["eps_occ"]->dimpi()[0], vectors_["eps_occ"]->dimpi()[0]);
    Focc->set_diagonal(vectors_["eps_occ"]);

    std::vector<int> ranges;
    ranges.push_back(0);
    ranges.push_back(vectors_["eps_focc"]->dimpi()[0]);
    ranges.push_back(vectors_["eps_occ"]->dimpi()[0]);

    std::shared_ptr<fisapt::IBOLocalizer2> local =
        fisapt::IBOLocalizer2::build(primary_, reference_->get_basisset("MINAO"), matrices_["Cocc"], options_);
    local->print_header();
    std::map<std::string, std::shared_ptr<Matrix> > ret = local->localize(matrices_["Cocc"], Focc, ranges);

    matrices_["Locc"] = ret["L"];
    matrices_["Qocc"] = ret["Q"];
}

void FISAPT::partition() {
    outfile->Printf("  ==> Partitioning <==\n\n");

    // => Sizing <= //

    std::shared_ptr<Molecule> mol = primary_->molecule();
    int nA = mol->natom();
    int na = matrices_["Locc"]->colspi()[0];

    // => Monomer Atoms <= //

    const std::vector<std::pair<int, int> >& fragment_list = mol->get_fragments();
    if (!(fragment_list.size() == 2 || fragment_list.size() == 3)) {
        throw PSIEXCEPTION("FISAPT: Molecular system must have 2 (A+B) or 3 (A+B+C) fragments");
    }

    std::vector<int> indA;
    std::vector<int> indB;
    std::vector<int> indC;

    for (int ind = fragment_list[0].first; ind < fragment_list[0].second; ind++) {
        indA.push_back(ind);
    }
    for (int ind = fragment_list[1].first; ind < fragment_list[1].second; ind++) {
        indB.push_back(ind);
    }
    if (fragment_list.size() == 3) {
        for (int ind = fragment_list[2].first; ind < fragment_list[2].second; ind++) {
            indC.push_back(ind);
        }
    }

    outfile->Printf("   => Atomic Partitioning <= \n\n");
    outfile->Printf("    Monomer A: %3zu atoms\n", indA.size());
    outfile->Printf("    Monomer B: %3zu atoms\n", indB.size());
    outfile->Printf("    Monomer C: %3zu atoms\n", indC.size());
    outfile->Printf("\n");

    // => Fragment Orbital Charges <= //

    auto QF = std::make_shared<Matrix>("QF", 3, na);
    double** QFp = QF->pointer();
    double** Qp = matrices_["Qocc"]->pointer();

    for (int ind = 0; ind < indA.size(); ind++) {
        for (int a = 0; a < na; a++) {
            QFp[0][a] += Qp[indA[ind]][a];
        }
    }

    for (int ind = 0; ind < indB.size(); ind++) {
        for (int a = 0; a < na; a++) {
            QFp[1][a] += Qp[indB[ind]][a];
        }
    }

    for (int ind = 0; ind < indC.size(); ind++) {
        for (int a = 0; a < na; a++) {
            QFp[2][a] += Qp[indC[ind]][a];
        }
    }

    // => Link Identification <= //

    std::string link_selection = options_.get_str("FISAPT_LINK_SELECTION");
    outfile->Printf("   => Link Bond Identification <=\n\n");
    outfile->Printf("    Link Bond Selection = %s\n\n", link_selection.c_str());

    std::vector<int> link_orbs;
    std::vector<std::pair<int, int> > link_atoms;
    std::vector<std::string> link_types;

    if (link_selection == "AUTOMATIC") {
        double delta = options_.get_double("FISAPT_CHARGE_COMPLETENESS");
        outfile->Printf("    Charge Completeness = %5.3f\n\n", delta);
        for (int a = 0; a < na; a++) {
            if (QFp[0][a] > delta || QFp[1][a] > delta || QFp[2][a] > delta) {
                ;
            } else if (QFp[0][a] + QFp[2][a] > delta) {
                link_orbs.push_back(a);
                link_types.push_back("AC");
            } else if (QFp[1][a] + QFp[2][a] > delta) {
                link_orbs.push_back(a);
                link_types.push_back("BC");
            } else if (QFp[0][a] + QFp[1][a] > delta) {
                link_orbs.push_back(a);
                link_types.push_back("AB");
            } else if (QFp[0][a] + QFp[1][a] > delta) {
                ;
            } else {
                throw PSIEXCEPTION("FISAPT: A, B, and C are bonded?! 3c-2e bonds are not cool.");
            }
        }

        for (int ind = 0; ind < link_orbs.size(); ind++) {
            int a = link_orbs[ind];
            std::vector<std::pair<double, int> > Qvals;
            for (int A = 0; A < nA; A++) {
                Qvals.push_back(std::pair<double, int>(Qp[A][a], A));
            }
            std::sort(Qvals.begin(), Qvals.end(), std::greater<std::pair<double, int> >());
            int A1 = Qvals[0].second;
            int A2 = Qvals[1].second;
            if (A2 < A1) std::swap(A1, A2);
            link_atoms.push_back(std::pair<int, int>(A1, A2));
        }

    } else if (link_selection == "MANUAL") {
        for (int ind = 0; ind < options_["FISAPT_MANUAL_LINKS"].size(); ind++) {
            link_atoms.push_back(std::pair<int, int>(options_["FISAPT_MANUAL_LINKS"][ind][0].to_integer() - 1,
                                                     options_["FISAPT_MANUAL_LINKS"][ind][1].to_integer() - 1));
        }

        for (int ind = 0; ind < link_atoms.size(); ind++) {
            int A1 = link_atoms[ind].first;
            int A2 = link_atoms[ind].second;
            if (A2 < A1) std::swap(A1, A2);

            double Qmax = 0.0;
            int aind = -1;
            for (int a = 0; a < na; a++) {
                double Qval = Qp[A1][a] * Qp[A2][a];
                if (Qmax < Qval) {
                    Qmax = Qval;
                    aind = a;
                }
            }
            link_orbs.push_back(aind);

            if (std::find(indA.begin(), indA.end(), A1) != indA.end() &&
                std::find(indC.begin(), indC.end(), A2) != indC.end()) {
                link_types.push_back("AC");
            } else if (std::find(indB.begin(), indB.end(), A1) != indB.end() &&
                       std::find(indC.begin(), indC.end(), A2) != indC.end()) {
                link_types.push_back("BC");
            } else if (std::find(indA.begin(), indA.end(), A1) != indA.end() &&
                       std::find(indB.begin(), indB.end(), A2) != indB.end()) {
                link_types.push_back("AB");
            } else {
                throw PSIEXCEPTION("FISAPT: FISAPT_MANUAL_LINKS contains a bond which is not AB, AC, or BC");
            }
        }

    } else {
        throw PSIEXCEPTION("FISAPT: Unrecognized FISAPT_LINK_SELECTION option.");
    }

    outfile->Printf("    Total Link Bonds = %zu\n\n", link_orbs.size());

    if (link_orbs.size()) {
        outfile->Printf("    --------------------------\n");
        outfile->Printf("    %-4s %4s %4s %5s %5s\n", "N", "Orb", "Type", "Aind1", "Aind2");
        outfile->Printf("    --------------------------\n");
        for (int ind = 0; ind < link_orbs.size(); ind++) {
            outfile->Printf("    %-4d %4d %4s %5d %5d\n", ind + 1, link_orbs[ind], link_types[ind].c_str(),
                            link_atoms[ind].first + 1, link_atoms[ind].second + 1);
        }
        outfile->Printf("    --------------------------\n");
        outfile->Printf("\n");
    }

    // => Nuclear Charge Targets <= //

    vectors_["ZA"] = std::make_shared<Vector>("ZA", nA);
    vectors_["ZB"] = std::make_shared<Vector>("ZB", nA);
    vectors_["ZC"] = std::make_shared<Vector>("ZC", nA);

    double* ZAp = vectors_["ZA"]->pointer();
    double* ZBp = vectors_["ZB"]->pointer();
    double* ZCp = vectors_["ZC"]->pointer();

    for (int ind = 0; ind < indA.size(); ind++) {
        ZAp[indA[ind]] = mol->Z(indA[ind]);
    }
    for (int ind = 0; ind < indB.size(); ind++) {
        ZBp[indB[ind]] = mol->Z(indB[ind]);
    }
    for (int ind = 0; ind < indC.size(); ind++) {
        ZCp[indC[ind]] = mol->Z(indC[ind]);
    }

    // => Local Orbital Targets <= //

    std::vector<int> orbsA;
    std::vector<int> orbsB;
    std::vector<int> orbsC;

    // => Assign Links <= //

    std::string link_assignment = options_.get_str("FISAPT_LINK_ASSIGNMENT");
    if (!(link_assignment == "AB" || link_assignment == "C")) {
        throw PSIEXCEPTION("FISAPT: FISAPT_LINK_ASSIGNMENT not recognized");
    }

    outfile->Printf("   => Link Bond Assignment <=\n\n");
    outfile->Printf("    Link Bond Assignment      = %s\n", link_assignment.c_str());
    outfile->Printf("\n");

    if (link_assignment == "C") {
        for (int ind = 0; ind < link_orbs.size(); ind++) {
            int a = link_orbs[ind];
            int A1 = link_atoms[ind].first;
            int A2 = link_atoms[ind].second;
            std::string type = link_types[ind];

            if (type == "AC") {
                ZAp[A1] -= 1.0;
                ZCp[A1] += 1.0;
                orbsC.push_back(a);
            } else if (type == "BC") {
                ZBp[A1] -= 1.0;
                ZCp[A1] += 1.0;
                orbsC.push_back(a);
            } else if (type == "AB") {
                ZAp[A1] -= 1.0;
                ZCp[A1] += 1.0;
                ZBp[A2] -= 1.0;
                ZCp[A2] += 1.0;
                orbsC.push_back(a);
            }
        }

    } else if (link_assignment == "AB") {
        for (int ind = 0; ind < link_orbs.size(); ind++) {
            int a = link_orbs[ind];
            int A1 = link_atoms[ind].first;
            int A2 = link_atoms[ind].second;
            std::string type = link_types[ind];

            if (type == "AC") {
                ZAp[A1] += 1.0;
                ZCp[A1] -= 1.0;
                orbsA.push_back(a);
            } else if (type == "BC") {
                ZBp[A1] += 1.0;
                ZCp[A1] -= 1.0;
                orbsB.push_back(a);
            } else if (type == "AB") {
                throw PSIEXCEPTION("FISAPT: AB link requires LINK_ASSIGNMENT C");
            }
        }
    }

    // => Remaining Orbitals <= //

    const std::vector<int>& fragment_charges = mol->get_fragment_charges();
    int CA2 = fragment_charges[0];
    int CB2 = fragment_charges[1];
    int CC2 = (fragment_charges.size() == 3 ? fragment_charges[2] : 0);

    double ZA2 = 0.0;
    double ZB2 = 0.0;
    double ZC2 = 0.0;
    for (int ind = 0; ind < nA; ind++) {
        ZA2 += ZAp[ind];
        ZB2 += ZBp[ind];
        ZC2 += ZCp[ind];
    }

    int EA2 = round(ZA2) - CA2;  // Number of needed electrons
    int EB2 = round(ZB2) - CB2;  // Number of needed electrons
    int EC2 = round(ZC2) - CC2;  // Number of needed electrons

    if (EA2 % 2 != 0) throw PSIEXCEPTION("FISAPT: Charge on A is incompatible with singlet");
    if (EB2 % 2 != 0) throw PSIEXCEPTION("FISAPT: Charge on B is incompatible with singlet");
    if (EC2 % 2 != 0) throw PSIEXCEPTION("FISAPT: Charge on C is incompatible with singlet");

    int NA2 = EA2 / 2;
    int NB2 = EB2 / 2;
    int NC2 = EC2 / 2;

    if (NA2 + NB2 + NC2 != na)
        throw PSIEXCEPTION("FISAPT: Sum of charges is incompatible with total number of electrons.");

    int RA2 = NA2 - orbsA.size();
    int RB2 = NB2 - orbsB.size();
    int RC2 = NC2 - orbsC.size();

    std::set<int> taken_orbs;
    for (int x : orbsA) taken_orbs.insert(x);
    for (int x : orbsB) taken_orbs.insert(x);
    for (int x : orbsC) taken_orbs.insert(x);

    std::vector<std::pair<double, int> > QCvals;
    for (int a = 0; a < na; a++) {
        if (taken_orbs.count(a)) continue;
        QCvals.push_back(std::pair<double, int>(QFp[2][a], a));
    }
    std::sort(QCvals.begin(), QCvals.end(), std::greater<std::pair<double, int> >());
    for (int ind = 0; ind < RC2; ind++) {
        int a = QCvals[ind].second;
        orbsC.push_back(a);
        taken_orbs.insert(a);
    }

    std::vector<std::pair<double, int> > QAvals;
    for (int a = 0; a < na; a++) {
        if (taken_orbs.count(a)) continue;
        QAvals.push_back(std::pair<double, int>(QFp[0][a], a));
    }
    std::sort(QAvals.begin(), QAvals.end(), std::greater<std::pair<double, int> >());
    for (int ind = 0; ind < RA2; ind++) {
        int a = QAvals[ind].second;
        orbsA.push_back(a);
        taken_orbs.insert(a);
    }

    std::vector<std::pair<double, int> > QBvals;
    for (int a = 0; a < na; a++) {
        if (taken_orbs.count(a)) continue;
        QBvals.push_back(std::pair<double, int>(QFp[1][a], a));
    }
    std::sort(QBvals.begin(), QBvals.end(), std::greater<std::pair<double, int> >());
    for (int ind = 0; ind < RB2; ind++) {
        int a = QBvals[ind].second;
        orbsB.push_back(a);
        taken_orbs.insert(a);
    }

    // => Orbital Subsets <= //

    std::sort(orbsA.begin(), orbsA.end());
    std::sort(orbsB.begin(), orbsB.end());
    std::sort(orbsC.begin(), orbsC.end());

    matrices_["LoccA"] = FISAPT::extract_columns(orbsA, matrices_["Locc"]);
    matrices_["LoccB"] = FISAPT::extract_columns(orbsB, matrices_["Locc"]);
    matrices_["LoccC"] = FISAPT::extract_columns(orbsC, matrices_["Locc"]);

    matrices_["LoccA"]->set_name("LoccA");
    matrices_["LoccB"]->set_name("LoccB");
    matrices_["LoccC"]->set_name("LoccC");

    // matrices_["LoccA"]->print();
    // matrices_["LoccB"]->print();
    // matrices_["LoccC"]->print();

    // => Summary <= //

    double ZA = 0.0;
    double ZB = 0.0;
    double ZC = 0.0;
    for (int ind = 0; ind < nA; ind++) {
        ZA += ZAp[ind];
        ZB += ZBp[ind];
        ZC += ZCp[ind];
    }
    ZA = round(ZA);
    ZB = round(ZB);
    ZC = round(ZC);

    double YA = round(2.0 * orbsA.size());
    double YB = round(2.0 * orbsB.size());
    double YC = round(2.0 * orbsC.size());

    outfile->Printf("   => Partition Summary <=\n\n");
    outfile->Printf("    Monomer A: %2d charge, %3d protons, %3d electrons, %3zu docc\n", (int)(ZA - YA), (int)ZA,
                    (int)YA, orbsA.size());
    outfile->Printf("    Monomer B: %2d charge, %3d protons, %3d electrons, %3zu docc\n", (int)(ZB - YB), (int)ZB,
                    (int)YB, orbsB.size());
    outfile->Printf("    Monomer C: %2d charge, %3d protons, %3d electrons, %3zu docc\n", (int)(ZC - YC), (int)ZC,
                    (int)YC, orbsC.size());
    outfile->Printf("\n");
}

void FISAPT::overlap() {
    outfile->Printf("  ==> Overlap Integrals <==\n\n");

    int nm = primary_->nbf();
    auto Tfact = std::make_shared<IntegralFactory>(primary_);
    std::shared_ptr<OneBodyAOInt> Tint = std::shared_ptr<OneBodyAOInt>(Tfact->ao_overlap());
    matrices_["S"] = std::make_shared<Matrix>("S", nm, nm);
    Tint->compute(matrices_["S"]);
}

void FISAPT::kinetic() {
    outfile->Printf("  ==> Kinetic Integrals <==\n\n");

    int nm = primary_->nbf();
    auto Tfact = std::make_shared<IntegralFactory>(primary_);
    std::shared_ptr<OneBodyAOInt> Tint = std::shared_ptr<OneBodyAOInt>(Tfact->ao_kinetic());
    matrices_["T"] = std::make_shared<Matrix>("T", nm, nm);
    Tint->compute(matrices_["T"]);
}

void FISAPT::nuclear() {
    outfile->Printf("  ==> Nuclear Integrals <==\n\n");

    // => Sizing <= //

    std::shared_ptr<Molecule> mol = primary_->molecule();
    int nA = mol->natom();
    int nm = primary_->nbf();

    // => Nuclear Potentials <= //

    auto Zxyz = std::make_shared<Matrix>("Zxyz", nA, 4);
    double** Zxyzp = Zxyz->pointer();

    auto Vfact = std::make_shared<IntegralFactory>(primary_);
    std::shared_ptr<PotentialInt> Vint;
    Vint = std::shared_ptr<PotentialInt>(static_cast<PotentialInt*>(Vfact->ao_potential()));
    Vint->set_charge_field(Zxyz);

    // > Molecular Centers < //

    for (int A = 0; A < nA; A++) {
        Zxyzp[A][1] = mol->x(A);
        Zxyzp[A][2] = mol->y(A);
        Zxyzp[A][3] = mol->z(A);
    }

    // > A < //

    double* ZAp = vectors_["ZA"]->pointer();
    for (int A = 0; A < nA; A++) {
        Zxyzp[A][0] = ZAp[A];
    }

    matrices_["VA"] = std::make_shared<Matrix>("VA", nm, nm);
    Vint->compute(matrices_["VA"]);

    // > B < //

    double* ZBp = vectors_["ZB"]->pointer();
    for (int A = 0; A < nA; A++) {
        Zxyzp[A][0] = ZBp[A];
    }

    matrices_["VB"] = std::make_shared<Matrix>("VB", nm, nm);
    Vint->compute(matrices_["VB"]);

    // > C < //

    double* ZCp = vectors_["ZC"]->pointer();
    for (int A = 0; A < nA; A++) {
        Zxyzp[A][0] = ZCp[A];
    }

    matrices_["VC"] = std::make_shared<Matrix>("VC", nm, nm);
    Vint->compute(matrices_["VC"]);

    // => Nuclear Repulsions <= //

    auto Zs = std::make_shared<Matrix>("Zs", nA, 3);
    double** Zsp = Zs->pointer();

    auto Rinv = std::make_shared<Matrix>("Rinv", nA, nA);
    double** Rinvp = Rinv->pointer();

    for (int A = 0; A < nA; A++) {
        Zsp[A][0] = ZAp[A];
        Zsp[A][1] = ZBp[A];
        Zsp[A][2] = ZCp[A];
    }

    for (int A = 0; A < nA; A++) {
        for (int B = 0; B < nA; B++) {
            if (A == B) continue;
            Rinvp[A][B] = 1.0 / mol->xyz(A).distance(mol->xyz(B));
        }
    }

    /// Nuclear repulsion for A, B, C,
    std::shared_ptr<Matrix> Enucs = linalg::triplet(Zs, Rinv, Zs, true, false, false);
    Enucs->scale(0.5);
    Enucs->set_name("E Nuc");
    matrices_["E NUC"] = Enucs;

    double** Enucsp = Enucs->pointer();
    double Etot = 0.0;
    for (int A = 0; A < 3; A++) {
        for (int B = 0; B < 3; B++) {
            Etot += Enucsp[A][B];
        }
    }

    // => External potential <= //

    if (reference_->external_pot()) {
        if (options_.get_bool("EXTERNAL_POTENTIAL_SYMMETRY") == false && reference_->nirrep() != 1)
            throw PSIEXCEPTION("SCF: External Fields are not consistent with symmetry. Set symmetry c1.");

        std::shared_ptr<Matrix> V_extern = reference_->external_pot()->computePotentialMatrix(primary_);

        if (options_.get_bool("EXTERNAL_POTENTIAL_SYMMETRY")) {
            // Attempt to apply symmetry. No error checking is performed.
            std::shared_ptr<Matrix> V_extern_sym = reference_->matrix_factory()->create_shared_matrix("External Potential");
            V_extern_sym->apply_symmetry(V_extern, reference_->aotoso());
            V_extern = V_extern_sym;
        }

        if (reference_->get_print()) {
            reference_->external_pot()->set_print(reference_->get_print());
            reference_->external_pot()->print();
        }
        if (reference_->get_print() > 3) V_extern->print();

        // Save external potential to add to one-electron SCF potential
        matrices_["VE"] = V_extern;

        // Extra nuclear repulsion
        std::vector<int> none;
        std::vector<int> frag_list(1);
        double Enuc_extern;
        char frag = '@'; // Next characters are 'A', 'B', 'C'
        for (int A = 0; A < mol->nfragments(); A++) {
            frag++;
            frag_list[0] = A;
            std::shared_ptr<Molecule> mol_frag = mol->extract_subsets(frag_list, none);
            Enuc_extern = reference_->external_pot()->computeNuclearEnergy(mol_frag);

            outfile->Printf("           Old Nuclear Repulsion %c: %24.16E [Eh]\n", frag, Enucsp[A][A]);
            outfile->Printf("    Additional Nuclear Repulsion %c: %24.16E [Eh]\n", frag, Enuc_extern);
            outfile->Printf("         Total Nuclear Repulsion %c: %24.16E [Eh]\n", frag, Enucsp[A][A] + Enuc_extern);

            Enucsp[A][A] += Enuc_extern;
            Etot += Enuc_extern;
        }
        matrices_["E NUC"] = Enucs;
        outfile->Printf("\n");
    }

    // => Print <= //

    // Zs->print();
    // Enucs->print();

    outfile->Printf("    Nuclear Repulsion Tot: %24.16E [Eh]\n", Etot);
    outfile->Printf("\n");
}

void FISAPT::coulomb() {
    outfile->Printf("  ==> Coulomb Integrals <==\n\n");

    // => Global JK Object <= //

    jk_ = JK::build_JK(primary_, reference_->get_basisset("DF_BASIS_SCF"), options_, false, doubles_);
    jk_->set_memory(doubles_);

    // => Build J and K for embedding <= //

    std::vector<SharedMatrix>& Cl = jk_->C_left();
    std::vector<SharedMatrix>& Cr = jk_->C_right();

    const std::vector<SharedMatrix>& J = jk_->J();
    const std::vector<SharedMatrix>& K = jk_->K();

    Cl.clear();
    Cr.clear();

    // => Prevent Failure if C, D, or E are empty <= //

    if (matrices_["LoccC"]->colspi()[0] > 0) {
        Cl.push_back(matrices_["LoccC"]);
        Cr.push_back(matrices_["LoccC"]);
    }

    jk_->set_do_J(true);
    jk_->set_do_K(true);
    jk_->initialize();
    jk_->print_header();

    jk_->compute();

    int nn = primary_->nbf();
    matrices_["JC"] = std::make_shared<Matrix>("JC", nn, nn);
    matrices_["KC"] = std::make_shared<Matrix>("KC", nn, nn);
    if (matrices_["LoccC"]->colspi()[0] > 0) {
        matrices_["JC"]->copy(J[0]);
        matrices_["KC"]->copy(K[0]);
    }
}

void FISAPT::scf() {
    outfile->Printf("  ==> Relaxed SCF Equations <==\n\n");

    // => Restricted Basis Sets with C Projected <= //

    std::vector<std::shared_ptr<Matrix> > Xs;
    Xs.push_back(matrices_["LoccA"]);
    Xs.push_back(matrices_["LoccB"]);
    Xs.push_back(matrices_["Cvir"]);
    matrices_["XC"] = linalg::horzcat(Xs);
    matrices_["XC"]->set_name("XC");

    // => Embedding Potential for C <= //

    std::shared_ptr<Matrix> WC(matrices_["VC"]->clone());
    WC->copy(matrices_["VC"]);
    WC->add(matrices_["JC"]);
    WC->add(matrices_["JC"]);
    WC->subtract(matrices_["KC"]);
    matrices_["WC"] = WC;

    // => A <= //

    outfile->Printf("  ==> SCF A: <==\n\n");
    std::shared_ptr<Matrix> VA_SCF(matrices_["VA"]->clone());
    VA_SCF->copy(matrices_["VA"]);
    if (reference_->external_pot()) VA_SCF->add(matrices_["VE"]);
    std::shared_ptr<FISAPTSCF> scfA =
        std::make_shared<FISAPTSCF>(jk_, matrices_["E NUC"]->get(0, 0), matrices_["S"], matrices_["XC"], matrices_["T"],
                                    VA_SCF, matrices_["WC"], matrices_["LoccA"], options_);
    scfA->compute_energy();

    scalars_["E0 A"] = scfA->scalars()["E SCF"];
    matrices_["Cocc0A"] = scfA->matrices()["Cocc"];
    matrices_["Cvir0A"] = scfA->matrices()["Cvir"];
    matrices_["J0A"] = scfA->matrices()["J"];
    matrices_["K0A"] = scfA->matrices()["K"];
    matrices_["F0A"] = scfA->matrices()["F"];
    vectors_["eps_occ0A"] = scfA->vectors()["eps_occ"];
    vectors_["eps_vir0A"] = scfA->vectors()["eps_vir"];

    // => B <= //

    outfile->Printf("  ==> SCF B: <==\n\n");
    std::shared_ptr<Matrix> VB_SCF(matrices_["VB"]->clone());
    VB_SCF->copy(matrices_["VB"]);
    if (reference_->external_pot()) VB_SCF->add(matrices_["VE"]);
    std::shared_ptr<FISAPTSCF> scfB =
        std::make_shared<FISAPTSCF>(jk_, matrices_["E NUC"]->get(1, 1), matrices_["S"], matrices_["XC"], matrices_["T"],
                                    VB_SCF, matrices_["WC"], matrices_["LoccB"], options_);
    scfB->compute_energy();

    scalars_["E0 B"] = scfB->scalars()["E SCF"];
    matrices_["Cocc0B"] = scfB->matrices()["Cocc"];
    matrices_["Cvir0B"] = scfB->matrices()["Cvir"];
    matrices_["J0B"] = scfB->matrices()["J"];
    matrices_["K0B"] = scfB->matrices()["K"];
    matrices_["F0B"] = scfB->matrices()["F"];
    vectors_["eps_occ0B"] = scfB->vectors()["eps_occ"];
    vectors_["eps_vir0B"] = scfB->vectors()["eps_vir"];
}

// Prep the computation to handle frozen core orbitals
void FISAPT::freeze_core() {
    outfile->Printf("  ==> Frozen Core <==\n\n");

    // => Frozen Core (for Disp) <= //

    std::shared_ptr<Molecule> mol = primary_->molecule();

    std::vector<int> none;
    std::vector<int> zero;
    zero.push_back(0);
    std::vector<int> one;
    one.push_back(1);
    std::shared_ptr<Molecule> molA = mol->extract_subsets(zero, none);
    std::shared_ptr<Molecule> molB = mol->extract_subsets(one, none);

    // std::shared_ptr<Molecule> molA = mol->extract_subsets({0},{});
    // std::shared_ptr<Molecule> molB = mol->extract_subsets({1},{});

    int nfocc0A = reference_->basisset()->n_frozen_core(options_.get_str("FREEZE_CORE"), molA);
    int nfocc0B = reference_->basisset()->n_frozen_core(options_.get_str("FREEZE_CORE"), molB);

    int nbf = matrices_["Cocc0A"]->rowspi()[0];
    int nocc0A = matrices_["Cocc0A"]->colspi()[0];
    int nocc0B = matrices_["Cocc0B"]->colspi()[0];
    int naocc0A = nocc0A - nfocc0A;
    int naocc0B = nocc0B - nfocc0B;
    int nvir0A = matrices_["Cvir0A"]->colspi()[0];
    int nvir0B = matrices_["Cvir0B"]->colspi()[0];
    int nmoA = nocc0A + nvir0A;
    int nmoB = nocc0B + nvir0B;

    outfile->Printf("\n");
    outfile->Printf("    ------------------\n");
    outfile->Printf("    %-6s %5s %5s\n", "Range", "A", "B");
    outfile->Printf("    ------------------\n");
    outfile->Printf("    %-6s %5d %5d\n", "nbf", nbf, nbf);
    outfile->Printf("    %-6s %5d %5d\n", "nmo", nmoA, nmoB);
    outfile->Printf("    %-6s %5d %5d\n", "nocc", nocc0A, nocc0B);
    outfile->Printf("    %-6s %5d %5d\n", "nvir", nvir0A, nvir0B);
    outfile->Printf("    %-6s %5d %5d\n", "nfocc", nfocc0A, nfocc0B);
    outfile->Printf("    %-6s %5d %5d\n", "naocc", naocc0A, naocc0B);
    outfile->Printf("    %-6s %5d %5d\n", "navir", nvir0A, nvir0B);
    outfile->Printf("    %-6s %5d %5d\n", "nfvir", 0, 0);
    outfile->Printf("    ------------------\n");
    outfile->Printf("\n");

    matrices_["Cfocc0A"] = std::make_shared<Matrix>("Cfocc0A", nbf, nfocc0A);
    matrices_["Caocc0A"] = std::make_shared<Matrix>("Caocc0A", nbf, naocc0A);
    matrices_["Cfocc0B"] = std::make_shared<Matrix>("Cfocc0B", nbf, nfocc0B);
    matrices_["Caocc0B"] = std::make_shared<Matrix>("Caocc0B", nbf, naocc0B);

    vectors_["eps_focc0A"] = std::make_shared<Vector>("eps_focc0A", nfocc0A);
    vectors_["eps_aocc0A"] = std::make_shared<Vector>("eps_aocc0A", naocc0A);
    vectors_["eps_focc0B"] = std::make_shared<Vector>("eps_focc0B", nfocc0B);
    vectors_["eps_aocc0B"] = std::make_shared<Vector>("eps_aocc0B", naocc0B);

    double** Cocc0Ap = matrices_["Cocc0A"]->pointer();
    double** Cocc0Bp = matrices_["Cocc0B"]->pointer();
    double** Cfocc0Ap = matrices_["Cfocc0A"]->pointer();
    double** Caocc0Ap = matrices_["Caocc0A"]->pointer();
    double** Cfocc0Bp = matrices_["Cfocc0B"]->pointer();
    double** Caocc0Bp = matrices_["Caocc0B"]->pointer();

    double* eps_occ0Ap = vectors_["eps_occ0A"]->pointer();
    double* eps_occ0Bp = vectors_["eps_occ0B"]->pointer();
    double* eps_focc0Ap = vectors_["eps_focc0A"]->pointer();
    double* eps_aocc0Ap = vectors_["eps_aocc0A"]->pointer();
    double* eps_focc0Bp = vectors_["eps_focc0B"]->pointer();
    double* eps_aocc0Bp = vectors_["eps_aocc0B"]->pointer();

    for (int m = 0; m < nbf; m++) {
        for (int a = 0; a < nfocc0A; a++) {
            Cfocc0Ap[m][a] = Cocc0Ap[m][a];
        }
        for (int a = 0; a < naocc0A; a++) {
            Caocc0Ap[m][a] = Cocc0Ap[m][a + nfocc0A];
        }
        for (int a = 0; a < nfocc0B; a++) {
            Cfocc0Bp[m][a] = Cocc0Bp[m][a];
        }
        for (int a = 0; a < naocc0B; a++) {
            Caocc0Bp[m][a] = Cocc0Bp[m][a + nfocc0B];
        }
    }

    for (int a = 0; a < nfocc0A; a++) {
        eps_focc0Ap[a] = eps_occ0Ap[a];
    }
    for (int a = 0; a < naocc0A; a++) {
        eps_aocc0Ap[a] = eps_occ0Ap[a + nfocc0A];
    }
    for (int a = 0; a < nfocc0B; a++) {
        eps_focc0Bp[a] = eps_occ0Bp[a];
    }
    for (int a = 0; a < naocc0B; a++) {
        eps_aocc0Bp[a] = eps_occ0Bp[a + nfocc0B];
    }

    // vectors_["eps_occ0A"]->print();
    // vectors_["eps_occ0B"]->print();
    // vectors_["eps_focc0A"]->print();
    // vectors_["eps_focc0B"]->print();
    // vectors_["eps_aocc0A"]->print();
    // vectors_["eps_aocc0B"]->print();
}


void FISAPT::unify() {
    outfile->Printf("  ==> Unification <==\n\n");

    std::shared_ptr<Matrix> Cocc_A = matrices_["Cocc0A"];
    std::shared_ptr<Matrix> Cocc_B = matrices_["Cocc0B"];
    std::shared_ptr<Matrix> Cocc_C = matrices_["LoccC"];

    std::shared_ptr<Matrix> D_A = linalg::doublet(Cocc_A, Cocc_A, false, true);
    std::shared_ptr<Matrix> D_B = linalg::doublet(Cocc_B, Cocc_B, false, true);
    std::shared_ptr<Matrix> D_C(D_A->clone());
    D_C->zero();
    if (Cocc_C->colspi()[0] > 0) {
        D_C = linalg::doublet(Cocc_C, Cocc_C, false, true);
    }

    matrices_["D_A"] = D_A;
    matrices_["D_B"] = D_B;
    matrices_["D_C"] = D_C;

    // Incorrect for this application: C is not frozen in these orbitals
    // std::shared_ptr<Matrix> P_A = linalg::doublet(matrices_["Cvir0A"], matrices_["Cvir0A"], false, true);
    // std::shared_ptr<Matrix> P_B = linalg::doublet(matrices_["Cvir0B"], matrices_["Cvir0B"], false, true);

    // PA and PB are used only to define the complement of DA and DB in the DCBS
    std::shared_ptr<Matrix> P_A =
        linalg::doublet(reference_->Ca_subset("AO", "ALL"), reference_->Ca_subset("AO", "ALL"), false, true);
    P_A->subtract(D_A);

    std::shared_ptr<Matrix> P_B =
        linalg::doublet(reference_->Ca_subset("AO", "ALL"), reference_->Ca_subset("AO", "ALL"), false, true);
    P_B->subtract(D_B);

    matrices_["P_A"] = P_A;
    matrices_["P_B"] = P_B;

    matrices_["Cocc_A"] = matrices_["Cocc0A"];
    matrices_["Cocc_B"] = matrices_["Cocc0B"];
    matrices_["Cocc_C"] = matrices_["Cocc0C"];

    matrices_["V_A"] = matrices_["VA"];
    matrices_["V_B"] = matrices_["VB"];
    matrices_["V_C"] = matrices_["VC"];
    matrices_["J_A"] = matrices_["J0A"];
    matrices_["J_B"] = matrices_["J0B"];
    matrices_["J_C"] = matrices_["JC"];
    matrices_["K_A"] = matrices_["K0A"];
    matrices_["K_B"] = matrices_["K0B"];
    matrices_["K_C"] = matrices_["KC"];
}

// Compute deltaHF contribution (counted as induction)
void FISAPT::dHF() {
    outfile->Printf("  ==> dHF <==\n\n");

    // => Pointers <= //

    std::shared_ptr<Matrix> T = matrices_["T"];

    std::shared_ptr<Matrix> D_A = matrices_["D_A"];
    std::shared_ptr<Matrix> D_B = matrices_["D_B"];
    std::shared_ptr<Matrix> D_C = matrices_["D_C"];
    std::shared_ptr<Matrix> V_A = matrices_["V_A"];
    std::shared_ptr<Matrix> V_B = matrices_["V_B"];
    std::shared_ptr<Matrix> V_C = matrices_["V_C"];
    std::shared_ptr<Matrix> J_A = matrices_["J_A"];
    std::shared_ptr<Matrix> J_B = matrices_["J_B"];
    std::shared_ptr<Matrix> J_C = matrices_["J_C"];
    std::shared_ptr<Matrix> K_A = matrices_["K_A"];
    std::shared_ptr<Matrix> K_B = matrices_["K_B"];
    std::shared_ptr<Matrix> K_C = matrices_["K_C"];

    double** Enuc2p = matrices_["E NUC"]->pointer();

    // => Dimer HF (Already done) <= //

    double EABC = reference_->energy();

    // => Monomer AC Energy <= //

    double EAC = 0.0;
    EAC += Enuc2p[0][0];
    EAC += Enuc2p[2][2];
    EAC += Enuc2p[0][2];
    EAC += Enuc2p[2][0];

    std::shared_ptr<Matrix> H_AC(D_A->clone());
    H_AC->copy(T);
    H_AC->add(V_A);
    H_AC->add(V_C);
    if (reference_->external_pot()) H_AC->add(matrices_["VE"]);

    std::shared_ptr<Matrix> F_AC(D_A->clone());
    F_AC->copy(H_AC);
    F_AC->add(J_A);
    F_AC->add(J_A);
    F_AC->add(J_C);
    F_AC->add(J_C);
    F_AC->subtract(K_A);
    F_AC->subtract(K_C);

    std::shared_ptr<Matrix> D_AC(D_A->clone());
    D_AC->copy(D_A);
    D_AC->add(D_C);

    EAC += D_AC->vector_dot(H_AC) + D_AC->vector_dot(F_AC);

    H_AC.reset();
    F_AC.reset();
    D_AC.reset();

    // => Monomer BC Energy <= //

    double EBC = 0.0;
    EBC += Enuc2p[1][1];
    EBC += Enuc2p[2][2];
    EBC += Enuc2p[1][2];
    EBC += Enuc2p[2][1];

    std::shared_ptr<Matrix> H_BC(D_B->clone());
    H_BC->copy(T);
    H_BC->add(V_B);
    H_BC->add(V_C);
    if (reference_->external_pot()) H_BC->add(matrices_["VE"]);

    std::shared_ptr<Matrix> F_BC(D_B->clone());
    F_BC->copy(H_BC);
    F_BC->add(J_B);
    F_BC->add(J_B);
    F_BC->add(J_C);
    F_BC->add(J_C);
    F_BC->subtract(K_B);
    F_BC->subtract(K_C);

    std::shared_ptr<Matrix> D_BC(D_B->clone());
    D_BC->copy(D_B);
    D_BC->add(D_C);

    EBC += D_BC->vector_dot(H_BC) + D_BC->vector_dot(F_BC);

    H_BC.reset();
    F_BC.reset();
    D_BC.reset();

    // We also compute the energies of the isolated A and B
    // fragments, so that the orbital deformation energy is available

    // => Monomer A Energy <= //

    double EA = 0.0;
    EA += Enuc2p[0][0];

    std::shared_ptr<Matrix> H_A(D_A->clone());
    H_A->copy(T);
    H_A->add(V_A);
    if (reference_->external_pot()) H_A->add(matrices_["VE"]);

    std::shared_ptr<Matrix> F_A(D_A->clone());
    F_A->copy(H_A);
    F_A->add(J_A);
    F_A->add(J_A);
    F_A->subtract(K_A);

    EA += D_A->vector_dot(H_A) + D_A->vector_dot(F_A);

    // => Monomer B Energy <= //

    double EB = 0.0;
    EB += Enuc2p[1][1];

    std::shared_ptr<Matrix> H_B(D_B->clone());
    H_B->copy(T);
    H_B->add(V_B);
    if (reference_->external_pot()) H_B->add(matrices_["VE"]);

    std::shared_ptr<Matrix> F_B(D_B->clone());
    F_B->copy(H_B);
    F_B->add(J_B);
    F_B->add(J_B);
    F_B->subtract(K_B);

    EB += D_B->vector_dot(H_B) + D_B->vector_dot(F_B);

    // => Monomer C Energy <= //

    double EC = 0.0;
    EC += Enuc2p[2][2];

    std::shared_ptr<Matrix> H_C(D_C->clone());
    H_C->copy(T);
    H_C->add(V_C);
    if (reference_->external_pot()) H_C->add(matrices_["VE"]);

    std::shared_ptr<Matrix> F_C(D_C->clone());
    F_C->copy(H_C);
    F_C->add(J_C);
    F_C->add(J_C);
    F_C->subtract(K_C);

    EC += D_C->vector_dot(H_C) + D_C->vector_dot(F_C);

    // => Delta HF <= //

    double EHF = EABC - EAC - EBC + EC;

    // => Monomer and dimer energies in the original ABC full system <= //

    // Compute density from A HF localized orbitals
    std::shared_ptr<Matrix> LoccA = matrices_["LoccA"];
    std::shared_ptr<Matrix> LoccB = matrices_["LoccB"];
    std::shared_ptr<Matrix> LD_A = linalg::doublet(LoccA, LoccA, false, true);
    std::shared_ptr<Matrix> LD_B = linalg::doublet(LoccB, LoccB, false, true);

    // Get J and K from A and B HF localized orbitals while we are at it
    std::vector<SharedMatrix>& Cl = jk_->C_left();
    std::vector<SharedMatrix>& Cr = jk_->C_right();

    const std::vector<SharedMatrix>& J = jk_->J();
    const std::vector<SharedMatrix>& K = jk_->K();

    Cl.clear();
    Cr.clear();

    Cl.push_back(LoccA);
    Cr.push_back(LoccA);
    Cl.push_back(LoccB);
    Cr.push_back(LoccB);

    jk_->compute();

    std::shared_ptr<Matrix> LJ_A(J[0]->clone());
    std::shared_ptr<Matrix> LK_A(K[0]->clone());
    std::shared_ptr<Matrix> LJ_B(J[1]->clone());
    std::shared_ptr<Matrix> LK_B(K[1]->clone());

    // We have all the ingredients, now we build everything
    // Monomer A localised energy
    double LE_A = 0.0;
    LE_A += Enuc2p[0][0];
    std::shared_ptr<Matrix> LH_A(T->clone());
    LH_A->copy(T);
    LH_A->add(V_A);
    if (reference_->external_pot()) LH_A->add(matrices_["VE"]);
    std::shared_ptr<Matrix> LF_A(LH_A->clone());
    LF_A->copy(LH_A);
    LF_A->add(LJ_A);
    LF_A->add(LJ_A);
    LF_A->subtract(LK_A);
    LE_A += LD_A->vector_dot(LH_A) + LD_A->vector_dot(LF_A);
    LH_A.reset();
    LF_A.reset();

    // Monomer B localised energy
    double LE_B = 0.0;
    LE_B += Enuc2p[1][1];
    std::shared_ptr<Matrix> LH_B(T->clone());
    LH_B->copy(T);
    LH_B->add(V_B);
    if (reference_->external_pot()) LH_B->add(matrices_["VE"]);
    std::shared_ptr<Matrix> LF_B(LH_B->clone());
    LF_B->copy(LH_B);
    LF_B->add(LJ_B);
    LF_B->add(LJ_B);
    LF_B->subtract(LK_B);
    LE_B += LD_B->vector_dot(LH_B) + LD_B->vector_dot(LF_B);
    LH_B.reset();
    LF_B.reset();

    // Dimer AC localised energy
    double LE_AC = 0.0;
    LE_AC += Enuc2p[0][0];
    LE_AC += Enuc2p[0][2];
    LE_AC += Enuc2p[2][2];
    LE_AC += Enuc2p[2][0];

    std::shared_ptr<Matrix> LH_AC(T->clone());
    LH_AC->copy(T);
    LH_AC->add(V_A);
    LH_AC->add(V_C);
    if (reference_->external_pot()) LH_AC->add(matrices_["VE"]);
    std::shared_ptr<Matrix> LF_AC(LH_AC->clone());
    LF_AC->copy(LH_AC);
    LF_AC->add(J_C);
    LF_AC->add(J_C);
    LF_AC->add(LJ_A);
    LF_AC->add(LJ_A);
    LF_AC->subtract(LK_A);
    LF_AC->subtract(K_C);
    std::shared_ptr<Matrix> LD_AC(LD_A->clone());
    LD_AC->copy(LD_A);
    LD_AC->add(D_C);

    LE_AC += LD_AC->vector_dot(LH_AC) + LD_AC->vector_dot(LF_AC);
    LD_AC.reset();
    LH_AC.reset();
    LF_AC.reset();

    // Dimer BC localised energy
    double LE_BC = 0.0;
    LE_BC += Enuc2p[1][1];
    LE_BC += Enuc2p[1][2];
    LE_BC += Enuc2p[2][2];
    LE_BC += Enuc2p[2][1];

    std::shared_ptr<Matrix> LH_BC(T->clone());
    LH_BC->copy(T);
    LH_BC->add(V_B);
    LH_BC->add(V_C);
    if (reference_->external_pot()) LH_BC->add(matrices_["VE"]);
    std::shared_ptr<Matrix> LF_BC(LH_BC->clone());
    LF_BC->copy(LH_BC);
    LF_BC->add(J_C);
    LF_BC->add(J_C);
    LF_BC->add(LJ_B);
    LF_BC->add(LJ_B);
    LF_BC->subtract(LK_B);
    LF_BC->subtract(K_C);
    std::shared_ptr<Matrix> LD_BC(LD_B->clone());
    LD_BC->copy(LD_B);
    LD_BC->add(D_C);

    LE_BC += LD_BC->vector_dot(LH_BC) + LD_BC->vector_dot(LF_BC);
    LD_BC.reset();
    LH_BC.reset();
    LF_BC.reset();

    // Dimer AB localised energy
    double LE_BA = 0.0;
    LE_BA += Enuc2p[1][1];
    LE_BA += Enuc2p[1][0];
    LE_BA += Enuc2p[0][0];
    LE_BA += Enuc2p[0][1];

    std::shared_ptr<Matrix> LH_BA(T->clone());
    LH_BA->copy(T);
    LH_BA->add(V_B);
    LH_BA->add(V_A);
    if (reference_->external_pot()) LH_BA->add(matrices_["VE"]);
    std::shared_ptr<Matrix> LF_BA(LH_BA->clone());
    LF_BA->copy(LH_BA);
    LF_BA->add(LJ_A);
    LF_BA->add(LJ_A);
    LF_BA->add(LJ_B);
    LF_BA->add(LJ_B);
    LF_BA->subtract(LK_B);
    LF_BA->subtract(LK_A);
    std::shared_ptr<Matrix> LD_BA(LD_B->clone());
    LD_BA->copy(LD_B);
    LD_BA->add(LD_A);

    LE_BA += LD_BA->vector_dot(LH_BA) + LD_BA->vector_dot(LF_BA);
    LD_BA.reset();
    LH_BA.reset();
    LF_BA.reset();

    // => Print <= //

    outfile->Printf("    E ABC(HF) = %24.16E [Eh]\n", EABC);
    outfile->Printf("    E AC(0)   = %24.16E [Eh]\n", EAC);
    outfile->Printf("    E BC(0)   = %24.16E [Eh]\n", EBC);
    outfile->Printf("    E A(0)    = %24.16E [Eh]\n", EA);
    outfile->Printf("    E B(0)    = %24.16E [Eh]\n", EB);
    outfile->Printf("    E AC(HF)  = %24.16E [Eh]\n", LE_AC);
    outfile->Printf("    E BC(HF)  = %24.16E [Eh]\n", LE_BC);
    outfile->Printf("    E AB(HF)  = %24.16E [Eh]\n", LE_BA);
    outfile->Printf("    E A(HF)   = %24.16E [Eh]\n", LE_A);
    outfile->Printf("    E B(HF)   = %24.16E [Eh]\n", LE_B);
    outfile->Printf("    E C       = %24.16E [Eh]\n", EC);
    outfile->Printf("    E HF      = %24.16E [Eh]\n", EHF);
    outfile->Printf("\n");

    scalars_["HF"] = EHF;
    scalars_["E_A"] = EA;
    scalars_["E_B"] = EB;

    // Export all components of dHF as Psi4 variables
    scalars_["E_C"] = EC;
    scalars_["E_AC"] = EAC;
    scalars_["E_BC"] = EBC;
    scalars_["E_ABC_HF"] = EABC;
    scalars_["E_AC_HF"] = LE_AC;
    scalars_["E_BC_HF"] = LE_BC;
    scalars_["E_AB_HF"] = LE_BA;
    scalars_["E_A_HF"] = LE_A;
    scalars_["E_B_HF"] = LE_B;
}

// Compute total electrostatics contribution
void FISAPT::elst() {
    outfile->Printf("  ==> Electrostatics <==\n\n");

    std::shared_ptr<Matrix> D_A = matrices_["D_A"];
    std::shared_ptr<Matrix> D_B = matrices_["D_B"];
    std::shared_ptr<Matrix> V_A = matrices_["V_A"];
    std::shared_ptr<Matrix> V_B = matrices_["V_B"];
    std::shared_ptr<Matrix> J_A = matrices_["J_A"];
    std::shared_ptr<Matrix> J_B = matrices_["J_B"];

    double Enuc = 0.0;
    double** Enuc2p = matrices_["E NUC"]->pointer();
    Enuc += 2.0 * Enuc2p[0][1];  // A - B

    double Elst10 = 0.0;
    std::vector<double> Elst10_terms;
    Elst10_terms.resize(4);
    Elst10_terms[0] += 2.0 * D_A->vector_dot(V_B);
    Elst10_terms[1] += 2.0 * D_B->vector_dot(V_A);
    Elst10_terms[2] += 4.0 * D_A->vector_dot(J_B);
    Elst10_terms[3] += 1.0 * Enuc;
    for (int k = 0; k < Elst10_terms.size(); k++) {
        Elst10 += Elst10_terms[k];
    }
    // for (int k = 0; k < Elst10_terms.size(); k++) {
    //    outfile->Printf("    Elst10,r (%1d)        = %18.12lf [Eh]\n",k+1,Elst10_terms[k]);
    //}
    scalars_["Elst10,r"] = Elst10;
    outfile->Printf("    Elst10,r            = %18.12lf [Eh]\n", Elst10);
    outfile->Printf("\n");
    // fflush(outfile);
}

// Compute total exchange contribution
void FISAPT::exch() {
    outfile->Printf("  ==> Exchange <==\n\n");

    // => Density and Potential Matrices <= //

    std::shared_ptr<Matrix> S = matrices_["S"];

    std::shared_ptr<Matrix> D_A = matrices_["D_A"];
    std::shared_ptr<Matrix> D_B = matrices_["D_B"];
    std::shared_ptr<Matrix> P_A = matrices_["P_A"];
    std::shared_ptr<Matrix> P_B = matrices_["P_B"];
    std::shared_ptr<Matrix> V_A = matrices_["V_A"];
    std::shared_ptr<Matrix> V_B = matrices_["V_B"];
    std::shared_ptr<Matrix> J_A = matrices_["J_A"];
    std::shared_ptr<Matrix> J_B = matrices_["J_B"];
    std::shared_ptr<Matrix> K_A = matrices_["K_A"];
    std::shared_ptr<Matrix> K_B = matrices_["K_B"];

    std::shared_ptr<Matrix> Cocc_A = matrices_["Cocc_A"];
    std::shared_ptr<Matrix> Cocc_B = matrices_["Cocc_B"];

    // ==> Exchange Terms (S^2, MCBS or DCBS) <== //

    std::shared_ptr<Matrix> C_O = linalg::triplet(D_B, S, Cocc_A);
    std::vector<SharedMatrix>& Cl = jk_->C_left();
    std::vector<SharedMatrix>& Cr = jk_->C_right();
    const std::vector<SharedMatrix>& J = jk_->J();
    const std::vector<SharedMatrix>& K = jk_->K();
    Cl.clear();
    Cr.clear();
    Cl.push_back(Cocc_A);
    Cr.push_back(C_O);
    jk_->compute();
    std::shared_ptr<Matrix> K_O = K[0];

    double Exch10_2M = 0.0;
    std::vector<double> Exch10_2M_terms;
    Exch10_2M_terms.resize(6);
    Exch10_2M_terms[0] -= 2.0 * D_A->vector_dot(K_B);
    Exch10_2M_terms[1] -= 2.0 * linalg::triplet(D_A, S, D_B)->vector_dot(V_A);
    Exch10_2M_terms[1] -= 4.0 * linalg::triplet(D_A, S, D_B)->vector_dot(J_A);
    Exch10_2M_terms[1] += 2.0 * linalg::triplet(D_A, S, D_B)->vector_dot(K_A);
    Exch10_2M_terms[2] -= 2.0 * linalg::triplet(D_B, S, D_A)->vector_dot(V_B);
    Exch10_2M_terms[2] -= 4.0 * linalg::triplet(D_B, S, D_A)->vector_dot(J_B);
    Exch10_2M_terms[2] += 2.0 * linalg::triplet(D_B, S, D_A)->vector_dot(K_B);
    Exch10_2M_terms[3] += 2.0 * linalg::triplet(linalg::triplet(D_B, S, D_A), S, D_B)->vector_dot(V_A);
    Exch10_2M_terms[3] += 4.0 * linalg::triplet(linalg::triplet(D_B, S, D_A), S, D_B)->vector_dot(J_A);
    Exch10_2M_terms[4] += 2.0 * linalg::triplet(linalg::triplet(D_A, S, D_B), S, D_A)->vector_dot(V_B);
    Exch10_2M_terms[4] += 4.0 * linalg::triplet(linalg::triplet(D_A, S, D_B), S, D_A)->vector_dot(J_B);
    Exch10_2M_terms[5] -= 2.0 * linalg::triplet(D_A, S, D_B)->vector_dot(K_O);
    for (int k = 0; k < Exch10_2M_terms.size(); k++) {
        Exch10_2M += Exch10_2M_terms[k];
    }
    // for (int k = 0; k < Exch10_2M_terms.size(); k++) {
    //    outfile->Printf("    Exch10(S^2) (%1d)     = %18.12lf [Eh]\n",k+1,Exch10_2M_terms[k]);
    //}
    // scalars_["Exch10(S^2)"] = Exch10_2;
    // outfile->Printf("    Exch10(S^2) [MCBS]  = %18.12lf [Eh]\n",Exch10_2M);
    // outfile->Printf("    Exch10(S^2)         = %18.12lf [Eh]\n",Exch10_2M);
    // fflush(outfile);

    // ==> Exchange Terms (S^2, DCBS only) <== //

    // => K_AS <= //

    std::shared_ptr<Matrix> C_AS = linalg::triplet(P_B, S, Cocc_A);
    Cl.clear();
    Cr.clear();
    Cl.push_back(Cocc_A);
    Cr.push_back(C_AS);
    jk_->compute();
    std::shared_ptr<Matrix> K_AS = K[0];

    // => Accumulation <= //

    double Exch10_2 = 0.0;
    std::vector<double> Exch10_2_terms;
    Exch10_2_terms.resize(3);
    Exch10_2_terms[0] -= 2.0 * linalg::triplet(linalg::triplet(D_A, S, D_B), S, P_A)->vector_dot(V_B);
    Exch10_2_terms[0] -= 4.0 * linalg::triplet(linalg::triplet(D_A, S, D_B), S, P_A)->vector_dot(J_B);
    Exch10_2_terms[1] -= 2.0 * linalg::triplet(linalg::triplet(D_B, S, D_A), S, P_B)->vector_dot(V_A);
    Exch10_2_terms[1] -= 4.0 * linalg::triplet(linalg::triplet(D_B, S, D_A), S, P_B)->vector_dot(J_A);
    Exch10_2_terms[2] -= 2.0 * linalg::triplet(P_A, S, D_B)->vector_dot(K_AS);
    for (int k = 0; k < Exch10_2_terms.size(); k++) {
        Exch10_2 += Exch10_2_terms[k];
    }
    // for (int k = 0; k < Exch10_2_terms.size(); k++) {
    //    outfile->Printf("    Exch10(S^2) (%1d)     = %18.12lf [Eh]\n",k+1,Exch10_2_terms[k]);
    //}
    scalars_["Exch10(S^2)"] = Exch10_2;
    // outfile->Printf("    Exch10(S^2) [DCBS]  = %18.12lf [Eh]\n",Exch10_2);
    outfile->Printf("    Exch10(S^2)         = %18.12lf [Eh]\n", Exch10_2);
    // fflush(outfile);

    // ==> Exchange Terms (S^\infty, MCBS or DCBS) <== //

    // => T Matrix <= //

    int na = matrices_["Cocc0A"]->colspi()[0];
    int nb = matrices_["Cocc0B"]->colspi()[0];
    int nbf = matrices_["Cocc0A"]->rowspi()[0];

    std::shared_ptr<Matrix> Sab = linalg::triplet(matrices_["Cocc0A"], S, matrices_["Cocc0B"], true, false, false);
    double** Sabp = Sab->pointer();
    auto T = std::make_shared<Matrix>("T", na + nb, na + nb);
    T->identity();
    double** Tp = T->pointer();
    for (int a = 0; a < na; a++) {
        for (int b = 0; b < nb; b++) {
            Tp[a][b + na] = Tp[b + na][a] = Sabp[a][b];
        }
    }
    // T->print();
    T->power(-1.0, 1.0E-12);
    Tp = T->pointer();
    for (int a = 0; a < na + nb; a++) {
        Tp[a][a] -= 1.0;
    }
    // T->print();

    auto C_T_A_n = std::make_shared<Matrix>("C_T_A_n", nbf, na);
    auto C_T_B_n = std::make_shared<Matrix>("C_T_A_n", nbf, nb);
    auto C_T_BA_n = std::make_shared<Matrix>("C_T_BA_n", nbf, nb);
    auto C_T_AB_n = std::make_shared<Matrix>("C_T_AB_n", nbf, na);

    C_DGEMM('N', 'N', nbf, na, na, 1.0, matrices_["Cocc0A"]->pointer()[0], na, &Tp[0][0], na + nb, 0.0,
            C_T_A_n->pointer()[0], na);
    C_DGEMM('N', 'N', nbf, nb, nb, 1.0, matrices_["Cocc0B"]->pointer()[0], nb, &Tp[na][na], na + nb, 0.0,
            C_T_B_n->pointer()[0], nb);
    C_DGEMM('N', 'N', nbf, nb, na, 1.0, matrices_["Cocc0A"]->pointer()[0], na, &Tp[0][na], na + nb, 0.0,
            C_T_BA_n->pointer()[0], nb);
    C_DGEMM('N', 'N', nbf, na, nb, 1.0, matrices_["Cocc0B"]->pointer()[0], nb, &Tp[na][0], na + nb, 0.0,
            C_T_AB_n->pointer()[0], na);

    // => K Terms <= //

    Cl.clear();
    Cr.clear();
    // J/K[T^A, S^\infty]
    Cl.push_back(matrices_["Cocc0A"]);
    Cr.push_back(C_T_A_n);
    // J/K[T^AB, S^\infty]
    Cl.push_back(matrices_["Cocc0A"]);
    Cr.push_back(C_T_AB_n);

    jk_->compute();

    std::shared_ptr<Matrix> J_T_A_n = J[0];
    std::shared_ptr<Matrix> K_T_A_n = K[0];
    std::shared_ptr<Matrix> J_T_AB_n = J[1];
    std::shared_ptr<Matrix> K_T_AB_n = K[1];

    std::shared_ptr<Matrix> T_A_n = linalg::doublet(matrices_["Cocc0A"], C_T_A_n, false, true);
    std::shared_ptr<Matrix> T_B_n = linalg::doublet(matrices_["Cocc0B"], C_T_B_n, false, true);
    std::shared_ptr<Matrix> T_BA_n = linalg::doublet(matrices_["Cocc0B"], C_T_BA_n, false, true);
    std::shared_ptr<Matrix> T_AB_n = linalg::doublet(matrices_["Cocc0A"], C_T_AB_n, false, true);

    double Exch10_n = 0.0;
    std::vector<double> Exch10_n_terms;
    Exch10_n_terms.resize(9);
    Exch10_n_terms[0] -= 2.0 * D_A->vector_dot(K_B);  // This needs to be the full D_A
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
    // for (int k = 0; k < Exch10_n_terms.size(); k++) {
    //    outfile->Printf("    Exch10 (%1d)          = %18.12lf [Eh]\n",k+1,Exch10_n_terms[k]);
    //}
    scalars_["Exch10"] = Exch10_n;
    // outfile->Printf("    Exch10      [MCBS]  = %18.12lf [Eh]\n",Exch10_n);
    outfile->Printf("    Exch10              = %18.12lf [Eh]\n", Exch10_n);
    outfile->Printf("\n");
    // fflush(outfile);

    if (options_.get_bool("SSAPT0_SCALE")) {
        sSAPT0_scale_ = scalars_["Exch10"] / scalars_["Exch10(S^2)"];
        sSAPT0_scale_ = pow(sSAPT0_scale_, 3.0);
        outfile->Printf("    Scaling F-SAPT Exch-Ind and Exch-Disp by %11.3E \n\n", sSAPT0_scale_);
    }
}

// Compute the total induction contribution
void FISAPT::ind() {
    outfile->Printf("  ==> Induction <==\n\n");

    // => Pointers <= //

    std::shared_ptr<Matrix> S = matrices_["S"];

    std::shared_ptr<Matrix> D_A = matrices_["D_A"];
    std::shared_ptr<Matrix> D_B = matrices_["D_B"];
    std::shared_ptr<Matrix> P_A = matrices_["P_A"];
    std::shared_ptr<Matrix> P_B = matrices_["P_B"];
    std::shared_ptr<Matrix> V_A = matrices_["V_A"];
    std::shared_ptr<Matrix> V_B = matrices_["V_B"];
    std::shared_ptr<Matrix> J_A = matrices_["J_A"];
    std::shared_ptr<Matrix> J_B = matrices_["J_B"];
    std::shared_ptr<Matrix> K_A = matrices_["K_A"];
    std::shared_ptr<Matrix> K_B = matrices_["K_B"];

    std::shared_ptr<Matrix> Cocc0A = matrices_["Cocc0A"];
    std::shared_ptr<Matrix> Cocc0B = matrices_["Cocc0B"];
    std::shared_ptr<Matrix> Cvir0A = matrices_["Cvir0A"];
    std::shared_ptr<Matrix> Cvir0B = matrices_["Cvir0B"];

    std::shared_ptr<Vector> eps_occ0A = vectors_["eps_occ0A"];
    std::shared_ptr<Vector> eps_occ0B = vectors_["eps_occ0B"];
    std::shared_ptr<Vector> eps_vir0A = vectors_["eps_vir0A"];
    std::shared_ptr<Vector> eps_vir0B = vectors_["eps_vir0B"];

    // => ExchInd perturbations <= //

    std::shared_ptr<Matrix> C_O_A = linalg::triplet(D_B, S, matrices_["Cocc_A"]);
    std::shared_ptr<Matrix> C_P_A = linalg::triplet(linalg::triplet(D_B, S, D_A), S, matrices_["Cocc_B"]);
    std::shared_ptr<Matrix> C_P_B = linalg::triplet(linalg::triplet(D_A, S, D_B), S, matrices_["Cocc_A"]);

    std::vector<SharedMatrix>& Cl = jk_->C_left();
    std::vector<SharedMatrix>& Cr = jk_->C_right();
    const std::vector<SharedMatrix>& J = jk_->J();
    const std::vector<SharedMatrix>& K = jk_->K();

    Cl.clear();
    Cr.clear();

    // J/K[O]
    Cl.push_back(matrices_["Cocc_A"]);
    Cr.push_back(C_O_A);
    // J/K[P_B]
    Cl.push_back(matrices_["Cocc_A"]);
    Cr.push_back(C_P_B);
    // J/K[P_B]
    Cl.push_back(matrices_["Cocc_B"]);
    Cr.push_back(C_P_A);

    // => Compute the JK matrices <= //

    jk_->compute();

    // => Unload the JK Object <= //

    std::shared_ptr<Matrix> J_O = J[0];
    std::shared_ptr<Matrix> J_P_B = J[1];
    std::shared_ptr<Matrix> J_P_A = J[2];

    std::shared_ptr<Matrix> K_O = K[0];
    std::shared_ptr<Matrix> K_P_B = K[1];
    std::shared_ptr<Matrix> K_P_A = K[2];

    // ==> Generalized ESP (Flat and Exchange) <== //

    std::map<std::string, std::shared_ptr<Matrix> > mapA;
    mapA["Cocc_A"] = Cocc0A;
    mapA["Cvir_A"] = Cvir0A;
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
    mapB["Cocc_A"] = Cocc0B;
    mapB["Cvir_A"] = Cvir0B;
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
        int na = eps_occ0A->dimpi()[0];
        int nb = eps_occ0B->dimpi()[0];
        int nr = eps_vir0A->dimpi()[0];
        int ns = eps_vir0B->dimpi()[0];

        double** xuAp = xuA->pointer();
        double** xuBp = xuB->pointer();
        double** wAp = wA->pointer();
        double** wBp = wB->pointer();
        double* eap = eps_occ0A->pointer();
        double* erp = eps_vir0A->pointer();
        double* ebp = eps_occ0B->pointer();
        double* esp = eps_vir0B->pointer();

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
    scalars_["Ind20,u (A<-B)"] = Ind20u_AB;
    scalars_["Ind20,u (B<-A)"] = Ind20u_BA;
    scalars_["Ind20,u"] = Ind20u;
    outfile->Printf("    Ind20,u (A<-B)      = %18.12lf [Eh]\n", Ind20u_AB);
    outfile->Printf("    Ind20,u (B<-A)      = %18.12lf [Eh]\n", Ind20u_BA);
    outfile->Printf("    Ind20,u             = %18.12lf [Eh]\n", Ind20u);
    // fflush(outfile);

    // => Exchange-Induction <= //

    double ExchInd20u_AB = 2.0 * xuA->vector_dot(uB);
    double ExchInd20u_BA = 2.0 * xuB->vector_dot(uA);
    double ExchInd20u = ExchInd20u_AB + ExchInd20u_BA;
    outfile->Printf("    Exch-Ind20,u (A<-B) = %18.12lf [Eh]\n", ExchInd20u_AB);
    outfile->Printf("    Exch-Ind20,u (B<-A) = %18.12lf [Eh]\n", ExchInd20u_BA);
    outfile->Printf("    Exch-Ind20,u        = %18.12lf [Eh]\n", ExchInd20u);
    outfile->Printf("\n");
    // fflush(outfile);
    if (options_.get_bool("SSAPT0_SCALE")) {
        double scale = sSAPT0_scale_;
        double sExchInd20u_AB = 2.0 * scale * xuA->vector_dot(uB);
        double sExchInd20u_BA = 2.0 * scale * xuB->vector_dot(uA);
        double sExchInd20u = sExchInd20u_AB + sExchInd20u_BA;
        outfile->Printf("    sExch-Ind20,u (A<-B) = %18.12lf [Eh]\n", sExchInd20u_AB);
        outfile->Printf("    sExch-Ind20,u (B<-A) = %18.12lf [Eh]\n", sExchInd20u_BA);
        outfile->Printf("    sExch-Ind20,u        = %18.12lf [Eh]\n", sExchInd20u);
        outfile->Printf("\n");
        scalars_["sExch-Ind20,u (A<-B)"] = sExchInd20u_AB;
        scalars_["sExch-Ind20,u (B<-A)"] = sExchInd20u_BA;
        scalars_["sExch-Ind20,u"] = sExchInd20u_AB + sExchInd20u_BA;
    }

    scalars_["Exch-Ind20,u (A<-B)"] = ExchInd20u_AB;
    scalars_["Exch-Ind20,u (B<-A)"] = ExchInd20u_BA;
    scalars_["Exch-Ind20,u"] = ExchInd20u_AB + ExchInd20u_BA;

    // => Coupled Induction <= //

    auto cphf = std::make_shared<CPHF_FISAPT>();

    // Effective constructor
    cphf->delta_ = options_.get_double("D_CONVERGENCE");
    cphf->maxiter_ = options_.get_int("MAXITER");
    cphf->jk_ = jk_;

    cphf->w_A_ = wB;  // Reversal of convention
    cphf->Cocc_A_ = Cocc0A;
    cphf->Cvir_A_ = Cvir0A;
    cphf->eps_occ_A_ = eps_occ0A;
    cphf->eps_vir_A_ = eps_vir0A;

    cphf->w_B_ = wA;  // Reversal of convention
    cphf->Cocc_B_ = Cocc0B;
    cphf->Cvir_B_ = Cvir0B;
    cphf->eps_occ_B_ = eps_occ0B;
    cphf->eps_vir_B_ = eps_vir0B;

    // Gogo CPKS
    cphf->compute_cphf();

    std::shared_ptr<Matrix> xA = cphf->x_A_;
    std::shared_ptr<Matrix> xB = cphf->x_B_;

    // Backward in Ed's convention
    xA->scale(-1.0);
    xB->scale(-1.0);

    // => Induction <= //

    double Ind20r_AB = 2.0 * xA->vector_dot(wB);
    double Ind20r_BA = 2.0 * xB->vector_dot(wA);
    double Ind20r = Ind20r_AB + Ind20r_BA;
    scalars_["Ind20,r (A<-B)"] = Ind20r_AB;
    scalars_["Ind20,r (B<-A)"] = Ind20r_BA;
    scalars_["Ind20,r"] = Ind20r;
    outfile->Printf("    Ind20,r (A<-B)      = %18.12lf [Eh]\n", Ind20r_AB);
    outfile->Printf("    Ind20,r (B<-A)      = %18.12lf [Eh]\n", Ind20r_BA);
    outfile->Printf("    Ind20,r             = %18.12lf [Eh]\n", Ind20r);
    // fflush(outfile);

    // => Exchange-Induction <= //

    double ExchInd20r_AB = 2.0 * xA->vector_dot(uB);
    double ExchInd20r_BA = 2.0 * xB->vector_dot(uA);
    double ExchInd20r = ExchInd20r_AB + ExchInd20r_BA;
    outfile->Printf("    Exch-Ind20,r (A<-B) = %18.12lf [Eh]\n", ExchInd20r_AB);
    outfile->Printf("    Exch-Ind20,r (B<-A) = %18.12lf [Eh]\n", ExchInd20r_BA);
    outfile->Printf("    Exch-Ind20,r        = %18.12lf [Eh]\n", ExchInd20r);
    outfile->Printf("\n");
    // fflush(outfile);

    scalars_["Exch-Ind20,r (A<-B)"] = ExchInd20r_AB;
    scalars_["Exch-Ind20,r (B<-A)"] = ExchInd20r_BA;
    scalars_["Exch-Ind20,r"] = ExchInd20r_AB + ExchInd20r_BA;

    if (options_.get_bool("SSAPT0_SCALE")) {
        double scale = sSAPT0_scale_;
        double sExchInd20r_AB = scale * ExchInd20r_AB;
        double sExchInd20r_BA = scale * ExchInd20r_BA;
        double sExchInd20r = sExchInd20r_AB + sExchInd20r_BA;
        outfile->Printf("    sExch-Ind20,r (A<-B) = %18.12lf [Eh]\n", sExchInd20r_AB);
        outfile->Printf("    sExch-Ind20,r (B<-A) = %18.12lf [Eh]\n", sExchInd20r_BA);
        outfile->Printf("    sExch-Ind20,r        = %18.12lf [Eh]\n", sExchInd20r);
        outfile->Printf("\n");
        scalars_["sExch-Ind20,r (A<-B)"] = sExchInd20r_AB;
        scalars_["sExch-Ind20,r (B<-A)"] = sExchInd20r_BA;
        scalars_["sExch-Ind20,r"] = sExchInd20r_AB + sExchInd20r_BA;
    }

    scalars_["delta HF,r (2)"] = 0.0;
    if (scalars_["HF"] != 0.0) {
        scalars_["delta HF,r (2)"] =
            scalars_["HF"] - scalars_["Elst10,r"] - scalars_["Exch10"] - scalars_["Ind20,r"] - scalars_["Exch-Ind20,r"];
    }

    // => Stash for ExchDisp <= //

    matrices_["J_O"] = J_O;
    matrices_["K_O"] = K_O;
    matrices_["J_P_A"] = J_P_A;
    matrices_["J_P_B"] = J_P_B;

    // => Kill the JK Object <= //

    jk_.reset();
}

// build potential for induction contribution
std::shared_ptr<Matrix> FISAPT::build_ind_pot(std::map<std::string, std::shared_ptr<Matrix> >& vars) {
    std::shared_ptr<Matrix> Ca = vars["Cocc_A"];
    std::shared_ptr<Matrix> Cr = vars["Cvir_A"];
    std::shared_ptr<Matrix> V_B = vars["V_B"];
    std::shared_ptr<Matrix> J_B = vars["J_B"];

    std::shared_ptr<Matrix> W(V_B->clone());
    W->copy(J_B);
    W->scale(2.0);
    W->add(V_B);

    return linalg::triplet(Ca, W, Cr, true, false, false);
}

// build exchange-induction potential
std::shared_ptr<Matrix> FISAPT::build_exch_ind_pot(std::map<std::string, std::shared_ptr<Matrix> >& vars) {
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

    std::shared_ptr<Matrix> J_O = vars["J_O"];  // J[D^A S D^B]
    std::shared_ptr<Matrix> K_O = vars["K_O"];  // K[D^A S D^B]
    std::shared_ptr<Matrix> J_P = vars["J_P"];  // J[D^B S D^A S D^B]

    std::shared_ptr<Matrix> W(K_B->clone());
    std::shared_ptr<Matrix> T;

    // 1
    W->copy(K_B);
    W->scale(-1.0);

    // 2
    T = linalg::triplet(S, D_B, J_A);
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
    T = linalg::triplet(S, D_B, K_A);
    T->scale(1.0);
    W->add(T);

    // 6
    T = linalg::triplet(J_B, D_B, S);
    T->scale(-2.0);
    W->add(T);

    // 7
    T = linalg::triplet(K_B, D_B, S);
    T->scale(1.0);
    W->add(T);

    // 8
    T = linalg::triplet(linalg::triplet(S, D_B, J_A), D_B, S);
    T->scale(2.0);
    W->add(T);

    // 9
    T = linalg::triplet(linalg::triplet(J_B, D_A, S), D_B, S);
    T->scale(2.0);
    W->add(T);

    // 10
    T = linalg::triplet(K_O, D_B, S);
    T->scale(-1.0);
    W->add(T);

    // 11
    T->copy(J_P);
    T->scale(2.0);
    W->add(T);

    // 12
    T = linalg::triplet(linalg::triplet(S, D_B, S), D_A, J_B);
    T->scale(2.0);
    W->add(T);

    // 13
    T = linalg::triplet(S, D_B, K_O, false, false, true);
    T->scale(-1.0);
    W->add(T);

    // 14
    T = linalg::triplet(S, D_B, V_A);
    T->scale(-1.0);
    W->add(T);

    // 15
    T = linalg::triplet(V_B, D_B, S);
    T->scale(-1.0);
    W->add(T);

    // 16
    T = linalg::triplet(linalg::triplet(S, D_B, V_A), D_B, S);
    T->scale(1.0);
    W->add(T);

    // 17
    T = linalg::triplet(linalg::triplet(V_B, D_A, S), D_B, S);
    T->scale(1.0);
    W->add(T);

    // 18
    T = linalg::triplet(linalg::triplet(S, D_B, S), D_A, V_B);
    T->scale(1.0);
    W->add(T);

    return linalg::triplet(Ca, W, Cr, true, false, false);
}

// Compute total dispersion contribution
void FISAPT::disp(std::map<std::string, SharedMatrix> matrix_cache, std::map<std::string, SharedVector> vector_cache,
                  bool do_print) {
    if (do_print) {
        outfile->Printf("  ==> Dispersion <==\n\n");
    }

    // => Auxiliary Basis Set <= //
    std::shared_ptr<BasisSet> auxiliary = reference_->get_basisset("DF_BASIS_SAPT");

    // => Pointers <= //

    std::shared_ptr<Matrix> Cocc0A = matrix_cache["Caocc0A"];
    std::shared_ptr<Matrix> Cocc0B = matrix_cache["Caocc0B"];
    std::shared_ptr<Matrix> Cvir0A = matrix_cache["Cvir0A"];
    std::shared_ptr<Matrix> Cvir0B = matrix_cache["Cvir0B"];

    std::shared_ptr<Vector> eps_occ0A = vector_cache["eps_aocc0A"];
    std::shared_ptr<Vector> eps_occ0B = vector_cache["eps_aocc0B"];
    std::shared_ptr<Vector> eps_vir0A = vector_cache["eps_vir0A"];
    std::shared_ptr<Vector> eps_vir0B = vector_cache["eps_vir0B"];

    // => Sizing <= //

    int nn = primary_->nbf();

    int na = Cocc0A->colspi()[0];
    int nb = Cocc0B->colspi()[0];
    int nr = Cvir0A->colspi()[0];
    int ns = Cvir0B->colspi()[0];
    int nQ = auxiliary->nbf();
    size_t nrQ = nr * (size_t)nQ;
    size_t nsQ = ns * (size_t)nQ;

    int nT = 1;
#ifdef _OPENMP
    nT = Process::environment.get_n_threads();
#endif

    // => Stashed Variables <= //

    std::shared_ptr<Matrix> S = matrix_cache["S"];
    std::shared_ptr<Matrix> D_A = matrix_cache["D_A"];
    std::shared_ptr<Matrix> P_A = matrix_cache["P_A"];
    std::shared_ptr<Matrix> V_A = matrix_cache["V_A"];
    std::shared_ptr<Matrix> J_A = matrix_cache["J_A"];
    std::shared_ptr<Matrix> K_A = matrix_cache["K_A"];
    std::shared_ptr<Matrix> D_B = matrix_cache["D_B"];
    std::shared_ptr<Matrix> P_B = matrix_cache["P_B"];
    std::shared_ptr<Matrix> V_B = matrix_cache["V_B"];
    std::shared_ptr<Matrix> J_B = matrix_cache["J_B"];
    std::shared_ptr<Matrix> K_B = matrix_cache["K_B"];
    std::shared_ptr<Matrix> K_O = matrix_cache["K_O"];

    // => Auxiliary C matrices <= //

    std::shared_ptr<Matrix> Cr1 = linalg::triplet(D_B, S, Cvir0A);
    Cr1->scale(-1.0);
    Cr1->add(Cvir0A);
    std::shared_ptr<Matrix> Cs1 = linalg::triplet(D_A, S, Cvir0B);
    Cs1->scale(-1.0);
    Cs1->add(Cvir0B);
    std::shared_ptr<Matrix> Ca2 = linalg::triplet(D_B, S, Cocc0A);
    std::shared_ptr<Matrix> Cb2 = linalg::triplet(D_A, S, Cocc0B);
    std::shared_ptr<Matrix> Cr3 = linalg::triplet(D_B, S, Cvir0A);
    std::shared_ptr<Matrix> CrX = linalg::triplet(linalg::triplet(D_A, S, D_B), S, Cvir0A);
    Cr3->subtract(CrX);
    Cr3->scale(2.0);
    std::shared_ptr<Matrix> Cs3 = linalg::triplet(D_A, S, Cvir0B);
    std::shared_ptr<Matrix> CsX = linalg::triplet(linalg::triplet(D_B, S, D_A), S, Cvir0B);
    Cs3->subtract(CsX);
    Cs3->scale(2.0);
    std::shared_ptr<Matrix> Ca4 = linalg::triplet(linalg::triplet(D_A, S, D_B), S, Cocc0A);
    Ca4->scale(-2.0);
    std::shared_ptr<Matrix> Cb4 = linalg::triplet(linalg::triplet(D_B, S, D_A), S, Cocc0B);
    Cb4->scale(-2.0);

    // => Auxiliary V matrices <= //

    std::shared_ptr<Matrix> Jbr = linalg::triplet(Cocc0B, J_A, Cvir0A, true, false, false);
    Jbr->scale(2.0);
    std::shared_ptr<Matrix> Kbr = linalg::triplet(Cocc0B, K_A, Cvir0A, true, false, false);
    Kbr->scale(-1.0);

    std::shared_ptr<Matrix> Jas = linalg::triplet(Cocc0A, J_B, Cvir0B, true, false, false);
    Jas->scale(2.0);
    std::shared_ptr<Matrix> Kas = linalg::triplet(Cocc0A, K_B, Cvir0B, true, false, false);
    Kas->scale(-1.0);

    std::shared_ptr<Matrix> KOas = linalg::triplet(Cocc0A, K_O, Cvir0B, true, false, false);
    KOas->scale(1.0);
    std::shared_ptr<Matrix> KObr = linalg::triplet(Cocc0B, K_O, Cvir0A, true, true, false);
    KObr->scale(1.0);

    std::shared_ptr<Matrix> JBas = linalg::triplet(linalg::triplet(Cocc0A, S, D_B, true, false, false), J_A, Cvir0B);
    JBas->scale(-2.0);
    std::shared_ptr<Matrix> JAbr = linalg::triplet(linalg::triplet(Cocc0B, S, D_A, true, false, false), J_B, Cvir0A);
    JAbr->scale(-2.0);

    std::shared_ptr<Matrix> Jbs = linalg::triplet(Cocc0B, J_A, Cvir0B, true, false, false);
    Jbs->scale(4.0);
    std::shared_ptr<Matrix> Jar = linalg::triplet(Cocc0A, J_B, Cvir0A, true, false, false);
    Jar->scale(4.0);

    std::shared_ptr<Matrix> JAas = linalg::triplet(linalg::triplet(Cocc0A, J_B, D_A, true, false, false), S, Cvir0B);
    JAas->scale(-2.0);
    std::shared_ptr<Matrix> JBbr = linalg::triplet(linalg::triplet(Cocc0B, J_A, D_B, true, false, false), S, Cvir0A);
    JBbr->scale(-2.0);

    // Get your signs right Hesselmann!
    std::shared_ptr<Matrix> Vbs = linalg::triplet(Cocc0B, V_A, Cvir0B, true, false, false);
    Vbs->scale(2.0);
    std::shared_ptr<Matrix> Var = linalg::triplet(Cocc0A, V_B, Cvir0A, true, false, false);
    Var->scale(2.0);
    std::shared_ptr<Matrix> VBas = linalg::triplet(linalg::triplet(Cocc0A, S, D_B, true, false, false), V_A, Cvir0B);
    VBas->scale(-1.0);
    std::shared_ptr<Matrix> VAbr = linalg::triplet(linalg::triplet(Cocc0B, S, D_A, true, false, false), V_B, Cvir0A);
    VAbr->scale(-1.0);
    std::shared_ptr<Matrix> VRas = linalg::triplet(linalg::triplet(Cocc0A, V_B, P_A, true, false, false), S, Cvir0B);
    VRas->scale(1.0);
    std::shared_ptr<Matrix> VSbr = linalg::triplet(linalg::triplet(Cocc0B, V_A, P_B, true, false, false), S, Cvir0A);
    VSbr->scale(1.0);

    std::shared_ptr<Matrix> Sas = linalg::triplet(Cocc0A, S, Cvir0B, true, false, false);
    std::shared_ptr<Matrix> Sbr = linalg::triplet(Cocc0B, S, Cvir0A, true, false, false);

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

    std::shared_ptr<Matrix> SBar = linalg::triplet(linalg::triplet(Cocc0A, S, D_B, true, false, false), S, Cvir0A);
    std::shared_ptr<Matrix> SAbs = linalg::triplet(linalg::triplet(Cocc0B, S, D_A, true, false, false), S, Cvir0B);

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

    // => Memory <= //

    std::vector<std::shared_ptr<Matrix> > Cs;
    Cs.push_back(Cocc0A);
    Cs.push_back(Cvir0A);
    Cs.push_back(Cocc0B);
    Cs.push_back(Cvir0B);
    Cs.push_back(Cr1);
    Cs.push_back(Cs1);
    Cs.push_back(Ca2);
    Cs.push_back(Cb2);
    Cs.push_back(Cr3);
    Cs.push_back(Cs3);
    Cs.push_back(Ca4);
    Cs.push_back(Cb4);

    size_t max_MO = 0, ncol = 0;
    for (auto& mat : Cs) {
        max_MO = std::max(max_MO, (size_t)mat->ncol());
        ncol += (size_t)mat->ncol();
    }

    // => Get integrals from DFHelper <= //
    auto dfh(std::make_shared<DFHelper>(primary_, auxiliary));
    dfh->set_memory(doubles_ - Cs[0]->nrow() * ncol);
    dfh->set_method("DIRECT_iaQ");
    dfh->set_nthreads(nT);
    dfh->initialize();
    dfh->print_header();

    dfh->add_space("a", Cs[0]);
    dfh->add_space("r", Cs[1]);
    dfh->add_space("b", Cs[2]);
    dfh->add_space("s", Cs[3]);
    dfh->add_space("r1", Cs[4]);
    dfh->add_space("s1", Cs[5]);
    dfh->add_space("a2", Cs[6]);
    dfh->add_space("b2", Cs[7]);
    dfh->add_space("r3", Cs[8]);
    dfh->add_space("s3", Cs[9]);
    dfh->add_space("a4", Cs[10]);
    dfh->add_space("b4", Cs[11]);

    dfh->add_transformation("Aar", "a", "r");
    dfh->add_transformation("Abs", "b", "s");
    dfh->add_transformation("Bas", "a", "s1");
    dfh->add_transformation("Bbr", "b", "r1");
    dfh->add_transformation("Cas", "a2", "s");
    dfh->add_transformation("Cbr", "b2", "r");
    dfh->add_transformation("Dar", "a", "r3");
    dfh->add_transformation("Dbs", "b", "s3");
    dfh->add_transformation("Ear", "a4", "r");
    dfh->add_transformation("Ebs", "b4", "s");

    dfh->transform();

    Cr1.reset();
    Cs1.reset();
    Ca2.reset();
    Cb2.reset();
    Cr3.reset();
    Cs3.reset();
    Ca4.reset();
    Cb4.reset();
    Cs.clear();
    dfh->clear_spaces();

    // => Blocking ... figure out how big a tensor slice to handle at a time <= //

    long int overhead = 0L;
    overhead += 2L * nT * nr * ns; // Thread work arrays Trs and Vrs below
    overhead += 2L * na * ns + 2L * nb * nr + 2L * na * nr + 2L * nb * ns; // Sas, Sbr, sBar, sAbs, Qas, Qbr, Qar, Qbs
    // account for a few of the smaller matrices already defined, but not exhaustively
    overhead += 12L * nn * nn; // D, V, J, K, P, and C matrices for A and B (neglecting C)

    long int rem = doubles_ - overhead;

    outfile->Printf("    %ld doubles - %ld overhead leaves %ld for dispersion\n", doubles_, overhead, rem);

    if (rem < 0L) {
        throw PSIEXCEPTION("Too little static memory for DFTSAPT::mp2_terms");
    }

    long int cost_a = 2L * nr * nQ + 2L * ns * nQ; // how much mem for Aar, Bas, Cas, Dar for a single a
    // cost_b would be the same value, and would be how much mem for Abs, Bbr, Cbr, Dbs for a single b
    long int max_a_l = rem / (2L * cost_a);
    long int max_b_l = max_a_l;
    int max_a = (max_a_l > na ? na : (int) max_a_l);
    int max_b = (max_b_l > nb ? nb : (int) max_b_l);
    if (max_a < 1 || max_b < 1) {
        throw PSIEXCEPTION("Too little dynamic memory for DFTSAPT::mp2_terms");
    }
    int nablocks = (na / max_a);
    if (na % max_a) nablocks++;
    int nbblocks = (nb / max_b);
    if (nb % max_b) nbblocks++;
    outfile->Printf("    Processing a single (a,b) pair requires %ld doubles\n", cost_a * 2L);
    outfile->Printf("    %d values of a processed in %d blocks of %d\n", na, nablocks, max_a);
    outfile->Printf("    %d values of b processed in %d blocks of %d\n\n", nb, nbblocks, max_b);

    // => Tensor Slices <= //

    auto Aar = std::make_shared<Matrix>("Aar", max_a * nr, nQ);
    auto Abs = std::make_shared<Matrix>("Abs", max_b * ns, nQ);
    auto Bas = std::make_shared<Matrix>("Bas", max_a * ns, nQ);
    auto Bbr = std::make_shared<Matrix>("Bbr", max_b * nr, nQ);
    auto Cas = std::make_shared<Matrix>("Cas", max_a * ns, nQ);
    auto Cbr = std::make_shared<Matrix>("Cbr", max_b * nr, nQ);
    auto Dar = std::make_shared<Matrix>("Dar", max_a * nr, nQ);
    auto Dbs = std::make_shared<Matrix>("Dbs", max_b * ns, nQ);

    // => Thread Work Arrays <= //

    std::vector<std::shared_ptr<Matrix> > Trs;
    std::vector<std::shared_ptr<Matrix> > Vrs;
    for (int t = 0; t < nT; t++) {
        Trs.push_back(std::make_shared<Matrix>("Trs", nr, ns));
        Vrs.push_back(std::make_shared<Matrix>("Vrs", nr, ns));
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

    double* eap = eps_occ0A->pointer();
    double* ebp = eps_occ0B->pointer();
    double* erp = eps_vir0A->pointer();
    double* esp = eps_vir0B->pointer();

    // => Slice D + E -> D <= //

    dfh->add_disk_tensor("Far", std::make_tuple(na, nr, nQ));

    for (size_t astart = 0; astart < na; astart += max_a) {
        size_t nablock = (astart + max_a >= na ? na - astart : max_a);

        dfh->fill_tensor("Dar", Dar, {astart, astart + nablock});
        dfh->fill_tensor("Ear", Aar, {astart, astart + nablock});

        double* D2p = Darp[0];
        double* A2p = Aarp[0];
        for (long int arQ = 0L; arQ < nablock * nrQ; arQ++) {
            (*D2p++) += (*A2p++);
        }
        dfh->write_disk_tensor("Far", Dar, {astart, astart + nablock});
    }

    dfh->add_disk_tensor("Fbs", std::make_tuple(na, nr, nQ));

    for (size_t bstart = 0; bstart < nb; bstart += max_b) {
        size_t nbblock = (bstart + max_b >= nb ? nb - bstart : max_b);

        dfh->fill_tensor("Dbs", Dbs, {bstart, bstart + nbblock});
        dfh->fill_tensor("Ebs", Abs, {bstart, bstart + nbblock});

        double* D2p = Dbsp[0];
        double* A2p = Absp[0];
        for (long int bsQ = 0L; bsQ < nbblock * nsQ; bsQ++) {
            (*D2p++) += (*A2p++);
        }
        dfh->write_disk_tensor("Fbs", Dbs, {bstart, bstart + nbblock});
    }

    // => Targets <= //

    double Disp20 = 0.0;
    double ExchDisp20 = 0.0;

    // ==> Master Loop <== //

    for (size_t astart = 0; astart < na; astart += max_a) {
        size_t nablock = (astart + max_a >= na ? na - astart : max_a);

        dfh->fill_tensor("Aar", Aar, {astart, astart + nablock});
        dfh->fill_tensor("Bas", Bas, {astart, astart + nablock});
        dfh->fill_tensor("Cas", Cas, {astart, astart + nablock});
        dfh->fill_tensor("Far", Dar, {astart, astart + nablock});

        for (size_t bstart = 0; bstart < nb; bstart += max_b) {
            size_t nbblock = (bstart + max_b >= nb ? nb - bstart : max_b);

            dfh->fill_tensor("Abs", Abs, {bstart, bstart + nbblock});
            dfh->fill_tensor("Bbr", Bbr, {bstart, bstart + nbblock});
            dfh->fill_tensor("Cbr", Cbr, {bstart, bstart + nbblock});
            dfh->fill_tensor("Fbs", Dbs, {bstart, bstart + nbblock});

            long int nab = nablock * nbblock;

#pragma omp parallel for schedule(dynamic) reduction(+ : Disp20, ExchDisp20)
            for (long int ab = 0L; ab < nab; ab++) {
                int a = ab / nbblock;
                int b = ab % nbblock;

                int thread = 0;
#ifdef _OPENMP
                thread = omp_get_thread_num();
#endif

                double** Trsp = Trs[thread]->pointer();
                double** Vrsp = Vrs[thread]->pointer();

                // => Amplitudes, Disp20 <= //

                C_DGEMM('N', 'T', nr, ns, nQ, 1.0, Aarp[(a)*nr], nQ, Absp[(b)*ns], nQ, 0.0, Vrsp[0], ns);

                for (int r = 0; r < nr; r++) {
                    for (int s = 0; s < ns; s++) {
                        Trsp[r][s] = Vrsp[r][s] / (eap[a + astart] + ebp[b + bstart] - erp[r] - esp[s]);
                        Disp20 += 4.0 * Trsp[r][s] * Vrsp[r][s];
                    }
                }

                // => Exch-Disp20 <= //

                // > Q1-Q3 < //

                C_DGEMM('N', 'T', nr, ns, nQ, 1.0, Bbrp[(b)*nr], nQ, Basp[(a)*ns], nQ, 0.0, Vrsp[0], ns);
                C_DGEMM('N', 'T', nr, ns, nQ, 1.0, Cbrp[(b)*nr], nQ, Casp[(a)*ns], nQ, 1.0, Vrsp[0], ns);
                C_DGEMM('N', 'T', nr, ns, nQ, 1.0, Aarp[(a)*nr], nQ, Dbsp[(b)*ns], nQ, 1.0, Vrsp[0], ns);
                C_DGEMM('N', 'T', nr, ns, nQ, 1.0, Darp[(a)*nr], nQ, Absp[(b)*ns], nQ, 1.0, Vrsp[0], ns);

                // > V,J,K < //

                C_DGER(nr, ns, 1.0, Qbrp[b + bstart], 1, Sasp[a + astart], 1, Vrsp[0], ns);
                C_DGER(nr, ns, 1.0, Sbrp[b + bstart], 1, Qasp[a + astart], 1, Vrsp[0], ns);
                C_DGER(nr, ns, 1.0, Qarp[a + astart], 1, SAbsp[b + bstart], 1, Vrsp[0], ns);
                C_DGER(nr, ns, 1.0, SBarp[a + astart], 1, Qbsp[b + bstart], 1, Vrsp[0], ns);

                for (int r = 0; r < nr; r++) {
                    for (int s = 0; s < ns; s++) {
                        ExchDisp20 -= 2.0 * Trsp[r][s] * Vrsp[r][s];
                    }
                }
            }
        }
    }

    scalars_["Disp20"] = Disp20;
    scalars_["Exch-Disp20"] = ExchDisp20;
    if (do_print) {
        outfile->Printf("    Disp20              = %18.12lf [Eh]\n", Disp20);
        outfile->Printf("    Exch-Disp20         = %18.12lf [Eh]\n", ExchDisp20);
        outfile->Printf("\n");
    }
}

// Compute total dispersion energy and S-infinity version of total exchange-dispersion
void FISAPT::sinf_disp(std::map<std::string, SharedMatrix> matrix_cache, std::map<std::string, SharedVector> vector_cache,
                        bool do_print) {
    if (do_print) {
        outfile->Printf("  ==> Dispersion <==\n\n");
    }

    // => Pointers <= //

    std::shared_ptr<Matrix> Cocc0A = matrix_cache["Caocc0A"];
    std::shared_ptr<Matrix> Cocc0B = matrix_cache["Caocc0B"];
    std::shared_ptr<Matrix> Cvir0A = matrix_cache["Cvir0A"];
    std::shared_ptr<Matrix> Cvir0B = matrix_cache["Cvir0B"];

    std::shared_ptr<Vector> eps_occ0A = vector_cache["eps_aocc0A"];
    std::shared_ptr<Vector> eps_occ0B = vector_cache["eps_aocc0B"];
    std::shared_ptr<Vector> eps_vir0A = vector_cache["eps_vir0A"];
    std::shared_ptr<Vector> eps_vir0B = vector_cache["eps_vir0B"];

    // => Auxiliary Basis Set <= //
    std::shared_ptr<BasisSet> auxiliary = reference_->get_basisset("DF_BASIS_SAPT");

    // => Sizing <= //

    int nn = primary_->nbf();
    int na = Cocc0A->colspi()[0];
    int nb = Cocc0B->colspi()[0];
    int nr = Cvir0A->colspi()[0];
    int ns = Cvir0B->colspi()[0];
    int nQ = auxiliary->nbf();
    size_t nrQ = nr * (size_t)nQ;
    size_t nsQ = ns * (size_t)nQ;

    int nT = 1;
#ifdef _OPENMP
    nT = Process::environment.get_n_threads();
#endif

    // => Stashed Variables <= //

    std::shared_ptr<Matrix> S = matrix_cache["S"];
    std::shared_ptr<Matrix> V_A = matrix_cache["V_A"];
    std::shared_ptr<Matrix> V_B = matrix_cache["V_B"];

    // => Intermolelcular overlap matrix and inverse <= //
    std::shared_ptr<Matrix> Sab = linalg::triplet(Cocc0A, S, Cocc0B, true, false, false);
    double** Sabp = Sab->pointer();
    auto D = std::make_shared<Matrix>("D", na + nb, na + nb);
    D->identity();
    double** Dp = D->pointer();
    for (int a = 0; a < na; a++) {
        for (int b = 0; b < nb; b++) {
            Dp[a][b + na] = Dp[b + na][a] = Sabp[a][b];
        }
    }
    D->power(-1.0, 1.0E-12);
    Dp = D->pointer();

    // => New Stuff <= //
    // Start with T's
    std::shared_ptr<Matrix> Sbr = linalg::triplet(Cocc0B, S, Cvir0A, true, false, false);
    std::shared_ptr<Matrix> Sas = linalg::triplet(Cocc0A, S, Cvir0B, true, false, false);
    auto Tar = std::make_shared<Matrix>("Tar", na, nr);
    auto Tbr = std::make_shared<Matrix>("Tbr", nb, nr);
    auto Tas = std::make_shared<Matrix>("Tas", na, ns);
    auto Tbs = std::make_shared<Matrix>("Tbs", nb, ns);

    C_DGEMM('N', 'N', na, nr, nb, 1.0, &Dp[0][na], na + nb, Sbr->pointer()[0], nr, 0.0,
            Tar->pointer()[0], nr);
    C_DGEMM('N', 'N', nb, nr, nb, 1.0, &Dp[na][na], na + nb, Sbr->pointer()[0], nr, 0.0,
            Tbr->pointer()[0], nr);
    C_DGEMM('N', 'N', na, ns, na, 1.0, &Dp[0][0], na + nb, Sas->pointer()[0], ns, 0.0,
            Tas->pointer()[0], ns);
    C_DGEMM('N', 'N', nb, ns, na, 1.0, &Dp[na][0], na + nb, Sas->pointer()[0], ns, 0.0,
            Tbs->pointer()[0], ns);

    // C1's and C2's from D's and C's.
    // C1's are C times D diagonal blocks.
    // C2's are times off-diagonal blocks.
    auto C1a = std::make_shared<Matrix>("C1a", nn, na);
    auto C1b = std::make_shared<Matrix>("C1b", nn, nb);
    auto C2a = std::make_shared<Matrix>("C2a", nn, na);
    auto C2b = std::make_shared<Matrix>("C2b", nn, nb);

    C_DGEMM('N', 'N', nn, na, na, 1.0, Cocc0A->pointer()[0], na, &Dp[0][0], na + nb, 0.0,
            C1a->pointer()[0], na);
    C_DGEMM('N', 'N', nn, nb, nb, 1.0, Cocc0B->pointer()[0], nb, &Dp[na][na], na + nb, 0.0,
            C1b->pointer()[0], nb);
    C_DGEMM('N', 'N', nn, na, nb, 1.0, Cocc0B->pointer()[0], nb, &Dp[na][0], na + nb, 0.0,
            C2a->pointer()[0], na);
    C_DGEMM('N', 'N', nn, nb, na, 1.0, Cocc0A->pointer()[0], na, &Dp[0][na], na + nb, 0.0,
            C2b->pointer()[0], nb);

    // Coeffs for all occupied
    std::vector<std::shared_ptr<Matrix> > hold_these;
    hold_these.push_back(Cocc0A);
    hold_these.push_back(Cocc0B);

    auto Cocc0AB = linalg::horzcat(hold_these);
    hold_these.clear();

    // Half transform D_ia and D_ib for JK
    auto D_Ni_a = std::make_shared<Matrix>("D_Ni_a", nn, na + nb);
    auto D_Ni_b = std::make_shared<Matrix>("D_Ni_b", nn, na + nb);

    C_DGEMM('N', 'N', nn, na + nb, na, 1.0, Cocc0A->pointer()[0], na, &Dp[0][0], na + nb, 0.0,
            D_Ni_a->pointer()[0], na + nb);
    C_DGEMM('N', 'N', nn, na + nb, nb, 1.0, Cocc0B->pointer()[0], nb, &Dp[na][0], na + nb, 0.0,
            D_Ni_b->pointer()[0], na + nb);

    // Make JK's
    jk_ = JK::build_JK(primary_, reference_->get_basisset("DF_BASIS_SCF"), options_, false, doubles_);
    jk_->set_memory(doubles_);
    jk_->set_do_J(true);
    jk_->set_do_K(true);
    jk_->initialize();
    jk_->print_header();

    std::vector<SharedMatrix>& Cl = jk_->C_left();
    std::vector<SharedMatrix>& Cr = jk_->C_right();
    const std::vector<SharedMatrix>& J = jk_->J();
    const std::vector<SharedMatrix>& K = jk_->K();

    Cl.clear();
    Cr.clear();
    Cl.push_back(Cocc0AB);
    Cr.push_back(D_Ni_a);
    Cl.push_back(Cocc0AB);
    Cr.push_back(D_Ni_b);
    jk_->compute();

    std::shared_ptr<Matrix> J_D_ia = J[0];
    std::shared_ptr<Matrix> K_D_ia = K[0];

    std::shared_ptr<Matrix> J_D_ib = J[1];
    std::shared_ptr<Matrix> K_D_ib = K[1];

    // Finish D_ia and D_ib transformation to make tilded C's
    auto D_ia = linalg::doublet(Cocc0AB, D_Ni_a, false, true);
    auto D_ib = linalg::doublet(Cocc0AB, D_Ni_b, false, true);

    auto Ct_Kr = linalg::triplet(D_ib, S, Cvir0A, false, false, false);
    Ct_Kr->scale(-1);
    Ct_Kr->add(Cvir0A);
    auto Ct_Ks = linalg::triplet(D_ia, S, Cvir0B, false, false, false);
    Ct_Ks->scale(-1);
    Ct_Ks->add(Cvir0B);

    // Make omega cores
    std::shared_ptr<Matrix> AJK(J_D_ia->clone());
    AJK->zero();
    AJK->add(J_D_ia);
    AJK->scale(2);
    AJK->add(V_A);
    AJK->subtract(K_D_ia);

    auto AJK_ar = linalg::triplet(C2a, AJK, Ct_Kr, true, false, false);
    auto AJK_as = linalg::triplet(C2a, AJK, Ct_Ks, true, false, false);
    auto AJK_br = linalg::triplet(C1b, AJK, Ct_Kr, true, false, false);
    auto AJK_bs = linalg::triplet(C1b, AJK, Ct_Ks, true, false, false);

    std::shared_ptr<Matrix> BJK(J_D_ib->clone());
    BJK->zero();
    BJK->add(J_D_ib);
    BJK->scale(2);
    BJK->add(V_B);
    BJK->subtract(K_D_ib);

    auto BJK_ar = linalg::triplet(C1a, BJK, Ct_Kr, true, false, false);
    auto BJK_as = linalg::triplet(C1a, BJK, Ct_Ks, true, false, false);
    auto BJK_br = linalg::triplet(C2b, BJK, Ct_Kr, true, false, false);
    auto BJK_bs = linalg::triplet(C2b, BJK, Ct_Ks, true, false, false);

    // Finish omega terms
    std::shared_ptr<Matrix> omega_ar(AJK_ar->clone());
    omega_ar->zero();
    omega_ar->add(AJK_ar);
    omega_ar->add(BJK_ar);
    omega_ar->scale(4);

    std::shared_ptr<Matrix> omega_as(AJK_as->clone());
    omega_as->zero();
    omega_as->add(AJK_as);
    omega_as->add(BJK_as);
    omega_as->scale(2);

    std::shared_ptr<Matrix> omega_br(AJK_br->clone());
    omega_br->zero();
    omega_br->add(AJK_br);
    omega_br->add(BJK_br);
    omega_br->scale(2);

    std::shared_ptr<Matrix> omega_bs(AJK_bs->clone());
    omega_bs->zero();
    omega_bs->add(AJK_bs);
    omega_bs->add(BJK_bs);
    omega_bs->scale(4);

    D.reset();
    Sbr.reset();
    Sas.reset();
    Cocc0AB.reset();
    D_Ni_a.reset();
    D_Ni_b.reset();
    J_D_ia.reset();
    K_D_ia.reset();
    J_D_ib.reset();
    K_D_ib.reset();
    D_ia.reset();
    D_ib.reset();
    AJK.reset();
    BJK.reset();
    AJK_ar.reset();
    AJK_as.reset();
    AJK_br.reset();
    AJK_bs.reset();
    BJK_ar.reset();
    BJK_as.reset();
    BJK_br.reset();
    BJK_bs.reset();

    // => Memory <= //

    std::vector<std::shared_ptr<Matrix> > Cs;
    Cs.push_back(Cocc0A);
    Cs.push_back(Cvir0A);
    Cs.push_back(Cocc0B);
    Cs.push_back(Cvir0B);
    Cs.push_back(C1a);
    Cs.push_back(C1b);
    Cs.push_back(C2a);
    Cs.push_back(C2b);
    Cs.push_back(Ct_Kr);
    Cs.push_back(Ct_Ks);

    size_t max_MO = 0, ncol = 0;
    for (auto& mat : Cs) {
        max_MO = std::max(max_MO, (size_t)mat->ncol());
        ncol += (size_t)mat->ncol();
    }

    // => Get integrals from DFHelper <= //
    auto dfh(std::make_shared<DFHelper>(primary_, auxiliary));
    dfh->set_memory(doubles_ - Cs[0]->nrow() * ncol);
    dfh->set_method("DIRECT_iaQ");
    dfh->set_nthreads(nT);
    dfh->initialize();
    dfh->print_header();

    dfh->add_space("a", Cs[0]);
    dfh->add_space("r", Cs[1]);
    dfh->add_space("b", Cs[2]);
    dfh->add_space("s", Cs[3]);
    dfh->add_space("a1", Cs[4]);
    dfh->add_space("b1", Cs[5]);
    dfh->add_space("a2", Cs[6]);
    dfh->add_space("b2", Cs[7]);
    dfh->add_space("r1", Cs[8]);
    dfh->add_space("s1", Cs[9]);

    dfh->add_transformation("Aar", "a", "r");
    dfh->add_transformation("Abs", "b", "s");
    dfh->add_transformation("Bar", "a1", "r1");
    dfh->add_transformation("Bbs", "b1", "s1");
    dfh->add_transformation("Cbr", "b2", "r1");
    dfh->add_transformation("Cas", "a2", "s1");
    dfh->add_transformation("Das", "a1", "s1");
    dfh->add_transformation("Dbr", "b1", "r1");
    dfh->add_transformation("Ebs", "b2", "s1");
    dfh->add_transformation("Ear", "a2", "r1");

    dfh->transform();

    C1a.reset();
    C1b.reset();
    C2a.reset();
    C2b.reset();
    Ct_Kr.reset();
    Ct_Ks.reset();
    Cs.clear();
    dfh->clear_spaces();

    // => Blocking <= //

    long int overhead = 0L;
    overhead += 2L * nT * nr * ns;
    overhead += 2L * na * ns + 2L * nb * nr + 2L * na * nr + 2L * nb * ns;
    long int rem = doubles_ - overhead;

    if (rem < 0L) {
        throw PSIEXCEPTION("Too little static memory for DFTSAPT::mp2_terms");
    }

    long int cost_a = 2L * nr * nQ + 2L * ns * nQ;
    long int max_a = rem / (2L * cost_a);
    long int max_b = max_a;
    max_a = (max_a > na ? na : max_a);
    max_b = (max_b > nb ? nb : max_b);
    if (max_a < 1L || max_b < 1L) {
        throw PSIEXCEPTION("Too little dynamic memory for DFTSAPT::mp2_terms");
    }

    // => Tensor Slices <= //

    auto Aar = std::make_shared<Matrix>("Aar", max_a * nr, nQ);
    auto Abs = std::make_shared<Matrix>("Abs", max_b * ns, nQ);
    auto Bar = std::make_shared<Matrix>("Bar", max_a * nr, nQ);
    auto Bbs = std::make_shared<Matrix>("Bbs", max_b * ns, nQ);
    auto Cbr = std::make_shared<Matrix>("Cbr", max_b * nr, nQ);
    auto Cas = std::make_shared<Matrix>("Cas", max_a * ns, nQ);
    auto Das = std::make_shared<Matrix>("Das", max_a * ns, nQ);
    auto Dbr = std::make_shared<Matrix>("Dbr", max_b * nr, nQ);
    auto Ebs = std::make_shared<Matrix>("Ebs", max_b * ns, nQ);
    auto Ear = std::make_shared<Matrix>("Ear", max_a * nr, nQ);

    // => Thread Work Arrays <= //

    std::vector<std::shared_ptr<Matrix> > Trs;
    std::vector<std::shared_ptr<Matrix> > Vrs;
    for (int t = 0; t < nT; t++) {
        Trs.push_back(std::make_shared<Matrix>("Trs", nr, ns));
        Vrs.push_back(std::make_shared<Matrix>("Vrs", nr, ns));
    }

    // => Pointers <= //
    double** Aarp = Aar->pointer();
    double** Absp = Abs->pointer();
    double** Barp = Bar->pointer();
    double** Bbsp = Bbs->pointer();
    double** Cbrp = Cbr->pointer();
    double** Casp = Cas->pointer();
    double** Dasp = Das->pointer();
    double** Dbrp = Dbr->pointer();
    double** Ebsp = Ebs->pointer();
    double** Earp = Ear->pointer();

    double** Tarp = Tar->pointer();
    double** Tbrp = Tbr->pointer();
    double** Tasp = Tas->pointer();
    double** Tbsp = Tbs->pointer();

    double** omega_arp = omega_ar->pointer();
    double** omega_asp = omega_as->pointer();
    double** omega_brp = omega_br->pointer();
    double** omega_bsp = omega_bs->pointer();

    double* eap = eps_occ0A->pointer();
    double* ebp = eps_occ0B->pointer();
    double* erp = eps_vir0A->pointer();
    double* esp = eps_vir0B->pointer();

    // => Targets <= //

    double Disp20 = 0.0;
    double CompleteDisp20 = 0.0;

    // ==> Master Loop <== //

    for (size_t astart = 0; astart < na; astart += max_a) {
        size_t nablock = (astart + max_a >= na ? na - astart : max_a);

        dfh->fill_tensor("Aar", Aar, {astart, astart + nablock});
        dfh->fill_tensor("Bar", Bar, {astart, astart + nablock});
        dfh->fill_tensor("Cas", Cas, {astart, astart + nablock});
        dfh->fill_tensor("Das", Das, {astart, astart + nablock});
        dfh->fill_tensor("Ear", Ear, {astart, astart + nablock});

        for (size_t bstart = 0; bstart < nb; bstart += max_b) {
            size_t nbblock = (bstart + max_b >= nb ? nb - bstart : max_b);

            dfh->fill_tensor("Abs", Abs, {bstart, bstart + nbblock});
            dfh->fill_tensor("Bbs", Bbs, {bstart, bstart + nbblock});
            dfh->fill_tensor("Cbr", Cbr, {bstart, bstart + nbblock});
            dfh->fill_tensor("Dbr", Dbr, {bstart, bstart + nbblock});
            dfh->fill_tensor("Ebs", Ebs, {bstart, bstart + nbblock});

            long int nab = nablock * nbblock;

#pragma omp parallel for schedule(dynamic) reduction(+ : Disp20, CompleteDisp20)
            for (long int ab = 0L; ab < nab; ab++) {
                int a = ab / nbblock;
                int b = ab % nbblock;

                int thread = 0;
#ifdef _OPENMP
                thread = omp_get_thread_num();
#endif

                double** Trsp = Trs[thread]->pointer();
                double** Vrsp = Vrs[thread]->pointer();

                // => Amplitudes, Disp20 <= //

                C_DGEMM('N', 'T', nr, ns, nQ, 1.0, Aarp[(a)*nr], nQ, Absp[(b)*ns], nQ, 0.0, Vrsp[0], ns);

                for (int r = 0; r < nr; r++) {
                    for (int s = 0; s < ns; s++) {
                        Trsp[r][s] = Vrsp[r][s] / (eap[a + astart] + ebp[b + bstart] - erp[r] - esp[s]);
                        Disp20 += 4.0 * Trsp[r][s] * Vrsp[r][s];
                    }
                }

                // => Exch-Disp20 <= //

                // > DF-Part < //

                C_DGEMM('N', 'T', nr, ns, nQ, 4.0, Barp[(a)*nr], nQ, Bbsp[(b)*ns], nQ, 0.0, Vrsp[0], ns);
                C_DGEMM('N', 'T', nr, ns, nQ, -2.0, Cbrp[(b)*nr], nQ, Casp[(a)*ns], nQ, 1.0, Vrsp[0], ns);
                C_DGEMM('N', 'T', nr, ns, nQ, -2.0, Dbrp[(b)*nr], nQ, Dasp[(a)*ns], nQ, 1.0, Vrsp[0], ns);
                C_DGEMM('N', 'T', nr, ns, nQ, 4.0, Earp[(a)*nr], nQ, Ebsp[(b)*ns], nQ, 1.0, Vrsp[0], ns);

                // > AO-Part < //

                C_DGER(nr, ns, 1.0, Tarp[a + astart], 1, omega_bsp[b + bstart], 1, Vrsp[0], ns);
                C_DGER(nr, ns, -1.0, omega_brp[b + bstart], 1, Tasp[a + astart], 1, Vrsp[0], ns);
                C_DGER(nr, ns, 1.0, omega_arp[a + astart], 1, Tbsp[b + bstart], 1, Vrsp[0], ns);
                C_DGER(nr, ns, -1.0, Tbrp[b + bstart], 1, omega_asp[a + astart], 1, Vrsp[0], ns);
                for (int r = 0; r < nr; r++) {
                    for (int s = 0; s < ns; s++) {
                        CompleteDisp20 += Trsp[r][s] * Vrsp[r][s];
                    }
                }
            }
        }
    }

    double ExchDisp20 = CompleteDisp20 - Disp20;

    scalars_["Disp20"] = Disp20;
    scalars_["Exch-Disp20 (S^inf)"] = ExchDisp20;
    Process::environment.globals["SAPT EXCH-DISP20(S^INF) ENERGY"] = scalars_["Exch-Disp20 (S^inf)"];
    if (do_print) {
        outfile->Printf("    Disp20              = %18.12lf [Eh]\n", Disp20);
        outfile->Printf("    Exch-Disp20 (S^inf) = %18.12lf [Eh]\n", ExchDisp20);
        outfile->Printf("\n");
    }
}



SharedMatrix submatrix_rows(SharedMatrix mat, const std::vector<int> &row_inds) {

    SharedMatrix mat_new = std::make_shared<Matrix>(mat->name(), row_inds.size(), mat->colspi(0));
    for(int r_new = 0; r_new < row_inds.size(); r_new++) {
        int r_old = row_inds[r_new];
        for(int c = 0; c < mat->colspi(0); c++) {
            mat_new->set(r_new, c, mat->get(r_old, c));
        }
    }
    return mat_new;
}

SharedMatrix submatrix_cols(SharedMatrix mat, const std::vector<int> &col_inds) {

    SharedMatrix mat_new = std::make_shared<Matrix>(mat->name(), mat->rowspi(0), col_inds.size());
    for(int r = 0; r < mat->rowspi(0); r++) {
        for(int c_new = 0; c_new < col_inds.size(); c_new++) {
            int c_old = col_inds[c_new];
            mat_new->set(r, c_new, mat->get(r, c_old));
        }
    }
    return mat_new;
}

SharedMatrix submatrix_rows_and_cols(SharedMatrix mat, const std::vector<int> &row_inds, const std::vector<int> &col_inds) {

    SharedMatrix mat_new = std::make_shared<Matrix>(mat->name(), row_inds.size(), col_inds.size());
    for(int r_new = 0; r_new < row_inds.size(); r_new++) {
        int r_old = row_inds[r_new];
        for(int c_new = 0; c_new < col_inds.size(); c_new++) {
            int c_old = col_inds[c_new];
            mat_new->set(r_new, c_new, mat->get(r_old, c_old));
        }
    }
    return mat_new;
}

std::vector<int> contract_lists(const std::vector<int> &y, const std::vector<std::vector<int>> &A_to_y) {

    // TODO: runtime is proportional to A_to_y size (system size, O(N))
    // could maybe reduce to &y size (domain size, O(1)), probably doesn't matter
    std::vector<int> yA;

    for(int a = 0, y_ind = 0; a < A_to_y.size(); ++a) {

        bool is_a = false;
        for(auto y_val : A_to_y[a]) {
            if (y_ind < y.size() && y[y_ind] == y_val) {
                y_ind++;
                is_a = true;
            }
        }

        if(is_a) {
            for(auto y_val : A_to_y[a]) {
                yA.push_back(y_val);
            }
        }

    }

    return yA;

}

/* Args: orthonormal orbitals C (ao x mo) and fock matrix F (ao x ao)
 * Return: transformation matrix X (mo x mo) and energy vector e (mo)
 *
 * CX are canonical orbitals (i.e. F(CX) = e(CX))
 */
std::pair<SharedMatrix, SharedVector> canonicalizer(SharedMatrix C, SharedMatrix F) {
    SharedMatrix X = std::make_shared<Matrix>("eigenvectors", C->colspi(0), C->colspi(0));
    SharedVector e = std::make_shared<Vector>("eigenvalues", C->colspi(0));

    auto temp = linalg::triplet(C, F, C, true, false, false);
    temp->diagonalize(X, e, descending);

    return std::make_pair(X, e);
}

std::pair<SharedMatrix, SharedVector> orthocanonicalizer(SharedMatrix S, SharedMatrix F) {
    
    double S_CUT = 1e-8;
    BasisSetOrthogonalization orthog(BasisSetOrthogonalization::PartialCholesky, S, 0.0, S_CUT, 0);
    auto X = orthog.basis_to_orthog_basis();

    int nmo_initial = X->rowspi(0);
    int nmo_final = X->colspi(0);

    SharedMatrix U = std::make_shared<Matrix>("eigenvectors", nmo_final, nmo_final);
    SharedVector e = std::make_shared<Vector>("eigenvalues", nmo_final);

    auto F_orth = linalg::triplet(X, F, X, true, false, false);
    F_orth->diagonalize(U, e, descending);

    X = linalg::doublet(X, U, false, false);

    return std::make_pair(X, e);
}

std::vector<SharedMatrix> update_amps(const std::vector<SharedMatrix> &t_abrs, 
                                      const std::vector<SharedMatrix> &e_abrs,
                                      const std::vector<SharedMatrix> &r_abrs) {
    int npair = t_abrs.size();
    std::vector<SharedMatrix> t_abrs_new(npair);

    //outfile->Printf("  !!  update_amps(), npair=%d\n", npair);
    for(int ab = 0; ab < npair; ab++) {
        t_abrs_new[ab] = r_abrs[ab]->clone();
        t_abrs_new[ab]->apply_denominator(e_abrs[ab]);
        t_abrs_new[ab]->scale(-1.0);
        t_abrs_new[ab]->add(t_abrs[ab]);
    }
    return t_abrs_new;
}

std::vector<SharedMatrix> calc_residual(const std::vector<SharedMatrix> &t_abrs, 
                                        const std::vector<SharedMatrix> &e_abrs, 
                                        const std::vector<SharedMatrix> &v_abrs,
                                        const std::vector<std::pair<int,int>> &ab_to_a_b,
                                        const std::vector<std::vector<int>> &a_b_to_ab,
                                        const std::vector<std::vector<SharedMatrix>> overlaps_aa,
                                        const std::vector<std::vector<SharedMatrix>> overlaps_bb,
                                        SharedMatrix FA_lmo,
                                        SharedMatrix FB_lmo) {
    int npair = ab_to_a_b.size();
    int na = a_b_to_ab.size();
    int nb = a_b_to_ab[0].size();
    std::vector<SharedMatrix> r_abrs(npair);

    for(int ab = 0; ab < npair; ab++) {
        int a, b;
        std::tie(a, b) = ab_to_a_b[ab];

        int nosv_a = t_abrs[ab]->rowspi()[0];
        int nosv_b = t_abrs[ab]->colspi()[0];

        r_abrs[ab] = t_abrs[ab]->clone();
        for(int r = 0; r < nosv_a; r++) {
            for(int s = 0; s < nosv_b; s++) {
                r_abrs[ab]->set(r, s, t_abrs[ab]->get(r, s) * e_abrs[ab]->get(r, s));
            }
        }
        r_abrs[ab]->add(v_abrs[ab]);

        for(int x = 0; x < na; x++) {
            int xb = a_b_to_ab[x][b];
            if(x == a || xb == -1) continue;
            //outfile->Printf("  !!      x=%d / %d\n", x, na);
            //auto t_xb = t_abrs[xb]->clone();
            auto t_xb = linalg::doublet(overlaps_aa[a][x], t_abrs[xb], false, false);
            t_xb->scale(-1.0 * FA_lmo->get(a, x));
            r_abrs[ab]->add(t_xb);
        }

        for(int y = 0; y < nb; y++) {
            int ay = a_b_to_ab[a][y];
            if(y == b || ay == -1) continue;
            //outfile->Printf("  !!      y=%d \n", y, nb);
            //auto t_ay = t_abrs[ay]->clone();
            auto t_ay = linalg::doublet(t_abrs[ay], overlaps_bb[y][b], false, false);
            t_ay->scale(-1.0 * FB_lmo->get(b, y));
            r_abrs[ab]->add(t_ay);
        }
    }

    return r_abrs;

}


                                        


void C_DGESV_wrapper(SharedMatrix A, SharedMatrix B) {
    int N = B->rowspi(0);
    int M = B->colspi(0);
    if (N == 0 || M == 0) return;

    // create a copy of B in fortran ordering
    std::vector<double> B_fortran(N * M, 0.0);
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            B_fortran[m * N + n] = B->get(n, m);
        }
    }

    // make the C_DGESV call, solving AX=B for X
    std::vector<int> ipiv(N);
    int errcode = C_DGESV(N, M, A->pointer()[0], N, ipiv.data(), B_fortran.data(), N);

    // copy the fortran-ordered X into the original matrix, reverting to C-ordering
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            B->set(n, m, B_fortran[m * N + n]);
        }
    }
}

void C_DGER_wrapper(SharedMatrix A, SharedVector rowvec, SharedVector colvec, double alpha=1.0) {

    int N = A->rowspi()[0];
    int M = A->colspi()[0];

    if(rowvec->dim() != N) throw PSIEXCEPTION("C_DGER dimension mismatch\n");
    if(colvec->dim() != M) throw PSIEXCEPTION("C_DGER dimension mismatch\n");


    //C_DGER(ns, nr, 2.0, tempv->pointer(), 1,  VA_bs->get_row(0, b)->pointer(), 1, vab_sr->get_pointer(), nr);
    C_DGER(N, M, alpha, rowvec->pointer(), 1,  colvec->pointer(), 1, A->get_pointer(), M);
}


SharedVector C_DGEMV_wrapper(SharedMatrix A, SharedVector X, bool transa = false) {

    int M = A->rowspi()[0];
    int N = A->colspi()[0];
    int xdim = transa ? M : N;
    int ydim = transa ? N : M;

    if(X->dim() != xdim) throw PSIEXCEPTION("C_DGEMV dimension mismatch\n");

    SharedVector Y = std::make_shared<Vector>(X->name(), ydim);

    char trans_char = (transa) ? 't' : 'n';
    C_DGEMV(trans_char, M, N, 1.0, A->get_pointer(), N, X->pointer(), 1, 0.0, Y->pointer(), 1);

    return Y;

}


// Compute total dispersion contribution
void FISAPT::local_disp(std::map<std::string, SharedMatrix> matrix_cache, std::map<std::string, SharedVector> vector_cache,
                  bool do_print) {

    /*
     * General Outline:
     *
     *  Part 1 (Dispersion)
     * 1. Form PAOs for monomers A and B
     * 2. Determine DOIs for: (a,a), (b,b), (a,b), (a,r), and (b,s)
     *    I don't think we need (a,s) or (b,r)
     * 3. Calculate dipole pair energies in SMALL PAO domains
     * 4. Do some sparsity
     *    Use pair energies and DOIs to get (a,b) pair list
     *    (Later, we'll need (a,a), (b,b), extended, etc. maybe
     * 5. Calculate all three-index integrals
     *    Store as K..
     *    Dense for now, sparse later
     * 6. Use Kar and Kbs integrals to compute OSV transformations
     * 7. For each pair, form (ar|br) integrals and SC amplitudes, transform to OSVs
     * 8. Compute OSV overlaps
     * 9. Solve amplitudes iteratively
     * 
     * Part 2 (Exchange Dispersion)
     * 1.
     * 
     */
    if (do_print) {
        outfile->Printf("  ==> Local Dispersion <==\n\n");
    }

    // => Auxiliary Basis Set <= //
    std::shared_ptr<BasisSet> auxiliary = reference_->get_basisset("DF_BASIS_SAPT");

    // => Orbitals <= //

    SharedMatrix CA_almo = matrices_["Laocc0A"];
    SharedMatrix CB_almo = matrices_["Laocc0B"];
    SharedMatrix CA_lmo = linalg::horzcat({ matrices_["Lfocc0A"] , matrices_["Laocc0A"]});
    SharedMatrix CB_lmo = linalg::horzcat({ matrices_["Lfocc0B"] , matrices_["Laocc0B"]});

    SharedMatrix CA_vir = matrix_cache["Cvir0A"];
    SharedMatrix CB_vir = matrix_cache["Cvir0B"];
    SharedVector eA_vir = vector_cache["eps_vir0A"];
    SharedVector eB_vir = vector_cache["eps_vir0B"];

    // => Cutoffs <= //
    
    double T_CUT_DO_PRE = 3e-2;
    double T_CUT_DO_ij = 1e-5;
    double T_CUT_PRE = 1e-5;
    double T_CUT_OSV = options_.get_double("T_CUT_OSV");

    // => Sizing <= //

    int nbf = primary_->nbf();
    int naux = auxiliary->nbf();
    int na = CA_lmo->colspi()[0];   // total occA
    int naa = CA_almo->colspi()[0]; // active occA
    int nfa = na - naa;             // frozen occA
    int nb = CB_lmo->colspi()[0];   // total occB
    int nab = CB_almo->colspi()[0]; // active occB
    int nfb = nb - nab;             // frozen occB
    int nr = CA_vir->colspi()[0];
    int ns = CB_vir->colspi()[0];
    size_t nrQ = nr * (size_t)naux;
    size_t nsQ = ns * (size_t)naux;

    int natom = molecule()->natom();

    outfile->Printf("  !! Monomer A occupied orbitals: %df / %da \n", nfa, naa);
    outfile->Printf("  !! Monomer B occupied orbitals: %df / %da \n", nfb, nab);
    outfile->Printf("  !! Monomer A virtual orbitals: %d \n", nr);
    outfile->Printf("  !! Monomer B virtual orbitals: %d \n", ns);

    int nT = 1;
#ifdef _OPENMP
    nT = Process::environment.get_n_threads();
#endif

    std::vector<std::vector<int>>atom_to_bf_(natom);
    for (size_t u = 0; u < nbf; ++u) {
        atom_to_bf_[primary_->function_to_center(u)].push_back(u);
    }

    // => Stashed Variables <= //

    std::shared_ptr<Matrix> S = matrix_cache["S"];
    std::shared_ptr<Matrix> D_A = matrix_cache["D_A"];
    std::shared_ptr<Matrix> P_A = matrix_cache["P_A"];
    std::shared_ptr<Matrix> V_A = matrix_cache["V_A"];
    std::shared_ptr<Matrix> J_A = matrix_cache["J_A"];
    std::shared_ptr<Matrix> K_A = matrix_cache["K_A"];
    std::shared_ptr<Matrix> D_B = matrix_cache["D_B"];
    std::shared_ptr<Matrix> P_B = matrix_cache["P_B"];
    std::shared_ptr<Matrix> V_B = matrix_cache["V_B"];
    std::shared_ptr<Matrix> J_B = matrix_cache["J_B"];
    std::shared_ptr<Matrix> K_B = matrix_cache["K_B"];
    std::shared_ptr<Matrix> K_O = matrix_cache["K_O"];

    SharedMatrix FA = matrix_cache["F0A"];
    SharedMatrix FA_almo = linalg::triplet(CA_almo, FA, CA_almo, true, false, false);
    SharedMatrix FA_lmo = linalg::triplet(CA_lmo, FA, CA_lmo, true, false, false); // TODO: redundant with above
    SharedMatrix FB = matrix_cache["F0B"];
    SharedMatrix FB_almo = linalg::triplet(CB_almo, FB, CB_almo, true, false, false);
    SharedMatrix FB_lmo = linalg::triplet(CB_lmo, FB, CB_lmo, true, false, false);

    // => Forming PAOs <= //

    // Form projected atomic orbitals by removing occupied space from the basis
    
    SharedMatrix CA_pao = std::make_shared<Matrix>("Monomer A Projected Atomic Orbitals", nbf, nbf);
    CA_pao->identity();
    CA_pao->subtract(linalg::triplet(CA_lmo, CA_lmo, S, false, true, false));

    // normalize PAOs
    SharedMatrix SA_pao = linalg::triplet(CA_pao, S, CA_pao, true, false, false);
    for (size_t i = 0; i < CA_pao->colspi(0); ++i) {
        CA_pao->scale_column(0, i, pow(SA_pao->get(i, i), -0.5));
    }
    SA_pao = linalg::triplet(CA_pao, S, CA_pao, true, false, false);
    SharedMatrix FA_pao = linalg::triplet(CA_pao, FA, CA_pao, true, false, false);

    SharedMatrix CB_pao = std::make_shared<Matrix>("Monomer B Projected Atomic Orbitals", nbf, nbf);
    CB_pao->identity();
    CB_pao->subtract(linalg::triplet(CB_lmo, CB_lmo, S, false, true, false));

    SharedMatrix SB_pao = linalg::triplet(CB_pao, S, CB_pao, true, false, false);
    for (size_t i = 0; i < CB_pao->colspi(0); ++i) {
        CB_pao->scale_column(0, i, pow(SB_pao->get(i, i), -0.5));
    }
    SB_pao = linalg::triplet(CB_pao, S, CB_pao, true, false, false);
    SharedMatrix FB_pao = linalg::triplet(CB_pao, FB, CB_pao, true, false, false);


    timer_on("Construct Grid");
    std::shared_ptr<DFTGrid> grid = std::make_shared<DFTGrid>(molecule(), primary_, options_);
    timer_off("Construct Grid");

    std::vector<std::shared_ptr<BasisFunctions>> point_funcs(nT);
    std::vector<SharedMatrix> DOI_aa_temps(nT);
    std::vector<SharedMatrix> DOI_bb_temps(nT);
    std::vector<SharedMatrix> DOI_ab_temps(nT);
    std::vector<SharedMatrix> DOI_ar_temps(nT);
    std::vector<SharedMatrix> DOI_bs_temps(nT);

    for (size_t thread = 0; thread < nT ; thread++) {
        point_funcs[thread] = std::make_shared<BasisFunctions>(primary_, grid->max_points(), nbf);
        DOI_aa_temps[thread] = std::make_shared<Matrix>("(a,a) Differential Overlap Integrals", na, na);
        DOI_ab_temps[thread] = std::make_shared<Matrix>("(a,b) Differential Overlap Integrals", na, nb);
        DOI_bb_temps[thread] = std::make_shared<Matrix>("(b,b) Differential Overlap Integrals", nb, nb);
        DOI_ar_temps[thread] = std::make_shared<Matrix>("(a,r) Differential Overlap Integrals", na, nbf);
        DOI_bs_temps[thread] = std::make_shared<Matrix>("(b,s) Differential Overlap Integrals", nb, nbf);
    }

    for (size_t Q = 0; Q < grid->blocks().size(); Q++) {
        size_t thread = 0;
        //thread = omp_get_thread_num();

        std::shared_ptr<BlockOPoints> block = grid->blocks()[Q];
        int nbf_block = block->local_nbf();
        int npoints_block = block->npoints();

        // compute values of each basis function at each point in this block
        point_funcs[thread]->compute_functions(block);

        // the values we just computed (max_points x max_functions)
        SharedMatrix point_values = point_funcs[thread]->basis_values()["PHI"];

        std::vector<int> bf_map = block->functions_local_to_global();

        // resize point_values buffer to size of this block
        SharedMatrix point_values_trim =
            std::make_shared<Matrix>("DFTGrid PHI Buffer", npoints_block, nbf_block);  // points x bf_block
        for (size_t p = 0; p < npoints_block; p++) {
            for (size_t k = 0; k < nbf_block; k++) {
                point_values_trim->set(p, k, point_values->get(p, k));
            }
        }

        SharedMatrix CA_lmo_slice = submatrix_rows(CA_lmo, bf_map);  // bf_block x naocc
        SharedMatrix CB_lmo_slice = submatrix_rows(CB_lmo, bf_map);  // bf_block x naocc
        SharedMatrix CA_pao_slice = submatrix_rows(CA_pao, bf_map);  // bf_block x npao
        SharedMatrix CB_pao_slice = submatrix_rows(CB_pao, bf_map);  // bf_block x npao

        // value of mo at each point squared
        CA_lmo_slice = linalg::doublet(point_values_trim, CA_lmo_slice, false, false);  // points x naocc
        CB_lmo_slice = linalg::doublet(point_values_trim, CB_lmo_slice, false, false);  // points x naocc
        CA_pao_slice = linalg::doublet(point_values_trim, CA_pao_slice, false, false);  // points x npao
        CB_pao_slice = linalg::doublet(point_values_trim, CB_pao_slice, false, false);  // points x npao

        // change to mat->square_this();
        for (size_t p = 0; p < npoints_block; p++) {
            for (size_t a = 0; a < na; ++a) {
                CA_lmo_slice->set(p, a, pow(CA_lmo_slice->get(p, a), 2));
            }
            for (size_t b = 0; b < nb; ++b) {
                CB_lmo_slice->set(p, b, pow(CB_lmo_slice->get(p, b), 2));
            }
            for (size_t r = 0; r < nbf; ++r) {
                CA_pao_slice->set(p, r, pow(CA_pao_slice->get(p, r), 2));
            }
            for (size_t s = 0; s < nbf; ++s) {
                CB_pao_slice->set(p, s, pow(CB_pao_slice->get(p, s), 2));
            }
        }

        SharedMatrix CA_lmo_slice_w = std::make_shared<Matrix>(CA_lmo_slice);  // points x na
        SharedMatrix CB_lmo_slice_w = std::make_shared<Matrix>(CB_lmo_slice);  // points x nb

        for (size_t p = 0; p < npoints_block; p++) {
            CA_lmo_slice_w->scale_row(0, p, block->w()[p]);
            CB_lmo_slice_w->scale_row(0, p, block->w()[p]);
        }

        DOI_aa_temps[thread]->add(linalg::doublet(CA_lmo_slice_w, CA_lmo_slice, true, false));  // na x na
        DOI_ab_temps[thread]->add(linalg::doublet(CA_lmo_slice_w, CB_lmo_slice, true, false));  // na x nb
        DOI_bb_temps[thread]->add(linalg::doublet(CB_lmo_slice_w, CB_lmo_slice, true, false));  // nb x nb
        DOI_ar_temps[thread]->add(linalg::doublet(CA_lmo_slice_w, CA_pao_slice, true, false));  // na x nbf
        DOI_bs_temps[thread]->add(linalg::doublet(CB_lmo_slice_w, CB_pao_slice, true, false));  // nb x nbf
    }

    SharedMatrix DOI_aa = std::make_shared<Matrix>("(a,a) Differential Overlap Integrals", na, na);
    SharedMatrix DOI_ab = std::make_shared<Matrix>("(a,b) Differential Overlap Integrals", na, nb);
    SharedMatrix DOI_bb = std::make_shared<Matrix>("(b,b) Differential Overlap Integrals", nb, nb);
    SharedMatrix DOI_ar = std::make_shared<Matrix>("(a,r) Differential Overlap Integrals", na, nbf);
    SharedMatrix DOI_bs = std::make_shared<Matrix>("(b,s) Differential Overlap Integrals", nb, nbf);

    for (size_t thread = 0; thread < nT; thread++) {
        DOI_aa->add(DOI_aa_temps[thread]);
        DOI_ab->add(DOI_ab_temps[thread]);
        DOI_bb->add(DOI_bb_temps[thread]);
        DOI_ar->add(DOI_ar_temps[thread]);
        DOI_bs->add(DOI_bs_temps[thread]);
    }

    for(size_t a = 0; a < na; a++) {
        for(size_t c = 0; c < na; c++) {
            DOI_aa->set(a, c, sqrt(DOI_aa->get(a,c)));
        }
    }

    for(size_t a = 0; a < na; a++) {
        for(size_t b = 0; b < nb; b++) {
            DOI_ab->set(a, b, sqrt(DOI_ab->get(a,b)));
        }
    }

    for(size_t b = 0; b < nb; b++) {
        for(size_t d = 0; d < nb; d++) {
            DOI_bb->set(b, d, sqrt(DOI_bb->get(b,d)));
        }
    }

    for(size_t a = 0; a < na; a++) {
        for(size_t r = 0; r < nbf; r++) {
            DOI_ar->set(a, r, sqrt(DOI_ar->get(a,r)));
        }
    }

    for(size_t b = 0; b < nb; b++) {
        for(size_t s = 0; s < nbf; s++) {
            DOI_bs->set(b, s, sqrt(DOI_bs->get(b,s)));
        }
    }

    // TODO: sqrt

    std::shared_ptr<MintsHelper> mints = std::make_shared<MintsHelper>(primary_, options_);
    std::vector<SharedMatrix> ao_dipole = mints->ao_dipole();

    SharedMatrix dipx_aa = linalg::triplet(CA_almo, ao_dipole[0], CA_almo, true, false, false);
    SharedMatrix dipy_aa = linalg::triplet(CA_almo, ao_dipole[1], CA_almo, true, false, false);
    SharedMatrix dipz_aa = linalg::triplet(CA_almo, ao_dipole[2], CA_almo, true, false, false);

    SharedMatrix dipx_ar = linalg::triplet(CA_almo, ao_dipole[0], CA_pao, true, false, false);
    SharedMatrix dipy_ar = linalg::triplet(CA_almo, ao_dipole[1], CA_pao, true, false, false);
    SharedMatrix dipz_ar = linalg::triplet(CA_almo, ao_dipole[2], CA_pao, true, false, false);

    SharedMatrix dipx_bb = linalg::triplet(CB_almo, ao_dipole[0], CB_almo, true, false, false);
    SharedMatrix dipy_bb = linalg::triplet(CB_almo, ao_dipole[1], CB_almo, true, false, false);
    SharedMatrix dipz_bb = linalg::triplet(CB_almo, ao_dipole[2], CB_almo, true, false, false);

    SharedMatrix dipx_bs = linalg::triplet(CB_almo, ao_dipole[0], CB_pao, true, false, false);
    SharedMatrix dipy_bs = linalg::triplet(CB_almo, ao_dipole[1], CB_pao, true, false, false);
    SharedMatrix dipz_bs = linalg::triplet(CB_almo, ao_dipole[2], CB_pao, true, false, false);

    // < a | dipole | a > and < b | dipole | b >
    std::vector<Vector3> R_a;
    std::vector<Vector3> R_b;

    // < a | dipole | r > and < b | dipole | s >
    std::vector<std::vector<Vector3>> dip_ar(naa);
    std::vector<std::vector<Vector3>> dip_bs(nab);

    // orbital energies
    std::vector<SharedVector> e_ar(naa);
    std::vector<SharedVector> e_bs(nab);

    for (size_t a = 0; a < naa; ++a) {
        R_a.push_back(Vector3(dipx_aa->get(a, a), dipy_aa->get(a, a), dipz_aa->get(a, a)));
    }
    for (size_t b = 0; b < nab; ++b) {
        R_b.push_back(Vector3(dipx_bb->get(b, b), dipy_bb->get(b, b), dipz_bb->get(b, b)));
    }

    for (size_t a = 0; a < naa; ++a) {
        std::vector<int> pao_inds;
        for (size_t r = 0; r < nbf; r++) {
            if (fabs(DOI_ar->get(a + nfa, r)) > T_CUT_DO_PRE) {
                pao_inds.push_back(r);
            }
        }
        pao_inds = contract_lists(pao_inds, atom_to_bf_);

        SharedMatrix CA_pao_i = submatrix_cols(CA_pao, pao_inds);
        SharedMatrix SA_pao_i = submatrix_rows_and_cols(SA_pao, pao_inds, pao_inds);
        SharedMatrix FA_pao_i = submatrix_rows_and_cols(FA_pao, pao_inds, pao_inds);

        SharedMatrix XA_pao_i;
        SharedVector eA_pao_i;
        std::tie(XA_pao_i, eA_pao_i) = orthocanonicalizer(SA_pao_i, FA_pao_i);

        SharedMatrix dipx_ar_i = submatrix_rows_and_cols(dipx_ar, {(int)a}, pao_inds);
        SharedMatrix dipy_ar_i = submatrix_rows_and_cols(dipy_ar, {(int)a}, pao_inds);
        SharedMatrix dipz_ar_i = submatrix_rows_and_cols(dipz_ar, {(int)a}, pao_inds);

        dipx_ar_i = linalg::doublet(dipx_ar_i, XA_pao_i);
        dipy_ar_i = linalg::doublet(dipy_ar_i, XA_pao_i);
        dipz_ar_i = linalg::doublet(dipz_ar_i, XA_pao_i);

        int npao_i = XA_pao_i->colspi(0);

        for (size_t r = 0; r < npao_i; r++) {
            dip_ar[a].push_back(Vector3(dipx_ar_i->get(0, r), dipy_ar_i->get(0, r), dipz_ar_i->get(0, r)));
        }
        e_ar[a] = eA_pao_i;
    }

    for (size_t b = 0; b < nab; ++b) {
        std::vector<int> pao_inds;
        for (size_t s = 0; s < nbf; s++) {
            if (fabs(DOI_bs->get(b + nfb, s)) > T_CUT_DO_PRE) {
                pao_inds.push_back(s);
            }
        }
        pao_inds = contract_lists(pao_inds, atom_to_bf_);

        SharedMatrix CB_pao_i = submatrix_cols(CB_pao, pao_inds);
        SharedMatrix SB_pao_i = submatrix_rows_and_cols(SB_pao, pao_inds, pao_inds);
        SharedMatrix FB_pao_i = submatrix_rows_and_cols(FB_pao, pao_inds, pao_inds);

        SharedMatrix XB_pao_i;
        SharedVector eB_pao_i;
        std::tie(XB_pao_i, eB_pao_i) = orthocanonicalizer(SB_pao_i, FB_pao_i);

        SharedMatrix dipx_bs_i = submatrix_rows_and_cols(dipx_bs, {(int)b}, pao_inds);
        SharedMatrix dipy_bs_i = submatrix_rows_and_cols(dipy_bs, {(int)b}, pao_inds);
        SharedMatrix dipz_bs_i = submatrix_rows_and_cols(dipz_bs, {(int)b}, pao_inds);

        dipx_bs_i = linalg::doublet(dipx_bs_i, XB_pao_i);
        dipy_bs_i = linalg::doublet(dipy_bs_i, XB_pao_i);
        dipz_bs_i = linalg::doublet(dipz_bs_i, XB_pao_i);

        int npao_i = XB_pao_i->colspi(0);

        for (size_t s = 0; s < npao_i; s++) {
            dip_bs[b].push_back(Vector3(dipx_bs_i->get(0, s), dipy_bs_i->get(0, s), dipz_bs_i->get(0, s)));
        }
        e_bs[b] = eB_pao_i;
    }

    SharedMatrix e_actual = std::make_shared<Matrix>("Dipole Dispersion Energies", naa, nab);
    SharedMatrix e_linear = std::make_shared<Matrix>("Parallel Dipole Dispersion Energies", naa, nab);

    for (size_t a = 0; a < naa; ++a) {
        for (size_t b = 0; b < nab; ++b) {
            Vector3 R_ab = R_a[a] - R_b[b];
            Vector3 Rh_ab = R_ab / R_ab.norm();

            double e_actual_temp = 0.0;
            double e_linear_temp = 0.0;

            for (int r = 0; r < dip_ar[a].size(); r++) {
                for (int s = 0; s < dip_bs[b].size(); s++) {
                    Vector3 u_ar = dip_ar[a][r];
                    Vector3 u_bs = dip_bs[b][s];

                    double num_actual = u_ar.dot(u_bs) - 3 * (u_ar.dot(Rh_ab) * u_bs.dot(Rh_ab));
                    num_actual *= num_actual;

                    double num_linear = -2 * u_ar.dot(u_bs);
                    num_linear *= num_linear;

                    double denom = (e_ar[a]->get(r) + e_bs[b]->get(s)) - (FA_almo->get(a, a) + FB_almo->get(b, b));

                    e_actual_temp += (num_actual / denom);
                    e_linear_temp += (num_linear / denom);
                }
            }

            e_actual_temp *= (-8 * pow(R_ab.norm(), -6));
            e_linear_temp *= (-8 * pow(R_ab.norm(), -6));

            e_actual->set(a, b, e_actual_temp);
            e_linear->set(a, b, e_linear_temp);
        }
    }

    double e_actual_sum = 0.0;
    double e_linear_sum = 0.0;
    for(size_t a = 0; a < naa; a++) {
        for(size_t b = 0; b < nab; b++) {
            e_actual_sum += e_actual->get(a,b);
            e_linear_sum += e_linear->get(a,b);
        }
    }

    outfile->Printf("  !! Estimated E_disp20 : %.6f kcal / mol\n", pc_hartree2kcalmol * e_actual_sum);
    outfile->Printf("  !!   Upper Bound : %.6f kcal / mol\n", pc_hartree2kcalmol * e_linear_sum);


    std::vector<std::vector<int>> a_b_to_ab(na, std::vector<int>(nb, -1));
    std::vector<std::pair<int,int>> ab_to_a_b;
    double de_dipole_ = 0.0;

    size_t overlap_count = 0;
    size_t energy_count = 0;
    size_t both_count = 0;

    for (size_t a = 0, ab = 0; a < naa; a++) {
        for (size_t b = 0; b < nab; b++) {
            bool overlap_big = (DOI_ab->get(nfa + a, nfb + b) > T_CUT_DO_ij);
            bool energy_big = (fabs(e_linear->get(a, b)) > T_CUT_PRE);

            if (overlap_big || energy_big) {
                a_b_to_ab[nfa + a][nfb + b] = ab;
                ab_to_a_b.push_back(std::make_pair(nfa + a, nfb + b));
                ab++;
            } else {
                de_dipole_ += e_actual->get(a, b);
            }

            if (overlap_big) overlap_count++;
            if (energy_big) energy_count++;
            if (overlap_big || energy_big) both_count++;
        }
    }

    int npair = ab_to_a_b.size();

    outfile->Printf("  !! Dipole Correction: %.6f kcal / mol\n", pc_hartree2kcalmol * de_dipole_);
    outfile->Printf("  !!   Overlap Criteria: %zu / %zu pairs \n", overlap_count, naa * nab);
    outfile->Printf("  !!   Energy  Criteria: %zu / %zu pairs \n", energy_count, naa * nab);
    outfile->Printf("  !!   Combined:         %zu / %zu pairs \n", both_count, naa * nab);

    auto factory = std::make_shared<IntegralFactory>(auxiliary, BasisSet::zero_ao_basis_set(), primary_, primary_);
    std::vector<std::shared_ptr<TwoBodyAOInt>> eris(nT);

    for (size_t thread = 0; thread < nT; thread++) {
        eris[thread] = std::shared_ptr<TwoBodyAOInt>(factory->eri());
    }

    // LMO/LMO integrals
    std::vector<SharedMatrix> Qaa(naux), Qbb(naux);
    std::vector<SharedMatrix> Qab(naux);

    // LMO/VIR integrals
    std::vector<SharedMatrix> Qar(naux), Qbs(naux);
    std::vector<SharedMatrix> Qas(naux), Qbr(naux);

    // LMO/PAO integrals
    std::vector<SharedMatrix> Qac(naux), Qbd(naux);
    std::vector<SharedMatrix> Qad(naux), Qbc(naux);

    outfile->Printf("  !! Starting Integral Transform... \n");

#pragma omp parallel for schedule(static, 1)
    for (int Q = 0; Q < auxiliary->nshell(); Q++) {
        int nq = auxiliary->shell(Q).nfunction();
        int qstart = auxiliary->shell(Q).function_index();
        int centerQ = auxiliary->shell_to_center(Q);

        size_t thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif
//        // sparse lists of non-screened basis functions
//        auto bf_map1 = riatom_to_bfs1[centerQ];
//        auto bf_map2 = riatom_to_bfs2[centerQ];
//
//        // inverse map, from global (non-screened) bf-index to Q-specific (screened) index
//        std::vector<int> bf_map1_inv(nbf, -1);
//        std::vector<int> bf_map2_inv(nbf, -1);
//        for (int m_ind = 0; m_ind < bf_map1.size(); m_ind++) {
//            bf_map1_inv[bf_map1[m_ind]] = m_ind;
//        }
//        for (int n_ind = 0; n_ind < bf_map2.size(); n_ind++) {
//            bf_map2_inv[bf_map2[n_ind]] = n_ind;
//        }
//
        for (size_t q = 0; q < nq; q++) {
//            qia[qstart + q] = std::make_shared<Matrix>("(mn|Q)", bf_map1.size(), bf_map2.size());
            Qar[qstart + q] = std::make_shared<Matrix>("(mn|Q)", nbf, nbf);
            Qbs[qstart + q] = std::make_shared<Matrix>("(mn|Q)", nbf, nbf);
            Qas[qstart + q] = std::make_shared<Matrix>("(mn|Q)", nbf, nbf);
            Qbr[qstart + q] = std::make_shared<Matrix>("(mn|Q)", nbf, nbf);
            Qaa[qstart + q] = std::make_shared<Matrix>("(mn|Q)", nbf, nbf);
            Qbb[qstart + q] = std::make_shared<Matrix>("(mn|Q)", nbf, nbf);
            Qab[qstart + q] = std::make_shared<Matrix>("(mn|Q)", nbf, nbf);

            Qac[qstart + q] = std::make_shared<Matrix>("(mn|Q)", nbf, nbf);
            Qbd[qstart + q] = std::make_shared<Matrix>("(mn|Q)", nbf, nbf);
            Qad[qstart + q] = std::make_shared<Matrix>("(mn|Q)", nbf, nbf);
            Qbc[qstart + q] = std::make_shared<Matrix>("(mn|Q)", nbf, nbf);
        }

        for (int M = 0; M < primary_->nshell(); M++) {
//        for (int M : riatom_to_shells1[centerQ]) {
            int nm = primary_->shell(M).nfunction();
            int mstart = primary_->shell(M).function_index();
            int centerM = primary_->shell_to_center(M);

            for (int N = 0; N < primary_->nshell(); N++) {
                int nn = primary_->shell(N).nfunction();
//            for (int N : riatom_to_shells2[centerQ]) {
                int nstart = primary_->shell(N).function_index();
                int centerN = primary_->shell_to_center(N);

//                // is (N in the list of M's) and (M in the list of N's)?
//                bool MN_symmetry =
//                    (riatom_to_atoms1_dense[centerQ][centerN] && riatom_to_atoms2_dense[centerQ][centerM]);

//                // if so, we want to exploit (MN|Q) <-> (NM|Q) symmetry
//                if (N < M && MN_symmetry) continue;
//
                eris[thread]->compute_shell(Q, 0, M, N);
                const double* buffer = eris[thread]->buffer();

                for (int q = 0, index = 0; q < nq; q++) {
                    for (int m = 0; m < nm; m++) {
                        for (int n = 0; n < nn; n++, index++) {
//                            qia[qstart + q]->set(bf_map1_inv[mstart + m], bf_map2_inv[nstart + n], buffer[index]);
                            Qar[qstart + q]->set(mstart + m, nstart + n, buffer[index]);
                            Qbs[qstart + q]->set(mstart + m, nstart + n, buffer[index]);
                            Qas[qstart + q]->set(mstart + m, nstart + n, buffer[index]);
                            Qbr[qstart + q]->set(mstart + m, nstart + n, buffer[index]);
                            Qaa[qstart + q]->set(mstart + m, nstart + n, buffer[index]);
                            Qbb[qstart + q]->set(mstart + m, nstart + n, buffer[index]);
                            Qab[qstart + q]->set(mstart + m, nstart + n, buffer[index]);

                            Qac[qstart + q]->set(mstart + m, nstart + n, buffer[index]);
                            Qbd[qstart + q]->set(mstart + m, nstart + n, buffer[index]);
                            Qad[qstart + q]->set(mstart + m, nstart + n, buffer[index]);
                            Qbc[qstart + q]->set(mstart + m, nstart + n, buffer[index]);
                        }
                    }
                }

//                // (MN|Q) <-> (NM|Q) symmetry
//                if (N > M && MN_symmetry) {
//                    for (int q = 0, index = 0; q < nq; q++) {
//                        for (int m = 0; m < nm; m++) {
//                            for (int n = 0; n < nn; n++, index++) {
//                                qia[qstart + q]->set(bf_map1_inv[nstart + n], bf_map2_inv[mstart + m], buffer[index]);
//                            }
//                        }
//                    }
//                }

            }  // N loop
        }      // M loop

//        SharedMatrix C_pao_slice = submatrix_rows(C_pao, riatom_to_bfs2[centerQ]);  // TODO: PAO slices

//        //// Here we'll refit the coefficients of C_lmo_slice to minimize residual from unscreened orbitals
//        //// This lets us get away with agressive coefficient screening
//        //// Boughton and Pulay 1992 JCC, Equation 3
//
//        // Solve for C_lmo_slice such that S[local,local] @ C_lmo_slice ~= S[local,all] @ C_lmo
//        SharedMatrix C_lmo_slice =
//            submatrix_rows_and_cols(SC_lmo, riatom_to_bfs1[centerQ], riatom_to_lmos_ext[centerQ]);
//        SharedMatrix S_aa =
//            submatrix_rows_and_cols(reference_wavefunction_->S(), riatom_to_bfs1[centerQ], riatom_to_bfs1[centerQ]);
//        C_DGESV_wrapper(S_aa, C_lmo_slice);

        // (mn|Q) C_mi C_nu -> (iu|Q)
        for (size_t q = 0; q < nq; q++) {
//            qia[qstart + q] = linalg::triplet(C_lmo_slice, qia[qstart + q], C_pao_slice, true, false, false);
            //Qar[qstart + q] = linalg::triplet(CA_lmo, Qar[qstart + q], CA_pao, true, false, false);
            //Qbs[qstart + q] = linalg::triplet(CB_lmo, Qbs[qstart + q], CB_pao, true, false, false);
            Qar[qstart + q] = linalg::triplet(CA_lmo, Qar[qstart + q], CA_vir, true, false, false);
            Qbs[qstart + q] = linalg::triplet(CB_lmo, Qbs[qstart + q], CB_vir, true, false, false);
            Qas[qstart + q] = linalg::triplet(CA_lmo, Qas[qstart + q], CB_vir, true, false, false);
            Qbr[qstart + q] = linalg::triplet(CB_lmo, Qbr[qstart + q], CA_vir, true, false, false);
            Qaa[qstart + q] = linalg::triplet(CA_lmo, Qaa[qstart + q], CA_lmo, true, false, false);
            Qbb[qstart + q] = linalg::triplet(CB_lmo, Qbb[qstart + q], CB_lmo, true, false, false);
            Qab[qstart + q] = linalg::triplet(CA_lmo, Qab[qstart + q], CB_lmo, true, false, false);

            Qac[qstart + q] = linalg::triplet(CA_lmo, Qac[qstart + q], CA_pao, true, false, false);
            Qbd[qstart + q] = linalg::triplet(CB_lmo, Qbd[qstart + q], CB_pao, true, false, false);
            Qad[qstart + q] = linalg::triplet(CA_lmo, Qad[qstart + q], CB_pao, true, false, false);
            Qbc[qstart + q] = linalg::triplet(CB_lmo, Qbc[qstart + q], CA_pao, true, false, false);
        }

    }  // Q loop

    // Compute the full metric, don't invert
    auto metric = std::make_shared<FittingMetric>(auxiliary, true);
    metric->form_fitting_metric();
    auto met = std::make_shared<Matrix>(metric->get_metric());

    // canonical transformation from PAOs -> virtuals
    std::vector<SharedMatrix> XA_can(na), XB_can(nb); 
    std::vector<SharedVector> eA_can(na), eB_can(nb); 

    // canonical transformation from PAOs -> orbital specific virtuals
    std::vector<SharedMatrix> XA_osv(na), XB_osv(nb);
    std::vector<SharedVector> eA_osv(na), eB_osv(nb);

    outfile->Printf("  !! Forming OSVs A... \n");

    for(size_t a = 0; a < na; a++) {

        std::tie(XA_can[a], eA_can[a]) = orthocanonicalizer(SA_pao, FA_pao); // TODO: use subset of SA_pao, FA_pao
        auto npao_dep_a = XA_can[a]->rowspi()[0]; // possibly linear dependent PAOs
        auto npao_a = XA_can[a]->colspi()[0]; // orthocanonicalized PAOs

        // get (ar|Q) integrals for fixed a
        auto Qar_a = std::make_shared<Matrix>("(ar|Q)_a", naux, npao_dep_a);
        for(size_t q = 0; q < naux; q++) {
            for(size_t r = 0; r < npao_dep_a; r++) {
                Qar_a->set(q, r, Qac[q]->get(a, r));
            }
        }
        Qar_a = linalg::doublet(Qar_a, XA_can[a]); // (naux x npao_pre_a) (npao_pre_a x npao_a)

        // contract Qar_a with inverse metric
        auto Jar_a = Qar_a->clone();
        auto local_met = met->clone();
        C_DGESV_wrapper(local_met, Jar_a);

        // (ar|ar')_a
        auto v_aa = linalg::doublet(Jar_a, Qar_a, true, false);

        // SC amplitudes
        auto t_aa = v_aa->clone();
        for(size_t r1 = 0; r1 < npao_a; r1++) {
            for(size_t r2 = 0; r2 < npao_a; r2++) {
                double denom = eA_can[a]->get(r1) + eA_can[a]->get(r2) - 2 * FA_lmo->get(a,a);
                t_aa->set(r1, r2, -1.0 * t_aa->get(r1, r2) / denom);
            }
        }

        // antisymmetrized SC amplitudes
        auto tt_aa = t_aa->clone();
        tt_aa->scale(2.0);
        tt_aa->subtract(t_aa->transpose());

        // virtual-virtual SC density matrix of pair aa
        auto d_aa = linalg::doublet(tt_aa, t_aa, true, false);
        d_aa->add(linalg::doublet(tt_aa, t_aa, false, true));

        // OSVs diagonalize this density matrix
        XA_osv[a]= std::make_shared<Matrix>("eigenvectors", npao_a, npao_a);
        SharedVector osv_occ = std::make_shared<Vector>("eigenvalues", npao_a);
        d_aa->diagonalize(XA_osv[a], osv_occ, descending);

        // truncate OSVs by occupation number
        int nosv_a = 0;
        for (size_t r = 0; r < npao_a; ++r) {
            if (fabs(osv_occ->get(r)) >= T_CUT_OSV) nosv_a ++;
        }
        Dimension dim_final = Dimension(1);
        dim_final.fill(nosv_a);
        XA_osv[a] = XA_osv[a]->get_block({Dimension(1), XA_osv[a]->rowspi()}, {Dimension(1), dim_final});

        // The (now truncated) OSVs are orthogonal, but not yet canonical
        XA_osv[a] = linalg::doublet(XA_can[a], XA_osv[a], false, false);

        // Now the transformation gives orbitals that are orthonormal and canonical
        SharedMatrix osv_canon;
        std::tie(osv_canon, eA_osv[a]) = canonicalizer(XA_osv[a], FA_pao);
        XA_osv[a] = linalg::doublet(XA_osv[a], osv_canon, false, false);

        outfile->Printf("  !! LMO %d: %4d ldPAOs -> %4d PAOS -> %d OSVs\n", a, npao_dep_a, npao_a, nosv_a);

    }

    outfile->Printf("  !! Forming OSVs B... \n");

    for(size_t b = 0; b < nb; b++) {

        std::tie(XB_can[b], eB_can[b]) = orthocanonicalizer(SB_pao, FB_pao); // TODO: use subset of SB_pao, FB_pao
        auto npao_dep_b = XB_can[b]->rowspi()[0]; // possibly linear dependent PAOs
        auto npao_b = XB_can[b]->colspi()[0]; // orthocanonicalized PAOs

        // get (bs|Q) integrals for fixed b
        auto Qbs_b = std::make_shared<Matrix>("(bs|Q)_b", naux, npao_dep_b);
        for(size_t q = 0; q < naux; q++) {
            for(size_t s = 0; s < npao_dep_b; s++) {
                Qbs_b->set(q, s, Qbd[q]->get(b, s));
            }
        }
        Qbs_b = linalg::doublet(Qbs_b, XB_can[b]); // (naux x npao_pre_a) (npao_pre_a x npao_b)

        // contract Qar_a with inverse metric
        auto Jbs_b = Qbs_b->clone();
        auto local_met = met->clone();
        C_DGESV_wrapper(local_met, Jbs_b);

        // (bs|bs')_b
        auto v_bb = linalg::doublet(Jbs_b, Qbs_b, true, false);

        // SC amplitudes
        auto t_bb = v_bb->clone();
        for(size_t s1 = 0; s1 < npao_b; s1++) {
            for(size_t s2 = 0; s2 < npao_b; s2++) {
                double denom = eB_can[b]->get(s1) + eB_can[b]->get(s2) - 2 * FB_lmo->get(b,b);
                t_bb->set(s1, s2, -1.0 * t_bb->get(s1, s2) / denom);
            }
        }

        // antisymmetrized SC amplitudes
        auto tt_bb = t_bb->clone();
        tt_bb->scale(2.0);
        tt_bb->subtract(t_bb->transpose());

        // virtual-virtual SC density matrix of pair bb
        auto d_bb = linalg::doublet(tt_bb, t_bb, true, false);
        d_bb->add(linalg::doublet(tt_bb, t_bb, false, true));

        // OSVs diagonalize this density matrix
        XB_osv[b]= std::make_shared<Matrix>("eigenvectors", npao_b, npao_b);
        SharedVector osv_occ = std::make_shared<Vector>("eigenvalues", npao_b);
        d_bb->diagonalize(XB_osv[b], osv_occ, descending);

        // truncate OSVs by occupation number
        int nosv_b = 0;
        for (size_t s = 0; s < npao_b; ++s) {
            if (fabs(osv_occ->get(s)) >= T_CUT_OSV) nosv_b ++;
        }
        Dimension dim_final = Dimension(1);
        dim_final.fill(nosv_b);
        XB_osv[b] = XB_osv[b]->get_block({Dimension(1), XB_osv[b]->rowspi()}, {Dimension(1), dim_final});

        // The (now truncated) OSVs are orthogonal, but not yet canonical
        XB_osv[b] = linalg::doublet(XB_can[b], XB_osv[b], false, false);

        // Now the transformation gives orbitals that are orthonormal and canonical
        SharedMatrix osv_canon;
        std::tie(osv_canon, eB_osv[b]) = canonicalizer(XB_osv[b], FB_pao);
        XB_osv[b] = linalg::doublet(XB_osv[b], osv_canon, false, false);

        outfile->Printf("  !! LMO %d: %4d ldPAOs -> %4d PAOS -> %d OSVs\n", b, npao_dep_b, npao_b, nosv_b);

    }

    std::vector<std::vector<SharedMatrix>> osv_overlaps_aa(na, std::vector<SharedMatrix>(na));
    std::vector<std::vector<SharedMatrix>> osv_overlaps_bb(nb, std::vector<SharedMatrix>(nb));

    for(size_t a = 0; a < na; a++) {
        for(size_t c = 0; c < na; c++) {
            osv_overlaps_aa[a][c] = linalg::triplet(XA_osv[a], SA_pao, XA_osv[c], true, false, false);
        }
    }

    for(size_t b = 0; b < nb; b++) {
        for(size_t d = 0; d < nb; d++) {
            osv_overlaps_bb[b][d] = linalg::triplet(XB_osv[b], SB_pao, XB_osv[d], true, false, false);
        }
    }

    std::vector<SharedMatrix> v_abrs(npair);  // v_abrs[ab]  = (ar|bs) 
    std::vector<SharedMatrix> e_abrs(npair);  // e_arbs[ab]  = (e_r + e_s - e_a - e_b)
    std::vector<SharedMatrix> t_abrs(npair);  // t_arbs[ab]  = amplitudes
    std::vector<SharedMatrix> tt_abrs(npair); // tt_arbs[ab] = 2t_arbs[ab] - t_arbs[ab].T
    std::vector<SharedMatrix> r_abrs(npair);  // r_arbs[ab]  = amplitude residuals

    outfile->Printf("  !! Iterating over (%d / %d) ab pairs \n", npair, naa * nab);

    for(int ab = 0; ab < npair; ab++) {
        int a, b;
        std::tie(a, b) = ab_to_a_b[ab];
        //outfile->Printf("  !!  Pair (%d / %d) : a=%d, b=%d    ", ab, npair, a, b);

        int npao_a = XA_osv[a]->rowspi()[0]; // number of PAOs
        int npao_b = XB_osv[b]->rowspi()[0]; // number of PAOs

        int nosv_a = XA_osv[a]->colspi()[0]; // number of OSVs
        int nosv_b = XB_osv[b]->colspi()[0]; // number of OSVs

        auto Qar_a = std::make_shared<Matrix>("(ar|Q)_a", naux, npao_a);
        for(size_t q = 0; q < naux; q++) {
            for(size_t r = 0; r < npao_a; r++) {
                Qar_a->set(q, r, Qac[q]->get(a,r));
            }
        }
        Qar_a = linalg::doublet(Qar_a, XA_osv[a], false, false);

        auto Qbs_b = std::make_shared<Matrix>("(bs|Q)_b", naux, npao_b);
        for(size_t q = 0; q < naux; q++) {
            for(size_t s = 0; s < npao_b; s++) {
                Qbs_b->set(q, s, Qbd[q]->get(b,s));
            }
        }
        Qbs_b = linalg::doublet(Qbs_b, XB_osv[b], false, false);

        auto local_met = met->clone();
        C_DGESV_wrapper(local_met, Qar_a);
        v_abrs[ab] = linalg::doublet(Qar_a, Qbs_b, true, false);

        e_abrs[ab] = std::make_shared<Matrix>("Energy Denominator", nosv_a, nosv_b);
        t_abrs[ab] = std::make_shared<Matrix>("Dispersion Amplitudes", nosv_a, nosv_b);
        for(size_t r = 0; r < nosv_a; r++) {
            for(size_t s = 0; s < nosv_b; s++) {
                double denom = (eA_osv[a]->get(r) + eB_osv[b]->get(s)) - (FA_lmo->get(a,a) + FB_lmo->get(b,b));
                e_abrs[ab]->set(r, s, denom);
                t_abrs[ab]->set(r, s, -1.0 * v_abrs[ab]->get(r,s) / denom);
            }
        }

    }

    //for(int ab = 0; ab < npair; ab++) {
    //    int a, b;
    //    std::tie(a, b) = ab_to_a_b[ab];
    //    //outfile->Printf("  !!  Pair (%d / %d) : a=%d, b=%d    ", ab, npair, a, b);

    //    int npao_a = XA_can[a]->colspi()[0]; // number of canonical orbitals
    //    int npao_b = XB_can[b]->colspi()[0]; // number of canonical orbitals

    //    //auto Qar_a = std::make_shared<Matrix>("3II slive", naux, nn);
    //    auto Qar_a = std::make_shared<Matrix>("3II slive", naux, nr);
    //    for(size_t q = 0; q < naux; q++) {
    //        //for(size_t r = 0; r < nn; r++) {
    //        for(size_t r = 0; r < nr; r++) {
    //            Qar_a->set(q, r, Qar[q]->get(a,r));
    //        }
    //    }
    //    //Qar_a = linalg::doublet(Qar_a, XA_pao[a], false, false);

    //    //auto Qbs_b = std::make_shared<Matrix>("3II slive", naux, nn);
    //    auto Qbs_b = std::make_shared<Matrix>("3II slive", naux, ns);
    //    for(size_t q = 0; q < naux; q++) {
    //        //for(size_t s = 0; s < nn; s++) {
    //        for(size_t s = 0; s < ns; s++) {
    //            Qbs_b->set(q, s, Qbs[q]->get(b,s));
    //        }
    //    }
    //    //Qbs_b = linalg::doublet(Qbs_b, XB_pao[b], false, false);
    //    auto local_met = met->clone();
    //    C_DGESV_wrapper(local_met, Qar_a);

    //    v_abrs[ab] = linalg::doublet(Qar_a, Qbs_b, true, false);
    //    //e_abrs[ab] = std::make_shared<Matrix>("Energy Denominator", XA_pao[a]->colspi()[0], XB_pao[b]->colspi()[0]);
    //    //for(size_t r = 0; r < XA_pao[a]->colspi()[0]; r++) {
    //    //    for(size_t s = 0; s < XB_pao[b]->colspi()[0]; s++) {
    //    //        e_abrs[ab]->set(r, s, eA_pao[a]->get(r) + eB_pao[b]->get(s) - FA_lmo->get(a,a) - FB_lmo->get(b,b));
    //    //    }
    //    //}
    //    e_abrs[ab] = std::make_shared<Matrix>("Energy Denominator", nr, ns);
    //    t_abrs[ab] = std::make_shared<Matrix>("Dispersion Amplitudes", nr, ns);
    //    for(size_t r = 0; r < nr; r++) {
    //        for(size_t s = 0; s < ns; s++) {
    //            double denom = (eA_vir->get(r) + eB_vir->get(s)) - (FA_lmo->get(a,a) + FB_lmo->get(b,b));
    //            e_abrs[ab]->set(r, s, denom);
    //            t_abrs[ab]->set(r, s, -1.0 * v_abrs[ab]->get(r,s) / denom);
    //        }
    //    }

    //}

    double disp_tot, disp_plain;

    for(int iteration=0; iteration < 10; iteration++) {

        //r_abrs = calc_residual(t_abrs, e_abrs, v_abrs, ab_to_a_b, a_b_to_ab, FA_lmo, FB_lmo);
        r_abrs = calc_residual(t_abrs, e_abrs, v_abrs, ab_to_a_b, a_b_to_ab, osv_overlaps_aa, osv_overlaps_bb, FA_lmo, FB_lmo);
        disp_tot = 0.0;
        disp_plain = 0.0;

        for(int ab = 0; ab < npair; ab++) {
            disp_tot += 4.0 * (t_abrs[ab]->vector_dot(v_abrs[ab]) + t_abrs[ab]->vector_dot(r_abrs[ab]));
            disp_plain += 4.0 * (t_abrs[ab]->vector_dot(v_abrs[ab]));
        }

        t_abrs = update_amps(t_abrs, e_abrs, r_abrs);

        outfile->Printf("  !! Iterative Disp: %.8f %.8f\n", disp_tot * pc_hartree2kcalmol, disp_plain * pc_hartree2kcalmol);

    }

    outfile->Printf("  !! Total Disp: %.8f %.8f\n", (disp_tot + de_dipole_) * pc_hartree2kcalmol, (disp_plain + de_dipole_) * pc_hartree2kcalmol);

    double e_exchdisp = 0.0;

    timer_on("ExchDisp");

    auto Qac_rect_all = std::make_shared<Matrix>("", naux * na, nbf);
    auto Qbd_rect_all = std::make_shared<Matrix>("", naux * nb, nbf);

    for(size_t q = 0, qa = 0; q < naux; q++) {
        for(size_t a = 0; a < na; a++, qa++) {
            for(size_t r = 0; r < nbf; r++) {
                Qac_rect_all->set(qa, r, Qac[q]->get(a, r));
            }
        }
    }

    for(size_t q = 0, qb = 0; q < naux; q++) {
        for(size_t b = 0; b < nb; b++, qb++) {
            for(size_t s = 0; s < nbf; s++) {
                Qbd_rect_all->set(qb, s, Qbd[q]->get(b, s));
            }
        }
    }

    outfile->Printf("  !! Made big int mats\n");

    for(int ab = 0; ab < npair; ab++) {
        int a, b;
        std::tie(a, b) = ab_to_a_b[ab];

        int npao_a = XA_osv[a]->rowspi()[0];
        int npao_b = XB_osv[b]->rowspi()[0];

        int nosv_a = XA_osv[a]->colspi()[0];
        int nosv_b = XB_osv[b]->colspi()[0];

        //outfile->Printf("  !! Pair: %d     PAOs: (%d,%d)     OSVs: (%d,%d)\n", ab, npao_a, npao_b, nosv_a, nosv_b);

        timer_on("Rect");
        // PAO -> OSV
        auto Qac_rect = linalg::doublet(Qac_rect_all, XA_osv[a]);
        auto Qbd_rect = linalg::doublet(Qbd_rect_all, XB_osv[b]);
        timer_off("Rect");

        timer_on("Overlap");
        auto S_ab = linalg::triplet(CA_lmo, S, CB_lmo, true, false, false); // na x ns
        auto S_as = linalg::triplet(CA_lmo, S, CB_pao, true, false, false); // na x ns
        auto S_br = linalg::triplet(CB_lmo, S, CA_pao, true, false, false); // na x ns
        auto S_rs = linalg::triplet(CA_pao, S, CB_pao, true, false, false); // na x ns

        S_as = linalg::doublet(S_as, XB_osv[b], false, false);
        S_br = linalg::doublet(S_br, XA_osv[a], false, false);
        S_rs = linalg::triplet(XA_osv[a], S_rs, XB_osv[b], true, false, false);

        auto S_ab_a = S_ab->get_row(0, a);
        auto S_ab_b = S_ab->get_column(0, b);
        auto S_as_a = S_as->get_row(0, a);
        auto S_br_b = S_br->get_row(0, b);

        auto VA_bs = linalg::triplet(CB_lmo, V_A, CB_pao, true, false, false);
        auto VB_ar = linalg::triplet(CA_lmo, V_B, CA_pao, true, false, false);

        VA_bs = linalg::doublet(VA_bs, XB_osv[b], false);
        VB_ar = linalg::doublet(VB_ar, XA_osv[a], false);

        timer_off("Overlap");

        timer_on("Copying");

        auto Qax_a = std::make_shared<Matrix>("3II slive", naux, na);
        for(size_t q = 0; q < naux; q++) {
            for(size_t x = 0; x < na; x++) {
                Qax_a->set(q, x, Qaa[q]->get(a,x));
            }
        }

        auto Qay_a = std::make_shared<Matrix>("3II slive", naux, nb);
        for(size_t q = 0; q < naux; q++) {
            for(size_t y = 0; y < nb; y++) {
                Qay_a->set(q, y, Qab[q]->get(a,y));
            }
        }

        auto Qar_a = std::make_shared<Matrix>("3II slive", naux, npao_a);
        for(size_t q = 0; q < naux; q++) {
            for(size_t r = 0; r < npao_a; r++) {
                Qar_a->set(q, r, Qac[q]->get(a,r));
            }
        }
        Qar_a = linalg::doublet(Qar_a, XA_osv[a], false, false);

        auto Qas_a = std::make_shared<Matrix>("3II slive", naux, npao_b);
        for(size_t q = 0; q < naux; q++) {
            for(size_t s = 0; s < npao_b; s++) {
                Qas_a->set(q, s, Qad[q]->get(a,s));
            }
        }
        Qas_a = linalg::doublet(Qas_a, XB_osv[b], false, false);

        //

        auto Qby_b = std::make_shared<Matrix>("3II slive", naux, nb);
        for(size_t q = 0; q < naux; q++) {
            for(size_t y = 0; y < nb; y++) {
                Qby_b->set(q, y, Qbb[q]->get(b,y));
            }
        }

        auto Qbr_b = std::make_shared<Matrix>("3II slive", naux, npao_a);
        for(size_t q = 0; q < naux; q++) {
            for(size_t r = 0; r < npao_a; r++) {
                Qbr_b->set(q, r, Qbc[q]->get(b,r));
            }
        }
        Qbr_b = linalg::doublet(Qbr_b, XA_osv[a], false, false);

        auto Qbs_b = std::make_shared<Matrix>("3II slive", naux, npao_b);
        for(size_t q = 0; q < naux; q++) {
            for(size_t s = 0; s < npao_b; s++) {
                Qbs_b->set(q, s, Qbd[q]->get(b,s));
            }
        }
        Qbs_b = linalg::doublet(Qbs_b, XB_osv[b], false, false);

        //

        auto Qxx = std::make_shared<Matrix>("3II slice", naux, na);
        for(size_t q = 0; q < naux; q++) {
            for(size_t x = 0; x < na; x++) {
                Qxx->set(q, x, Qaa[q]->get(x,x));
            }
        }

        auto Qyy = std::make_shared<Matrix>("3II slice", naux, nb);
        for(size_t q = 0; q < naux; q++) {
            for(size_t y = 0; y < nb; y++) {
                Qyy->set(q, y, Qbb[q]->get(y,y));
            }
        }

        //

        auto Qxb_b = std::make_shared<Matrix>("3II slice", naux, na);
        for(size_t q = 0; q < naux; q++) {
            for(size_t x = 0; x < na; x++) {
                Qxb_b->set(q, x, Qab[q]->get(x,b));
            }
        }

        auto Qya_a = std::make_shared<Matrix>("3II slice", naux, nb);
        for(size_t q = 0; q < naux; q++) {
            for(size_t y = 0; y < nb; y++) {
                Qya_a->set(q, y, Qab[q]->get(a,y));
            }
        }

        timer_off("Copying");

        timer_on("Copying2");

        //std::vector<SharedMatrix> rQa(nr), sQb(ns);
        std::vector<SharedMatrix> rQa(nosv_a), sQb(nosv_b);
        std::vector<SharedMatrix> aQr(na), bQs(nb);

        //for(size_t r = 0; r < nr; r++) {
        for(size_t r = 0; r < nosv_a; r++) {
            rQa[r] = std::make_shared<Matrix>("(ar|Q)_r", naux, na);
            for(size_t q = 0, qx = 0; q < naux; q++) {
                for(size_t x = 0; x < na; x++, qx++) {
                    //rQa[r]->set(q, x, Qar[q]->get(x, r));
                    rQa[r]->set(q, x, Qac_rect->get(qx, r));
                }
            }
        }

        for(size_t x = 0; x < na; x++) {
            //aQr[x] = std::make_shared<Matrix>("(ar|Q)_a", naux, nr);
            aQr[x] = std::make_shared<Matrix>("(ar|Q)_a", naux, nosv_a);
            for(size_t q = 0; q < naux; q++) {
                //for(size_t r = 0; r < nr; r++) {
                size_t qx = q * na + x;
                for(size_t r = 0; r < nosv_a; r++) {
                    aQr[x]->set(q, r, Qac_rect->get(qx, r));
                }
            }
        }

        //for(size_t s = 0; s < ns; s++) {
        for(size_t s = 0; s < nosv_b; s++) {
            sQb[s] = std::make_shared<Matrix>("(bs|Q)_s", naux, nb);
            for(size_t q = 0, qy=0; q < naux; q++) {
                for(size_t y = 0; y < nb; y++, qy++) {
                    //sQb[s]->set(q, y, Qbs[q]->get(y, s));
                    sQb[s]->set(q, y, Qbd_rect->get(qy, s));
                }
            }
        }

        for(size_t y = 0; y < nb; y++) {
            //bQs[y] = std::make_shared<Matrix>("(bs|Q)_b", naux, ns);
            bQs[y] = std::make_shared<Matrix>("(bs|Q)_b", naux, nosv_b);
            for(size_t q = 0; q < naux; q++) {
                size_t qy = q * nb + y;
                //for(size_t s = 0; s < ns; s++) {
                for(size_t s = 0; s < nosv_b; s++) {
                    //bQs[y]->set(q, s, Qbs[q]->get(y, s));
                    bQs[y]->set(q, s, Qbd_rect->get(qy, s));
                }
            }
        }

        timer_off("Copying2");

        timer_on("Inverting");
        auto local_met = met->clone();
        auto Jas_a = Qas_a->clone();
        C_DGESV_wrapper(local_met, Jas_a);

        local_met = met->clone();
        auto Jx = Qxx->clone()->collapse(1);
        C_DGESV_wrapper(local_met, Jx);

        local_met = met->clone();
        auto Jy = Qyy->clone()->collapse(1);
        C_DGESV_wrapper(local_met, Jy);

        local_met = met->clone();
        auto Jxb_b = Qxb_b->clone();
        C_DGESV_wrapper(local_met, Jxb_b);

        local_met = met->clone();
        auto Jya_a = Qya_a->clone();
        C_DGESV_wrapper(local_met, Jya_a);

        local_met = met->clone();
        auto Jby_b = Qby_b->clone();
        C_DGESV_wrapper(local_met, Jby_b);

        local_met = met->clone();
        auto Jax_a = Qax_a->clone();
        C_DGESV_wrapper(local_met, Jax_a);

        local_met = met->clone();
        auto Jay_a = Qay_a->clone();
        C_DGESV_wrapper(local_met, Jay_a);

        local_met = met->clone();
        auto Jar_a = Qar_a->clone();
        C_DGESV_wrapper(local_met, Jar_a);
        timer_off("Inverting");


        SharedVector tempv, tempv2;
        SharedMatrix temp, temp2;

        auto Qy = Qyy->collapse(1); // (naux x nb) -> (naux x 1)

        // part 1

        timer_on("Part 1");
        auto vab_sr = linalg::doublet(Jas_a, Qbr_b, true, false);
        timer_off("Part 1");


        // part 2

        timer_on("Part 2");
        tempv = C_DGEMV_wrapper(Qbr_b, Jx->get_column(0,0), true);
        tempv->scale(2.0);
        //for(size_t r = 0; r < nr; r++) {
        for(size_t r = 0; r < nosv_a; r++) {
            tempv->add(r, -1.0 * rQa[r]->vector_dot(Jxb_b));
        }
        C_DGER_wrapper(vab_sr, S_as_a, tempv, 1.0);

        temp = linalg::doublet(Jxb_b, Qar_a, true, false); // na x nr
        temp2 = linalg::doublet(Jax_a, Qbr_b, true, false); // na x nr
        temp->scale(2.0);
        temp->subtract(temp2);
        temp = linalg::doublet(S_as, temp, true, false);
        vab_sr->add(temp);
        timer_off("Part 2");

        // part 3
        
        timer_on("Part 3");
        tempv = C_DGEMV_wrapper(Qas_a, Jy->get_column(0,0), true);
        tempv->scale(2.0);
        //for(size_t s = 0; s < ns; s++) {
        for(size_t s = 0; s < nosv_b; s++) {
            tempv->add(s, -1.0 * sQb[s]->vector_dot(Jya_a));
        }
        C_DGER_wrapper(vab_sr, tempv, S_br_b, 1.0);

        temp = linalg::doublet(Jay_a, Qbs_b, true, false); // nb x ns
        temp2 = linalg::doublet(Jby_b, Qas_a, true, false); // nb x ns
        temp->scale(2.0);
        temp->subtract(temp2);
        temp = linalg::doublet(S_br, temp, true, false);
        vab_sr->add(temp->transpose());
        timer_off("Part 3");

        // part 4

        timer_on("Part 4");
        temp = linalg::doublet(Jax_a, S_ab, false, false); // (naux x na) x (na x nb)
        //tempv = std::make_shared<Vector>("", ns);
        tempv = std::make_shared<Vector>("", nosv_b);
        //for(size_t s = 0; s < ns; s++) {
        for(size_t s = 0; s < nosv_b; s++) {
            tempv->set(s, +1.0 * sQb[s]->vector_dot(temp));
        }
        C_DGER_wrapper(vab_sr, tempv, S_br_b, +1.0);

        temp = linalg::triplet(S_ab, Jax_a, Qbs_b, true, true, false); // nb x ns
        temp = linalg::doublet(S_br, temp, true, false); // nr x ns
        temp->scale(-2.0);
        vab_sr->add(temp->transpose());

        //tempv = std::make_shared<Vector>("", ns);
        tempv = std::make_shared<Vector>("", nosv_b);
        //for(size_t s = 0; s < ns; s++) {
        for(size_t s = 0; s < nosv_b; s++) {
            tempv2 = C_DGEMV_wrapper(sQb[s], Jx->get_column(0, 0), true);
            tempv->set(s, tempv2->vector_dot(S_ab_a));
        }
        C_DGER_wrapper(vab_sr, tempv, S_br_b, -2.0);

        tempv = C_DGEMV_wrapper(Qbs_b, Jx->get_column(0, 0), true);
        tempv2 = C_DGEMV_wrapper(S_br, S_ab_a, true);
        C_DGER_wrapper(vab_sr, tempv, tempv2, 4.0);
        timer_off("Part 4");

        // part 5

        timer_on("Part 5");
        temp = linalg::doublet(Jby_b, S_ab, false, true); // (naux x nb) (nb x na)
        //tempv = std::make_shared<Vector>("", nr);
        tempv = std::make_shared<Vector>("", nosv_a);
        //for(size_t r = 0; r < nr; r++) {
        for(size_t r = 0; r < nosv_a; r++) {
            tempv->set(r, +1.0 * rQa[r]->vector_dot(temp));
        }
        C_DGER_wrapper(vab_sr, S_as_a, tempv, +1.0);

        temp = linalg::triplet(Jar_a, Qby_b, S_ab, true, false, true); //(nr x naux) (naux x nb) (nb x na)
        temp = linalg::doublet(S_as, temp, true, true) ; // (ns x na) (na x nr)
        temp->scale(-2.0);
        vab_sr->add(temp);

        //tempv = std::make_shared<Vector>("", nr);
        tempv = std::make_shared<Vector>("", nosv_a);
        //for(size_t r = 0; r < nr; r++) {
        for(size_t r = 0; r < nosv_a; r++) {
            tempv2 = C_DGEMV_wrapper(rQa[r], Jy->get_column(0, 0), true);
            tempv->set(r, tempv2->vector_dot(S_ab_b));
        }
        C_DGER_wrapper(vab_sr, S_as_a, tempv, -2.0);

        tempv = C_DGEMV_wrapper(Qar_a, Jy->get_column(0, 0), true);
        tempv2 = C_DGEMV_wrapper(S_as, S_ab_b, true);
        C_DGER_wrapper(vab_sr, tempv2, tempv, 4.0);
        timer_off("Part 5");

        // part 6
        
        timer_on("Part 6");
        temp = linalg::triplet(Jax_a, Qby_b, S_br, true, false, false); // (na x naux) (naux x nb) (nb x nr)
        temp = linalg::doublet(S_as, temp, true, false); // (ns x na) (na x nr)
        vab_sr->add(temp);

        tempv = C_DGEMV_wrapper(Jax_a, Qy->get_column(0, 0), true);
        tempv = C_DGEMV_wrapper(S_as, tempv, true);
        C_DGER_wrapper(vab_sr, tempv, S_br_b, -2.0);

        tempv = C_DGEMV_wrapper(Qby_b, Jx->get_column(0, 0), true);
        tempv = C_DGEMV_wrapper(S_br, tempv, true);
        C_DGER_wrapper(vab_sr, S_as_a, tempv, -2.0);
        timer_off("Part 6");

        // part 7
        
        timer_on("Part 7");
        //temp = std::make_shared<Matrix>("blah", naux, nr);
        temp = std::make_shared<Matrix>("blah", naux, nosv_a);
        for(size_t x = 0; x < na; x++) {
            temp->axpy(S_ab->get(x, b), aQr[x]);
        }

        timer_on("Inverse 1");
        local_met = met->clone();
        C_DGESV_wrapper(local_met, temp);
        timer_off("Inverse 1");

        //temp2 = std::make_shared<Matrix>("blah", naux, ns);
        temp2 = std::make_shared<Matrix>("blah", naux, nosv_b);
        for(size_t y = 0; y < nb; y++) {
            temp2->axpy(S_ab->get(a, y), bQs[y]);
        }

        temp = linalg::doublet(temp2, temp, true, false);
        vab_sr->add(temp);

        tempv = C_DGEMV_wrapper(S_ab, S_ab_b, true); // (a x b).T x (a)
        //temp2 = std::make_shared<Matrix>("blah", naux, ns);
        temp2 = std::make_shared<Matrix>("blah", naux, nosv_b);
        for(size_t y = 0; y < nb; y++) {
            temp2->axpy(tempv->get(y), bQs[y]);
        }
        temp = linalg::doublet(temp2, Jar_a, true, false);
        temp->scale(-2.0);
        vab_sr->add(temp);

        tempv = C_DGEMV_wrapper(S_ab, S_ab_a, false); // (a x b) x (b)
        //temp2 = std::make_shared<Matrix>("blah", naux, nr);
        temp2 = std::make_shared<Matrix>("blah", naux, nosv_a);
        for(size_t x = 0; x < na; x++) {
            temp2->axpy(tempv->get(x), aQr[x]);
        }

        // redo this one

        timer_on("Inverse 2");
        local_met = met->clone();
        C_DGESV_wrapper(local_met, temp2);
        timer_off("Inverse 2");

        temp = linalg::doublet(Qbs_b, temp2, true, false);
        temp->scale(-2.0);
        vab_sr->add(temp);
        timer_off("Part 7");

        // part 8

        timer_on("Part 8");
        tempv = C_DGEMV_wrapper(S_br, S_ab_a, true); // (b x r).T x (b)
        C_DGER_wrapper(vab_sr, VA_bs->get_row(0, b), tempv, 2.0);

        tempv = C_DGEMV_wrapper(S_as, S_ab_b, true);
        C_DGER_wrapper(vab_sr, tempv, VB_ar->get_row(0, a), 2.0);

        tempv = C_DGEMV_wrapper(VA_bs, S_ab_a, true);
        C_DGER_wrapper(vab_sr, tempv, S_br_b, -1.0);

        tempv = C_DGEMV_wrapper(VB_ar, S_ab_b, true);
        C_DGER_wrapper(vab_sr, S_as_a, tempv, -1.0);

        tempv = C_DGEMV_wrapper(S_rs, VA_bs->get_row(0, b), false);
        C_DGER_wrapper(vab_sr, S_as_a, tempv, +1.0);

        tempv = C_DGEMV_wrapper(S_rs, VB_ar->get_row(0, a), true);
        C_DGER_wrapper(vab_sr, tempv, S_br_b, +1.0);
        timer_off("Part 8");

        e_exchdisp += -2.0 * t_abrs[ab]->vector_dot(vab_sr->transpose());

    }

    timer_off("ExchDisp");

    outfile->Printf("  !! Exch Disp: %.8f\n", e_exchdisp * pc_hartree2kcalmol);
    outfile->Printf("  !! Disp + Exch Disp: %.8f\n", (disp_tot + de_dipole_ + e_exchdisp) * pc_hartree2kcalmol);

    return;

    double Disp20 = 0.0;
    double ExchDisp20 = 0.0;

    scalars_["Disp20"] = Disp20;
    scalars_["Exch-Disp20"] = ExchDisp20;
    if (do_print) {
        outfile->Printf("    Disp20              = %18.12lf [Eh]\n", Disp20);
        outfile->Printf("    Exch-Disp20         = %18.12lf [Eh]\n", ExchDisp20);
        outfile->Printf("\n");
    }
}

void FISAPT::print_trailer() {
    scalars_["Electrostatics"] = scalars_["Elst10,r"];
    scalars_["Exchange"] = scalars_["Exch10"];
    scalars_["Induction"] = scalars_["Ind20,r"] + scalars_["Exch-Ind20,r"] + scalars_["delta HF,r (2)"];
    scalars_["sInduction"] = scalars_["Ind20,r"] + scalars_["sExch-Ind20,r"] + scalars_["delta HF,r (2)"];
    scalars_["Dispersion"] = scalars_["Disp20"] + scalars_["Exch-Disp20"];
    scalars_["sDispersion"] = scalars_["Disp20"] + scalars_["sExch-Disp20"];
    scalars_["SAPT"] =
        scalars_["Electrostatics"] + scalars_["Exchange"] + scalars_["Induction"] + scalars_["Dispersion"];
    scalars_["sSAPT"] =
        scalars_["Electrostatics"] + scalars_["Exchange"] + scalars_["sInduction"] + scalars_["sDispersion"];

    double Sdelta = scalars_["Induction"] / (scalars_["Ind20,r"] + scalars_["Exch-Ind20,r"]);
    scalars_["Induction (A<-B)"] = Sdelta * (scalars_["Ind20,r (A<-B)"] + scalars_["Exch-Ind20,r (A<-B)"]);
    scalars_["Induction (B<-A)"] = Sdelta * (scalars_["Ind20,r (B<-A)"] + scalars_["Exch-Ind20,r (B<-A)"]);

    outfile->Printf("  ==> Results <==\n\n");

    outfile->Printf("\n    SAPT Results  \n");
    std::string scaled = "   ";
    outfile->Printf(
        "  --------------------------------------------------------------------------------------------------------\n");
    outfile->Printf("    Electrostatics            %16.8lf [mEh] %16.8lf [kcal/mol] %16.8lf [kJ/mol]\n",
                    scalars_["Electrostatics"] * 1000.0, scalars_["Electrostatics"] * pc_hartree2kcalmol,
                    scalars_["Electrostatics"] * pc_hartree2kJmol);
    outfile->Printf("      Elst10,r                %16.8lf [mEh] %16.8lf [kcal/mol] %16.8lf [kJ/mol]\n\n",
                    scalars_["Elst10,r"] * 1000.0, scalars_["Elst10,r"] * pc_hartree2kcalmol,
                    scalars_["Elst10,r"] * pc_hartree2kJmol);

    outfile->Printf("    Exchange %3s              %16.8lf [mEh] %16.8lf [kcal/mol] %16.8lf [kJ/mol]\n", scaled.c_str(),
                    scalars_["Exchange"] * 1000.0, scalars_["Exchange"] * pc_hartree2kcalmol,
                    scalars_["Exchange"] * pc_hartree2kJmol);
    outfile->Printf("      Exch10                  %16.8lf [mEh] %16.8lf [kcal/mol] %16.8lf [kJ/mol]\n",
                    scalars_["Exch10"] * 1000.0, scalars_["Exch10"] * pc_hartree2kcalmol,
                    scalars_["Exch10"] * pc_hartree2kJmol);
    outfile->Printf("      Exch10(S^2)             %16.8lf [mEh] %16.8lf [kcal/mol] %16.8lf [kJ/mol]\n\n",
                    scalars_["Exch10(S^2)"] * 1000.0, scalars_["Exch10(S^2)"] * pc_hartree2kcalmol,
                    scalars_["Exch10(S^2)"] * pc_hartree2kJmol);

    outfile->Printf("    Induction %3s             %16.8lf [mEh] %16.8lf [kcal/mol] %16.8lf [kJ/mol]\n", scaled.c_str(),
                    scalars_["Induction"] * 1000.0, scalars_["Induction"] * pc_hartree2kcalmol,
                    scalars_["Induction"] * pc_hartree2kJmol);
    outfile->Printf("      Ind20,r                 %16.8lf [mEh] %16.8lf [kcal/mol] %16.8lf [kJ/mol]\n",
                    scalars_["Ind20,r"] * 1000.0, scalars_["Ind20,r"] * pc_hartree2kcalmol,
                    scalars_["Ind20,r"] * pc_hartree2kJmol);
    outfile->Printf("      Exch-Ind20,r %3s        %16.8lf [mEh] %16.8lf [kcal/mol] %16.8lf [kJ/mol]\n", scaled.c_str(),
                    scalars_["Exch-Ind20,r"] * 1000.0, scalars_["Exch-Ind20,r"] * pc_hartree2kcalmol,
                    scalars_["Exch-Ind20,r"] * pc_hartree2kJmol);
    outfile->Printf("      delta HF,r (2) %3s      %16.8lf [mEh] %16.8lf [kcal/mol] %16.8lf [kJ/mol]\n", scaled.c_str(),
                    scalars_["delta HF,r (2)"] * 1000.0, scalars_["delta HF,r (2)"] * pc_hartree2kcalmol,
                    scalars_["delta HF,r (2)"] * pc_hartree2kJmol);
    outfile->Printf("      Induction (A<-B) %3s    %16.8lf [mEh] %16.8lf [kcal/mol] %16.8lf [kJ/mol]\n", scaled.c_str(),
                    scalars_["Induction (A<-B)"] * 1000.0, scalars_["Induction (A<-B)"] * pc_hartree2kcalmol,
                    scalars_["Induction (A<-B)"] * pc_hartree2kJmol);
    outfile->Printf("      Induction (B<-A) %3s    %16.8lf [mEh] %16.8lf [kcal/mol] %16.8lf [kJ/mol]\n\n",
                    scaled.c_str(), scalars_["Induction (B<-A)"] * 1000.0,
                    scalars_["Induction (B<-A)"] * pc_hartree2kcalmol, scalars_["Induction (B<-A)"] * pc_hartree2kJmol);

    outfile->Printf("    Dispersion %3s            %16.8lf [mEh] %16.8lf [kcal/mol] %16.8lf [kJ/mol]\n", scaled.c_str(),
                    scalars_["Dispersion"] * 1000.0, scalars_["Dispersion"] * pc_hartree2kcalmol,
                    scalars_["Dispersion"] * pc_hartree2kJmol);
    outfile->Printf("      Disp20                  %16.8lf [mEh] %16.8lf [kcal/mol] %16.8lf [kJ/mol]\n",
                    scalars_["Disp20"] * 1000.0, scalars_["Disp20"] * pc_hartree2kcalmol,
                    scalars_["Disp20"] * pc_hartree2kJmol);
    outfile->Printf("      Exch-Disp20 %3s         %16.8lf [mEh] %16.8lf [kcal/mol] %16.8lf [kJ/mol]\n\n",
                    scaled.c_str(), scalars_["Exch-Disp20"] * 1000.0, scalars_["Exch-Disp20"] * pc_hartree2kcalmol,
                    scalars_["Exch-Disp20"] * pc_hartree2kJmol);

    outfile->Printf("  Total HF                    %16.8lf [mEh] %16.8lf [kcal/mol] %16.8lf [kJ/mol]\n",
                    scalars_["HF"] * 1000.0, scalars_["HF"] * pc_hartree2kcalmol, scalars_["HF"] * pc_hartree2kJmol);
    outfile->Printf("  Total SAPT0 %3s             %16.8lf [mEh] %16.8lf [kcal/mol] %16.8lf [kJ/mol]\n", scaled.c_str(),
                    scalars_["SAPT"] * 1000.0, scalars_["SAPT"] * pc_hartree2kcalmol,
                    scalars_["SAPT"] * pc_hartree2kJmol);
    if (options_.get_bool("SSAPT0_SCALE")) {
        outfile->Printf("  Total sSAPT0 %3s            %16.8lf [mEh] %16.8lf [kcal/mol] %16.8lf [kJ/mol]\n",
                        scaled.c_str(), scalars_["sSAPT"] * 1000.0, scalars_["sSAPT"] * pc_hartree2kcalmol,
                        scalars_["sSAPT"] * pc_hartree2kJmol);
    }
    outfile->Printf("\n");
    outfile->Printf(
        "  --------------------------------------------------------------------------------------------------------\n");

    outfile->Printf("    Han Solo: This is *not* gonna work.\n");
    outfile->Printf("    Luke Skywalker: Why didn't you say so before?\n");
    outfile->Printf("    Han Solo: I *did* say so before.\n");

    Process::environment.globals["SAPT ELST ENERGY"] = scalars_["Electrostatics"];
    Process::environment.globals["SAPT ELST10,R ENERGY"] = scalars_["Elst10,r"];

    Process::environment.globals["SAPT EXCH ENERGY"] = scalars_["Exchange"];
    Process::environment.globals["SAPT EXCH10 ENERGY"] = scalars_["Exch10"];
    Process::environment.globals["SAPT EXCH10(S^2) ENERGY"] = scalars_["Exch10(S^2)"];

    Process::environment.globals["SAPT IND ENERGY"] = scalars_["Induction"];
    Process::environment.globals["SAPT IND20,R ENERGY"] = scalars_["Ind20,r"];
    Process::environment.globals["SAPT EXCH-IND20,R ENERGY"] = scalars_["Exch-Ind20,r"];
    Process::environment.globals["SAPT IND20,U ENERGY"] = scalars_["Ind20,u"];
    Process::environment.globals["SAPT EXCH-IND20,U ENERGY"] = scalars_["Exch-Ind20,u"];

    Process::environment.globals["SAPT DISP ENERGY"] = scalars_["Dispersion"];
    Process::environment.globals["SAPT DISP20 ENERGY"] = scalars_["Disp20"];
    Process::environment.globals["SAPT EXCH-DISP20 ENERGY"] = scalars_["Exch-Disp20"];

    Process::environment.globals["SAPT0 TOTAL ENERGY"] = scalars_["SAPT"];
    Process::environment.globals["SAPT TOTAL ENERGY"] = scalars_["SAPT"];
    Process::environment.globals["CURRENT ENERGY"] = Process::environment.globals["SAPT TOTAL ENERGY"];

    // Export the components of dHF to Psi4 variables
    Process::environment.globals["SAPT HF(2) ENERGY ABC(HF)"] = scalars_["E_ABC_HF"];
    Process::environment.globals["SAPT HF(2) ENERGY AC(0)"] = scalars_["E_AC"];
    Process::environment.globals["SAPT HF(2) ENERGY BC(0)"] = scalars_["E_BC"];
    Process::environment.globals["SAPT HF(2) ENERGY A(0)"] = scalars_["E_A"];
    Process::environment.globals["SAPT HF(2) ENERGY B(0)"] = scalars_["E_B"];
    Process::environment.globals["SAPT HF(2) ENERGY AC(HF)"] = scalars_["E_AC_HF"];
    Process::environment.globals["SAPT HF(2) ENERGY BC(HF)"] = scalars_["E_BC_HF"];
    Process::environment.globals["SAPT HF(2) ENERGY AB(HF)"] = scalars_["E_AB_HF"];
    Process::environment.globals["SAPT HF(2) ENERGY A(HF)"] = scalars_["E_A_HF"];
    Process::environment.globals["SAPT HF(2) ENERGY B(HF)"] = scalars_["E_B_HF"];
    Process::environment.globals["SAPT HF(2) ENERGY C"] = scalars_["E_C"];
    Process::environment.globals["SAPT HF(2) ENERGY HF"] = scalars_["HF"];
}

void FISAPT::raw_plot(const std::string& filepath) {
    outfile->Printf("  ==> Scalar Field Plots <==\n\n");

    outfile->Printf("    F-SAPT Plot Filepath = %s\n\n", filepath.c_str());

    auto csg = std::make_shared<CubicScalarGrid>(primary_, options_);
    csg->set_filepath(filepath);
    csg->print_header();
    csg->set_auxiliary_basis(reference_->get_basisset("DF_BASIS_SCF"));

    /// Zeroth-order wavefunctions
    std::shared_ptr<Matrix> D_A = matrices_["D_A"];
    std::shared_ptr<Matrix> D_B = matrices_["D_B"];
    std::shared_ptr<Matrix> D_C = matrices_["D_C"];

    /// Fully interacting wavefunctions
    std::shared_ptr<Matrix> DFA = linalg::doublet(matrices_["LoccA"], matrices_["LoccA"], false, true);
    std::shared_ptr<Matrix> DFB = linalg::doublet(matrices_["LoccB"], matrices_["LoccB"], false, true);

    // => Density Fields <= //

    csg->compute_density(D_A, "DA");
    csg->compute_density(D_B, "DB");
    csg->compute_density(D_C, "DC");
    csg->compute_density(DFA, "DFA");
    csg->compute_density(DFB, "DFB");

    // => Difference Density Fields <= //

    DFA->subtract(D_A);
    DFB->subtract(D_B);

    csg->compute_density(DFA, "dDA");
    csg->compute_density(DFB, "dDB");

    // => ESP Fields <= //

    double* ZAp = vectors_["ZA"]->pointer();
    double* ZBp = vectors_["ZB"]->pointer();
    double* ZCp = vectors_["ZC"]->pointer();

    std::shared_ptr<Molecule> mol = primary_->molecule();

    std::vector<double> w_A(mol->natom());
    std::vector<double> w_B(mol->natom());
    std::vector<double> w_C(mol->natom());

    for (int A = 0; A < mol->natom(); A++) {
        w_A[A] = (ZAp[A]) / mol->Z(A);
        w_B[A] = (ZBp[A]) / mol->Z(A);
        w_C[A] = (ZCp[A]) / mol->Z(A);
    }

    D_A->scale(2.0);
    D_B->scale(2.0);
    D_C->scale(2.0);

    csg->compute_esp(D_A, w_A, "VA");
    csg->compute_esp(D_B, w_B, "VB");
    csg->compute_esp(D_C, w_C, "VC");

    D_A->scale(0.5);
    D_B->scale(0.5);
    D_C->scale(0.5);
}

void FISAPT::flocalize() {
    outfile->Printf("  ==> F-SAPT Localization (IBO) <==\n\n");

    // Currently always separating core and valence
    {
        outfile->Printf("  Local Orbitals for Monomer A:\n\n");

        int nn = matrices_["Caocc0A"]->rowspi()[0];
        int nf = matrices_["Cfocc0A"]->colspi()[0];
        int na = matrices_["Caocc0A"]->colspi()[0];
        int nm = nf + na;

        std::vector<int> ranges;
        ranges.push_back(0);
        ranges.push_back(nf);
        ranges.push_back(nm);

        std::shared_ptr<Matrix> Focc(
            new Matrix("Focc", vectors_["eps_occ0A"]->dimpi()[0], vectors_["eps_occ0A"]->dimpi()[0]));
        Focc->set_diagonal(vectors_["eps_occ0A"]);

        std::shared_ptr<fisapt::IBOLocalizer2> local =
            fisapt::IBOLocalizer2::build(primary_, reference_->get_basisset("MINAO"), matrices_["Cocc0A"], options_);
        local->print_header();
        std::map<std::string, std::shared_ptr<Matrix> > ret = local->localize(matrices_["Cocc0A"], Focc, ranges);

        matrices_["Locc0A"] = ret["L"];
        matrices_["Uocc0A"] = ret["U"];
        matrices_["Qocc0A"] = ret["Q"];

        matrices_["Lfocc0A"] = std::make_shared<Matrix>("Lfocc0A", nn, nf);
        matrices_["Laocc0A"] = std::make_shared<Matrix>("Laocc0A", nn, na);
        matrices_["Ufocc0A"] = std::make_shared<Matrix>("Ufocc0A", nf, nf);
        matrices_["Uaocc0A"] = std::make_shared<Matrix>("Uaocc0A", na, na);

        double** Lp = matrices_["Locc0A"]->pointer();
        double** Lfp = matrices_["Lfocc0A"]->pointer();
        double** Lap = matrices_["Laocc0A"]->pointer();
        double** Up = matrices_["Uocc0A"]->pointer();
        double** Ufp = matrices_["Ufocc0A"]->pointer();
        double** Uap = matrices_["Uaocc0A"]->pointer();

        for (int n = 0; n < nn; n++) {
            for (int i = 0; i < nf; i++) {
                Lfp[n][i] = Lp[n][i];
            }
            for (int i = 0; i < na; i++) {
                Lap[n][i] = Lp[n][i + nf];
            }
        }

        for (int i = 0; i < nf; i++) {
            for (int j = 0; j < nf; j++) {
                Ufp[i][j] = Up[i][j];
            }
        }

        for (int i = 0; i < na; i++) {
            for (int j = 0; j < na; j++) {
                Uap[i][j] = Up[i + nf][j + nf];
            }
        }

        matrices_["Locc0A"]->set_name("Locc0A");
        matrices_["Lfocc0A"]->set_name("Lfocc0A");
        matrices_["Laocc0A"]->set_name("Laocc0A");
        matrices_["Uocc0A"]->set_name("Uocc0A");
        matrices_["Ufocc0A"]->set_name("Ufocc0A");
        matrices_["Uaocc0A"]->set_name("Uaocc0A");
        matrices_["Qocc0A"]->set_name("Qocc0A");
    }

    {
        outfile->Printf("  Local Orbitals for Monomer B:\n\n");

        int nn = matrices_["Caocc0B"]->rowspi()[0];
        int nf = matrices_["Cfocc0B"]->colspi()[0];
        int na = matrices_["Caocc0B"]->colspi()[0];
        int nm = nf + na;

        std::vector<int> ranges;
        ranges.push_back(0);
        ranges.push_back(nf);
        ranges.push_back(nm);

        std::shared_ptr<Matrix> Focc(
            new Matrix("Focc", vectors_["eps_occ0B"]->dimpi()[0], vectors_["eps_occ0B"]->dimpi()[0]));
        Focc->set_diagonal(vectors_["eps_occ0B"]);

        std::shared_ptr<fisapt::IBOLocalizer2> local =
            fisapt::IBOLocalizer2::build(primary_, reference_->get_basisset("MINAO"), matrices_["Cocc0B"], options_);
        local->print_header();
        std::map<std::string, std::shared_ptr<Matrix> > ret = local->localize(matrices_["Cocc0B"], Focc, ranges);

        matrices_["Locc0B"] = ret["L"];
        matrices_["Uocc0B"] = ret["U"];
        matrices_["Qocc0B"] = ret["Q"];

        matrices_["Lfocc0B"] = std::make_shared<Matrix>("Lfocc0B", nn, nf);
        matrices_["Laocc0B"] = std::make_shared<Matrix>("Laocc0B", nn, na);
        matrices_["Ufocc0B"] = std::make_shared<Matrix>("Ufocc0B", nf, nf);
        matrices_["Uaocc0B"] = std::make_shared<Matrix>("Uaocc0B", na, na);

        double** Lp = matrices_["Locc0B"]->pointer();
        double** Lfp = matrices_["Lfocc0B"]->pointer();
        double** Lap = matrices_["Laocc0B"]->pointer();
        double** Up = matrices_["Uocc0B"]->pointer();
        double** Ufp = matrices_["Ufocc0B"]->pointer();
        double** Uap = matrices_["Uaocc0B"]->pointer();

        for (int n = 0; n < nn; n++) {
            for (int i = 0; i < nf; i++) {
                Lfp[n][i] = Lp[n][i];
            }
            for (int i = 0; i < na; i++) {
                Lap[n][i] = Lp[n][i + nf];
            }
        }

        for (int i = 0; i < nf; i++) {
            for (int j = 0; j < nf; j++) {
                Ufp[i][j] = Up[i][j];
            }
        }

        for (int i = 0; i < na; i++) {
            for (int j = 0; j < na; j++) {
                Uap[i][j] = Up[i + nf][j + nf];
            }
        }

        matrices_["Locc0B"]->set_name("Locc0B");
        matrices_["Lfocc0B"]->set_name("Lfocc0B");
        matrices_["Laocc0B"]->set_name("Laocc0B");
        matrices_["Uocc0B"]->set_name("Uocc0B");
        matrices_["Ufocc0B"]->set_name("Ufocc0B");
        matrices_["Uaocc0B"]->set_name("Uaocc0B");
        matrices_["Qocc0B"]->set_name("Qocc0B");
    }
}

// Compute fragment-fragment partitioning of electrostatic contribution
void FISAPT::felst() {
    outfile->Printf("  ==> F-SAPT Electrostatics <==\n\n");

    // => Sizing <= //

    std::shared_ptr<Molecule> mol = primary_->molecule();
    int nn = primary_->nbf();
    int nA = mol->natom();
    int nB = mol->natom();
    int na = matrices_["Locc0A"]->colspi()[0];
    int nb = matrices_["Locc0B"]->colspi()[0];

    // => Targets <= //

    double Elst10 = 0.0;
    std::vector<double> Elst10_terms;
    Elst10_terms.resize(4);

    matrices_["Elst_AB"] = std::make_shared<Matrix>("Elst_AB", nA + na, nB + nb);
    double** Ep = matrices_["Elst_AB"]->pointer();

    // => A <-> B <= //

    double* ZAp = vectors_["ZA"]->pointer();
    double* ZBp = vectors_["ZB"]->pointer();
    for (int A = 0; A < nA; A++) {
        for (int B = 0; B < nB; B++) {
            if (A == B) continue;
            double E = ZAp[A] * ZBp[B] / mol->xyz(A).distance(mol->xyz(B));
            Ep[A][B] += E;
            Elst10_terms[3] += E;
        }
    }

    // => a <-> b <= //

    int nT = 1;
#ifdef _OPENMP
    nT = Process::environment.get_n_threads();
#endif

    // => Get integrals from DFHelper <= //
    dfh_ = std::make_shared<DFHelper>(primary_, reference_->get_basisset("DF_BASIS_SCF"));
    dfh_->set_memory(doubles_);
    dfh_->set_method("DIRECT_iaQ");
    dfh_->set_nthreads(nT);
    dfh_->initialize();
    dfh_->print_header();

    dfh_->add_space("a", matrices_["Locc0A"]);
    dfh_->add_space("b", matrices_["Locc0B"]);

    dfh_->add_transformation("Aaa", "a", "a");
    dfh_->add_transformation("Abb", "b", "b");

    dfh_->transform();

    size_t nQ = dfh_->get_naux();
    auto QaC = std::make_shared<Matrix>("QaC", na, nQ);
    double** QaCp = QaC->pointer();
    for (size_t a = 0; a < na; a++) {
        dfh_->fill_tensor("Aaa", QaCp[a], {a, a + 1}, {a, a + 1});
    }

    auto QbC = std::make_shared<Matrix>("QbC", nb, nQ);
    double** QbCp = QbC->pointer();
    for (size_t b = 0; b < nb; b++) {
        dfh_->fill_tensor("Abb", QbCp[b], {b, b + 1}, {b, b + 1});
    }

    std::shared_ptr<Matrix> Elst10_3 = linalg::doublet(QaC, QbC, false, true);
    double** Elst10_3p = Elst10_3->pointer();
    for (int a = 0; a < na; a++) {
        for (int b = 0; b < nb; b++) {
            double E = 4.0 * Elst10_3p[a][b];
            Elst10_terms[2] += E;
            Ep[a + nA][b + nB] += E;
        }
    }

    matrices_["Vlocc0A"] = QaC;
    matrices_["Vlocc0B"] = QbC;

    // => Nuclear Part (PITA) <= //

    auto Zxyz2 = std::make_shared<Matrix>("Zxyz", 1, 4);
    double** Zxyz2p = Zxyz2->pointer();
    auto Vfact2 = std::make_shared<IntegralFactory>(primary_);
    std::shared_ptr<PotentialInt> Vint2(static_cast<PotentialInt*>(Vfact2->ao_potential()));
    Vint2->set_charge_field(Zxyz2);
    auto Vtemp2 = std::make_shared<Matrix>("Vtemp2", nn, nn);

    // => A <-> b <= //

    for (int A = 0; A < nA; A++) {
        if (ZAp[A] == 0.0) continue;
        Vtemp2->zero();
        Zxyz2p[0][0] = ZAp[A];
        Zxyz2p[0][1] = mol->x(A);
        Zxyz2p[0][2] = mol->y(A);
        Zxyz2p[0][3] = mol->z(A);
        Vint2->compute(Vtemp2);
        std::shared_ptr<Matrix> Vbb =
            linalg::triplet(matrices_["Locc0B"], Vtemp2, matrices_["Locc0B"], true, false, false);
        double** Vbbp = Vbb->pointer();
        for (int b = 0; b < nb; b++) {
            double E = 2.0 * Vbbp[b][b];
            Elst10_terms[1] += E;
            Ep[A][b + nB] += E;
        }
    }

    // => a <-> B <= //

    for (int B = 0; B < nB; B++) {
        if (ZBp[B] == 0.0) continue;
        Vtemp2->zero();
        Zxyz2p[0][0] = ZBp[B];
        Zxyz2p[0][1] = mol->x(B);
        Zxyz2p[0][2] = mol->y(B);
        Zxyz2p[0][3] = mol->z(B);
        Vint2->compute(Vtemp2);
        std::shared_ptr<Matrix> Vaa =
            linalg::triplet(matrices_["Locc0A"], Vtemp2, matrices_["Locc0A"], true, false, false);
        double** Vaap = Vaa->pointer();
        for (int a = 0; a < na; a++) {
            double E = 2.0 * Vaap[a][a];
            Elst10_terms[0] += E;
            Ep[a + nA][B] += E;
        }
    }

    // Prepare DFHelper object for the next module
    dfh_->clear_spaces();

    // => Summation <= //

    for (int k = 0; k < Elst10_terms.size(); k++) {
        Elst10 += Elst10_terms[k];
    }
    // for (int k = 0; k < Elst10_terms.size(); k++) {
    //    outfile->Printf("    Elst10,r (%1d)        = %18.12lf [Eh]\n",k+1,Elst10_terms[k]);
    //}
    // scalars_["Elst10,r"] = Elst10;
    outfile->Printf("    Elst10,r            = %18.12lf [Eh]\n", Elst10);
    outfile->Printf("\n");
    // fflush(outfile);
}

// Compute fragment-fragment partitioning of exchange contribution
void FISAPT::fexch() {
    outfile->Printf("  ==> F-SAPT Exchange <==\n\n");

    // => Sizing <= //

    std::shared_ptr<Molecule> mol = primary_->molecule();
    int nn = primary_->nbf();
    int nA = mol->natom();
    int nB = mol->natom();
    int na = matrices_["Locc0A"]->colspi()[0];
    int nb = matrices_["Locc0B"]->colspi()[0];
    int nr = matrices_["Cvir0A"]->colspi()[0];
    int ns = matrices_["Cvir0B"]->colspi()[0];

    // => Targets <= //

    double Exch10_2 = 0.0;
    std::vector<double> Exch10_2_terms;
    Exch10_2_terms.resize(3);

    matrices_["Exch_AB"] = std::make_shared<Matrix>("Exch_AB", nA + na, nB + nb);
    double** Ep = matrices_["Exch_AB"]->pointer();

    // ==> Stack Variables <== //

    std::shared_ptr<Matrix> S = matrices_["S"];
    std::shared_ptr<Matrix> V_A = matrices_["V_A"];
    std::shared_ptr<Matrix> J_A = matrices_["J_A"];
    std::shared_ptr<Matrix> V_B = matrices_["V_B"];
    std::shared_ptr<Matrix> J_B = matrices_["J_B"];

    std::shared_ptr<Matrix> LoccA = matrices_["Locc0A"];
    std::shared_ptr<Matrix> LoccB = matrices_["Locc0B"];
    std::shared_ptr<Matrix> CvirA = matrices_["Cvir0A"];
    std::shared_ptr<Matrix> CvirB = matrices_["Cvir0B"];

    // ==> DF ERI Setup (JKFIT Type, in Full Basis) <== //

    int nT = 1;
#ifdef _OPENMP
    nT = Process::environment.get_n_threads();
#endif

    std::vector<std::shared_ptr<Matrix> > Cs;
    Cs.push_back(LoccA);
    Cs.push_back(CvirA);
    Cs.push_back(LoccB);
    Cs.push_back(CvirB);

    size_t max_MO = 0;
    for (auto& mat : Cs) max_MO = std::max(max_MO, (size_t)mat->ncol());

    dfh_->add_space("a", Cs[0]);
    dfh_->add_space("r", Cs[1]);
    dfh_->add_space("b", Cs[2]);
    dfh_->add_space("s", Cs[3]);

    dfh_->add_transformation("Aar", "a", "r");
    dfh_->add_transformation("Abs", "b", "s");

    dfh_->transform();

    // ==> Electrostatic Potentials <== //

    std::shared_ptr<Matrix> W_A(J_A->clone());
    W_A->copy(J_A);
    W_A->scale(2.0);
    W_A->add(V_A);

    std::shared_ptr<Matrix> W_B(J_B->clone());
    W_B->copy(J_B);
    W_B->scale(2.0);
    W_B->add(V_B);

    std::shared_ptr<Matrix> WAbs = linalg::triplet(LoccB, W_A, CvirB, true, false, false);
    std::shared_ptr<Matrix> WBar = linalg::triplet(LoccA, W_B, CvirA, true, false, false);
    double** WBarp = WBar->pointer();
    double** WAbsp = WAbs->pointer();

    W_A.reset();
    W_B.reset();

    // ==> Exchange S^2 Computation <== //

    std::shared_ptr<Matrix> Sab = linalg::triplet(LoccA, S, LoccB, true, false, false);
    std::shared_ptr<Matrix> Sba = linalg::triplet(LoccB, S, LoccA, true, false, false);
    std::shared_ptr<Matrix> Sas = linalg::triplet(LoccA, S, CvirB, true, false, false);
    std::shared_ptr<Matrix> Sbr = linalg::triplet(LoccB, S, CvirA, true, false, false);
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

    // E_exch1->print();
    // E_exch2->print();

    size_t nQ = dfh_->get_naux();
    auto TrQ = std::make_shared<Matrix>("TrQ", nr, nQ);
    double** TrQp = TrQ->pointer();
    auto TsQ = std::make_shared<Matrix>("TsQ", ns, nQ);
    double** TsQp = TsQ->pointer();
    auto TbQ = std::make_shared<Matrix>("TbQ", nb, nQ);
    double** TbQp = TbQ->pointer();
    auto TaQ = std::make_shared<Matrix>("TaQ", na, nQ);
    double** TaQp = TaQ->pointer();

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

    // E_exch3->print();

    // => Totals <= //

    for (int a = 0; a < na; a++) {
        for (int b = 0; b < nb; b++) {
            Ep[a + nA][b + nB] = E_exch1p[a][b] + E_exch2p[a][b] + E_exch3p[a][b];
            Exch10_2_terms[0] += E_exch1p[a][b];
            Exch10_2_terms[1] += E_exch2p[a][b];
            Exch10_2_terms[2] += E_exch3p[a][b];
        }
    }

    for (int k = 0; k < Exch10_2_terms.size(); k++) {
        Exch10_2 += Exch10_2_terms[k];
    }
    // for (int k = 0; k < Exch10_2_terms.size(); k++) {
    //    outfile->Printf("    Exch10(S^2) (%1d)     = %18.12lf [Eh]\n",k+1,Exch10_2_terms[k]);
    //}
    // scalars_["Exch10(S^2)"] = Exch10_2;
    outfile->Printf("    Exch10(S^2)         = %18.12lf [Eh]\n", Exch10_2);
    outfile->Printf("\n");
    // fflush(outfile);

    // => Exchange scaling <= //

    if (options_.get_bool("FISAPT_FSAPT_EXCH_SCALE")) {
        double scale = scalars_["Exch10"] / scalars_["Exch10(S^2)"];
        matrices_["Exch_AB"]->scale(scale);
        outfile->Printf("    Scaling F-SAPT Exch10(S^2) by %11.3E to match Exch10\n\n", scale);
    }
    if (options_.get_bool("SSAPT0_SCALE")) {
        sSAPT0_scale_ = scalars_["Exch10"] / scalars_["Exch10(S^2)"];
        sSAPT0_scale_ = pow(sSAPT0_scale_, 3.0);
        outfile->Printf("    Scaling F-SAPT Exch-Ind and Exch-Disp by %11.3E \n\n", sSAPT0_scale_);
    }

    // Prepare DFHelper object for the next module
    dfh_->clear_spaces();
}

// Compute fragment-fragment partitioning of induction contribution
void FISAPT::find() {
    outfile->Printf("  ==> F-SAPT Induction <==\n\n");

    // => Options <= //

    bool ind_resp = options_.get_bool("FISAPT_FSAPT_IND_RESPONSE");
    bool ind_scale = options_.get_bool("FISAPT_FSAPT_IND_SCALE");

    // => Sizing <= //

    std::shared_ptr<Molecule> mol = primary_->molecule();
    int nn = primary_->nbf();
    int nA = mol->natom();
    int nB = mol->natom();
    int na = matrices_["Locc0A"]->colspi()[0];
    int nb = matrices_["Locc0B"]->colspi()[0];
    int nr = matrices_["Cvir0A"]->colspi()[0];
    int ns = matrices_["Cvir0B"]->colspi()[0];

    // => Pointers <= //

    std::shared_ptr<Matrix> Locc_A = matrices_["Locc0A"];
    std::shared_ptr<Matrix> Locc_B = matrices_["Locc0B"];

    std::shared_ptr<Matrix> Uocc_A = matrices_["Uocc0A"];
    std::shared_ptr<Matrix> Uocc_B = matrices_["Uocc0B"];

    std::shared_ptr<Matrix> Cocc_A = matrices_["Cocc0A"];
    std::shared_ptr<Matrix> Cocc_B = matrices_["Cocc0B"];
    std::shared_ptr<Matrix> Cvir_A = matrices_["Cvir0A"];
    std::shared_ptr<Matrix> Cvir_B = matrices_["Cvir0B"];

    std::shared_ptr<Vector> eps_occ_A = vectors_["eps_occ0A"];
    std::shared_ptr<Vector> eps_occ_B = vectors_["eps_occ0B"];
    std::shared_ptr<Vector> eps_vir_A = vectors_["eps_vir0A"];
    std::shared_ptr<Vector> eps_vir_B = vectors_["eps_vir0B"];

    // => DFHelper = DF + disk tensors <= //

    int nT = 1;
#ifdef _OPENMP
    nT = Process::environment.get_n_threads();
#endif

    size_t nQ = dfh_->get_naux();

    // => ESPs <= //

    dfh_->add_disk_tensor("WBar", std::make_tuple(nB + nb, na, nr));
    dfh_->add_disk_tensor("WAbs", std::make_tuple(nA + na, nb, ns));

    // => Nuclear Part (PITA) <= //

    auto Zxyz2 = std::make_shared<Matrix>("Zxyz", 1, 4);
    double** Zxyz2p = Zxyz2->pointer();
    auto Vfact2 = std::make_shared<IntegralFactory>(primary_);
    std::shared_ptr<PotentialInt> Vint2(static_cast<PotentialInt*>(Vfact2->ao_potential()));
    Vint2->set_charge_field(Zxyz2);
    auto Vtemp2 = std::make_shared<Matrix>("Vtemp2", nn, nn);

    double* ZAp = vectors_["ZA"]->pointer();
    for (size_t A = 0; A < nA; A++) {
        Vtemp2->zero();
        Zxyz2p[0][0] = ZAp[A];
        Zxyz2p[0][1] = mol->x(A);
        Zxyz2p[0][2] = mol->y(A);
        Zxyz2p[0][3] = mol->z(A);
        Vint2->compute(Vtemp2);
        std::shared_ptr<Matrix> Vbs = linalg::triplet(Cocc_B, Vtemp2, Cvir_B, true, false, false);
        dfh_->write_disk_tensor("WAbs", Vbs, {A, A + 1});
    }

    double* ZBp = vectors_["ZB"]->pointer();
    for (size_t B = 0; B < nB; B++) {
        Vtemp2->zero();
        Zxyz2p[0][0] = ZBp[B];
        Zxyz2p[0][1] = mol->x(B);
        Zxyz2p[0][2] = mol->y(B);
        Zxyz2p[0][3] = mol->z(B);
        Vint2->compute(Vtemp2);
        std::shared_ptr<Matrix> Var = linalg::triplet(Cocc_A, Vtemp2, Cvir_A, true, false, false);
        dfh_->write_disk_tensor("WBar", Var, {B, B + 1});
    }

    // ==> DFHelper Setup (JKFIT Type, in Full Basis) <== //

    std::vector<std::shared_ptr<Matrix> > Cs;
    Cs.push_back(Cocc_A);
    Cs.push_back(Cvir_A);
    Cs.push_back(Cocc_B);
    Cs.push_back(Cvir_B);

    size_t max_MO = 0;
    for (auto& mat : Cs) max_MO = std::max(max_MO, (size_t)mat->ncol());

    dfh_->add_space("a", Cs[0]);
    dfh_->add_space("r", Cs[1]);
    dfh_->add_space("b", Cs[2]);
    dfh_->add_space("s", Cs[3]);

    dfh_->add_transformation("Aar", "a", "r");
    dfh_->add_transformation("Abs", "b", "s");

    dfh_->transform();

    // => Electronic Part (Massive PITA) <= //

    double** RaCp = matrices_["Vlocc0A"]->pointer();
    double** RbDp = matrices_["Vlocc0B"]->pointer();

    auto TsQ = std::make_shared<Matrix>("TsQ", ns, nQ);
    auto T1As = std::make_shared<Matrix>("T1As", na, ns);
    double** TsQp = TsQ->pointer();
    double** T1Asp = T1As->pointer();
    for (size_t b = 0; b < nb; b++) {
        dfh_->fill_tensor("Abs", TsQ, {b, b + 1});
        C_DGEMM('N', 'T', na, ns, nQ, 2.0, RaCp[0], nQ, TsQp[0], nQ, 0.0, T1Asp[0], ns);
        for (size_t a = 0; a < na; a++) {
            dfh_->write_disk_tensor("WAbs", T1Asp[a], {nA + a, nA + a + 1}, {b, b + 1});
        }
    }

    auto TrQ = std::make_shared<Matrix>("TrQ", nr, nQ);
    auto T1Br = std::make_shared<Matrix>("T1Br", nb, nr);
    double** TrQp = TrQ->pointer();
    double** T1Brp = T1Br->pointer();
    for (size_t a = 0; a < na; a++) {
        dfh_->fill_tensor("Aar", TrQ, {a, a + 1});
        C_DGEMM('N', 'T', nb, nr, nQ, 2.0, RbDp[0], nQ, TrQp[0], nQ, 0.0, T1Brp[0], nr);
        for (size_t b = 0; b < nb; b++) {
            dfh_->write_disk_tensor("WBar", T1Brp[b], {nB + b, nB + b + 1}, {a, a + 1});
        }
    }

    // ==> Stack Variables <== //

    double* eap = eps_occ_A->pointer();
    double* ebp = eps_occ_B->pointer();
    double* erp = eps_vir_A->pointer();
    double* esp = eps_vir_B->pointer();

    std::shared_ptr<Matrix> S = matrices_["S"];
    std::shared_ptr<Matrix> D_A = matrices_["D_A"];
    std::shared_ptr<Matrix> V_A = matrices_["V_A"];
    std::shared_ptr<Matrix> J_A = matrices_["J_A"];
    std::shared_ptr<Matrix> K_A = matrices_["K_A"];
    std::shared_ptr<Matrix> D_B = matrices_["D_B"];
    std::shared_ptr<Matrix> V_B = matrices_["V_B"];
    std::shared_ptr<Matrix> J_B = matrices_["J_B"];
    std::shared_ptr<Matrix> K_B = matrices_["K_B"];
    std::shared_ptr<Matrix> J_O = matrices_["J_O"];
    std::shared_ptr<Matrix> K_O = matrices_["K_O"];
    std::shared_ptr<Matrix> J_P_A = matrices_["J_P_A"];
    std::shared_ptr<Matrix> J_P_B = matrices_["J_P_B"];

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
    mapA["Cocc_A"] = Locc_A;
    mapA["Cvir_A"] = Cvir_A;
    mapA["Cocc_B"] = Locc_B;
    mapA["Cvir_B"] = Cvir_B;
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
    mapB["Cocc_A"] = Locc_B;
    mapB["Cvir_A"] = Cvir_B;
    mapB["Cocc_B"] = Locc_A;
    mapB["Cvir_B"] = Cvir_A;
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

    auto Ind20u_AB_terms = std::make_shared<Matrix>("Ind20 [A<-B] (a x B)", na, nB + nb);
    auto Ind20u_BA_terms = std::make_shared<Matrix>("Ind20 [B<-A] (A x b)", nA + na, nb);
    double** Ind20u_AB_termsp = Ind20u_AB_terms->pointer();
    double** Ind20u_BA_termsp = Ind20u_BA_terms->pointer();

    double Ind20u_AB = 0.0;
    double Ind20u_BA = 0.0;

    auto ExchInd20u_AB_terms = std::make_shared<Matrix>("ExchInd20 [A<-B] (a x B)", na, nB + nb);
    auto ExchInd20u_BA_terms = std::make_shared<Matrix>("ExchInd20 [B<-A] (A x b)", nA + na, nb);
    double** ExchInd20u_AB_termsp = ExchInd20u_AB_terms->pointer();
    double** ExchInd20u_BA_termsp = ExchInd20u_BA_terms->pointer();

    double ExchInd20u_AB = 0.0;
    double ExchInd20u_BA = 0.0;

    int sna = 0;
    int snB = 0;
    int snb = 0;
    int snA = 0;

    if (options_.get_bool("SSAPT0_SCALE")) {
        sna = na;
        snB = nB;
        snb = nb;
        snA = nA;
    }

    std::shared_ptr<Matrix> sExchInd20u_AB_terms =
        std::make_shared<Matrix>("sExchInd20 [A<-B] (a x B)", sna, snB + snb);
    std::shared_ptr<Matrix> sExchInd20u_BA_terms =
        std::make_shared<Matrix>("sExchInd20 [B<-A] (A x b)", snA + sna, snb);
    double** sExchInd20u_AB_termsp = sExchInd20u_AB_terms->pointer();
    double** sExchInd20u_BA_termsp = sExchInd20u_BA_terms->pointer();

    double sExchInd20u_AB = 0.0;
    double sExchInd20u_BA = 0.0;

    auto Indu_AB_terms = std::make_shared<Matrix>("Ind [A<-B] (a x B)", na, nB + nb);
    auto Indu_BA_terms = std::make_shared<Matrix>("Ind [B<-A] (A x b)", nA + na, nb);
    double** Indu_AB_termsp = Indu_AB_terms->pointer();
    double** Indu_BA_termsp = Indu_BA_terms->pointer();

    double Indu_AB = 0.0;
    double Indu_BA = 0.0;

    auto sIndu_AB_terms = std::make_shared<Matrix>("sInd [A<-B] (a x B)", sna, snB + snb);
    auto sIndu_BA_terms = std::make_shared<Matrix>("sInd [B<-A] (A x b)", snA + sna, snb);
    double** sIndu_AB_termsp = sIndu_AB_terms->pointer();
    double** sIndu_BA_termsp = sIndu_BA_terms->pointer();

    double sIndu_AB = 0.0;
    double sIndu_BA = 0.0;

    // ==> A <- B Uncoupled <== //

    for (size_t B = 0; B < nB + nb; B++) {
        // ESP
        dfh_->fill_tensor("WBar", wB, {B, B + 1});

        // Uncoupled amplitude
        for (int a = 0; a < na; a++) {
            for (int r = 0; r < nr; r++) {
                xAp[a][r] = wBp[a][r] / (eap[a] - erp[r]);
            }
        }

        // Backtransform the amplitude to LO
        std::shared_ptr<Matrix> x2A = linalg::doublet(Uocc_A, xA, true, false);
        double** x2Ap = x2A->pointer();

        // Zip up the Ind20 contributions
        for (int a = 0; a < na; a++) {
            double Jval = 2.0 * C_DDOT(nr, x2Ap[a], 1, wBTp[a], 1);
            double Kval = 2.0 * C_DDOT(nr, x2Ap[a], 1, uBTp[a], 1);
            Ind20u_AB_termsp[a][B] = Jval;
            Ind20u_AB += Jval;
            ExchInd20u_AB_termsp[a][B] = Kval;
            ExchInd20u_AB += Kval;
            if (options_.get_bool("SSAPT0_SCALE")) {
                sExchInd20u_AB_termsp[a][B] = Kval;
                sExchInd20u_AB += Kval;
                sIndu_AB_termsp[a][B] = Jval + Kval;
                sIndu_AB += Jval + Kval;
            }

            Indu_AB_termsp[a][B] = Jval + Kval;
            Indu_AB += Jval + Kval;
        }
    }

    // ==> B <- A Uncoupled <== //

    for (size_t A = 0; A < nA + na; A++) {
        // ESP
        dfh_->fill_tensor("WAbs", wA, {A, A + 1});

        // Uncoupled amplitude
        for (int b = 0; b < nb; b++) {
            for (int s = 0; s < ns; s++) {
                xBp[b][s] = wAp[b][s] / (ebp[b] - esp[s]);
            }
        }

        // Backtransform the amplitude to LO
        std::shared_ptr<Matrix> x2B = linalg::doublet(Uocc_B, xB, true, false);
        double** x2Bp = x2B->pointer();

        // Zip up the Ind20 contributions
        for (int b = 0; b < nb; b++) {
            double Jval = 2.0 * C_DDOT(ns, x2Bp[b], 1, wATp[b], 1);
            double Kval = 2.0 * C_DDOT(ns, x2Bp[b], 1, uATp[b], 1);
            Ind20u_BA_termsp[A][b] = Jval;
            Ind20u_BA += Jval;
            ExchInd20u_BA_termsp[A][b] = Kval;
            ExchInd20u_BA += Kval;
            if (options_.get_bool("SSAPT0_SCALE")) {
                sExchInd20u_BA_termsp[A][b] = Kval;
                sExchInd20u_BA += Kval;
                sIndu_BA_termsp[A][b] = Jval + Kval;
                sIndu_BA += Jval + Kval;
            }
            Indu_BA_termsp[A][b] = Jval + Kval;
            Indu_BA += Jval + Kval;
        }
    }

    double Ind20u = Ind20u_AB + Ind20u_BA;
    outfile->Printf("    Ind20,u (A<-B)      = %18.12lf [Eh]\n", Ind20u_AB);
    outfile->Printf("    Ind20,u (B<-A)      = %18.12lf [Eh]\n", Ind20u_BA);
    outfile->Printf("    Ind20,u             = %18.12lf [Eh]\n", Ind20u);
    // fflush(outfile);

    double ExchInd20u = ExchInd20u_AB + ExchInd20u_BA;
    outfile->Printf("    Exch-Ind20,u (A<-B) = %18.12lf [Eh]\n", ExchInd20u_AB);
    outfile->Printf("    Exch-Ind20,u (B<-A) = %18.12lf [Eh]\n", ExchInd20u_BA);
    outfile->Printf("    Exch-Ind20,u        = %18.12lf [Eh]\n", ExchInd20u);
    outfile->Printf("\n");
    // fflush(outfile);
    if (options_.get_bool("SSAPT0_SCALE")) {
        double sExchInd20u = sExchInd20u_AB + sExchInd20u_BA;
        outfile->Printf("    sExch-Ind20,u (A<-B) = %18.12lf [Eh]\n", sExchInd20u_AB);
        outfile->Printf("    sExch-Ind20,u (B<-A) = %18.12lf [Eh]\n", sExchInd20u_BA);
        outfile->Printf("    sExch-Ind20,u        = %18.12lf [Eh]\n", sExchInd20u);
        outfile->Printf("\n");
    }

    double Ind = Ind20u + ExchInd20u;
    std::shared_ptr<Matrix> Ind_AB_terms = Indu_AB_terms;
    std::shared_ptr<Matrix> Ind_BA_terms = Indu_BA_terms;
    std::shared_ptr<Matrix> sInd_AB_terms = sIndu_AB_terms;
    std::shared_ptr<Matrix> sInd_BA_terms = sIndu_BA_terms;

    if (ind_resp) {
        outfile->Printf("  COUPLED INDUCTION (You asked for it!):\n\n");

        // ==> Coupled Targets <== //

        auto Ind20r_AB_terms = std::make_shared<Matrix>("Ind20 [A<-B] (a x B)", na, nB + nb);
        auto Ind20r_BA_terms = std::make_shared<Matrix>("Ind20 [B<-A] (A x b)", nA + na, nb);
        double** Ind20r_AB_termsp = Ind20r_AB_terms->pointer();
        double** Ind20r_BA_termsp = Ind20r_BA_terms->pointer();

        double Ind20r_AB = 0.0;
        double Ind20r_BA = 0.0;

        auto ExchInd20r_AB_terms = std::make_shared<Matrix>("ExchInd20 [A<-B] (a x B)", na, nB + nb);
        auto ExchInd20r_BA_terms = std::make_shared<Matrix>("ExchInd20 [B<-A] (A x b)", nA + na, nb);
        double** ExchInd20r_AB_termsp = ExchInd20r_AB_terms->pointer();
        double** ExchInd20r_BA_termsp = ExchInd20r_BA_terms->pointer();

        double ExchInd20r_AB = 0.0;
        double ExchInd20r_BA = 0.0;

        auto Indr_AB_terms = std::make_shared<Matrix>("Ind [A<-B] (a x B)", na, nB + nb);
        auto Indr_BA_terms = std::make_shared<Matrix>("Ind [B<-A] (A x b)", nA + na, nb);
        double** Indr_AB_termsp = Indr_AB_terms->pointer();
        double** Indr_BA_termsp = Indr_BA_terms->pointer();

        double Indr_AB = 0.0;
        double Indr_BA = 0.0;

        // => JK Object <= //

        // TODO: Account for 2-index overhead in memory
        auto nso = primary_->nbf();
        auto jk_memory = (long int)doubles_;
        jk_memory -= 24 * nso * nso;
        jk_memory -= 4 * na * nso;
        jk_memory -= 4 * nb * nso;
        if (jk_memory < 0L) {
            throw PSIEXCEPTION("Too little static memory for FISAPT::induction");
        }

        std::shared_ptr<JK> jk =
            JK::build_JK(primary_, reference_->get_basisset("DF_BASIS_SCF"), options_, false, (size_t)jk_memory);

        jk->set_memory((size_t)jk_memory);
        jk->set_do_J(true);
        jk->set_do_K(true);
        jk->initialize();
        jk->print_header();

        // ==> Master Loop over perturbing atoms <== //

        int nC = std::max(nA + na, nB + nb);

        for (size_t C = 0; C < nC; C++) {
            if (C < nB + nb) dfh_->fill_tensor("WBar", wB, {C, C + 1});
            if (C < nA + na) dfh_->fill_tensor("WAbs", wB, {C, C + 1});

            outfile->Printf("    Responses for (A <- Source B = %3zu) and (B <- Source A = %3zu)\n\n",
                            (C < nB + nb ? C : nB + nb - 1), (C < nA + na ? C : nA + na - 1));

            auto cphf = std::make_shared<CPHF_FISAPT>();

            // Effective constructor
            cphf->delta_ = options_.get_double("D_CONVERGENCE");
            cphf->maxiter_ = options_.get_int("MAXITER");
            cphf->jk_ = jk;

            cphf->w_A_ = wB;  // Reversal of convention
            cphf->Cocc_A_ = Cocc_A;
            cphf->Cvir_A_ = Cvir_A;
            cphf->eps_occ_A_ = eps_occ_A;
            cphf->eps_vir_A_ = eps_vir_A;

            cphf->w_B_ = wA;  // Reversal of convention
            cphf->Cocc_B_ = Cocc_B;
            cphf->Cvir_B_ = Cvir_B;
            cphf->eps_occ_B_ = eps_occ_B;
            cphf->eps_vir_B_ = eps_vir_B;

            // Gogo CPKS
            cphf->compute_cphf();

            xA = cphf->x_A_;
            xB = cphf->x_B_;

            xA->scale(-1.0);
            xB->scale(-1.0);

            if (C < nB + nb) {
                // Backtransform the amplitude to LO
                std::shared_ptr<Matrix> x2A = linalg::doublet(Uocc_A, xA, true, false);
                double** x2Ap = x2A->pointer();

                // Zip up the Ind20 contributions
                for (int a = 0; a < na; a++) {
                    double Jval = 2.0 * C_DDOT(nr, x2Ap[a], 1, wBTp[a], 1);
                    double Kval = 2.0 * C_DDOT(nr, x2Ap[a], 1, uBTp[a], 1);
                    Ind20r_AB_termsp[a][C] = Jval;
                    Ind20r_AB += Jval;
                    ExchInd20r_AB_termsp[a][C] = Kval;
                    ExchInd20r_AB += Kval;
                    Indr_AB_termsp[a][C] = Jval + Kval;
                    Indr_AB += Jval + Kval;
                }
            }

            if (C < nA + na) {
                // Backtransform the amplitude to LO
                std::shared_ptr<Matrix> x2B = linalg::doublet(Uocc_B, xB, true, false);
                double** x2Bp = x2B->pointer();

                // Zip up the Ind20 contributions
                for (int b = 0; b < nb; b++) {
                    double Jval = 2.0 * C_DDOT(ns, x2Bp[b], 1, wATp[b], 1);
                    double Kval = 2.0 * C_DDOT(ns, x2Bp[b], 1, uATp[b], 1);
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
        outfile->Printf("    Ind20,r (A<-B)      = %18.12lf [Eh]\n", Ind20r_AB);
        outfile->Printf("    Ind20,r (B<-A)      = %18.12lf [Eh]\n", Ind20r_BA);
        outfile->Printf("    Ind20,r             = %18.12lf [Eh]\n", Ind20r);
        // fflush(outfile);

        double ExchInd20r = ExchInd20r_AB + ExchInd20r_BA;
        outfile->Printf("    Exch-Ind20,r (A<-B) = %18.12lf [Eh]\n", ExchInd20r_AB);
        outfile->Printf("    Exch-Ind20,r (B<-A) = %18.12lf [Eh]\n", ExchInd20r_BA);
        outfile->Printf("    Exch-Ind20,r        = %18.12lf [Eh]\n", ExchInd20r);
        outfile->Printf("\n");
        // fflush(outfile);

        Ind = Ind20r + ExchInd20r;
        Ind_AB_terms = Indr_AB_terms;
        Ind_BA_terms = Indr_BA_terms;
    }

    // => Induction scaling <= //

    if (ind_scale) {
        double dHF = 0.0;
        if (scalars_["HF"] != 0.0) {
            dHF = scalars_["HF"] - scalars_["Elst10,r"] - scalars_["Exch10"] - scalars_["Ind20,r"] -
                  scalars_["Exch-Ind20,r"];
        }
        double IndHF = scalars_["Ind20,r"] + scalars_["Exch-Ind20,r"] + dHF;
        double IndSAPT0 = scalars_["Ind20,r"] + scalars_["Exch-Ind20,r"];

        double Sdelta = IndHF / IndSAPT0;
        double SrAB = (ind_resp ? 1.0
                                : (scalars_["Ind20,r (A<-B)"] + scalars_["Exch-Ind20,r (A<-B)"]) /
                                      (scalars_["Ind20,u (A<-B)"] + scalars_["Exch-Ind20,u (A<-B)"]));
        double SrBA = (ind_resp ? 1.0
                                : (scalars_["Ind20,r (B<-A)"] + scalars_["Exch-Ind20,r (B<-A)"]) /
                                      (scalars_["Ind20,u (B<-A)"] + scalars_["Exch-Ind20,u (B<-A)"]));

        double sIndHF = scalars_["Ind20,r"] + scalars_["sExch-Ind20,r"] + dHF;
        double sIndSAPT0 = scalars_["Ind20,r"] + scalars_["sExch-Ind20,r"];

        double sSdelta = sIndHF / IndSAPT0;

        double sSrAB = (ind_resp ? 1.0
                                 : (scalars_["Ind20,r (A<-B)"] + scalars_["sExch-Ind20,r (A<-B)"]) /
                                       (scalars_["Ind20,u (A<-B)"] + scalars_["sExch-Ind20,u (A<-B)"]));
        double sSrBA = (ind_resp ? 1.0
                                 : (scalars_["Ind20,r (B<-A)"] + scalars_["sExch-Ind20,r (B<-A)"]) /
                                       (scalars_["Ind20,u (B<-A)"] + scalars_["sExch-Ind20,u (B<-A)"]));

        outfile->Printf("    Scaling for delta HF        = %11.3E\n", Sdelta);
        outfile->Printf("    Scaling for response (A<-B) = %11.3E\n", SrAB);
        outfile->Printf("    Scaling for response (B<-A) = %11.3E\n", SrBA);
        outfile->Printf("    Scaling for total (A<-B)    = %11.3E\n", Sdelta * SrAB);
        outfile->Printf("    Scaling for total (B<-A)    = %11.3E\n", Sdelta * SrBA);
        outfile->Printf("\n");

        Ind_AB_terms->scale(Sdelta * SrAB);
        Ind_BA_terms->scale(Sdelta * SrBA);
        Ind20u_AB_terms->scale(Sdelta * SrAB);
        ExchInd20u_AB_terms->scale(Sdelta * SrAB);
        Ind20u_BA_terms->scale(Sdelta * SrBA);
        ExchInd20u_BA_terms->scale(Sdelta * SrBA);
        sInd_AB_terms->scale(sSdelta * SrAB);
        sInd_BA_terms->scale(sSdelta * SrBA);
    }

    matrices_["IndAB_AB"] = std::make_shared<Matrix>("IndAB_AB", nA + na, nB + nb);
    matrices_["IndBA_AB"] = std::make_shared<Matrix>("IndBA_AB", nA + na, nB + nb);
    matrices_["Ind20u_AB_terms"] = std::make_shared<Matrix>("Ind20uAB_AB", nA + na, nB + nb);
    matrices_["ExchInd20u_AB_terms"] = std::make_shared<Matrix>("ExchInd20uAB_AB", nA + na, nB + nb);
    matrices_["Ind20u_BA_terms"] = std::make_shared<Matrix>("Ind20uBA_AB", nA + na, nB + nb);
    matrices_["ExchInd20u_BA_terms"] = std::make_shared<Matrix>("ExchInd20uBA_AB", nA + na, nB + nb);
    double** EABp = matrices_["IndAB_AB"]->pointer();
    double** EBAp = matrices_["IndBA_AB"]->pointer();
    double** Ind20ABp = matrices_["Ind20u_AB_terms"]->pointer();
    double** ExchInd20ABp = matrices_["ExchInd20u_AB_terms"]->pointer();
    double** Ind20BAp = matrices_["Ind20u_BA_terms"]->pointer();
    double** ExchInd20BAp = matrices_["ExchInd20u_BA_terms"]->pointer();
    double** EAB2p = Ind_AB_terms->pointer();
    double** EBA2p = Ind_BA_terms->pointer();
    double** Ind20AB2p = Ind20u_AB_terms->pointer();
    double** ExchInd20AB2p = ExchInd20u_AB_terms->pointer();
    double** Ind20BA2p = Ind20u_BA_terms->pointer();
    double** ExchInd20BA2p = ExchInd20u_BA_terms->pointer();

    for (int a = 0; a < na; a++) {
        for (int B = 0; B < nB + nb; B++) {
            EABp[a + nA][B] = EAB2p[a][B];
            Ind20ABp[a + nA][B] = Ind20AB2p[a][B];
            ExchInd20ABp[a + nA][B] = ExchInd20AB2p[a][B];
        }
    }

    for (int A = 0; A < nA + na; A++) {
        for (int b = 0; b < nb; b++) {
            EBAp[A][b + nB] = EBA2p[A][b];
            Ind20BAp[A][b + nB] = Ind20BA2p[A][b];
            ExchInd20BAp[A][b + nB] = ExchInd20BA2p[A][b];
        }
    }

    matrices_["sIndAB_AB"] = std::make_shared<Matrix>("sIndAB_AB", snA + sna, snB + snb);
    matrices_["sIndBA_AB"] = std::make_shared<Matrix>("sIndBA_AB", snA + sna, snB + snb);
    double** sEABp = matrices_["sIndAB_AB"]->pointer();
    double** sEBAp = matrices_["sIndBA_AB"]->pointer();
    double** sEAB2p = sInd_AB_terms->pointer();
    double** sEBA2p = sInd_BA_terms->pointer();

    for (int a = 0; a < sna; a++) {
        for (int B = 0; B < snB + snb; B++) {
            sEABp[a + snA][B] = sEAB2p[a][B];
        }
    }

    for (int A = 0; A < snA + sna; A++) {
        for (int b = 0; b < snb; b++) {
            sEBAp[A][b + snB] = sEBA2p[A][b];
        }
    }
    // We're done with dfh_'s integrals
    dfh_->clear_all();
}

// Compute fragment-fragment partitioning of dispersion contribution
void FISAPT::fdisp() {
    outfile->Printf("  ==> F-SAPT Dispersion <==\n\n");

    // => Auxiliary Basis Set <= //

    std::shared_ptr<BasisSet> auxiliary = reference_->get_basisset("DF_BASIS_SAPT");

    // => Sizing <= //

    std::shared_ptr<Molecule> mol = primary_->molecule();
    int nn = primary_->nbf();
    int nA = mol->natom();
    int nB = mol->natom();
    int na = matrices_["Laocc0A"]->colspi()[0];
    int nb = matrices_["Laocc0B"]->colspi()[0];
    int nr = matrices_["Cvir0A"]->colspi()[0];
    int ns = matrices_["Cvir0B"]->colspi()[0];
    int nQ = auxiliary->nbf();
    size_t naQ = na * (size_t)nQ;
    size_t nbQ = nb * (size_t)nQ;

    int nfa = matrices_["Lfocc0A"]->colspi()[0];
    int nfb = matrices_["Lfocc0B"]->colspi()[0];

    int nT = 1;
#ifdef _OPENMP
    nT = Process::environment.get_n_threads();
#endif

    // => Targets <= //

    matrices_["Disp_AB"] = std::make_shared<Matrix>("Disp_AB", nA + nfa + na, nB + nfb + nb);
    double** Ep = matrices_["Disp_AB"]->pointer();

    int snA = 0;
    int snfa = 0;
    int sna = 0;
    int snB = 0;
    int snfb = 0;
    int snb = 0;

    if (options_.get_bool("SSAPT0_SCALE")) {
        snA = nA;
        snfa = nfa;
        sna = na;
        snB = nB;
        snfb = nfb;
        snb = nb;
    }

    matrices_["sDisp_AB"] = std::make_shared<Matrix>("Disp_AB", snA + snfa + sna, snB + snfb + snb);
    double** sEp = matrices_["sDisp_AB"]->pointer();

    // => Stashed Variables <= //

    std::shared_ptr<Matrix> S = matrices_["S"];
    std::shared_ptr<Matrix> D_A = matrices_["D_A"];
    std::shared_ptr<Matrix> P_A = matrices_["P_A"];
    std::shared_ptr<Matrix> V_A = matrices_["V_A"];
    std::shared_ptr<Matrix> J_A = matrices_["J_A"];
    std::shared_ptr<Matrix> K_A = matrices_["K_A"];
    std::shared_ptr<Matrix> D_B = matrices_["D_B"];
    std::shared_ptr<Matrix> P_B = matrices_["P_B"];
    std::shared_ptr<Matrix> V_B = matrices_["V_B"];
    std::shared_ptr<Matrix> J_B = matrices_["J_B"];
    std::shared_ptr<Matrix> K_B = matrices_["K_B"];
    std::shared_ptr<Matrix> K_O = matrices_["K_O"];

    std::shared_ptr<Matrix> Caocc_A = matrices_["Caocc0A"];
    std::shared_ptr<Matrix> Caocc_B = matrices_["Caocc0B"];
    std::shared_ptr<Matrix> Cavir_A = matrices_["Cvir0A"];
    std::shared_ptr<Matrix> Cavir_B = matrices_["Cvir0B"];

    std::shared_ptr<Vector> eps_aocc_A = vectors_["eps_aocc0A"];
    std::shared_ptr<Vector> eps_aocc_B = vectors_["eps_aocc0B"];
    std::shared_ptr<Vector> eps_avir_A = vectors_["eps_vir0A"];
    std::shared_ptr<Vector> eps_avir_B = vectors_["eps_vir0B"];

    std::shared_ptr<Matrix> Uaocc_A = matrices_["Uaocc0A"];
    std::shared_ptr<Matrix> Uaocc_B = matrices_["Uaocc0B"];

    // => Auxiliary C matrices <= //

    std::shared_ptr<Matrix> Cr1 = linalg::triplet(D_B, S, Cavir_A);
    Cr1->scale(-1.0);
    Cr1->add(Cavir_A);
    std::shared_ptr<Matrix> Cs1 = linalg::triplet(D_A, S, Cavir_B);
    Cs1->scale(-1.0);
    Cs1->add(Cavir_B);
    std::shared_ptr<Matrix> Ca2 = linalg::triplet(D_B, S, Caocc_A);
    std::shared_ptr<Matrix> Cb2 = linalg::triplet(D_A, S, Caocc_B);
    std::shared_ptr<Matrix> Cr3 = linalg::triplet(D_B, S, Cavir_A);
    std::shared_ptr<Matrix> CrX = linalg::triplet(linalg::triplet(D_A, S, D_B), S, Cavir_A);
    Cr3->subtract(CrX);
    Cr3->scale(2.0);
    std::shared_ptr<Matrix> Cs3 = linalg::triplet(D_A, S, Cavir_B);
    std::shared_ptr<Matrix> CsX = linalg::triplet(linalg::triplet(D_B, S, D_A), S, Cavir_B);
    Cs3->subtract(CsX);
    Cs3->scale(2.0);
    std::shared_ptr<Matrix> Ca4 = linalg::triplet(linalg::triplet(D_A, S, D_B), S, Caocc_A);
    Ca4->scale(-2.0);
    std::shared_ptr<Matrix> Cb4 = linalg::triplet(linalg::triplet(D_B, S, D_A), S, Caocc_B);
    Cb4->scale(-2.0);

    // => Auxiliary V matrices <= //

    std::shared_ptr<Matrix> Jbr = linalg::triplet(Caocc_B, J_A, Cavir_A, true, false, false);
    Jbr->scale(2.0);
    std::shared_ptr<Matrix> Kbr = linalg::triplet(Caocc_B, K_A, Cavir_A, true, false, false);
    Kbr->scale(-1.0);

    std::shared_ptr<Matrix> Jas = linalg::triplet(Caocc_A, J_B, Cavir_B, true, false, false);
    Jas->scale(2.0);
    std::shared_ptr<Matrix> Kas = linalg::triplet(Caocc_A, K_B, Cavir_B, true, false, false);
    Kas->scale(-1.0);

    std::shared_ptr<Matrix> KOas = linalg::triplet(Caocc_A, K_O, Cavir_B, true, false, false);
    KOas->scale(1.0);
    std::shared_ptr<Matrix> KObr = linalg::triplet(Caocc_B, K_O, Cavir_A, true, true, false);
    KObr->scale(1.0);

    std::shared_ptr<Matrix> JBas = linalg::triplet(linalg::triplet(Caocc_A, S, D_B, true, false, false), J_A, Cavir_B);
    JBas->scale(-2.0);
    std::shared_ptr<Matrix> JAbr = linalg::triplet(linalg::triplet(Caocc_B, S, D_A, true, false, false), J_B, Cavir_A);
    JAbr->scale(-2.0);

    std::shared_ptr<Matrix> Jbs = linalg::triplet(Caocc_B, J_A, Cavir_B, true, false, false);
    Jbs->scale(4.0);
    std::shared_ptr<Matrix> Jar = linalg::triplet(Caocc_A, J_B, Cavir_A, true, false, false);
    Jar->scale(4.0);

    std::shared_ptr<Matrix> JAas = linalg::triplet(linalg::triplet(Caocc_A, J_B, D_A, true, false, false), S, Cavir_B);
    JAas->scale(-2.0);
    std::shared_ptr<Matrix> JBbr = linalg::triplet(linalg::triplet(Caocc_B, J_A, D_B, true, false, false), S, Cavir_A);
    JBbr->scale(-2.0);

    // Get your signs right Hesselmann!
    std::shared_ptr<Matrix> Vbs = linalg::triplet(Caocc_B, V_A, Cavir_B, true, false, false);
    Vbs->scale(2.0);
    std::shared_ptr<Matrix> Var = linalg::triplet(Caocc_A, V_B, Cavir_A, true, false, false);
    Var->scale(2.0);
    std::shared_ptr<Matrix> VBas = linalg::triplet(linalg::triplet(Caocc_A, S, D_B, true, false, false), V_A, Cavir_B);
    VBas->scale(-1.0);
    std::shared_ptr<Matrix> VAbr = linalg::triplet(linalg::triplet(Caocc_B, S, D_A, true, false, false), V_B, Cavir_A);
    VAbr->scale(-1.0);
    std::shared_ptr<Matrix> VRas = linalg::triplet(linalg::triplet(Caocc_A, V_B, P_A, true, false, false), S, Cavir_B);
    VRas->scale(1.0);
    std::shared_ptr<Matrix> VSbr = linalg::triplet(linalg::triplet(Caocc_B, V_A, P_B, true, false, false), S, Cavir_A);
    VSbr->scale(1.0);

    std::shared_ptr<Matrix> Sas = linalg::triplet(Caocc_A, S, Cavir_B, true, false, false);
    std::shared_ptr<Matrix> Sbr = linalg::triplet(Caocc_B, S, Cavir_A, true, false, false);

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

    std::shared_ptr<Matrix> SBar = linalg::triplet(linalg::triplet(Caocc_A, S, D_B, true, false, false), S, Cavir_A);
    std::shared_ptr<Matrix> SAbs = linalg::triplet(linalg::triplet(Caocc_B, S, D_A, true, false, false), S, Cavir_B);

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

    // => Integrals from DFHelper <= //

    std::vector<std::shared_ptr<Matrix> > Cs;
    Cs.push_back(Caocc_A);
    Cs.push_back(Cavir_A);
    Cs.push_back(Caocc_B);
    Cs.push_back(Cavir_B);
    Cs.push_back(Cr1);
    Cs.push_back(Cs1);
    Cs.push_back(Ca2);
    Cs.push_back(Cb2);
    Cs.push_back(Cr3);
    Cs.push_back(Cs3);
    Cs.push_back(Ca4);
    Cs.push_back(Cb4);

    size_t max_MO = 0, ncol = 0;
    for (auto& mat : Cs) {
        max_MO = std::max(max_MO, (size_t)mat->ncol());
        ncol += (size_t)mat->ncol();
    }

    auto dfh(std::make_shared<DFHelper>(primary_, auxiliary));
    dfh->set_memory(doubles_ - Cs[0]->nrow() * ncol);
    dfh->set_method("DIRECT_iaQ");
    dfh->set_nthreads(nT);
    dfh->initialize();
    dfh->print_header();

    dfh->add_space("a", Cs[0]);
    dfh->add_space("r", Cs[1]);
    dfh->add_space("b", Cs[2]);
    dfh->add_space("s", Cs[3]);
    dfh->add_space("r1", Cs[4]);
    dfh->add_space("s1", Cs[5]);
    dfh->add_space("a2", Cs[6]);
    dfh->add_space("b2", Cs[7]);
    dfh->add_space("r3", Cs[8]);
    dfh->add_space("s3", Cs[9]);
    dfh->add_space("a4", Cs[10]);
    dfh->add_space("b4", Cs[11]);

    dfh->add_transformation("Aar", "r", "a");
    dfh->add_transformation("Abs", "s", "b");
    dfh->add_transformation("Bas", "s1", "a");
    dfh->add_transformation("Bbr", "r1", "b");
    dfh->add_transformation("Cas", "s", "a2");
    dfh->add_transformation("Cbr", "r", "b2");
    dfh->add_transformation("Dar", "r3", "a");
    dfh->add_transformation("Dbs", "s3", "b");
    dfh->add_transformation("Ear", "r", "a4");
    dfh->add_transformation("Ebs", "s", "b4");

    dfh->transform();

    Cr1.reset();
    Cs1.reset();
    Ca2.reset();
    Cb2.reset();
    Cr3.reset();
    Cs3.reset();
    Ca4.reset();
    Cb4.reset();
    Cs.clear();
    dfh->clear_spaces();

    // => Blocking ... figure out how big a tensor slice to handle at a time <= //

    long int overhead = 0L;
    overhead += 5L * nT * na * nb; // Tab, Vab, T2ab, V2ab, and Iab work arrays below
    overhead += 2L * na * ns + 2L * nb * nr + 2L * na * nr + 2L * nb * ns; // Sas, Sbr, sBar, sAbs, Qas, Qbr, Qar, Qbs
    // the next few matrices allocated here don't take too much room (but might if large numbers of threads)
    overhead += 2L * na * nb * (nT + 1); // E_disp20 and E_exch_disp20 thread work and final matrices
    overhead += 1L * sna * snb * (nT + 1); // sE_exch_disp20 thread work and final matrices
    overhead += 1L * (nA + nfa + na) * (nB + nfb + nb); // Disp_AB
    overhead += 1L * (snA + snfa + sna) * (snB + snfb + snb); // sDisp_AB
    // account for a few of the smaller matrices already defined, but not exhaustively
    overhead += 12L * nn * nn; // D, V, J, K, P, and C matrices for A and B (neglecting C)
    long int rem = doubles_ - overhead;

    outfile->Printf("    %ld doubles - %ld overhead leaves %ld for dispersion\n", doubles_, overhead, rem);
    
    if (rem < 0L) {
        throw PSIEXCEPTION("Too little static memory for DFTSAPT::mp2_terms");
    }

    // cost_r is how much mem for Aar, Bbr, Cbr, Dar for a single r
    // cost_s would be the same value, and is the mem requirement for Abs, Bas, Cas, and Dbs for single s
    long int cost_r = 2L * na * nQ + 2L * nb * nQ; 
    long int max_r_l = rem / (2L * cost_r); // 2 b/c need to hold both an r and an s
    long int max_s_l = max_r_l;
    int max_r = (max_r_l > nr ? nr : (int) max_r_l);
    int max_s = (max_s_l > ns ? ns : (int) max_s_l);
    if (max_r < 1 || max_s < 1) {
        throw PSIEXCEPTION("Too little dynamic memory for DFTSAPT::mp2_terms");
    }
    int nrblocks = (nr / max_r);
    if (nr % max_r) nrblocks++;
    int nsblocks = (ns / max_s);
    if (ns % max_s) nsblocks++;
    outfile->Printf("    Processing a single (r,s) pair requires %ld doubles\n", cost_r * 2L);
    outfile->Printf("    %d values of r processed in %d blocks of %d\n", nr, nrblocks, max_r);
    outfile->Printf("    %d values of s processed in %d blocks of %d\n\n", ns, nsblocks, max_s);

    // => Tensor Slices <= //

    auto Aar = std::make_shared<Matrix>("Aar", max_r * na, nQ);
    auto Abs = std::make_shared<Matrix>("Abs", max_s * nb, nQ);
    auto Bas = std::make_shared<Matrix>("Bas", max_s * na, nQ);
    auto Bbr = std::make_shared<Matrix>("Bbr", max_r * nb, nQ);
    auto Cas = std::make_shared<Matrix>("Cas", max_s * na, nQ);
    auto Cbr = std::make_shared<Matrix>("Cbr", max_r * nb, nQ);
    auto Dar = std::make_shared<Matrix>("Dar", max_r * na, nQ);
    auto Dbs = std::make_shared<Matrix>("Dbs", max_s * nb, nQ);

    // => Thread Work Arrays <= //

    std::vector<std::shared_ptr<Matrix> > Tab;
    std::vector<std::shared_ptr<Matrix> > Vab;
    std::vector<std::shared_ptr<Matrix> > T2ab;
    std::vector<std::shared_ptr<Matrix> > V2ab;
    std::vector<std::shared_ptr<Matrix> > Iab;
    for (int t = 0; t < nT; t++) {
        Tab.push_back(std::make_shared<Matrix>("Tab", na, nb));
        Vab.push_back(std::make_shared<Matrix>("Vab", na, nb));
        T2ab.push_back(std::make_shared<Matrix>("T2ab", na, nb));
        V2ab.push_back(std::make_shared<Matrix>("V2ab", na, nb));
        Iab.push_back(std::make_shared<Matrix>("Iab", na, nb));
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

    double* eap = eps_aocc_A->pointer();
    double* ebp = eps_aocc_B->pointer();
    double* erp = eps_avir_A->pointer();
    double* esp = eps_avir_B->pointer();

    // => Slice D + E -> D <= //

    dfh->add_disk_tensor("Far", std::make_tuple(nr, na, nQ));

    for (size_t rstart = 0; rstart < nr; rstart += max_r) {
        size_t nrblock = (rstart + max_r >= nr ? nr - rstart : max_r);

        dfh->fill_tensor("Dar", Dar, {rstart, rstart + nrblock});
        dfh->fill_tensor("Ear", Aar, {rstart, rstart + nrblock});

        double* D2p = Darp[0];
        double* A2p = Aarp[0];
        for (long int arQ = 0L; arQ < nrblock * naQ; arQ++) {
            (*D2p++) += (*A2p++);
        }
        dfh->write_disk_tensor("Far", Dar, {rstart, rstart + nrblock});
    }

    dfh->add_disk_tensor("Fbs", std::make_tuple(ns, nb, nQ));

    for (size_t sstart = 0; sstart < ns; sstart += max_s) {
        size_t nsblock = (sstart + max_s >= ns ? ns - sstart : max_s);

        dfh->fill_tensor("Dbs", Dbs, {sstart, sstart + nsblock});
        dfh->fill_tensor("Ebs", Abs, {sstart, sstart + nsblock});

        double* D2p = Dbsp[0];
        double* A2p = Absp[0];
        for (long int bsQ = 0L; bsQ < nsblock * nbQ; bsQ++) {
            (*D2p++) += (*A2p++);
        }
        dfh->write_disk_tensor("Fbs", Dbs, {sstart, sstart + nsblock});
    }

    // => Targets <= //

    double Disp20 = 0.0;
    double ExchDisp20 = 0.0;
    double sExchDisp20 = 0.0;

    // => Local Targets <= //

    std::vector<std::shared_ptr<Matrix> > E_disp20_threads;
    std::vector<std::shared_ptr<Matrix> > E_exch_disp20_threads;
    std::vector<std::shared_ptr<Matrix> > sE_exch_disp20_threads;
    for (int t = 0; t < nT; t++) {
        E_disp20_threads.push_back(std::make_shared<Matrix>("E_disp20", na, nb));
        E_exch_disp20_threads.push_back(std::make_shared<Matrix>("E_exch_disp20", na, nb));
        sE_exch_disp20_threads.push_back(std::make_shared<Matrix>("sE_exch_disp20", sna, snb));
    }

    // => MO => LO Transform <= //

    double** UAp = Uaocc_A->pointer();
    double** UBp = Uaocc_B->pointer();

    // ==> Master Loop <== //

    double scale = 1.0;
    if (options_.get_bool("SSAPT0_SCALE")) {
        scale = sSAPT0_scale_;
    }

    for (size_t rstart = 0; rstart < nr; rstart += max_r) {
        size_t nrblock = (rstart + max_r >= nr ? nr - rstart : max_r);

        dfh->fill_tensor("Aar", Aar, {rstart, rstart + nrblock});
        dfh->fill_tensor("Far", Dar, {rstart, rstart + nrblock});
        dfh->fill_tensor("Bbr", Bbr, {rstart, rstart + nrblock});
        dfh->fill_tensor("Cbr", Cbr, {rstart, rstart + nrblock});

        for (size_t sstart = 0; sstart < ns; sstart += max_s) {
            size_t nsblock = (sstart + max_s >= ns ? ns - sstart : max_s);

            dfh->fill_tensor("Abs", Abs, {sstart, sstart + nsblock});
            dfh->fill_tensor("Fbs", Dbs, {sstart, sstart + nsblock});
            dfh->fill_tensor("Bas", Bas, {sstart, sstart + nsblock});
            dfh->fill_tensor("Cas", Cas, {sstart, sstart + nsblock});

            long int nrs = nrblock * nsblock;

#pragma omp parallel for schedule(dynamic) reduction(+ : Disp20, ExchDisp20, sExchDisp20)
            for (long int rs = 0L; rs < nrs; rs++) {
                int r = rs / nsblock;
                int s = rs % nsblock;

                int thread = 0;
#ifdef _OPENMP
                thread = omp_get_thread_num();
#endif

                double** E_disp20Tp = E_disp20_threads[thread]->pointer();
                double** E_exch_disp20Tp = E_exch_disp20_threads[thread]->pointer();
                double** sE_exch_disp20Tp = sE_exch_disp20_threads[thread]->pointer();

                double** Tabp = Tab[thread]->pointer();
                double** Vabp = Vab[thread]->pointer();
                double** T2abp = T2ab[thread]->pointer();
                double** V2abp = V2ab[thread]->pointer();
                double** Iabp = Iab[thread]->pointer();

                // => Amplitudes, Disp20 <= //

                C_DGEMM('N', 'T', na, nb, nQ, 1.0, Aarp[(r)*na], nQ, Absp[(s)*nb], nQ, 0.0, Vabp[0], nb);
                for (int a = 0; a < na; a++) {
                    for (int b = 0; b < nb; b++) {
                        Tabp[a][b] = Vabp[a][b] / (eap[a] + ebp[b] - erp[r + rstart] - esp[s + sstart]);
                    }
                }

                C_DGEMM('N', 'N', na, nb, nb, 1.0, Tabp[0], nb, UBp[0], nb, 0.0, Iabp[0], nb);
                C_DGEMM('T', 'N', na, nb, na, 1.0, UAp[0], na, Iabp[0], nb, 0.0, T2abp[0], nb);
                C_DGEMM('N', 'N', na, nb, nb, 1.0, Vabp[0], nb, UBp[0], nb, 0.0, Iabp[0], nb);
                C_DGEMM('T', 'N', na, nb, na, 1.0, UAp[0], na, Iabp[0], nb, 0.0, V2abp[0], nb);

                for (int a = 0; a < na; a++) {
                    for (int b = 0; b < nb; b++) {
                        E_disp20Tp[a][b] += 4.0 * T2abp[a][b] * V2abp[a][b];
                        Disp20 += 4.0 * T2abp[a][b] * V2abp[a][b];
                    }
                }

                // => Exch-Disp20 <= //

                // > Q1-Q3 < //

                C_DGEMM('N', 'T', na, nb, nQ, 1.0, Basp[(s)*na], nQ, Bbrp[(r)*nb], nQ, 0.0, Vabp[0], nb);
                C_DGEMM('N', 'T', na, nb, nQ, 1.0, Casp[(s)*na], nQ, Cbrp[(r)*nb], nQ, 1.0, Vabp[0], nb);
                C_DGEMM('N', 'T', na, nb, nQ, 1.0, Aarp[(r)*na], nQ, Dbsp[(s)*nb], nQ, 1.0, Vabp[0], nb);
                C_DGEMM('N', 'T', na, nb, nQ, 1.0, Darp[(r)*na], nQ, Absp[(s)*nb], nQ, 1.0, Vabp[0], nb);

                // > V,J,K < //

                C_DGER(na, nb, 1.0, &Sasp[0][s + sstart], ns, &Qbrp[0][r + rstart], nr, Vabp[0], nb);
                C_DGER(na, nb, 1.0, &Qasp[0][s + sstart], ns, &Sbrp[0][r + rstart], nr, Vabp[0], nb);
                C_DGER(na, nb, 1.0, &Qarp[0][r + rstart], nr, &SAbsp[0][s + sstart], ns, Vabp[0], nb);
                C_DGER(na, nb, 1.0, &SBarp[0][r + rstart], nr, &Qbsp[0][s + sstart], ns, Vabp[0], nb);

                C_DGEMM('N', 'N', na, nb, nb, 1.0, Vabp[0], nb, UBp[0], nb, 0.0, Iabp[0], nb);
                C_DGEMM('T', 'N', na, nb, na, 1.0, UAp[0], na, Iabp[0], nb, 0.0, V2abp[0], nb);

                for (int a = 0; a < na; a++) {
                    for (int b = 0; b < nb; b++) {
                        E_exch_disp20Tp[a][b] -= 2.0 * T2abp[a][b] * V2abp[a][b];
                        if (options_.get_bool("SSAPT0_SCALE"))
                            sE_exch_disp20Tp[a][b] -= scale * 2.0 * T2abp[a][b] * V2abp[a][b];
                        ExchDisp20 -= 2.0 * T2abp[a][b] * V2abp[a][b];
                        sExchDisp20 -= scale * 2.0 * T2abp[a][b] * V2abp[a][b];
                    }
                }
            }
        }
    }

    auto E_disp20 = std::make_shared<Matrix>("E_disp20", na, nb);
    auto E_exch_disp20 = std::make_shared<Matrix>("E_exch_disp20", na, nb);
    double** E_disp20p = E_disp20->pointer();
    double** E_exch_disp20p = E_exch_disp20->pointer();

    for (int t = 0; t < nT; t++) {
        E_disp20->add(E_disp20_threads[t]);
        E_exch_disp20->add(E_exch_disp20_threads[t]);
    }

    for (int a = 0; a < na; a++) {
        for (int b = 0; b < nb; b++) {
            Ep[a + nfa + nA][b + nfb + nB] = E_disp20p[a][b] + E_exch_disp20p[a][b];
        }
    }

    if (options_.get_bool("SSAPT0_SCALE")) {
        auto sE_exch_disp20 = std::make_shared<Matrix>("sE_exch_disp20", na, nb);
        sE_exch_disp20->copy(E_exch_disp20);
        double** sE_exch_disp20p = sE_exch_disp20->pointer();
        sE_exch_disp20->scale(sSAPT0_scale_);

        for (int a = 0; a < na; a++) {
            for (int b = 0; b < nb; b++) {
                sEp[a + nfa + nA][b + nfb + nB] = E_disp20p[a][b] + sE_exch_disp20p[a][b];
            }
        }
    }

    // E_disp20->print();
    // E_exch_disp20->print();

    scalars_["Disp20"] = Disp20;
    scalars_["Exch-Disp20"] = ExchDisp20;
    if (options_.get_bool("SSAPT0_SCALE")) scalars_["sExch-Disp20"] = sExchDisp20;
    outfile->Printf("    Disp20              = %18.12lf [Eh]\n", Disp20);
    outfile->Printf("    Exch-Disp20         = %18.12lf [Eh]\n", ExchDisp20);
    if (options_.get_bool("SSAPT0_SCALE")) outfile->Printf("    sExch-Disp20         = %18.12lf [Eh]\n", sExchDisp20);
    outfile->Printf("\n");
    // fflush(outfile);
}

std::shared_ptr<Matrix> FISAPT::extract_columns(const std::vector<int>& cols, std::shared_ptr<Matrix> A) {
    int nm = A->rowspi()[0];
    int na = A->colspi()[0];
    int ni = cols.size();

    auto A2 = std::make_shared<Matrix>("A2", nm, ni);
    double** Ap = A->pointer();
    double** A2p = A2->pointer();

    for (int m = 0; m < nm; m++) {
        for (int i = 0; i < ni; i++) {
            A2p[m][i] = Ap[m][cols[i]];
        }
    }

    return A2;
}

FISAPTSCF::FISAPTSCF(std::shared_ptr<JK> jk, double enuc, std::shared_ptr<Matrix> S, std::shared_ptr<Matrix> X,
                     std::shared_ptr<Matrix> T, std::shared_ptr<Matrix> V, std::shared_ptr<Matrix> W,
                     std::shared_ptr<Matrix> C, Options& options)
    : options_(options), jk_(jk) {
    scalars_["E NUC"] = enuc;
    matrices_["S"] = S;
    matrices_["X"] = X;
    matrices_["T"] = T;
    matrices_["V"] = V;
    matrices_["W"] = W;
    matrices_["C0"] = C;
}

FISAPTSCF::~FISAPTSCF() {}
void FISAPTSCF::compute_energy() {
    // => Sizing <= //

    int nbf = matrices_["X"]->rowspi()[0];
    int nmo = matrices_["X"]->colspi()[0];
    int nocc = matrices_["C0"]->colspi()[0];
    int nvir = nmo - nocc;

    // => One-electron potential <= //

    matrices_["H"] = std::shared_ptr<Matrix>(matrices_["T"]->clone());
    matrices_["H"]->set_name("H");
    matrices_["H"]->copy(matrices_["T"]);
    matrices_["H"]->add(matrices_["V"]);
    // matrices_["H"]->add(matrices_["W"]);

    // => Fock Matrix <= //

    matrices_["F"] = std::shared_ptr<Matrix>(matrices_["T"]->clone());
    matrices_["F"]->set_name("F");

    // => For Convenience <= //

    std::shared_ptr<Matrix> H = matrices_["H"];
    std::shared_ptr<Matrix> F = matrices_["F"];
    std::shared_ptr<Matrix> S = matrices_["S"];
    std::shared_ptr<Matrix> X = matrices_["X"];
    std::shared_ptr<Matrix> W = matrices_["W"];

    // matrices_["S"]->print();
    // matrices_["X"]->print();
    // matrices_["T"]->print();
    // matrices_["V"]->print();
    // matrices_["W"]->print();
    // matrices_["C0"]->print();

    // => Guess <= //

    std::shared_ptr<Matrix> Cocc2(matrices_["C0"]->clone());
    Cocc2->copy(matrices_["C0"]);

    // => Convergence Criteria <= //

    int maxiter = options_.get_int("MAXITER");
    double Etol = options_.get_double("E_CONVERGENCE");
    double Gtol = options_.get_double("D_CONVERGENCE");
    bool converged = false;
    double Eold = 0.0;

    outfile->Printf("    Maxiter = %11d\n", maxiter);
    outfile->Printf("    E Tol   = %11.3E\n", Etol);
    outfile->Printf("    D Tol   = %11.3E\n", Gtol);
    outfile->Printf("\n");

    // => DIIS Setup <= //

    int max_diis_vectors = options_.get_int("DIIS_MAX_VECS");
    outfile->Printf("    Max DIIS Vectors = %d\n", max_diis_vectors);
    outfile->Printf("\n");

    bool diised = false;
    auto Gsize = std::make_shared<Matrix>("Gsize", nmo, nmo);
    auto diis = std::make_shared<DIISManager>(max_diis_vectors, "FISAPT DIIS");
    diis->set_error_vector_size(1, DIISEntry::Matrix, Gsize.get());
    diis->set_vector_size(1, DIISEntry::Matrix, F.get());
    Gsize.reset();

    // ==> Master Loop <== //

    outfile->Printf("    Iter %3s: %24s %11s %11s\n", "N", "E", "dE", "|D|");
    for (int iter = 1; iter <= maxiter; iter++) {
        // => Compute Density Matrix <= //

        std::shared_ptr<Matrix> D = linalg::doublet(Cocc2, Cocc2, false, true);

        // => Compute Fock Matrix <= //

        std::vector<SharedMatrix>& Cl = jk_->C_left();
        std::vector<SharedMatrix>& Cr = jk_->C_right();

        const std::vector<SharedMatrix>& Js = jk_->J();
        const std::vector<SharedMatrix>& Ks = jk_->K();

        Cl.clear();
        Cr.clear();

        Cl.push_back(Cocc2);
        Cr.push_back(Cocc2);

        jk_->compute();

        std::shared_ptr<Matrix> J = Js[0];
        std::shared_ptr<Matrix> K = Ks[0];

        F->copy(H);
        F->add(W);
        F->add(J);
        F->add(J);
        F->subtract(K);

        // => Compute Energy <= //

        double E = scalars_["E NUC"] + D->vector_dot(H) + D->vector_dot(F) + D->vector_dot(W);
        double Ediff = E - Eold;
        scalars_["E SCF"] = E;

        // => Compute Orbital Gradient <= //

        std::shared_ptr<Matrix> G1 = linalg::triplet(F, D, S);
        std::shared_ptr<Matrix> G2 = linalg::triplet(S, D, F);
        G1->subtract(G2);
        std::shared_ptr<Matrix> G3 = linalg::triplet(X, G1, X, true, false, false);
        double Gnorm = G3->rms();

        // => Print and Check Convergence <= //

        outfile->Printf("    Iter %3d: %24.16E %11.3E %11.3E %s\n", iter, E, Ediff, Gnorm, (diised ? "DIIS" : ""));

        if (std::fabs(Ediff) < Etol && std::fabs(Gnorm) < Gtol) {
            converged = true;
            break;
        }

        Eold = E;

        // => DIIS <= //

        diis->add_entry(2, G3.get(), F.get());
        diised = diis->extrapolate(1, F.get());

        // => Diagonalize Fock Matrix <= //

        std::shared_ptr<Matrix> F2 = linalg::triplet(X, F, X, true, false, false);
        auto U2 = std::make_shared<Matrix>("C", nmo, nmo);
        auto e2 = std::make_shared<Vector>("eps", nmo);
        F2->diagonalize(U2, e2, ascending);
        std::shared_ptr<Matrix> C = linalg::doublet(X, U2, false, false);

        // => Assign New Orbitals <= //

        double** Coccp = Cocc2->pointer();
        double** Cp = C->pointer();
        for (int m = 0; m < nbf; m++) {
            for (int i = 0; i < nocc; i++) {
                Coccp[m][i] = Cp[m][i];
            }
        }

        matrices_["C"] = C;
        vectors_["eps"] = e2;
    }
    outfile->Printf("\n");

    if (converged) {
        outfile->Printf("    FISAPTSCF Converged.\n\n");
    } else {
        outfile->Printf("    FISAPTSCF Failed.\n\n");
    }

    // => Post Results <= //

    std::shared_ptr<Vector> eps = vectors_["eps"];
    auto eps_occ = std::make_shared<Vector>("eps_occ", nocc);
    auto eps_vir = std::make_shared<Vector>("eps_vir", nvir);

    double* ep = eps->pointer();
    double* eop = eps_occ->pointer();
    double* evp = eps_vir->pointer();

    for (int i = 0; i < nocc; i++) {
        eop[i] = ep[i];
    }

    for (int a = 0; a < nvir; a++) {
        evp[a] = ep[a + nocc];
    }

    vectors_["eps_occ"] = eps_occ;
    vectors_["eps_vir"] = eps_vir;

    std::shared_ptr<Matrix> C = matrices_["C"];
    auto Cocc = std::make_shared<Matrix>("Cocc", nbf, nocc);
    auto Cvir = std::make_shared<Matrix>("Cvir", nbf, nvir);

    double** Cp = C->pointer();
    double** Cop = Cocc->pointer();
    double** Cvp = Cvir->pointer();

    for (int m = 0; m < nbf; m++) {
        for (int i = 0; i < nocc; i++) {
            Cop[m][i] = Cp[m][i];
        }
    }

    for (int m = 0; m < nbf; m++) {
        for (int a = 0; a < nvir; a++) {
            Cvp[m][a] = Cp[m][a + nocc];
        }
    }

    matrices_["Cocc"] = Cocc;
    matrices_["Cvir"] = Cvir;

    const std::vector<SharedMatrix>& Js = jk_->J();
    const std::vector<SharedMatrix>& Ks = jk_->K();

    matrices_["J"] = std::shared_ptr<Matrix>(Js[0]->clone());
    matrices_["K"] = std::shared_ptr<Matrix>(Ks[0]->clone());
    matrices_["J"]->copy(Js[0]);
    matrices_["K"]->copy(Ks[0]);
    matrices_["J"]->set_name("J");
    matrices_["K"]->set_name("K");

    // => Print Final Info <= //

    outfile->Printf("    Final SCF Energy: %24.16E [Eh]\n\n", scalars_["E SCF"]);

    print_orbitals("Occupied Orbital Energies", 1, eps_occ);
    print_orbitals("Virtual Orbital Energies", nocc + 1, eps_vir);
}
void FISAPTSCF::print_orbitals(const std::string& header, int start, std::shared_ptr<Vector> eps) {
    outfile->Printf("   => %s <=\n\n", header.c_str());
    outfile->Printf("    ");
    int n = eps->dimpi()[0];
    double* ep = eps->pointer();
    int count = 0;
    for (int i = 0; i < n; i++) {
        outfile->Printf("%4d %11.6f  ", i + start, ep[i]);
        if (count++ % 3 == 2 && count != n) outfile->Printf("\n    ");
    }
    outfile->Printf("\n\n");
}

CPHF_FISAPT::CPHF_FISAPT() {}
CPHF_FISAPT::~CPHF_FISAPT() {}
void CPHF_FISAPT::compute_cphf() {
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

    preconditioner(r_A, z_A, eps_occ_A_, eps_vir_A_);
    preconditioner(r_B, z_B, eps_occ_B_, eps_vir_B_);

    // Uncoupled value
    // outfile->Printf("(A<-B): %24.16E\n", -2.0 * z_A->vector_dot(w_A_));
    // outfile->Printf("(B<-A): %24.16E\n", -2.0 * z_B->vector_dot(w_B_));

    p_A->copy(z_A);
    p_B->copy(z_B);

    double zr_old_A = z_A->vector_dot(r_A);
    double zr_old_B = z_B->vector_dot(r_B);

    double r2A = 1.0;
    double r2B = 1.0;

    double b2A = sqrt(w_A_->vector_dot(w_A_));
    double b2B = sqrt(w_B_->vector_dot(w_B_));

    outfile->Printf("  ==> CPHF Iterations <==\n\n");

    outfile->Printf("    Maxiter     = %11d\n", maxiter_);
    outfile->Printf("    Convergence = %11.3E\n", delta_);
    outfile->Printf("\n");

    std::time_t start;
    std::time_t stop;

    start = std::time(nullptr);

    outfile->Printf("    -----------------------------------------\n");
    outfile->Printf("    %-4s %11s  %11s  %10s\n", "Iter", "Monomer A", "Monomer B", "Time [s]");
    outfile->Printf("    -----------------------------------------\n");
    // fflush(outfile);

    int iter;
    for (iter = 0; iter < maxiter_; iter++) {
        std::map<std::string, std::shared_ptr<Matrix> > b;
        if (r2A > delta_) {
            b["A"] = p_A;
        }
        if (r2B > delta_) {
            b["B"] = p_B;
        }

        std::map<std::string, std::shared_ptr<Matrix> > s = product(b);

        if (r2A > delta_) {
            std::shared_ptr<Matrix> s_A = s["A"];
            double alpha = r_A->vector_dot(z_A) / p_A->vector_dot(s_A);
            if (alpha < 0.0) {
                throw PSIEXCEPTION("Monomer A: A Matrix is not SPD");
            }
            size_t no = x_A_->nrow();
            size_t nv = x_A_->ncol();
            double** xp = x_A_->pointer();
            double** rp = r_A->pointer();
            double** pp = p_A->pointer();
            double** sp = s_A->pointer();
            C_DAXPY(no * nv, alpha, pp[0], 1, xp[0], 1);
            C_DAXPY(no * nv, -alpha, sp[0], 1, rp[0], 1);
            r2A = sqrt(C_DDOT(no * nv, rp[0], 1, rp[0], 1)) / b2A;
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
            C_DAXPY(no * nv, alpha, pp[0], 1, xp[0], 1);
            C_DAXPY(no * nv, -alpha, sp[0], 1, rp[0], 1);
            r2B = sqrt(C_DDOT(no * nv, rp[0], 1, rp[0], 1)) / b2B;
        }

        stop = std::time(nullptr);
        outfile->Printf("    %-4d %11.3E%1s %11.3E%1s %10ld\n", iter + 1, r2A, (r2A < delta_ ? "*" : " "), r2B,
                        (r2B < delta_ ? "*" : " "), stop - start);
        // fflush(outfile);

        if (r2A <= delta_ && r2B <= delta_) {
            break;
        }

        if (r2A > delta_) {
            preconditioner(r_A, z_A, eps_occ_A_, eps_vir_A_);
            double zr_new = z_A->vector_dot(r_A);
            double beta = zr_new / zr_old_A;
            zr_old_A = zr_new;
            int no = x_A_->nrow();
            int nv = x_A_->ncol();
            double** pp = p_A->pointer();
            double** zp = z_A->pointer();
            C_DSCAL(no * nv, beta, pp[0], 1);
            C_DAXPY(no * nv, 1.0, zp[0], 1, pp[0], 1);
        }

        if (r2B > delta_) {
            preconditioner(r_B, z_B, eps_occ_B_, eps_vir_B_);
            double zr_new = z_B->vector_dot(r_B);
            double beta = zr_new / zr_old_B;
            zr_old_B = zr_new;
            int no = x_B_->nrow();
            int nv = x_B_->ncol();
            double** pp = p_B->pointer();
            double** zp = z_B->pointer();
            C_DSCAL(no * nv, beta, pp[0], 1);
            C_DAXPY(no * nv, 1.0, zp[0], 1, pp[0], 1);
        }
    }

    outfile->Printf("    -----------------------------------------\n");
    outfile->Printf("\n");
    // fflush(outfile);

    if (iter == maxiter_) throw PSIEXCEPTION("CPHF did not converge.");
}

void CPHF_FISAPT::preconditioner(std::shared_ptr<Matrix> r, std::shared_ptr<Matrix> z, std::shared_ptr<Vector> o,
                                 std::shared_ptr<Vector> v) {
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

std::map<std::string, std::shared_ptr<Matrix> > CPHF_FISAPT::product(
    std::map<std::string, std::shared_ptr<Matrix> > b) {
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
        auto T = std::make_shared<Matrix>("T", nso, no);
        double** Tp = T->pointer();
        C_DGEMM('N', 'T', nso, no, nv, 1.0, Cp[0], nv, bp[0], nv, 0.0, Tp[0], no);
        Cr.push_back(T);
    }

    if (do_B) {
        Cl.push_back(Cocc_B_);
        int no = b["B"]->nrow();
        int nv = b["B"]->ncol();
        int nso = Cvir_B_->nrow();
        double** Cp = Cvir_B_->pointer();
        double** bp = b["B"]->pointer();
        auto T = std::make_shared<Matrix>("T", nso, no);
        double** Tp = T->pointer();
        C_DGEMM('N', 'T', nso, no, nv, 1.0, Cp[0], nv, bp[0], nv, 0.0, Tp[0], no);
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
        auto T = std::make_shared<Matrix>("T", no, nso);
        s["A"] = std::make_shared<Matrix>("S", no, nv);
        double** Cop = Cocc_A_->pointer();
        double** Cvp = Cvir_A_->pointer();
        double** Jp = Jv->pointer();
        double** Tp = T->pointer();
        double** Sp = s["A"]->pointer();
        C_DGEMM('T', 'N', no, nso, nso, 1.0, Cop[0], no, Jp[0], nso, 0.0, Tp[0], nso);
        C_DGEMM('N', 'N', no, nv, nso, 1.0, Tp[0], nso, Cvp[0], nv, 0.0, Sp[0], nv);

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
        auto T = std::make_shared<Matrix>("T", no, nso);
        s["B"] = std::make_shared<Matrix>("S", no, nv);
        double** Cop = Cocc_B_->pointer();
        double** Cvp = Cvir_B_->pointer();
        double** Jp = Jv->pointer();
        double** Tp = T->pointer();
        double** Sp = s["B"]->pointer();
        C_DGEMM('T', 'N', no, nso, nso, 1.0, Cop[0], no, Jp[0], nso, 0.0, Tp[0], nso);
        C_DGEMM('N', 'N', no, nv, nso, 1.0, Tp[0], nso, Cvp[0], nv, 0.0, Sp[0], nv);

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

}  // Namespace fisapt

}  // Namespace psi

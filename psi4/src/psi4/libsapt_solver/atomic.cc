/*
 *@BEGIN LICENSE
 *
 * PSI4: an ab initio quantum chemistry software package
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *@END LICENSE
 */

#include "atomic.h"
//#include <psi4/libmints/mints.h>
#include <psi4/libmints/molecule.h>
#include <psi4/libmints/basisset.h>
#include "psi4/libmints/vector.h"
#include <psi4/libfock/cubature.h>
#include <psi4/libfock/points.h>
#include <psi4/libdiis/diismanager.h>
#include <psi4/libqt/qt.h>
#include <psi4/psi4-dec.h>
#include <psi4/physconst.h>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"

#include <sstream>
#include <fstream>
#include <iostream>
#include <regex>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace psi;
using namespace std;

namespace psi {

AtomicDensity::AtomicDensity() : 
    print_(1),
    debug_(0),
    bench_(0)
{
}
AtomicDensity::~AtomicDensity() 
{
}
std::shared_ptr<AtomicDensity> AtomicDensity::build(const std::string& type, std::shared_ptr<BasisSet> basis, Options& options)
{
    AtomicDensity* target;
    if (type == "STOCKHOLDER") {
        StockholderDensity* isa = new StockholderDensity();
        //isa->convergence_ = options.get_double("ISA_CONVERGENCE");
        isa->convergence_ = 1.0e-6;
        //isa->maxiter_ = options.get_int("ISA_MAXITER");
        isa->maxiter_ = 1000;
        //isa->diis_ = options.get_bool("ISA_DIIS");
        isa->diis_ = false;
        //isa->diis_min_vecs_ = options.get_int("ISA_DIIS_MIN_VECS");
        isa->diis_min_vecs_ = 100;
        //isa->diis_max_vecs_ = options.get_int("ISA_DIIS_MAX_VECS");
        isa->diis_max_vecs_ = 100;
        //isa->diis_flush_ = options.get_int("ISA_DIIS_FLUSH");
        isa->diis_flush_ = 100;
        //isa->diis_flush_maxiter_ = options.get_int("ISA_DIIS_FLUSH_MAXITER");
        isa->diis_flush_maxiter_ = 500;
        //isa->guess_ = options.get_str("ISA_GUESS");
        isa->guess_ = "NONE";
        target = static_cast<AtomicDensity*>(isa); 
    } else {
        throw PSIEXCEPTION("AtomicDensity::build: Unrecognized Atomic Density Type");
    }

    target->set_print(options.get_int("PRINT"));
    target->set_debug(options.get_int("DEBUG"));
    target->set_bench(options.get_int("BENCH"));

    target->primary_ = basis;
    target->molecule_ = basis->molecule();
    target->grid_ = std::shared_ptr<MolecularGrid>(new DFTGrid(basis->molecule(),basis,options));
    return std::shared_ptr<AtomicDensity>(target);
}
void AtomicDensity::compute_total_density()
{
    // Fast grid sizing
    int npoints2 = grid_->npoints(); // may be less than NA x 75 x 302 b/c pruning
    double* x2p = grid_->x();
    double* y2p = grid_->y();
    double* z2p = grid_->z();
    double* w2p = grid_->w();
    int* index = grid_->index();  // this is empty !?!?
    outfile->Printf("Total number of points is %d (npoints2) \n", npoints2);

    // Density computer (in blocks)
    std::shared_ptr<RKSFunctions> points = std::shared_ptr<RKSFunctions>(new RKSFunctions(primary_,grid_->max_points(),grid_->max_functions()));
    points->set_ansatz(0);
    points->set_pointers(D_);
    std::shared_ptr<Vector> rho3 = points->point_value("RHO_A");
    double* rho3p = rho3->pointer();

    // Build density (in fast ordering)
    std::shared_ptr<Vector> rho2(new Vector("rho2", npoints2));
    double* rho2p = rho2->pointer();
    const std::vector<std::shared_ptr<BlockOPoints> >& blocks = grid_->blocks();    
    size_t offset = 0L;
    for (int ind = 0; ind < blocks.size(); ind++) {
        int npoints = blocks[ind]->npoints();
        points->compute_points(blocks[ind]);
        C_DAXPY(npoints,1.0,rho3p,1,&rho2p[offset],1);        
        offset += npoints;
        //rho3->print_out();
    }

    // How many clean points?
    const std::vector<std::vector<std::shared_ptr<SphericalGrid> > >& spheres = grid_->spherical_grids();
    int npoints = 0;
    for (int A = 0; A < spheres.size(); A++) {
        for (int R = 0; R < spheres[A].size(); R++) {
            npoints += spheres[A][R]->npoints();
        }
    }
    outfile->Printf("Total number of points is %d (npoints) \n", npoints);

    // Targets
    x_ = std::shared_ptr<Vector>(new Vector("x",npoints)); 
    y_ = std::shared_ptr<Vector>(new Vector("y",npoints)); 
    z_ = std::shared_ptr<Vector>(new Vector("z",npoints)); 
    w_ = std::shared_ptr<Vector>(new Vector("w",npoints)); 
    rho_ = std::shared_ptr<Vector>(new Vector("rho",npoints)); 
    double* xp = x_->pointer();
    double* yp = y_->pointer();
    double* zp = z_->pointer();
    double* wp = w_->pointer();
    double* rhop = rho_->pointer();

    //std::cout << "npoints: " << npoints << " npoints2: " << npoints2 << std::endl;

    //for(int i = 0; i < npoints; ++i) {
    //    outfile->Printf(" %6d: (%15.5f %15.5f %15.5f %15.5f ) vs  (%15.5f %15.5f %15.5f %15.5f) \n", i, xp[i], yp[i], zp[i], wp[i], x2p[i], y2p[i], z2p[i], w2p[i]);
    //}

    for (int i = 0; i < npoints2; i++) {
        //xp[index[i]] = x2p[i];
        //yp[index[i]] = y2p[i];
        //zp[index[i]] = z2p[i];
        //wp[index[i]] = w2p[i];
        //rhop[index[i]] = rho2p[i];
        xp[index[i]] = x2p[i];
        yp[index[i]] = y2p[i];
        zp[index[i]] = z2p[i];
        wp[index[i]] = w2p[i];
        rhop[index[i]] = rho2p[i];
        //outfile->Printf(" In the code %d %d %f \n", i, index[i], rho2p[i]);
    }

    // Debug stuff 
    //x_->print();
    //y_->print();
    //z_->print();
    //w_->print();
    //rho_->print();
}

StockholderDensity::StockholderDensity() : AtomicDensity() 
{
}
StockholderDensity::~StockholderDensity()
{
}
void StockholderDensity::print_header() const 
{
    outfile->Printf("  ==> Stockholder Atomic Densities <==\n\n");
    molecule_->print();
    primary_->print();
    grid_->print();

}
void StockholderDensity::compute(std::shared_ptr<Matrix> D)
{
    D_ = D;

    print_header();

    //D->print_out();

    compute_total_density(); // I don't think this correctly tracks grid points w.r.t central atom

    // ==> Main ISA Algorithm <== //

    // Where are the true atoms?
    std::vector<int> Aind;  // [0,  1,  2,              6,  7,      5]
    std::vector<int> Aind2; // [0,  1,  2, -1, -1, -1,  6,  7, -1,  9]
    for (int A = 0; A < molecule_->natom(); A++) {
        if (molecule_->Z(A) != 0) {
            Aind2.push_back(Aind.size());
            Aind.push_back(A);
        } else {
            Aind2.push_back(-1);
        }
    }
    int nA = Aind.size(); // number of non ghost atoms
    int nA2 = molecule_->natom(); // number of total atoms

    // Grid indexing
    int nP = x_->dimpi()[0];
    double* xp = x_->pointer();
    double* yp = y_->pointer();
    double* zp = z_->pointer();
    double* wp = w_->pointer();
    double* rhop = rho_->pointer();

    outfile->Printf("  Electron count is %24.16E\n", C_DDOT(nP,wp,1,rhop,1));

    const std::vector<std::shared_ptr<RadialGrid> >& rads = grid_->radial_grids();
    const std::vector<std::vector<std::shared_ptr<SphericalGrid> > >& spheres = grid_->spherical_grids();

    // True atom spherical quadratures 
    std::vector<std::vector<int> > orders;
    std::vector<std::vector<double> > wrs;
    rs_.resize(nA);
    ws_.resize(nA);
    wrs.resize(nA);
    orders.resize(nA);
    int nstate = 0;
    for (int A = 0; A < nA; A++) { // For each real atom
        int Aabs = Aind[A];
        std::vector<double> rs2;  // Grid radii (75 points)
        std::vector<double> ws2;  // ISA weights (75 points)
        std::vector<double> wrs2; // Quadrature weights (75 points)
        std::vector<std::pair<double, int> > index;  // (Grid radius, Grid radius index)
        outfile->Printf("%5d Points\n",rads[Aabs]->npoints());
        for (int R = 0; R < rads[Aabs]->npoints(); R++) { // For all 75 R values
            rs2.push_back(rads[Aabs]->r()[R]);
            wrs2.push_back(rads[Aabs]->w()[R]);
            ws2.push_back(1.0); // Initial guess if NONE is selected
            index.push_back(std::pair<double, int>(rs2[R],R));
            nstate++;
        } 
        std::sort(index.begin(), index.end());
        for (int R = 0; R < rads[Aabs]->npoints(); R++) {
            int Rold = index[R].second;
            rs_[A].push_back(rs2[Rold]);
            ws_[A].push_back(ws2[Rold]);
            wrs[A].push_back(wrs2[Rold]);
            orders[A].push_back(Rold);
        } 
    }
    std::vector<std::vector<double> > grads = ws_;

    std::vector<std::vector<int> > orders2;
    orders2.resize(nA);
    for (int A = 0; A < nA; A++) {
        orders2[A].resize(orders[A].size());
        for (int R = 0; R < orders[A].size(); R++) {
            orders2[A][orders[A][R]] = R;
        }
    }    

    // Low-memory atomic charges target
    int max_points = 0;
    std::vector<std::vector<std::vector<double> > > Q;
    std::vector<std::vector<std::vector<double> > > P;
    std::vector<std::vector<std::vector<double> > > T;
    std::vector<int> atomic_points;
    Q.resize(nA);
    P.resize(nA);
    T.resize(nA);
    atomic_points.resize(nA);
    for (int A = 0; A < nA; A++) {
        int Aabs = Aind[A];
        Q[A].resize(orders[A].size());
        P[A].resize(orders[A].size());
        T[A].resize(orders[A].size());
        int atom_points = 0;
        for (int R = 0; R < orders[A].size(); R++) {
            int Rabs = orders[A][R];
            Q[A][R].resize(spheres[Aabs][Rabs]->npoints());
            P[A][R].resize(spheres[Aabs][Rabs]->npoints());
            T[A][R].resize(spheres[Aabs][Rabs]->npoints());
            atom_points += spheres[Aabs][Rabs]->npoints();
        }
        atomic_points[A] = atom_points;
        max_points = (max_points >= atom_points ? max_points : atom_points);
    }
    
    // Temps 
    std::shared_ptr<Matrix> Q2(new Matrix("Q2", 1, max_points));
    double** Q2p = Q2->pointer();

    std::shared_ptr<Matrix> rho0(new Matrix("rho0", 1, max_points));
    double** rho0p = rho0->pointer();

    // DIIS Setup
    std::shared_ptr<Matrix> state(new Matrix("State", nstate, 1));
    std::shared_ptr<Matrix> error(new Matrix("Error", nstate, 1));
    double** statep = state->pointer();
    double** errorp = error->pointer();

    std::shared_ptr<DIISManager> diis_manager(new DIISManager(diis_max_vecs_, "ISA DIIS vector", DIISManager::LargestError, DIISManager::OnDisk));
    diis_manager->set_error_vector_size(1, DIISEntry::Matrix, error.get());
    diis_manager->set_vector_size(1, DIISEntry::Matrix, state.get());

    // Store last iteration
    std::vector<std::vector<double> > ws_old = ws_;

    // New DIIS vector
    std::vector<std::vector<double> > ws_diis = ws_;

    // => Master Loop <= //

    outfile->Printf("   > ISA Iterations <\n\n");
    outfile->Printf("    Guess:              %11s\n", guess_.c_str());
    outfile->Printf("    Convergence:        %11.3E\n", convergence_);
    outfile->Printf("    Maximum iterations: %11d\n", maxiter_);   
    outfile->Printf("    DIIS:               %11s\n", (diis_ ? "Yes" : "No"));
    outfile->Printf("    DIIS Min Vecs:      %11d\n", diis_min_vecs_);
    outfile->Printf("    DIIS Max Vecs:      %11d\n", diis_max_vecs_);
    outfile->Printf("    DIIS Flush Vecs:    %11d\n", diis_flush_);
    outfile->Printf("    DIIS Flush Maxiter: %11d\n", diis_flush_maxiter_);
    outfile->Printf("\n");

    // => Advanced Guess <= //
    
    guess();

    std::vector<std::vector<double> > wref = ws_; // To weight the gradient

    bool converged = false;
    outfile->Printf("    %14s %24s %24s %24s %2s\n", "@ISA Iteration", "Delta", "Gradient", "Metric", "ID");
    for (int iter = 0, diis_iter = 0; iter <= maxiter_; iter++) {

        // Checkpoint last iteration
        ws_old = ws_;
    
        // Needed atomic weights
        int offset = 0;
        for (int A = 0; A < nA2; A++) {
            if (molecule_->Z(A) != 0) {
                int Arel = Aind2[A]; 
                compute_weights(atomic_points[Arel],&xp[offset],&yp[offset],&zp[offset],Q2p,&rhop[offset],Arel,rho0p[0]);
                int offset2 = 0;
                for (int R = 0; R < orders[Arel].size(); R++) {
                    for (int k = 0; k < spheres[A][orders[Arel][R]]->npoints(); k++) {
                        Q[Arel][orders[Arel][R]][k] = Q2p[0][offset2];
                        P[Arel][orders[Arel][R]][k] = rhop[offset2 + offset];
                        T[Arel][orders[Arel][R]][k] = rho0p[0][offset2];
                        offset2++; 
                    }
                }
            }
            for (int R = 0; R < spheres[A].size(); R++) {
                offset += spheres[A][R]->npoints();
            }
        }

        // Approximate Hirshfelder metric and gradient norm
        double O = 0.0;
        double grad = 0.0;
        for (int A = 0; A < Q.size(); A++) {
            for (int R = 0; R < Q[A].size(); R++) { 
                double val0 = 0.0;
                double val1 = 0.0;
                for (int k = 0; k < Q[A][R].size(); k++) {
                    if (T[A][R][k] > 1.0E-300 && P[A][R][k] > 1.0E-300) {
                        val0 += T[A][R][k] - P[A][R][k] * log(T[A][R][k] / P[A][R][k]);
                        val1 += 1.0 - P[A][R][k] / T[A][R][k];
                    } else {
                        val0 += 0.0;
                        val1 += 1.0;
                    }
                }
                val0 *= wrs[A][R] / Q[A][R].size();
                val1 *= wref[A][R] * wrs[A][R] / Q[A][R].size();
                grad += val1 * val1;
                grads[A][R] = val1;
                O += val0;
            }
        }
        grad = sqrt(1.0 / Q.size() * grad);

        std::string signal = "";
        if (iter > 1 && iter < diis_flush_maxiter_ && iter % diis_flush_ == 0) {

            // Approximate Newton-Raphson Thing
            for (int A = 0; A < Q.size(); A++) {
                for (int R = 0; R < Q[A].size(); R++) { 
                    double val1 = 0.0;
                    double val2 = 0.0;
                    for (int k = 0; k < Q[A][R].size(); k++) {
                        val1 += 1.0 - P[A][R][k] / T[A][R][k];
                        val2 += P[A][R][k] / (T[A][R][k] * T[A][R][k]);
                    }
                    ws_[A][R] -= val1 / val2;
                    if (ws_[A][R] <= 0.0) {
                        double val = 0.0;
                        for (int k = 0; k < Q[A][R].size(); k++) {
                            val += Q[A][R][k];
                        }
                        val /= Q[A][R].size();
                        ws_[A][R] = val;
                    }
                }
            }

            signal = "NR";

        } else {

            // Spherical averaging
            for (int A = 0; A < Q.size(); A++) {
                for (int R = 0; R < Q[A].size(); R++) { 
                    double val = 0.0;
                    for (int k = 0; k < Q[A][R].size(); k++) {
                        val += Q[A][R][k];
                    }
                    val /= Q[A][R].size();
                    ws_[A][R] = val;
                }
            }

            signal = "RT";
        }

        // DIIS error
        for (int A = 0; A < Q.size(); A++) {
            for (int R = 0; R < Q[A].size(); R++) { 
                double val = 0.0;
                for (int k = 0; k < Q[A][R].size(); k++) {
                    val += 1.0 - P[A][R][k] / ws_[A][R];
                }
                val /= Q[A][R].size();
                ws_diis[A][R] = wrs[A][R] * val;
            }
        }

        // Residual and error norm
        double norm = 0.0;
        for (int A = 0; A < ws_.size(); A++) {
            double dwval = 0.0;
            for (int R = 0; R < ws_[A].size(); R++) {
                dwval += wrs[A][R] * (ws_[A][R] - ws_old[A][R]) * (ws_[A][R] - ws_old[A][R]);
            }
            norm += 1.0 / (double) ws_.size() * sqrt(dwval);
        }

        // Print iterative trace
        outfile->Printf("    @ISA Iter %4d %20.12E %20.12E %20.12E %2s\n", iter, norm, grad, O, signal.c_str());

        // Convergence check
        if (norm < convergence_) { 
            converged = true;
            break; 
        }

        // Periodically flush the DIIS subspace
        if (diis_ && iter > 0 && iter % diis_flush_ == 0) {
            diis_manager->reset_subspace();
            diis_iter = 0;
        }

        // DIIS Add
        if (diis_ && iter > 0) {
            int offset2 = 0;
            for (int A = 0; A < ws_.size(); A++) {
                for (int R = 0; R < ws_[A].size(); R++) {
                    statep[0][offset2] = (ws_[A][R] == 0.0 ? -std::numeric_limits<double>::infinity() : log(ws_[A][R]));
                    //statep[0][offset2] = ws_[A][R];
                    errorp[0][offset2] = ws_diis[A][R];
                    offset2++;
                }
            }
            diis_manager->add_entry(2,error.get(),state.get());
            diis_iter++;
        }

        // DIIS Extrapolate
        if (diis_ && diis_iter >= diis_min_vecs_) {
            diis_manager->extrapolate(1,state.get());
            int offset2 = 0;
            for (int A = 0; A < ws_.size(); A++) {
                for (int R = 0; R < ws_[A].size(); R++) {
                    ws_[A][R] = exp(statep[0][offset2]);
                    //ws_[A][R] = statep[0][offset2];
                    offset2++;
                }
            }
            outfile->Printf("DIIS");
        }

 
    }

    //if (bench_) {
    //    std::stringstream ss1;
    //    ss1 << "ISA_" << name_ << "_ws.dat";
    //    FILE* fh1 = fopen(ss1.str().c_str(), "w");
    //    for (int A = 0; A < ws_.size(); A++) {
    //        for (int R = 0; R < ws_[A].size(); R++) {
    //            fh1->Printf("%24.16E ", ws_[A][R]);
    //        }
    //        fh1->Printf("\n");
    //    }
    //    fclose(fh1);
    //    
    //    std::stringstream ss2;
    //    ss2 << "ISA_" << name_ << "_rs.dat";
    //    FILE* fh2 = fopen(ss2.str().c_str(), "w");
    //    for (int A = 0; A < ws_.size(); A++) {
    //        for (int R = 0; R < ws_[A].size(); R++) {
    //            outfile->Printf(fh2, "%24.16E ", rs_[A][R]);
    //        }
    //        outfile->Printf(fh2,"\n");
    //    }
    //    fclose(fh2);
    //    
    //    std::stringstream ss3;
    //    ss3 << "ISA_" << name_ << "_gs.dat";
    //    FILE* fh3 = fopen(ss3.str().c_str(), "w");
    //    for (int A = 0; A < ws_.size(); A++) {
    //        for (int R = 0; R < ws_[A].size(); R++) {
    //            outfile->Printf(fh3, "%24.16E ", grads[A][R]);
    //        }
    //        outfile->Printf(fh3,"\n");
    //    }
    //    fclose(fh3);
    //}

    diis_manager->delete_diis_file();

    outfile->Printf("\n");
    if (converged) { 
        outfile->Printf("    ISA Converged.\n\n"); 
    } else {
        outfile->Printf("    ISA Failed.\n\n"); 
    }

    // => Compute normalizations for later <= //
    
    // Target
    N_ = std::shared_ptr<Vector>(new Vector("N", nA));
    double* Np = N_->pointer();
    
    std::shared_ptr<Matrix> Q3(new Matrix("Q3", nA, max_points));
    double** Q3p = Q3->pointer();

    for (int index = 0; index < nP; index+=max_points) {
        int points = (index + max_points >= nP ? nP - index : max_points);
        compute_weights(points,&xp[index],&yp[index],&zp[index],Q3p,&rhop[index]);
        for (int A = 0; A < nA; A++) {
            Np[A] += C_DDOT(points,&wp[index],1,Q3p[A],1);
        }
    }
}
void StockholderDensity::compute_weights(int nP, double* xp, double* yp, double* zp, double** wp, double* rhop, int atom, double* rho0p)
{
    // Where are the true atoms?
    std::vector<int> Aind;
    std::vector<int> Aind2;
    for (int A = 0; A < molecule_->natom(); A++) {
        if (molecule_->Z(A) != 0) {
            Aind2.push_back(Aind.size());
            Aind.push_back(A);
        } else {
            Aind2.push_back(-1);
        }
    }
    int nA = Aind.size();
    int nA2 = molecule_->natom();
    
    // I like to work in log space for interpolation window root finding
    std::vector<std::vector<double> > ls;
    ls.resize(nA);
    for (int A = 0; A < rs_.size(); A++) {
        for (int R = 0; R < rs_[A].size(); R++) {
            ls[A].push_back(log(rs_[A][R]));
        }
    }

    // Doesn't like being on the stack?
    int nthreads = 1;
    //#ifdef _OPENMP
    //    nthreads = omp_get_max_threads();
    //#endif

    std::vector<double> wA(nA, 0.0);
    
    // Compute Q_A^P via W_A^P
    //#pragma omp parallel for schedule(dynamic) 
    for (int P = 0; P < nP; P++) {
        double xc = xp[P];
        double yc = yp[P];
        double zc = zp[P];

        int thread = 0;
        //#ifdef _OPENMP
        //    thread = omp_get_thread_num();
        //#endif
        double wT = 0.0;
    
        for (int A = 0; A < nA; A++) {
            int Aabs = Aind[A];

            // Default value
            wA[A] = 0.0;

            // What is R_A^P?
            double xA = molecule_->x(Aabs);
            double yA = molecule_->y(Aabs);
            double zA = molecule_->z(Aabs);
            double R = sqrt((xA - xc) * (xA - xc) +
                            (yA - yc) * (yA - yc) +
                            (zA - zc) * (zA - zc));

            // Interpolation problem
            const std::vector<double>& rv = rs_[A];
            const std::vector<double>& lv = ls[A];
            const std::vector<double>& wv = ws_[A];
            int nR = rv.size();

            // Root solving to determine window (log-transformed Regula Falsi/Illinois, like a ninja)
            int ind = 0;
            double lR = log(R);
            if (R <= rv[0]) {
                if (fabs(R - rv[0]) < 1.0E-12 * rv[0]) {
                    ind = 0; // Within epsilon of the inside shell
                } else {
                    continue; // Too close inside
                }
            } else if (R >= rv[nR - 1]) {
                if (fabs(R - rv[nR - 1]) < 1.0E-12 * rv[nR - 1]) {
                    ind = nR - 2; // Within epsilon of the outside shell
                } else {
                    continue; // Too far outside
                }
            } else {
                double dl = lv[0] - lR; // Negative
                double dr = lv[nR - 1] - lR; // Positive
                double xl = 0;
                double xr = nR-1;
                int retl = 0;
                int retr = 0;
                do {
                    double xc = xr - dr * (xr - xl) / (dr - dl);
                    int xind = (int) xc;
                    if (xind == nR - 1 && lv[xind - 1] - lR <= 0.0) {
                        ind = nR - 2;
                        break;
                    } else if (lv[xind] - lR <= 0.0 && lv[xind + 1] - lR > 0.0) {
                        ind = xind;
                        break;
                    }
                    double dc = lv[xind] - lR;
                    if (dc <= 0.0) {
                        xl = (double) xind;
                        dl = dc;
                        retl = 0;
                        retr++;
                    } else {
                        xr = (double) xind;
                        dr = dc;
                        retr = 0;
                        retl++;
                    }
                    if (retr > 1) {
                        dr *= 0.5;
                    } 
                    if (retl > 1) {
                        dl *= 0.5;
                    } 
                } while (true);
            }

            // Exponential interpolation pair
            double Rl = rs_[A][ind];
            double Rr = rs_[A][ind+1];
            double wl = ws_[A][ind];
            double wr = ws_[A][ind+1];

            if (wl == 0.0 || wr == 0.0) continue;

            double lwl = log(wl);
            double lwr = log(wr);

            wA[A] = exp(lwl + (lwr - lwl)*(R - Rl)/(Rr - Rl));

            wT += wA[A];
        }
        if (wT == 0.0) wT = 1.0; // Don't NaN due to no density, yo?
        
        // => Output <= //

        double scale = (rhop == NULL ? 1.0 / wT : rhop[P] / wT);
        if (atom == -1) {
            for (int A = 0; A < nA; A++) {
                wp[A][P] = wA[A] * scale;
            }
        } else {
            wp[0][P] = wA[atom] * scale;
        }

        if (rho0p != NULL) {
            rho0p[P] = wT;
        }
    }

}
void StockholderDensity::compute_charges(double scale) 
{
    // Where are the true atoms?
    std::vector<int> Aind;
    std::vector<int> Aind2;
    for (int A = 0; A < molecule_->natom(); A++) {
        if (molecule_->Z(A) != 0) {
            Aind2.push_back(Aind.size());
            Aind.push_back(A);
        } else {
            Aind2.push_back(-1);
        }
    }
    int nA = Aind.size();
    int nA2 = molecule_->natom();

    // Where are the atomic charges?
    double* Np = N_->pointer();

    // Print    
    outfile->Printf("   > Atomic Charges <\n\n");
    outfile->Printf("    %4s %3s %11s %11s %11s\n", 
        "N", "Z", "Nuclear", "Electronic", "Atomic");
    double Ztot = 0.0;
    double Qtot = 0.0;
    for (int A = 0; A < nA; A++) {
        int Aabs = Aind[A];
        double Z = molecule_->Z(Aabs);
        double Q = -scale * Np[A];
        outfile->Printf("    %4d %3s %11.3E %11.3E %11.3E\n", 
            Aabs+1, molecule_->symbol(Aabs).c_str(), Z, Q, Z + Q);
        Ztot += Z;
        Qtot += Q;
    }
    outfile->Printf("    %8s %11.3E %11.3E %11.3E\n", 
            "Total", Ztot, Qtot, Ztot + Qtot);
    outfile->Printf("\n");

    outfile->Printf("    True Molecular Charge: %11.3E\n", (double) molecule_->molecular_charge());
    outfile->Printf("    Grid Molecular Charge: %11.3E\n", Ztot + Qtot);
    outfile->Printf("    Grid Error:            %11.3E\n", Ztot + Qtot - (double) molecule_->molecular_charge());
    outfile->Printf("\n");

}
std::shared_ptr<Matrix> StockholderDensity::charges(double scale) 
{
    // Where are the true atoms?
    std::vector<int> Aind;
    std::vector<int> Aind2;
    for (int A = 0; A < molecule_->natom(); A++) {
        if (molecule_->Z(A) != 0) {
            Aind2.push_back(Aind.size());
            Aind.push_back(A);
        } else {
            Aind2.push_back(-1);
        }
    }
    int nA = Aind.size();
    int nA2 = molecule_->natom();

    // Where are the atomic charges?
    double* Np = N_->pointer();

    std::shared_ptr<Matrix> T(new Matrix("Q", nA, 1));
    double* Tp = T->pointer()[0];

    for (int A = 0; A < nA; A++) {
        int Aabs = Aind[A];
        double Z = molecule_->Z(Aabs);
        double Q = -scale * Np[A];
        Tp[A] = Z + Q;
    }
    
    return T;
}
// the third parameter of from_string() should be
// one of std::hex, std::dec or std::oct
template <class T>
bool from_string(T& t,
                 const std::string& s,
                 std::ios_base& (*f)(std::ios_base&))
{
    std::istringstream iss(s);
    return !(iss >> f >> t).fail();
}

void StockholderDensity::guess()
{
    if (guess_ == "NONE") 
        return;

    outfile->Printf("   => ISA Guess <=\n\n");

    // Where are the true atoms?
    std::vector<int> Aind;
    std::vector<int> Aind2;
    for (int A = 0; A < molecule_->natom(); A++) {
        if (molecule_->Z(A) != 0) {
            Aind2.push_back(Aind.size());
            Aind.push_back(A);
        } else {
            Aind2.push_back(-1);
        }
    }
    int nA = Aind.size();
    int nA2 = molecule_->natom();

    std::string PSIDATADIR = Process::environment.get_datadir();

    // Read each atom in and interpolate separately
    for (int A = 0; A < nA; A++) {

        // Which file?
        std::stringstream ss;
        ss << molecule_->symbol(Aind[A]) << "_" << guess_ << ".dat";       
        std::string filename = PSIDATADIR +  "/atoms/" + ss.str();
        outfile->Printf("    Reading guess for %2s from %s.\n", molecule_->symbol(Aind[A]).c_str(),
            filename.c_str());

        // Read the file
        std::vector<std::string> lines;
        std::string text;
        ifstream infile(filename.c_str());
        if (!infile)
            throw PSIEXCEPTION("ISA Guess: unable to open file " + filename);
        while (infile.good()) {
            getline(infile, text);
            lines.push_back(text);
        }

        // Parse the file
        regex numberline("^\\s*" NUMBER "\\s+" NUMBER "\\s*$");
        smatch what;
        std::vector<double> rref;        
        std::vector<double> wref;        
        for (int index = 0; index < lines.size(); index++) {
            if (!regex_match(lines[index], what, numberline))
                continue; // Hope the user is not an idiot
            double r;
            if (!from_string<double>(r, what[1].str(), std::dec))
                throw PSIEXCEPTION("ISA Guess: Unable to convert grid file line\n");
            double w;
            if (!from_string<double>(w, what[2].str(), std::dec))
                throw PSIEXCEPTION("ISA Guess: Unable to convert grid file line\n");
            rref.push_back(r);
            wref.push_back(w);
        }

        // Do the interpolation
        for (int k = 0; k < ws_[A].size(); k++) {
            double r = rs_[A][k];
            int kind = -1;
            for (int k2 = 0; k2 < rref.size()-1; k2++) {
                if (rref[k2] < r && rref[k2+1] >= r) {
                    kind = k2;
                    break;
                }
            }
            if (kind == -1) {
                throw PSIEXCEPTION("ISA Guess: Tabulated grid does not span\n");
            }
            
            double Rl = rref[kind];
            double Rr = rref[kind+1];
            double wl = wref[kind];
            double wr = wref[kind+1];

            if (wl == 0.0 || wr == 0.0) {
                throw PSIEXCEPTION("ISA Guess: Who designed this grid guess?\n");
            }

            double lwl = log(wl);
            double lwr = log(wr);

            ws_[A][k] = exp(lwl + (lwr - lwl)*(r - Rl)/(Rr - Rl));
        }
    } 

    outfile->Printf("\n");
}

}

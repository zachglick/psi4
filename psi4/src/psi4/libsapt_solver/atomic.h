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

#ifndef ASAPT_ATOMIC_H
#define ASAPT_ATOMIC_H

#include <psi4/libmints/typedefs.h>
#include <psi4/libmints/wavefunction.h>
#include "psi4/libqt/qt.h"
#include <map>
#include <set>

namespace psi {

class DFTGrid;
class MolecularGrid;

class AtomicDensity {

protected:

    // ==> Utility Metadata <== //

    /// Name for bench files
    std::string name_;
    
    /// Print flag
    int print_;
    /// Debug flag
    int debug_;
    /// Bench flag
    int bench_;

    // ==> Input Specification <== //

    /// Molecule this atomic density is built around
    std::shared_ptr<Molecule> molecule_;
    /// Basis this atomic density is built around
    std::shared_ptr<BasisSet> primary_;
    /// Density matrix this atomic density is built around
    std::shared_ptr<Matrix> D_;
    /// MolecularGrid to collocate on (Becke-style, yo?)
    std::shared_ptr<MolecularGrid> grid_;

    // ==> Targets (All reordered) <== //

    /// X coordinates of grid
    std::shared_ptr<Vector> x_;
    /// Y coordinates of grid
    std::shared_ptr<Vector> y_;
    /// Z coordinates of grid
    std::shared_ptr<Vector> z_;
    /// Weights of grid
    std::shared_ptr<Vector> w_;
    /// Total density of grid
    std::shared_ptr<Vector> rho_;
    /// Atomic normalizations (true atoms)
    std::shared_ptr<Vector> N_;

    // ==> Utility Routines <== //

    /// Allocate grid and total density, evaluate, and sort to classical ordering
    void compute_total_density();
        
    /// Ubiqitous header
    virtual void print_header() const = 0;
   
    /// Protected constructor
    AtomicDensity();

public:
    // ==> Master Routines <== //

    /// Master destructor
    virtual ~AtomicDensity();

    /// Master builder
    static std::shared_ptr<AtomicDensity> build(const std::string& type, std::shared_ptr<BasisSet> basis, Options& options);
    /// Master compute routine
    virtual void compute(std::shared_ptr<Matrix> D) = 0;
    /// Compute weights for npoints at (x,y,z), and place in w (nA x npoints). Also multiplies in rhop if not NULL
    virtual void compute_weights(int npoints, double* x, double* y, double* z, double** w, double* rhop = NULL, int atom = -1, double* rho0p = NULL) = 0;
    /// Compute and disply the atomic charges, multiplying the electronic part by scale
    virtual void compute_charges(double scale = 2.0) = 0; 
    /// Compute and return the atomic charges, multiplying the electronic part by scale
    virtual std::shared_ptr<Matrix> charges(double scale = 2.0) = 0;

    // ==> Accessors <== //

    std::shared_ptr<Vector> x()   const { return x_; }
    std::shared_ptr<Vector> y()   const { return y_; }
    std::shared_ptr<Vector> z()   const { return z_; }
    std::shared_ptr<Vector> w()   const { return w_; }
    std::shared_ptr<Vector> rho() const { return rho_; }
    std::shared_ptr<Vector> N()   const { return N_; }

    std::shared_ptr<Molecule> molecule() const { return molecule_; }
    std::shared_ptr<BasisSet> primary() const { return primary_; }
    std::shared_ptr<Matrix> D() const { return D_; }
    std::shared_ptr<MolecularGrid> grid() { return grid_; }

    // ==> Setters <== //
    
    void set_print(int print) { print_ = print; }
    void set_debug(int debug) { debug_ = debug; }
    void set_bench(int bench) { bench_ = bench; }
    void set_name(const std::string& name) { name_ = name; }

};

class StockholderDensity : public AtomicDensity {

friend class AtomicDensity;

protected:

    // => Convergence Stuff <= //

    /// Convergence criterion
    double convergence_;
    /// Maximum iterations
    int maxiter_;
    /// Use DIIS?
    bool diis_;
    /// Minimum iterations to DIIS at?
    int diis_min_vecs_; 
    /// Maximum DIIS subspace size?
    int diis_max_vecs_; 
    /// Number of iterations between flushes
    int diis_flush_;
    /// Maximum iteration to do the flush trick
    int diis_flush_maxiter_;
    /// Name of guess files or NONE
    std::string guess_;

    // => State variables <= //

    /// Shell radii
    std::vector<std::vector<double> > rs_;
    /// Shell densities
    std::vector<std::vector<double> > ws_;

    /// UHF atomic tablulated guess
    void guess();

    /// Ubiqitous header
    virtual void print_header() const;
    /// Protected constructor
    StockholderDensity();
    
public:
    /// Subclass destructor
    virtual ~StockholderDensity();

    /// Master compute routine
    virtual void compute(std::shared_ptr<Matrix> D);
    /// Compute weights for npoints at (x,y,z), and place in w (nA x npoints). Also multiplies in rhop if not NULL
    virtual void compute_weights(int npoints, double* x, double* y, double* z, double** w, double* rhop = NULL, int atom = -1, double* rho0p = NULL);
    /// Compute and disply the atomic charges, multiplying the electronic part by scale
    virtual void compute_charges(double scale = 2.0); 
    /// Compute and return the atomic charges, multiplying the electronic part by scale
    virtual std::shared_ptr<Matrix> charges(double scale = 2.0);

};

} // End namespace

#endif


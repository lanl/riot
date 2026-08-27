//========================================================================================
// (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License, see licenses/bsd_athenak.txt file for details
//========================================================================================
// NOTE(@pdmullen): The linear mode pgen was largely borrowed from AthenaK, ported here to
// Parthenon and adapted to multiple materials
// This file was made in part with generative AI.

// C/C++ headers
#include <algorithm> // min, max
#include <cmath>     // sqrt()
#include <cstdio>    // fopen(), fprintf(), freopen()
#include <iostream>  // endl
#include <limits>    // numeric_limits
#include <limits>
#include <sstream> // stringstream
#include <string>  // c_str()

#include "hydro/hydro.hpp"
#include "riot_driver.hpp"
#include "riot_pgen/pgen.hpp"
#include "riot_utils/riot_utils.hpp"
#include <globals.hpp>
#include <singularity-eos/eos/eos.hpp>

//----------------------------------------------------------------------------------------
void HydroEigensystem(const Real d, const Real v1, const Real v2, const Real v3,
                      const Real p, const Real gamma, Real eigenvalues[5],
                      Real right_eigenmatrix[5][5]);

namespace {

//----------------------------------------------------------------------------------------
//! \struct LinWaveVariables
//! \brief container for variables shared with vector potential and error functions
struct LinWaveVariables {
  Real d0, p0, v1_0, b1_0, b2_0, b3_0, dby, dbz, k_par;
  Real cos_a2, cos_a3, sin_a2, sin_a3;
  int wave_flag;
  Real amp, vflow, lambda;
  Real rem[5][5];
  Real ev[5];
  Real gamma, gm1;
};

} // end anonymous namespace

namespace linear_modes {

LinWaveVariables lwv;

//----------------------------------------------------------------------------------------
//! \fn  void linear_modes::ProblemGenerator
//! \brief
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  // read global parameters
  const Mesh *pmesh = pmb->pmy_mesh;
  const int ndim = pmesh->ndim;
  const int iprob = pin->GetInteger("problem", "iprob");
  const Real nperiod = pin->GetOrAddReal("problem", "nperiod", 1);
  const bool along_x1 = pin->GetOrAddBoolean("problem", "along_x1", false);
  const bool along_x2 = pin->GetOrAddBoolean("problem", "along_x2", false);
  const bool along_x3 = pin->GetOrAddBoolean("problem", "along_x3", false);
  // error check input flags
  if ((along_x1 && (along_x2 || along_x3)) || (along_x2 && along_x3)) {
    PARTHENON_FAIL("Can only specify one of along_x1/2/3 to be true");
  }
  if ((along_x2 || along_x3) && (ndim == 1)) {
    PARTHENON_FAIL("Cannot specify waves along x2 or x3 axis in 1D");
  }
  if (along_x3 && (ndim == 2)) {
    PARTHENON_FAIL("Cannot specify waves along x3 axis in 2D");
  }

  // Triggers the 3T version of this test
  const bool do_ionization = pin->GetBoolean("physics", "ionization");

  // Code below will automatically calculate wavevector along grid diagonal, imposing the
  // conditions of periodicity and exactly one wavelength along each grid direction
  const Real x1size = pmesh->mesh_size.xmax(X1DIR) - pmesh->mesh_size.xmin(X1DIR);
  const Real x2size = pmesh->mesh_size.xmax(X2DIR) - pmesh->mesh_size.xmin(X2DIR);
  const Real x3size = pmesh->mesh_size.xmax(X3DIR) - pmesh->mesh_size.xmin(X3DIR);

  // start with wavevector along x1 axis
  lwv.cos_a3 = 1.0;
  lwv.sin_a3 = 0.0;
  lwv.cos_a2 = 1.0;
  lwv.sin_a2 = 0.0;
  if ((ndim >= 2) && !(along_x1)) {
    Real ang_3 = std::atan(x1size / x2size);
    lwv.sin_a3 = std::sin(ang_3);
    lwv.cos_a3 = std::cos(ang_3);
  }
  if ((ndim == 3) && !(along_x1)) {
    Real ang_2 = std::atan(0.5 * (x1size * lwv.cos_a3 + x2size * lwv.sin_a3) / x3size);
    lwv.sin_a2 = std::sin(ang_2);
    lwv.cos_a2 = std::cos(ang_2);
  }

  // hardcode wavevector along x2 axis, override ang_2, ang_3
  if (along_x2) {
    lwv.cos_a3 = 0.0;
    lwv.sin_a3 = 1.0;
    lwv.cos_a2 = 1.0;
    lwv.sin_a2 = 0.0;
  }

  // hardcode wavevector along x3 axis, override ang_2, ang_3
  if (along_x3) {
    lwv.cos_a3 = 0.0;
    lwv.sin_a3 = 1.0;
    lwv.cos_a2 = 0.0;
    lwv.sin_a2 = 1.0;
  }

  // choose the smallest projection of the wavelength in each direction that is > 0
  lwv.lambda = std::numeric_limits<Real>::max();
  if (lwv.cos_a2 * lwv.cos_a3 > 0.0) {
    lwv.lambda = std::min(lwv.lambda, x1size * lwv.cos_a2 * lwv.cos_a3);
  }
  if (lwv.cos_a2 * lwv.sin_a3 > 0.0) {
    lwv.lambda = std::min(lwv.lambda, x2size * lwv.cos_a2 * lwv.sin_a3);
  }
  if (lwv.sin_a2 > 0.0) lwv.lambda = std::min(lwv.lambda, x3size * lwv.sin_a2);

  // Initialize k_parallel
  lwv.k_par = 2.0 * (M_PI) / lwv.lambda;

  // Wave properties
  lwv.wave_flag = pin->GetInteger("problem", "wave_flag");
  lwv.amp = pin->GetReal("problem", "amp");
  lwv.vflow = pin->GetOrAddReal("problem", "vflow", 0.0);

  // Set background state: v1_0 is parallel to wavevector.
  lwv.d0 = 1.0;
  lwv.v1_0 = lwv.vflow;
  lwv.b1_0 = 1.0;
  lwv.b2_0 = std::sqrt(2.0);
  lwv.b3_0 = 0.5;

  // capture variables for kernel
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  // Adiabatic index (assuming single material)
  auto mat_pkg = pmb->packages.Get("materials");
  auto eos_vec = mat_pkg->Param<std::vector<RiotEOS::EOS>>("h.h.EOS");
  lwv.gm1 = eos_vec[0].GruneisenParamFromDensityTemperature(lwv.d0, 1.);
  lwv.gamma = lwv.gm1 + 1.0;
  lwv.p0 = 1.0 / lwv.gamma;

  // Compute eigenvectors in hydrodynamics
  HydroEigensystem(lwv.d0, lwv.v1_0, 0.0, 0.0, lwv.p0, lwv.gamma, lwv.ev, lwv.rem);

  // set tlim as number of wave periods for evolution
  pin->SetReal("parthenon/time", "tlim",
               nperiod * (std::abs(lwv.lambda / lwv.ev[lwv.wave_flag])));

  // prep setting meshblock data
  using parthenon::MakePackDescriptor;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  auto &rc = pmb->meshblock_data.Get();

  for (auto &var : rc->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }

  // packing
  static auto desc =
      MakePackDescriptor<ccmat::rho, ccmat::internal_energy, ccmat::volume_fraction,
                         ccbulk::total_material_energy, ccbulk::momentum,
                         // electrons
                         ccbulk::electron_internal_energy>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(rc.get());

  const int nummat =
      v.GetUpperBoundHost(0, ccmat::rho()) - v.GetLowerBoundHost(0, ccmat::rho()) + 1;
  if (iprob == 1) {
    if (nummat != 1) PARTHENON_FAIL("iprob=1 requires a single material");
  } else if (iprob == 2 || iprob == 3 || iprob == 4) {
    if (nummat != 2)
      PARTHENON_FAIL("iprob=2,3,4 requires 2 materials (with the same EOS)");
  } else {
    PARTHENON_FAIL("iprob specification not recognized");
  }

  // coordinates and problem setup
  auto &coords = pmb->coords;
  auto dlw = lwv;

  // set initial condition
  pmb->par_for(
      "ProblemGenerator::linear_modes", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // cell-centered coordinates
        const Real x1v = coords.Xc<X1DIR>(i);
        const Real x2v = coords.Xc<X2DIR>(j);
        const Real x3v = coords.Xc<X3DIR>(k);

        // inclined wavevector
        const Real x =
            dlw.cos_a2 * (x1v * dlw.cos_a3 + x2v * dlw.sin_a3) + x3v * dlw.sin_a2;
        const Real sn = std::sin(dlw.k_par * x);
        const Real mx = dlw.d0 * dlw.vflow + dlw.amp * sn * dlw.rem[1][dlw.wave_flag];
        const Real my = dlw.amp * sn * dlw.rem[2][dlw.wave_flag];
        const Real mz = dlw.amp * sn * dlw.rem[3][dlw.wave_flag];

        // set bulk quantities
        v(0, ccbulk::momentum(0), k, j, i) =
            mx * dlw.cos_a2 * dlw.cos_a3 - my * dlw.sin_a3 - mz * dlw.sin_a2 * dlw.cos_a3;
        v(0, ccbulk::momentum(1), k, j, i) =
            mx * dlw.cos_a2 * dlw.sin_a3 + my * dlw.cos_a3 - mz * dlw.sin_a2 * dlw.sin_a3;
        v(0, ccbulk::momentum(2), k, j, i) = mx * dlw.sin_a2 + mz * dlw.cos_a2;
        v(0, ccbulk::total_material_energy(), k, j, i) =
            dlw.p0 / dlw.gm1 + 0.5 * dlw.d0 * (dlw.v1_0) * (dlw.v1_0) +
            dlw.amp * sn * dlw.rem[4][dlw.wave_flag];

        // set material quantities
        const Real drho = dlw.amp * sn * dlw.rem[0][dlw.wave_flag];
        const Real rho_bulk = dlw.d0 + drho;
        const Real KE = 0.5 *
                        (SQR(v(0, ccbulk::momentum(0), k, j, i)) +
                         SQR(v(0, ccbulk::momentum(1), k, j, i)) +
                         SQR(v(0, ccbulk::momentum(2), k, j, i))) /
                        rho_bulk;
        const Real u_bulk = v(0, ccbulk::total_material_energy(), k, j, i) - KE;

        // iprob==2 discontinous vfracs, iprob==3 uniform vfracs, iprob==4 smoothly
        // varying vfracs
        if (iprob > 1) {
          Real vfrac_0 = 1.0;
          if (iprob == 2) {
            vfrac_0 = (drho >= 0.0) ? 1.0 : 0.0;
          } else if (iprob == 3) {
            vfrac_0 = 0.1;
          } else { // iprob == 4
            vfrac_0 = 0.25 * sn + 0.5;
          }
          const Real vfrac_1 = 1.0 - vfrac_0;
          v(0, ccmat::volume_fraction(0), k, j, i) = vfrac_0;
          v(0, ccmat::volume_fraction(1), k, j, i) = vfrac_1;
          v(0, ccmat::rho(0), k, j, i) = vfrac_0 * rho_bulk;
          v(0, ccmat::rho(1), k, j, i) = vfrac_1 * rho_bulk;
        } else { // iprob==1, single material
          v(0, ccmat::volume_fraction(0), k, j, i) = 1.0;
          v(0, ccmat::rho(0), k, j, i) = rho_bulk;
        }

        // ionization correction
        if (do_ionization) {
          // put half the energy into electrons
          v(0, ccbulk::electron_internal_energy(), k, j, i) = 0.5 * u_bulk;
        }
      });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void linear_modes::ProblemModifier
//! \brief
void ProblemModifier(parthenon::ParthenonManager *pman) {
  pman->app_input->UserWorkAfterLoop = linear_modes::UserWorkAfterLoop;
}

//----------------------------------------------------------------------------------------
//! \fn  std::shared_ptr<StateDescriptor> linear_modes::ProblemPackage
//! \brief
std::shared_ptr<StateDescriptor> ProblemPackage(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("linear_modes");
  pkg->CheckRefinementBlock = linear_modes::ProblemCheckRefinementBlock;
  return pkg;
}

//----------------------------------------------------------------------------------------
//! \fn  void linear_modes::UserWorkAfterLoop
//! \brief
void UserWorkAfterLoop(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {
  // linear wave params
  auto dlw = lwv;

  // errors
  const int nvars = 5;

  // packing
  using parthenon::MakePackDescriptor;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace mdname = riot::container_names;

  auto &md = pmesh->mesh_data.GetOrAdd("base", 0);
  if (md->NumBlocks() == 0) return;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto v = riot::MakePack<ccbulk::rho, ccbulk::momentum, ccbulk::total_material_energy>(
      md.get());

  // indexing
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  // Error reductions
  RiotUtils::array_type<Real, nvars> l1_err;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "LinearModesErrors", DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    RiotUtils::array_type<Real, nvars> &rsum) {
        // Capture coordinates this Meshblock
        auto &coords = v.GetCoordinates(b);
        Real x1v = coords.Xc<X1DIR>(i);
        Real x2v = coords.Xc<X2DIR>(j);
        Real x3v = coords.Xc<X3DIR>(k);
        Real vol = coords.CellVolume(k, j, i);

        // Analytic solution
        // TODO(@pdmullen): remove duplicated code...
        const Real x =
            dlw.cos_a2 * (x1v * dlw.cos_a3 + x2v * dlw.sin_a3) + x3v * dlw.sin_a2;
        const Real sn = std::sin(dlw.k_par * x);
        const Real mx = dlw.d0 * dlw.vflow + dlw.amp * sn * dlw.rem[1][dlw.wave_flag];
        const Real my = dlw.amp * sn * dlw.rem[2][dlw.wave_flag];
        const Real mz = dlw.amp * sn * dlw.rem[3][dlw.wave_flag];
        const Real tdens = dlw.d0 + dlw.amp * sn * dlw.rem[0][dlw.wave_flag];
        const Real tmom1 =
            mx * dlw.cos_a2 * dlw.cos_a3 - my * dlw.sin_a3 - mz * dlw.sin_a2 * dlw.cos_a3;
        const Real tmom2 =
            mx * dlw.cos_a2 * dlw.sin_a3 + my * dlw.cos_a3 - mz * dlw.sin_a2 * dlw.sin_a3;
        const Real tmom3 = mx * dlw.sin_a2 + mz * dlw.cos_a2;
        const Real tener = dlw.p0 / dlw.gm1 + 0.5 * dlw.d0 * (dlw.v1_0) * (dlw.v1_0) +
                           dlw.amp * sn * dlw.rem[4][dlw.wave_flag];

        // numerical values
        const Real rho_num = v(b, ccbulk::rho(), k, j, i);
        const Real p0_num = v(b, ccbulk::momentum(0), k, j, i);
        const Real p1_num = v(b, ccbulk::momentum(1), k, j, i);
        const Real p2_num = v(b, ccbulk::momentum(2), k, j, i);
        const Real eng_num = v(b, ccbulk::total_material_energy(), k, j, i);

        // Compute errors
        rsum.my_array[0] += vol * fabs(rho_num - tdens);
        rsum.my_array[1] += vol * fabs(p0_num - tmom1);
        rsum.my_array[2] += vol * fabs(p1_num - tmom2);
        rsum.my_array[3] += vol * fabs(p2_num - tmom3);
        rsum.my_array[4] += vol * fabs(eng_num - tener);
      },
      RiotUtils::GlobalSum<Real, Kokkos::HostSpace, nvars>(l1_err));
  Kokkos::fence();

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &(l1_err.my_array[0]), nvars, MPI_PARTHENON_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif // MPI_PARALLEL

  // normalize errors by number of cells
  Real vol = (pmesh->mesh_size.xmax(X1DIR) - pmesh->mesh_size.xmin(X1DIR)) *
             (pmesh->mesh_size.xmax(X2DIR) - pmesh->mesh_size.xmin(X2DIR)) *
             (pmesh->mesh_size.xmax(X3DIR) - pmesh->mesh_size.xmin(X3DIR));
  for (int i = 0; i < nvars; ++i)
    l1_err.my_array[i] = l1_err.my_array[i] / vol;

  // compute rms error
  Real rms_err = 0.0;
  for (int i = 0; i < nvars; ++i) {
    rms_err += SQR(l1_err.my_array[i]);
  }
  rms_err = std::sqrt(rms_err);

  // root process opens output file and writes out errors
  if (parthenon::Globals::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("parthenon/job", "problem_id"));
    fname.append("-errs.dat");
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        PARTHENON_FAIL("Error output file could not be opened");
      }

      // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        PARTHENON_FAIL("Error output file could not be opened");
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3   Ncycle  RMS-L1       ");
      std::fprintf(pfile, "d_L1         M1_L1         M2_L1         M3_L1         E_L1");
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%04d", pmesh->mesh_size.nx(X1DIR));
    std::fprintf(pfile, "  %04d", pmesh->mesh_size.nx(X2DIR));
    std::fprintf(pfile, "  %04d", pmesh->mesh_size.nx(X3DIR));
    std::fprintf(pfile, "  %05d  %e", tm.ncycle, rms_err);
    for (int i = 0; i < nvars; ++i) {
      std::fprintf(pfile, "  %e", l1_err.my_array[i]);
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  parthenon::AmrTag linear_modes::ProblemCheckRefinementBlock
//! \brief
parthenon::AmrTag ProblemCheckRefinementBlock(MeshBlockData<Real> *mbd) {
  // linear wave params
  auto dlw = lwv;

  namespace ccbulk = cell_variables::cell_averaged::bulk;
  IndexRange ib = mbd->GetBoundsI(IndexDomain::interior);
  IndexRange jb = mbd->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = mbd->GetBoundsK(IndexDomain::interior);

  // pack descriptors are expensive to compute but always the same
  using parthenon::MakePackDescriptor;
  auto pmb = mbd->GetBlockPointer();
  static auto desc = MakePackDescriptor<ccbulk::rho>((pmb->resolved_packages).get());
  auto vbulk = desc.GetPack(mbd);

  Real min_dens_diff = std::numeric_limits<Real>::max();
  constexpr int b = 0;
  constexpr int var = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "Hydro::EstimateTimestep", DevExecSpace(),
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &min_rho) {
        min_rho = std::abs(std::min(min_rho, vbulk(b, var, k, j, i) - dlw.d0));
      },
      Kokkos::Min<Real>(min_dens_diff));

  if (min_dens_diff < 0.3 * lwv.amp) return AmrTag::refine;
  if (min_dens_diff > 0.6 * lwv.amp) return AmrTag::derefine;
  return AmrTag::same;
}

} // namespace linear_modes

//----------------------------------------------------------------------------------------
//! \fn void HydroEigensystem()
//! \brief computes eigenvectors of linear waves in ideal gas/isothermal hydrodynamics
void HydroEigensystem(const Real d, const Real v1, const Real v2, const Real v3,
                      const Real p, const Real gamma, Real eigenvalues[5],
                      Real right_eigenmatrix[5][5]) {
  Real vsq = v1 * v1 + v2 * v2 + v3 * v3;
  Real h = (p / (gamma - 1.0) + 0.5 * d * vsq + p) / d;
  Real a = std::sqrt(gamma * p / d);

  // Compute eigenvalues (eq. B2)
  eigenvalues[0] = v1 - a;
  eigenvalues[1] = v1;
  eigenvalues[2] = v1;
  eigenvalues[3] = v1;
  eigenvalues[4] = v1 + a;

  // Right-eigenvectors, stored as COLUMNS (eq. B3)
  right_eigenmatrix[0][0] = 1.0;
  right_eigenmatrix[1][0] = v1 - a;
  right_eigenmatrix[2][0] = v2;
  right_eigenmatrix[3][0] = v3;
  right_eigenmatrix[4][0] = h - v1 * a;

  right_eigenmatrix[0][1] = 0.0;
  right_eigenmatrix[1][1] = 0.0;
  right_eigenmatrix[2][1] = 1.0;
  right_eigenmatrix[3][1] = 0.0;
  right_eigenmatrix[4][1] = v2;

  right_eigenmatrix[0][2] = 0.0;
  right_eigenmatrix[1][2] = 0.0;
  right_eigenmatrix[2][2] = 0.0;
  right_eigenmatrix[3][2] = 1.0;
  right_eigenmatrix[4][2] = v3;

  right_eigenmatrix[0][3] = 1.0;
  right_eigenmatrix[1][3] = v1;
  right_eigenmatrix[2][3] = v2;
  right_eigenmatrix[3][3] = v3;
  right_eigenmatrix[4][3] = 0.5 * vsq;

  right_eigenmatrix[0][4] = 1.0;
  right_eigenmatrix[1][4] = v1 + a;
  right_eigenmatrix[2][4] = v2;
  right_eigenmatrix[3][4] = v3;
  right_eigenmatrix[4][4] = h + v1 * a;

  return;
}

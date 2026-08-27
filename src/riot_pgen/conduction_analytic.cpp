//========================================================================================
// (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
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
// This file was made in part with generative AI.

// conductive thermal diffusion of sinusiodal electron temperature.
// analytic solution = 1 + beta*sin(2*n*pi*x/L) *
// exp(-(2*n*pi/L)^2 * (Ke/(rho*cve) * t)

#include <singularity-eos/eos/eos.hpp>

#include "riot_pgen/pgen.hpp"

namespace conduction_analytic {

using parthenon::ParArray1D;
using namespace RiotEOS;

//----------------------------------------------------------------------------------------
//! \fn  void conduction_analytic::ProblemGenerator
//! \brief
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto &rc = pmb->meshblock_data.Get();

  for (auto &var : rc->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }

  static auto desc = MakePackDescriptor<
      cm::lT_cache, cm::lr_cache, cm::rho, ccmat::rho, ccmat::internal_energy,
      ccmat::volume_fraction, ccbulk::total_material_energy, ccbulk::momentum,
      ccbulk::temperature, ccbulk::electron_temperature, ccbulk::electron_internal_energy,
      ccbulk::electron_pressure, ccbulk::pressure, ccbulk::electron_number_density,
      ccmat::ionization_zbar, cm::ionization_zbar>((pmb->resolved_packages).get());
  auto v = desc.GetPack(rc.get());

  // EoS (assuming single material)
  auto hydro_pkg = pmb->packages.Get("hydro");
  auto mat_pkg = pmb->packages.Get("materials");
  const auto &ion_eos = mat_pkg->Param<RiotEOS::EOS_Array_t>("d.d.EOS");
  const auto &electron_eos = mat_pkg->Param<RiotEOS::EOS_Array_t>("d.d.electron_EOS");
  const auto &eos_from_matid =
      mat_pkg->Param<parthenon::ParArray1D<int>>("d.EOS_from_matid");

  // Problem parameters
  const Real rho0 = pin->GetReal("conduction_analytic", "rho0");
  const Real constant = pin->GetReal("conduction_analytic", "constant");
  const Real beta = pin->GetReal("conduction_analytic", "amplitude");
  const int nx = pin->GetInteger("conduction_analytic", "nx");
  const int ny = pin->GetInteger("conduction_analytic", "ny");
  const int nz = pin->GetInteger("conduction_analytic", "nz");
  const auto &ionization_options = pmb->packages.Get("ionization");
  const Real Ke = ionization_options->Param<Real>("electron_conductivity");

  const Mesh *pmesh = pmb->pmy_mesh;
  const parthenon::RegionSize &mesh_size = pmesh->mesh_size;
  const Real dx1_mesh = (mesh_size.xmax(X1DIR) - mesh_size.xmin(X1DIR));
  const Real dx2_mesh = (mesh_size.xmax(X2DIR) - mesh_size.xmin(X2DIR));
  const Real dx3_mesh = (mesh_size.xmax(X3DIR) - mesh_size.xmin(X3DIR));

  // wavevector
  const Real kx = 2.0 * M_PI * nx / dx1_mesh;
  const Real ky = 2.0 * M_PI * ny / dx2_mesh;
  const Real kz = 2.0 * M_PI * nz / dx3_mesh;

  // Indexing
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;

  const int mat_id = 0;

  pmb->par_for(
      "ProblemGenerator::conduction_analytic", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<parthenon::X1DIR>(i);
        const Real y = coords.Xc<parthenon::X2DIR>(j);
        const Real z = coords.Xc<parthenon::X3DIR>(k);

        const Real dx = coords.Dxf<parthenon::X1DIR>(i);
        const Real dy = coords.Dxf<parthenon::X2DIR>(j);
        const Real dz = coords.Dxf<parthenon::X3DIR>(k);

        // EoS parameters, lambdas and electron specific heat
        const Real abar = ion_eos[mat_id].MeanAtomicMass();
        const Real zbar = ion_eos[mat_id].MeanAtomicNumber();
        const int eos_id = eos_from_matid(mat_id);
        auto &eose = electron_eos(eos_id);
        auto &eosi = ion_eos(eos_id);
        RiotEOS::LambdaIndexerMulti<decltype(v)> lambda(v, 0, k, j, i);
        // specific heat of electrons - should be constant so just pass any
        // temperature
        const Real cve =
            eose.SpecificHeatFromDensityTemperature(rho0, 1., lambda[mat_id]);

        // diffusion coefficient
        const Real D = Ke / (rho0 * cve);

        // density and vol frac
        v(0, ccmat::rho(0), k, j, i) = rho0;
        v(0, ccmat::volume_fraction(mat_id), k, j, i) = 1.0;

        // electron temperature - Fourier mode
        // solution at t=0
        const Real Te = constant + cell_average_solution(x, y, z, dx, dy, dz, kx, ky, kz,
                                                         beta, D, 0.);
        v(0, ccbulk::electron_temperature(), k, j, i) = Te;
        // set Ti = Te
        const Real Ti = Te;
        v(0, ccbulk::temperature(), k, j, i) = Ti;

        // assume fully ionized
        const Real ne = rho0 / abar / 1.66054e-24 * zbar;
        v(0, ccbulk::electron_number_density(), k, j, i) = ne;
        v(0, ccmat::ionization_zbar(mat_id), k, j, i) = rho0 * zbar;
        // have to set prim zbar also because it's used in the eos calls below
        v(0, cm::ionization_zbar(mat_id), k, j, i) = zbar;

        // internal energy density
        const Real sie_e =
            eose.InternalEnergyFromDensityTemperature(rho0, Te, lambda[mat_id]);
        const Real sie_i =
            eosi.InternalEnergyFromDensityTemperature(rho0, Ti, lambda[mat_id]);
        const Real ue = sie_e * rho0;
        const Real ui = sie_i * rho0;

        // electron and total material energy
        v(0, ccbulk::electron_internal_energy(), k, j, i) = ue;
        v(0, ccbulk::total_material_energy(), k, j, i) = ui + ue;

        // electron and total pressure
        const Real pe = eose.PressureFromDensityTemperature(rho0, Te, lambda[mat_id]);
        const Real pi = eosi.PressureFromDensityTemperature(rho0, Ti, lambda[mat_id]);
        v(0, ccbulk::electron_pressure(), k, j, i) = pe;
        v(0, ccbulk::pressure(), k, j, i) = pi + pe;
      });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void conduction_analytic::ProblemModifier
//! \brief
void ProblemModifier(parthenon::ParthenonManager *pman) {
  pman->app_input->UserWorkAfterLoop = conduction_analytic::UserWorkAfterLoop;
}

//----------------------------------------------------------------------------------------
//! \fn  void conduction_analytic::UserWorkAfterLoop
//! \brief
void UserWorkAfterLoop(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {
  // packing
  using parthenon::MakePackDescriptor;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  const int nvars = 1;
  auto &md = pmesh->mesh_data.Get();
  // static auto desc = riot::MakePackDescriptor<cm::lT_cache, cm::lr_cache,
  // ccbulk::electron_temperature,
  // cm::rho, ccmat::rho, ccmat::volume_fraction>(md.get());
  static auto desc = MakePackDescriptor<
      cm::lT_cache, cm::lr_cache, cm::rho, ccmat::rho, ccmat::internal_energy,
      ccmat::volume_fraction, ccbulk::total_material_energy, ccbulk::momentum,
      ccbulk::temperature, ccbulk::electron_temperature, ccbulk::electron_internal_energy,
      ccbulk::electron_pressure, ccbulk::pressure, ccbulk::electron_number_density,
      cm::ionization_zbar>(md.get());
  auto v = desc.GetPack(md.get());

  // EoS (assuming single material)
  auto hydro_pkg = pmesh->packages.Get("hydro");
  auto mat_pkg = pmesh->packages.Get("materials");
  const auto &ion_eos = mat_pkg->Param<RiotEOS::EOS_Array_t>("d.d.EOS");
  const auto &electron_eos = mat_pkg->Param<RiotEOS::EOS_Array_t>("d.d.electron_EOS");
  const auto &eos_from_matid =
      mat_pkg->Param<parthenon::ParArray1D<int>>("d.EOS_from_matid");

  // Problem parameters
  const Real rho0 = pin->GetReal("conduction_analytic", "rho0");
  const Real constant = pin->GetReal("conduction_analytic", "constant");
  const Real beta = pin->GetReal("conduction_analytic", "amplitude");
  const int nx = pin->GetInteger("conduction_analytic", "nx");
  const int ny = pin->GetInteger("conduction_analytic", "ny");
  const int nz = pin->GetInteger("conduction_analytic", "nz");
  const auto &ionization_options = pmesh->packages.Get("ionization");
  const Real Ke = ionization_options->Param<Real>("electron_conductivity");

  const parthenon::RegionSize &mesh_size = pmesh->mesh_size;
  const Real dx1_mesh = (mesh_size.xmax(X1DIR) - mesh_size.xmin(X1DIR));
  const Real dx2_mesh = (mesh_size.xmax(X2DIR) - mesh_size.xmin(X2DIR));
  const Real dx3_mesh = (mesh_size.xmax(X3DIR) - mesh_size.xmin(X3DIR));

  // wavevector
  const Real kx = 2.0 * M_PI * nx / dx1_mesh;
  const Real ky = 2.0 * M_PI * ny / dx2_mesh;
  const Real kz = 2.0 * M_PI * nz / dx3_mesh;

  const int mat_id = 0;

  // indexing
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  // Error reductions
  RiotUtils::array_type<Real, nvars> l1_err;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "conduction_analytic errors", DevExecSpace(),
      0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    RiotUtils::array_type<Real, nvars> &rsum) {
        // Capture coordinates this Meshblock
        auto coords = v.GetCoordinates(b);
        const Real x = coords.Xc<X1DIR>(i);
        const Real y = coords.Xc<X2DIR>(j);
        const Real z = coords.Xc<X3DIR>(k);
        const Real dx = coords.Dxf<parthenon::X1DIR>(i);
        const Real dy = coords.Dxf<parthenon::X2DIR>(j);
        const Real dz = coords.Dxf<parthenon::X3DIR>(k);
        const Real vol = dx * dy * dz;

        const int eos_id = eos_from_matid(mat_id);
        auto &eose = electron_eos(eos_id);
        auto &eosi = ion_eos(eos_id);
        RiotEOS::LambdaIndexerMulti<decltype(v)> lambda(v, 0, k, j, i);

        // specific heat of electrons - should be constant so just pass any
        // temperature
        const Real cve =
            eose.SpecificHeatFromDensityTemperature(rho0, 1., lambda[mat_id]);
        // diffusion coefficient
        const Real D = Ke / (rho0 * cve);

        // Analytic solution
        const Real Te_true = constant + cell_average_solution(x, y, z, dx, dy, dz, kx, ky,
                                                              kz, beta, D, tm.time);

        // Numerical solution
        const Real Te_num = v(b, ccbulk::electron_temperature(), k, j, i);

        // Compute errors
        rsum.my_array[0] += vol * std::abs(Te_true - Te_num);
      },
      RiotUtils::GlobalSum<Real, Kokkos::HostSpace, nvars>(l1_err));
  Kokkos::fence();

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &(l1_err.my_array[0]), nvars, MPI_PARTHENON_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif // MPI_PARALLEL

  // normalize errors by number of cells
  Real vol = dx1_mesh * dx2_mesh * dx3_mesh;
  l1_err.my_array[0] = l1_err.my_array[0] / vol;

  const Real final_err = l1_err.my_array[0];

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
      std::fprintf(pfile, "# Nx1  Nx2  Nx3   Ncycle  Te-L1       ");
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%04d", pmesh->mesh_size.nx(X1DIR));
    std::fprintf(pfile, "  %04d", pmesh->mesh_size.nx(X2DIR));
    std::fprintf(pfile, "  %04d", pmesh->mesh_size.nx(X3DIR));
    std::fprintf(pfile, "  %05d  %e", tm.ncycle, final_err);
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}

} // namespace conduction_analytic

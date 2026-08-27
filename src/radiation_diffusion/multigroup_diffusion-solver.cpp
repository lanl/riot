//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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

// C++ includes
#include <cmath>
#include <limits>

// Parthenon includes
#include <bvals/boundary_conditions_generic.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/cg_solver.hpp>
#include <solvers/mg_solver.hpp>
#include <solvers/solver_base.hpp>

// Riot includes
#include "hydro/hydro.hpp"
#include "microphysics/eos_riot.hpp"
#include "microphysics/opacity_models.hpp"
#include "radiation_diffusion/diffusion_equation.hpp"
#include "radiation_diffusion/multigroup_diffusion.hpp"
#include "riot_driver.hpp"

using namespace parthenon::driver::prelude;

namespace RadiationDiffusion {

// using solver_t = preconditioner_t;

template <class temperature>
std::shared_ptr<parthenon::solvers::SolverBase> GetSolverSptr(ParameterInput *pin) {
  using namespace MultiGroupVars;
  using equations_coupled_t =
      LinearizedRadiationDiffusionEquation<Egroup, diag_loc, D, D, true>;
  using equations_uncoupled_t =
      LinearizedRadiationDiffusionEquation<Egroup, diag_loc, D, D, false>;
  using prolongator_t = parthenon::solvers::ProlongationBlockInteriorZeroDirichlet;
  using restrictor_t = parthenon::solvers::RestrictionCombined;
  using preconditioner_t =
      parthenon::solvers::MGSolver<equations_uncoupled_t, prolongator_t, restrictor_t>;

  // using solver_t =
  //     parthenon::solvers::TridiagSolver<equations_t>;
  using solver_t =
      parthenon::solvers::BiCGSTABSolver<equations_uncoupled_t, preconditioner_t>;
  using MG = MultiGroup<temperature>;
  return std::make_shared<solver_t>(
      MG::multigroup_base_container, MG::multigroup_u_container,
      MG::multigroup_rhs_container, pin, "diffusion/linear_solver_params");
}

template std::shared_ptr<parthenon::solvers::SolverBase>
GetSolverSptr<cell_variables::cell_averaged::bulk::temperature>(ParameterInput *);
template std::shared_ptr<parthenon::solvers::SolverBase>
GetSolverSptr<cell_variables::cell_averaged::bulk::electron_temperature>(
    ParameterInput *);

} // namespace RadiationDiffusion

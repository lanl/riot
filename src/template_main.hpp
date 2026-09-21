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
#ifndef TEMPLATE_MAIN_HPP_
#define TEMPLATE_MAIN_HPP_
// This file was made in part with generative AI.

#include <defs.hpp>
#include <parthenon_manager.hpp>

#include "diagnostics/diagnostics.hpp"
#include "riot.hpp"
#include "riot_pgen/pgen.hpp"
#ifdef RIOT_ENABLE_PYTHON
#include "riot_utils/py_init.hpp"
#endif

namespace riot {

//----------------------------------------------------------------------------------------
//! \fn  int riot::main
//! \brief
template <typename T>
int main(parthenon::ParthenonManager &pman) {
  using namespace riot;

#ifdef RIOT_ENABLE_PYTHON
  // startup python
  Python::Init(pman.pinput.get());
#endif

  if constexpr (std::is_same_v<T, RiotDriver>) {
    // Register pgens
    RegisterPgens();
    // Handle ProblemGenerator user-defined modifiers
    ProblemModifier(&pman);
  }

  // Tell pman to register reflecting boundaries
  pman.app_input->RegisterDefaultReflectingBoundaryConditions();

  // Redefine parthenon defaults
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  pman.app_input->ProcessPackages = ProcessPackages;
  if constexpr (std::is_same_v<T, RiotDriver>) {
    pman.app_input->ProblemGenerator = ProblemGenerator;
  }
  pman.app_input->PostStepDiagnosticsInLoop = diagnostics::PostStepDiagnosticsInLoop;

  // call ParthenonInit to set up the mesh
  pman.ParthenonInitPackagesAndMesh();

  // Initialize the driver
  T driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());

  // This line actually runs the simulation
  auto driver_status = driver.Execute();

  // print out some diagnostics from the run
  if constexpr (std::is_same_v<T, RiotDriver>) {
    driver.ReportBlockHistogram();
    driver.ReportMemUsage();
  }

  // call MPI_Finalize and Kokkos::finalize if necessary
  pman.ParthenonFinalize();

  // MPI and Kokkos can no longer be used

  return (0);
}

} // namespace riot

#endif // TEMPLATE_MAIN_HPP_

//========================================================================================
// (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
//
// This file was made in part with generative AI.
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

#include "analysis_driver.hpp"
#include "riot_driver.hpp"
#include "template_main.hpp"

//----------------------------------------------------------------------------------------
//! \fn  int main
//! \brief
int main(int argc, char *argv[]) {
  using namespace riot;
  parthenon::ParthenonManager pman;

  // Set up kokkos and read pin
  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  int main_status;
  if (pman.IsAnalysis()) {
    printf("Calling AnalysisDriver\n");
    main_status = main<AnalysisDriver>(pman);
  } else {
    printf("Calling RiotDriver\n");
    main_status = main<RiotDriver>(pman);
  }

  return main_status;
}

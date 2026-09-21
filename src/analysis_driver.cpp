//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
#include "diagnostics/riot_viz.hpp"

namespace riot {

AnalysisDriver::AnalysisDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
  : Driver(pin, app_in, pm) {
  auto &pkgs = pm->packages.AllPackages();
  do_viz = pkgs.contains("riot_viz");
  // Current time was put into pin upon "restart"
  current_time = pin->GetReal("parthenon/time", "start_time");
}

DriverStatus AnalysisDriver::Execute() {
  TaskListStatus task_status;
  if (do_viz) task_status = riot_viz::Render(pmesh, current_time).Execute();

  DriverStatus status = (task_status == TaskListStatus::complete
                         ? DriverStatus::complete
                         : DriverStatus::failed);
  return status;
}

} // namespace riot

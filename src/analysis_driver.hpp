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
#ifndef ANALYSIS_DRIVER_HPP_
#define ANALYSIS_DRIVER_HPP_
// This file was made in part with generative AI.

#include <memory>

#include <parthenon/driver.hpp>
using namespace parthenon::driver::prelude;
using parthenon::Packages_t;
using parthenon::StateDescriptor;

#include "plugins.hpp"
#include "variables.hpp"

namespace riot {

class AnalysisDriver : public Driver {
 public:
  AnalysisDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm);
  DriverStatus Execute();

 private:
  bool do_viz;
  Real current_time;
};

} // namespace riot

#endif // ANALYSIS_DRIVER_HPP_

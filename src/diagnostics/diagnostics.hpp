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
#ifndef DIAGNOSTICS_DIAGNOSTICS_HPP_
#define DIAGNOSTICS_DIAGNOSTICS_HPP_
// This file was made in part with generative AI.

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;
using parthenon::Packages_t;

#include "variables.hpp"

#define FOREACH_DIAG                                                                     \
  DIAG(dsplanar)                                                                         \
  DIAG(doubleshell)                                                                      \
  DIAG(masses)                                                                           \
  DIAG(energies)

#define DIAG(name)                                                                       \
  namespace name {                                                                       \
  std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);                      \
  } // namespace name
FOREACH_DIAG
#undef DIAG

namespace diagnostics {
using ppkg_t = std::function<std::shared_ptr<StateDescriptor>(ParameterInput *)>;
#define DIAG(name) {#name, name::Initialize},
static std::unordered_map<std::string, ppkg_t> diagnostics_init({FOREACH_DIAG});
#undef DIAG

void AddDiagnostics(ParameterInput *pin, Packages_t &packages);

} // namespace diagnostics

#undef FOREACH_DIAG

#endif // DIAGNOSTICS_DIAGNOSTICS_HPP_

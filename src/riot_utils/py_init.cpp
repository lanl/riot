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
// This file was made in part with generative AI.

#define PY_ARRAY_UNIQUE_SYMBOL PCALL_NUMPY_API
// IMPORTANT: do NOT define NO_IMPORT_ARRAY in this TU
#include <numpy/arrayobject.h>

#include "py_init.hpp"

#include <parthenon/driver.hpp>
using namespace parthenon::driver::prelude;
// #include <parthenon/package.hpp>
//  using namespace parthenon::package::prelude;

//----------------------------------------------------------------------------------------
//! \fn  int pcall_init_numpy_capi
//! \brief
extern "C" int pcall_init_numpy_capi() {
  // import_array() initializes PyArray_API;
  // returns void on success (or returns NULL via macros)
  // _import_array() returns < 0 on failure
  return _import_array();
}

namespace Python {

//----------------------------------------------------------------------------------------
//! \fn  void Python::Init
//! \brief
void Init(ParameterInput *pin) {
  auto python_path = pin->GetOrAddVector<std::string>(
      "riot", "python_path", {}, "a list of paths for python to search");
  python_path.push_back(".");
  PythonEnv::initialize(python_path);
  pcall_init_numpy_capi();
}

} // namespace Python

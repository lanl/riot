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

#include <cfenv>
// ...

//----------------------------------------------------------------------------------------
//! \fn  int setup_floating_point
//! \brief
int setup_floating_point(bool trap_fpe) {
  int ret = 0;
  if (trap_fpe) {
#if defined(_GNU_SOURCE) && !defined(__APPLE__)
    if (-1 == feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW)) {
      // flog(warn) << "error when trying to trap floating-point exceptions"
      //<< std::endl;
      PARTHENON_WARN("error when trying to trap floating-point exceptions");
      ret = 1;
    }
#else
    PARTHENON_WARN("no support for trapping floating-point exceptions");
    ret = 1;
#endif
  }
  return ret;
}

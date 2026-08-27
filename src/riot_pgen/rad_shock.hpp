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
#ifndef RIOT_PGEN_RAD_SHOCK_HPP_
#define RIOT_PGEN_RAD_SHOCK_HPP_
// This file was made in part with generative AI.

// C++ includes
#include <memory>

// Parthenon includes
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

namespace rad_shock {
using namespace parthenon::package::prelude;
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
void ProblemModifier(parthenon::ParthenonManager *pman);
void ShockInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void ShockOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
} // namespace rad_shock

#endif // RIOT_PGEN_RAD_SHOCK_HPP_

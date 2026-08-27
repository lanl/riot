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
#ifndef RIOT_PGEN_LINEAR_MODES_HPP_
#define RIOT_PGEN_LINEAR_MODES_HPP_
// This file was made in part with generative AI.

// Parthenon includes
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

namespace linear_modes {
using namespace parthenon::package::prelude;
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
void ProblemModifier(parthenon::ParthenonManager *pman);
std::shared_ptr<StateDescriptor> ProblemPackage(ParameterInput *pin);
void UserWorkAfterLoop(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm);
parthenon::AmrTag ProblemCheckRefinementBlock(MeshBlockData<Real> *mbd);
} // namespace linear_modes

#endif // RIOT_PGEN_LINEAR_MODES_HPP_

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
#ifndef MULTIPHYSICS_FILL_SHARED_DERIVED_HPP_
#define MULTIPHYSICS_FILL_SHARED_DERIVED_HPP_
// This file was made in part with generative AI.

#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

namespace Multiphysics {

void PostCommsFillDerived(MeshData<Real> *rc);
void FillInteriorDerived(MeshData<Real> *rc);
void FillInteriorBlockDerived(std::shared_ptr<MeshBlockData<Real>> pmbd);

} // namespace Multiphysics

#endif // MULTIPHYSICS_FILL_SHARED_DERIVED_HPP_

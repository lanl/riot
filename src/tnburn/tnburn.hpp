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
#ifndef TNBURN_TNBURN_HPP_
#define TNBURN_TNBURN_HPP_
// This file was made in part with generative AI.

// C++ includes
#include <memory>
#include <string>

// Parthenon includes
#include <parthenon/package.hpp>

// Riot includes
#include "ratelib/read_isotope_data.hpp"

using namespace parthenon::package::prelude;

static const std::string default_tn_fname = "isotope_data.hdf5";

namespace TNBurn {
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            std::vector<std::vector<int>> isotope_names);
TaskStatus CalculateTNBurnSource(MeshData<Real> *state, MeshData<Real> *src,
                                 const Real dt);
void FillDerived(MeshData<Real> *md);
TaskStatus SharedSources(MeshData<Real> *sourcein, MeshData<Real> *sourceout,
                         MeshData<Real> *state);
Real IntegratedReactionCount(MeshData<Real> *md, const int mat, const int reaction);
} // namespace TNBurn

#endif

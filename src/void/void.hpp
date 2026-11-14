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
#ifndef VOID_VOID_HPP_
#define VOID_VOID_HPP_
// This file was made in part with generative AI.

#include <memory>

#include <parthenon/package.hpp>

namespace Void {
using namespace parthenon::package::prelude;

// The void material takes up volume but carries no pressure, energy, heat capacity or
// stiffness, and never enters PTE. The closure is the complementarity condition
//   alpha_V >= 0,  p >= p_void,  alpha_V (p - p_void) = 0
// i.e. either the void is closed (ordinary PTE over the other materials) or it is open
// and every other material sits on its p_void isobar at a common temperature.
struct Params {
  int matid = -1;        // void material id (-1: no void)
  Real pressure = 0.0;   // p_void
  Real density = 1.e-20; // tracer density of the void
  bool freeze = true;    // zero the velocity of pure-void cells
  Real tol = 1.e-8;      // relative tolerance of the open-void solves
  int max_iter = 100;    // iteration limit of the open-void solves
};

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

// The void params from the input deck, a disabled set if there is no void (for setup,
// before the void package exists)
Params ParamsFromInput(ParameterInput *pin);
inline int MatidFromInput(ParameterInput *pin) { return ParamsFromInput(pin).matid; }

// The void has no EOS or opacity: point its material block at placeholders, which are
// never evaluated. Called by the materials package before it reads the material blocks.
void SetPlaceholderMaterial(ParameterInput *pin);

// The void params, or a disabled set if the void package is not enabled
inline Params GetParams(const Mesh *pm) {
  if (pm->packages.AllPackages().count("void") == 0) return Params();
  return pm->packages.Get("void")->Param<Params>("params");
}

// Slice index of the void material in block b of a pack, or -1 if it is not allocated
template <typename F, typename V>
KOKKOS_INLINE_FUNCTION int SliceIndex(const V &v, const int b, const int nmat,
                                      const int matid) {
  for (int m = 0; m < nmat; m++) {
    if (v(b, F(m)).sparse_id == matid) return m;
  }
  return -1;
}

} // namespace Void

#endif // VOID_VOID_HPP_

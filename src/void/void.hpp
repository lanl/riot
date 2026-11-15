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

#include <algorithm>
#include <memory>

#include <parthenon/package.hpp>

#include "variables.hpp"

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
  // Matter density at or below which a cell is frozen like pure void. Round-off leaves
  // |m| ~ eps p_void dt/dx, so set it to ~10-100x eps p_void C / S_ref^2 (C the CFL
  // number, S_ref a lower bound on the fastest signal speed).
  Real freeze_rho = 0.0;
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

// Whether a cell of matter (non-void) density rho_matter is frozen like pure void
KOKKOS_INLINE_FUNCTION bool Frozen(const Params &p, const Real rho_matter) {
  return p.freeze && p.matid >= 0 && !(rho_matter > p.freeze_rho);
}

// Put a frozen cell at rest, discarding its kinetic energy, at most the energy the matter
// holds (E - e_void)
template <typename PV, typename PVV, typename I>
KOKKOS_INLINE_FUNCTION void FreezeCell(const Params &p, const PV &pv, const PVV &pv_void,
                                       const bool has_void, const I kji) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  const Real E = pv(ccbulk::total_material_energy(), kji);
  const Real ke = 0.5 *
                  (SQR(pv(ccbulk::momentum(0), kji)) + SQR(pv(ccbulk::momentum(1), kji)) +
                   SQR(pv(ccbulk::momentum(2), kji))) /
                  (pv(ccbulk::rho(), kji) + 1.e-100);
  const Real e_void =
      has_void ? -p.pressure * pv_void(ccmat::volume_fraction(), kji) : 0.0;
  for (int d = 0; d < 3; d++) {
    pv(ccbulk::velocity(d), kji) = 0.0;
  }
  pv(ccbulk::internal_energy(), kji) = E - std::min(ke, std::max(E - e_void, 0.0));
}

} // namespace Void

#endif // VOID_VOID_HPP_

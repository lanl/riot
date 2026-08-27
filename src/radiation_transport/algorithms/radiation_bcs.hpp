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
#ifndef RADIATION_ALGORITHMS_RADIATION_BCS_HPP_
#define RADIATION_ALGORITHMS_RADIATION_BCS_HPP_
// This file was made in part with generative AI.

// C++ headers
#include <limits>
#include <memory>
#include <string>
#include <vector>

// Parthenon headers
#include <bvals/boundary_conditions_generic.hpp>
#include <parthenon/package.hpp>

// Riot headers
#include "radiation_transport/angular_grids/geodesic_grid.hpp"
#include "radiation_transport/angular_grids/latlon_grid.hpp"
#include "radiation_transport/transport_utils/transport_utils.hpp"

using namespace parthenon::package::prelude;

namespace Drive {
using Side = parthenon::BoundaryFunction::BCSide;

//----------------------------------------------------------------------------------------
//! \fn constexpr IndexDomain BndryDomain
//! \brief Maps (DIR, SIDE) to the corresponding ghost-zone IndexDomain
template <parthenon::CoordinateDirection DIR, parthenon::BoundaryFunction::BCSide SIDE>
inline constexpr IndexDomain BndryDomain() {
  constexpr bool inner = (SIDE == parthenon::BoundaryFunction::BCSide::Inner);
  if constexpr (DIR == parthenon::X1DIR) {
    return inner ? IndexDomain::inner_x1 : IndexDomain::outer_x1;
  } else if constexpr (DIR == parthenon::X2DIR) {
    return inner ? IndexDomain::inner_x2 : IndexDomain::outer_x2;
  } else {
    return inner ? IndexDomain::inner_x3 : IndexDomain::outer_x3;
  }
}

//----------------------------------------------------------------------------------------
//! \fn KOKKOS_FORCEINLINE_FUNCTION void RefIdx
//! \brief Returns the (k,j,i) index triple with the DIR-normal component replaced by the
//! interior reference index `ref`, i.e. the cell copied/reflected into the ghost zone
template <parthenon::CoordinateDirection DIR>
KOKKOS_FORCEINLINE_FUNCTION void RefIdx(const int k, const int j, const int i,
                                        const int ref, int &kr, int &jr, int &ir) {
  kr = (DIR == parthenon::X3DIR) ? ref : k;
  jr = (DIR == parthenon::X2DIR) ? ref : j;
  ir = (DIR == parthenon::X1DIR) ? ref : i;
}

//----------------------------------------------------------------------------------------
//! \fn IndexRange InteriorBounds
//! \brief Returns the interior IndexRange along the DIR-normal axis
template <parthenon::CoordinateDirection DIR>
inline const IndexRange InteriorBounds(const parthenon::IndexShape &bounds) {
  if constexpr (DIR == parthenon::X1DIR) {
    return bounds.GetBoundsI(IndexDomain::interior);
  } else if constexpr (DIR == parthenon::X2DIR) {
    return bounds.GetBoundsJ(IndexDomain::interior);
  } else {
    return bounds.GetBoundsK(IndexDomain::interior);
  }
}

//----------------------------------------------------------------------------------------
//! \fn template <CoordinateDirection DIR, BCSide SIDE> void DriveBCImpl
//! \brief Unified drive boundary condition for all six faces
template <parthenon::CoordinateDirection DIR, parthenon::BoundaryFunction::BCSide SIDE>
inline void DriveBCImpl(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  namespace ccrad = cell_variables::cell_averaged::rad;
  auto pmb = mbd->GetBlockPointer();
  const auto nb = IndexRange{0, 0};

  // Indexing for outflow: reference the first (inner) or last (outer) interior cell
  constexpr bool inner = (SIDE == parthenon::BoundaryFunction::BCSide::Inner);
  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto range = InteriorBounds<DIR>(bounds);
  const int ref = inner ? range.s : range.e;
  constexpr IndexDomain domain = BndryDomain<DIR, SIDE>();

  // Radiation package
  const auto rad_pkg = GetRadPackage(pmb->packages);
  const int ngroups = rad_pkg->Param<int>("ngroups");
  const int nangles = rad_pkg->Param<int>("nangles");

  // Intensity
  static auto descr = GetBoundaryPackDescriptorMap<ccrad::intensity>(mbd);
  auto vr = descr[coarse].GetPack(mbd.get());
  if (vr.GetMaxNumberOfVars() > 0) {
    const Real trad = rad_pkg->Param<Real>("drive_trad");
    const auto unit_utils = rad_pkg->Param<UnitUtils>("unit_utils");
    const auto fbnd = *(rad_pkg->MutableParam<ParArray1D<Real>>("fbnd_d"));
    pmb->par_for_bndry(
        "DriveBC-intensity", nb, domain, parthenon::TopologicalElement::CC, coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          int kr, jr, ir;
          RefIdx<DIR>(k, j, i, ref, kr, jr, ir);
          for (int gg = 0; gg < ngroups; ++gg) {
            const Real ee =
                (trad > 0) ? Emissivity(gg, trad, fbnd, ngroups, unit_utils) : 0.0;
            for (int aa = 0; aa < nangles; ++aa) {
              vr(0, ccrad::intensity(GAI(nangles, gg, aa)), k, j, i) =
                  std::max(ee, vr(0, ccrad::intensity(GAI(nangles, gg, aa)), kr, jr, ir));
            }
          }
        });
  }

  // Opacity: by default zero at the boundary to force a purely upwind flux (even when
  // radiation_transport/beta > 0)
  static auto desco = GetBoundaryPackDescriptorMap<ccrad::aa, ccrad::ss>(mbd);
  auto vo = desco[coarse].GetPack(mbd.get());
  if (vo.GetMaxNumberOfVars() > 0) {
    const bool dp_force_upwind_flux_bc =
        rad_pkg->Param<bool>("drive_force_upwind_flux_bc");
    const Real copy = (dp_force_upwind_flux_bc) ? 0.0 : 1.0;
    pmb->par_for_bndry(
        "DriveBC-opacity", nb, domain, parthenon::TopologicalElement::CC, coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          int kr, jr, ir;
          RefIdx<DIR>(k, j, i, ref, kr, jr, ir);
          for (int gg = 0; gg < ngroups; ++gg) {
            vo(0, ccrad::aa(gg), k, j, i) = copy * vo(0, ccrad::aa(gg), kr, jr, ir);
            vo(0, ccrad::ss(gg), k, j, i) = copy * vo(0, ccrad::ss(gg), kr, jr, ir);
          }
        });
  }
}

template <parthenon::CoordinateDirection DIR, parthenon::BoundaryFunction::BCSide SIDE>
inline auto DriveBC() {
  return [](std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) -> void {
    using namespace parthenon;
    using namespace parthenon::BoundaryFunction;
    DriveBCImpl<DIR, SIDE>(mbd, coarse);
  };
}

} // namespace Drive

//----------------------------------------------------------------------------------------
// Shared initialization for the radiation transport BCs (default, drive).
//----------------------------------------------------------------------------------------
namespace RadiationBC {

// The input blocks from which the shared radiation BCs are read.
constexpr char drive_block[] = "radiation_transport/drive";

// Individual shared BC options.
namespace BCOption {

//----------------------------------------------------------------------------------------
//! \fn void AddDriveParams
//! \brief
inline void AddDriveParams(ParameterInput *pin, Params &params) {
  // Extract custom BC parameters
  params.Add("drive_trad",
             pin->GetOrAddReal(drive_block, "trad_bc", 0.0,
                               "Specifies the uniform radiation temperature (in K) for "
                               "the drive boundary condition."));
  params.Add("drive_force_upwind_flux_bc",
             pin->GetOrAddBoolean(
                 drive_block, "force_upwind_flux_bc", true,
                 "Zero the boundary opacity to force a purely upwind flux (even when "
                 "radiation_transport/beta > 0) for the drive boundary condition; if "
                 "false, copy the interior opacity instead"));
}

} // namespace BCOption

//----------------------------------------------------------------------------------------
//! \struct Helper struct for boundary enrollment
//! \brief
struct BoundarySpec {
  const char *name;
  BoundaryFace face;
};

//----------------------------------------------------------------------------------------
//! \fn EnrollRadiationBC
//! \brief
inline void EnrollRadiationBC(StateDescriptor *rad_pkg, ParameterInput *pin,
                              Params &params) {
  using Side = parthenon::BoundaryFunction::BCSide;
  using RadBFunc_t = typename std::remove_reference_t<
      decltype(rad_pkg->UserBoundaryFunctions[BoundaryFace::inner_x1])>::value_type;

  // Boundary faces
  const std::array<BoundarySpec, 6> bndry_faces = {{
      {"ix1_bc", BoundaryFace::inner_x1},
      {"ox1_bc", BoundaryFace::outer_x1},
      {"ix2_bc", BoundaryFace::inner_x2},
      {"ox2_bc", BoundaryFace::outer_x2},
      {"ix3_bc", BoundaryFace::inner_x3},
      {"ox3_bc", BoundaryFace::outer_x3},
  }};

  // Drive BC
  bool drive_bc_enrolled = false;
  const std::array<RadBFunc_t, 6> drive_bcs = {
      Drive::DriveBC<X1DIR, Side::Inner>(), Drive::DriveBC<X1DIR, Side::Outer>(),
      Drive::DriveBC<X2DIR, Side::Inner>(), Drive::DriveBC<X2DIR, Side::Outer>(),
      Drive::DriveBC<X3DIR, Side::Inner>(), Drive::DriveBC<X3DIR, Side::Outer>()};

  // Enroll custom BC
  for (int i = 0; i < bndry_faces.size(); ++i) {
    const auto &[name, face] = bndry_faces[i];
    if (pin->GetOrAddString(drive_block, name, "default") == "drive") {
      rad_pkg->UserBoundaryFunctions[face].push_back(drive_bcs[i]);
      drive_bc_enrolled = true;
    }
  }

  // Enroll custom BC params
  if (drive_bc_enrolled) {
    BCOption::AddDriveParams(pin, params);
  }
}

} // namespace RadiationBC

#endif // RADIATION_ALGORITHMS_RADIATION_BCS_HPP_

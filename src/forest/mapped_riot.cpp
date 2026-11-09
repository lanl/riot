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

#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "forest/mapped_riot.hpp"

using namespace parthenon::package::prelude;

namespace RiotForest {

namespace {
constexpr char node_coordinates_name[] = "n.coordinates";

// Physical (R, z, 0) (or (x, y, z)) of node (k, j, i). Not in the kernel, as nvcc does
// not let a device lambda first capture a variable in an if constexpr branch
template <typename Coords>
KOKKOS_INLINE_FUNCTION void NodePosition(const Coords &coords, const int k, const int j,
                                         const int i, Real p[3]) {
  if constexpr (parthenon::IsCoord<parthenon::MappedCartesian>()) {
    coords.Vertex(k, j, i, p[0], p[1], p[2]);
  } else {
    coords.Vertex(j, i, p[0], p[1]);
    p[2] = 0.0;
  }
}

template <typename Coords>
void FillNodeCoordinates(MeshBlock *pmb, const Coords &coords) {
  if constexpr (parthenon::IsMappedCoord()) {
    using TE = parthenon::TopologicalElement;
    auto x = pmb->meshblock_data.Get()->Get(node_coordinates_name).data;
    const auto &cb = pmb->cellbounds;
    const auto ib = cb.GetBoundsI(IndexDomain::entire, TE::NN);
    const auto jb = cb.GetBoundsJ(IndexDomain::entire, TE::NN);
    const auto kb = cb.GetBoundsK(IndexDomain::entire, TE::NN);
    pmb->par_for(
        "FillNodeCoordinates", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          Real p[3];
          NodePosition(coords, k, j, i, p);
          for (int d = 0; d < 3; ++d)
            x(d, k, j, i) = p[d];
        });
  }
}

// Physical (R, z, 0) (or (x, y, z)) of the nodes, which outputs use as the mesh geometry
void FillNodeCoordinatesMesh(Mesh *pm, ParameterInput *pin, parthenon::SimTime &tm) {
  for (auto &pmb : pm->block_list)
    FillNodeCoordinates(pmb.get(), pmb->coords);
}
} // namespace

void Initialize(ParameterInput *pin, StateDescriptor *riot) {
  if constexpr (parthenon::IsMappedCoord()) {
    // physics that is not yet written for mapped (non-rectilinear block) meshes
    auto on = [&](const std::string &key) {
      return riot->AllParams().hasKey(key) && riot->Param<bool>(key);
    };
    for (const std::string key : {"do_mix", "do_multigroup_diffusion", "do_levelsets"}) {
      PARTHENON_REQUIRE(!on(key), std::string(parthenon::Coordinates_t::name_) +
                                      " coordinates only support hydro, strength, thinc, "
                                      "gravity, radiation transport, tn burn, passive "
                                      "scalars, tracers, lasers and ionization (" +
                                      key + ")");
    }
    // ... and of ionization, the transport of electrons and ions
    for (const std::string key :
         {"electron_thermal_conduction", "ion_thermal_conduction", "plasma_viscosity"}) {
      PARTHENON_REQUIRE(!on("do_ionization") ||
                            !pin->GetOrAddBoolean("ionization", key, false),
                        "Thermal conduction and plasma viscosity are not supported on "
                        "mapped meshes (" +
                            key + ")");
    }
    // The reflecting radiation boundary flips the component of the angles along the axis
    // of its direction, which on mapped meshes is the normal of the boundary only on the
    // axis of (R, z) meshes, the inner x1 faces of their trees
    for (const std::string key :
         {"ix1_bc", "ox1_bc", "ix2_bc", "ox2_bc", "ix3_bc", "ox3_bc"}) {
      const bool axis =
          !parthenon::IsCoord<parthenon::MappedCartesian>() && key == "ix1_bc";
      PARTHENON_REQUIRE(axis || !pin->DoesParameterExist("radiation_transport/bc", key) ||
                            pin->GetString("radiation_transport/bc", key) != "reflect",
                        "On mapped meshes radiation_transport/bc/" + key +
                            " cannot be reflect");
    }
    if constexpr (parthenon::IsCoord<parthenon::MappedCartesian>()) {
      PARTHENON_REQUIRE(pin->GetOrAddInteger("parthenon/mesh", "nx3", 1) > 1,
                        "MappedCartesian coordinates require a 3D mesh");
    } else {
      PARTHENON_REQUIRE(pin->GetOrAddInteger("parthenon/mesh", "nx3", 1) == 1,
                        "riot does not support more than two dimensions with "
                        "cylindrical coordinates");
    }

    // On mapped meshes, reconstruct the velocity in the frame of each face (the mesh's
    // own basis) rather than the global basis
    const std::string vel_frame =
        pin->GetOrAddString("hydro", "mapped_velocity_frame", "face",
                            std::vector<std::string>{"face", "global"},
                            "Velocity reconstruction basis on mapped meshes");
    riot->AddParam("mapped_face_frame", vel_frame == "face");

    // physical node positions, the output geometry on mapped meshes
    riot->AddField(node_coordinates_name,
                   Metadata({Metadata::Node, Metadata::Vector, Metadata::Derived,
                             Metadata::OneCopy, Metadata::CoordinatesVec},
                            std::vector<int>{3}));
    riot->UserWorkBeforeOutputMesh = FillNodeCoordinatesMesh;
  }
}

} // namespace RiotForest

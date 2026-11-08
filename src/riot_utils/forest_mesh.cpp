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

#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <coordinates/coordinates.hpp>
#include <utils/error_checking.hpp>

#include "riot_utils/forest_mesh.hpp"

namespace RiotMesh {

using parthenon::MappedPatch;
using parthenon::Real;
using parthenon::forest::Edge;
using parthenon::forest::ForestDefinition;
using parthenon::forest::Node;

namespace {

// Builds nodes keyed by an integer label, so that shared corners are the same Node
class NodeSet {
 public:
  std::shared_ptr<Node> Get(int key, Real r, Real z) {
    auto it = nodes_.find(key);
    if (it != nodes_.end()) return it->second;
    auto n = Node::create(next_id_++, {r, z});
    nodes_[key] = n;
    pos_[key] = {r, z};
    return n;
  }
  std::array<Real, 2> Pos(int key) const { return pos_.at(key); }

 private:
  std::map<int, std::shared_ptr<Node>> nodes_;
  std::map<int, std::array<Real, 2>> pos_;
  int next_id_ = 0;
};

ForestDefinition MakeBall2D(parthenon::ParameterInput *pin) {
  constexpr char block[] = "riot/forest";
  const Real r_inner = pin->GetReal(block, "r_inner");
  const Real a = pin->GetOrAddReal(block, "core_half_width", 0.55 * r_inner);
  auto radii = pin->GetVector<Real>(block, "shell_radii");
  const bool log_radial = pin->GetOrAddBoolean(block, "log_radial", true);
  const std::string wedge_edge = pin->GetOrAddString(block, "core_edge", "line");
  PARTHENON_REQUIRE(wedge_edge == "line" || wedge_edge == "equiangle",
                    "riot/forest/core_edge must be line or equiangle");
  const int core_edge =
      (wedge_edge == "line") ? MappedPatch::Line : MappedPatch::EquiAngle;
  PARTHENON_REQUIRE(a > 0.0 && a < r_inner / std::sqrt(2.0),
                    "riot/forest/core_half_width must be in (0, r_inner/sqrt(2))");
  PARTHENON_REQUIRE(radii.size() >= 2 && std::abs(radii[0] - r_inner) <= 1.e-12 * r_inner,
                    "riot/forest/shell_radii must start at r_inner and have >= 2 entries");
  for (int s = 1; s < radii.size(); ++s)
    PARTHENON_REQUIRE(radii[s] > radii[s - 1], "riot/forest/shell_radii must increase");

  ForestDefinition fd;
  fd.SetRootLevel(pin->GetOrAddInteger(block, "root_level", 0));
  const std::array<Real, 3> lmin{0.0, 0.0, pin->GetReal("parthenon/mesh", "x3min")};
  const std::array<Real, 3> lmax{1.0, 1.0, pin->GetReal("parthenon/mesh", "x3max")};

  NodeSet ns;
  // Node keys: core nodes 0..5, then 5 nodes per radius at 10 + 10 * s + p for polar
  // position p = 0 (top, +z axis), 1 (45 deg), 2 (equator), 3 (-45 deg), 4 (-z axis)
  enum { O = 0, Ca = 1, Cb = 2, Q0 = 3, Q1 = 4, Q2 = 5 };
  ns.Get(O, 0.0, 0.0);
  ns.Get(Ca, 0.0, a);
  ns.Get(Cb, 0.0, -a);
  ns.Get(Q0, a, 0.0);
  ns.Get(Q1, a, a);
  ns.Get(Q2, a, -a);
  const std::array<Real, 5> th{M_PI / 2, M_PI / 4, 0.0, -M_PI / 4, -M_PI / 2};
  // unit (R, z) of each polar position, exact on the axis and equator
  const Real c45 = std::sqrt(0.5);
  const std::array<Real, 5> ur{0.0, c45, 1.0, c45, 0.0};
  const std::array<Real, 5> uz{1.0, c45, 0.0, -c45, -1.0};
  auto rkey = [](int s, int p) { return 10 + 10 * s + p; };
  for (int s = 0; s < radii.size(); ++s) {
    for (int p = 0; p < 5; ++p) {
      ns.Get(rkey(s, p), radii[s] * ur[p], radii[s] * uz[p]);
    }
  }

  int id = 0;
  auto add_coons = [&](std::array<int, 4> keys, std::array<int, 4> edges) {
    Real x[4], y[4];
    for (int n = 0; n < 4; ++n) {
      const auto p = ns.Pos(keys[n]);
      x[n] = p[0];
      y[n] = p[1];
    }
    fd.AddFace(id++,
               {ns.Get(keys[0], 0, 0), ns.Get(keys[1], 0, 0), ns.Get(keys[2], 0, 0),
                ns.Get(keys[3], 0, 0)},
               lmin, lmax, MappedPatch::MakeCoons(x, y, edges.data()));
  };
  auto add_polar = [&](std::array<int, 4> keys, Real r0, Real r1, Real th0, Real th1) {
    fd.AddFace(id++,
               {ns.Get(keys[0], 0, 0), ns.Get(keys[1], 0, 0), ns.Get(keys[2], 0, 0),
                ns.Get(keys[3], 0, 0)},
               lmin, lmax, MappedPatch::MakePolar(r0, r1, th0, th1, log_radial));
  };
  auto edge = [&](int k0, int k1) { return Edge({ns.Get(k0, 0, 0), ns.Get(k1, 0, 0)}); };
  constexpr int L = MappedPatch::Line, A = MappedPatch::Arc;
  const int E = core_edge;

  // Core: edges are {eta = 0, eta = 1, xi = 0, xi = 1}
  add_coons({O, Q0, Ca, Q1}, {L, E, L, E}); // upper: x1 = R, x2 = z
  add_coons({O, Q0, Cb, Q2}, {L, E, L, E}); // lower: x1 = R, x2 = -z
  fd.AddBC(edge(O, Ca));
  fd.AddBC(edge(O, Cb));

  // Wedges from the core to r_inner: x1 runs from the axis (or 45 deg) towards the
  // equator in each hemisphere, x2 outward
  const int s0 = 0;
  add_coons({Ca, Q1, rkey(s0, 0), rkey(s0, 1)}, {E, A, L, L});
  add_coons({Q1, Q0, rkey(s0, 1), rkey(s0, 2)}, {E, A, L, L});
  add_coons({Q2, Q0, rkey(s0, 3), rkey(s0, 2)}, {E, A, L, L});
  add_coons({Cb, Q2, rkey(s0, 4), rkey(s0, 3)}, {E, A, L, L});
  fd.AddBC(edge(Ca, rkey(s0, 0)));
  fd.AddBC(edge(Cb, rkey(s0, 4)));

  // Polar shells
  for (int s = 0; s + 1 < radii.size(); ++s) {
    const Real r0 = radii[s], r1 = radii[s + 1];
    add_polar({rkey(s, 0), rkey(s, 1), rkey(s + 1, 0), rkey(s + 1, 1)}, r0, r1, th[0],
              th[1]);
    add_polar({rkey(s, 1), rkey(s, 2), rkey(s + 1, 1), rkey(s + 1, 2)}, r0, r1, th[1],
              th[2]);
    add_polar({rkey(s, 3), rkey(s, 2), rkey(s + 1, 3), rkey(s + 1, 2)}, r0, r1, th[3],
              th[2]);
    add_polar({rkey(s, 4), rkey(s, 3), rkey(s + 1, 4), rkey(s + 1, 3)}, r0, r1, th[4],
              th[3]);
    fd.AddBC(edge(rkey(s, 0), rkey(s + 1, 0)));
    fd.AddBC(edge(rkey(s, 4), rkey(s + 1, 4)));
  }
  // Outer sphere
  const int so = radii.size() - 1;
  for (int p = 0; p < 4; ++p) {
    if (p < 2) {
      fd.AddBC(edge(rkey(so, p), rkey(so, p + 1)));
    } else {
      fd.AddBC(edge(rkey(so, p + 1), rkey(so, p)));
    }
  }
  return fd;
}

} // namespace

std::optional<ForestDefinition> MakeForest(parthenon::ParameterInput *pin) {
  const std::string type = pin->GetOrAddString("riot/forest", "type", "none");
  if (type == "none") return std::nullopt;
  PARTHENON_REQUIRE(type == "ball2d", "Unknown riot/forest/type " + type);
  PARTHENON_REQUIRE(parthenon::IsCoord<parthenon::MappedCylindrical>(),
                    "riot/forest/type = ball2d requires MappedCylindrical coordinates");
  return MakeBall2D(pin);
}

} // namespace RiotMesh

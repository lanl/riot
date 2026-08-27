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
#ifndef TRACERS_TRACERS_HPP_
#define TRACERS_TRACERS_HPP_
// This file was made in part with generative AI.

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "parthenon/driver.hpp"
#include <parthenon/package.hpp>

#include "reconstruction/reconstruction.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon::driver::prelude;

namespace Tracers {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t &packages);
TaskCollection PushTracers(Mesh *pm, parthenon::SimTime &tm, Real dt);
TaskStatus TransportParticles(MeshData<Real> *md, const std::string &swarm_name,
                              const Real dt);
template <typename T>
TaskStatus SampleFieldsAtTracers(T *data, const std::string &swarm_name);
void PostAMRInitialization(MeshBlock *pmb, ParameterInput *pin);

//----------------------------------------------------------------------------------------
//! \fn  void AddSamplingSwarmValues
//  \brief Register swarm variables and create device arrays for field metadata
inline void AddSamplingSwarmValues(const std::vector<std::string> &sample_fields,
                                   const Packages_t &packages, StateDescriptor *tracers,
                                   const std::string &swarm_name, Params &params) {
  const int n_fields = sample_fields.size();
  parthenon::ParArray1D<bool> is_sparse_d("is_sparse_" + swarm_name, n_fields);
  parthenon::ParArray1D<int> ncomps_d("ncomps_" + swarm_name, n_fields);
  parthenon::ParArray1D<int> swarm_offset_d("swarm_offset_" + swarm_name, n_fields);
  parthenon::ParArray1D<int> total_size_d("total_size_" + swarm_name, n_fields);
  std::vector<std::string> all_fields, all_swarm_vars;
  auto is_sparse_h = is_sparse_d.GetHostMirror();
  auto ncomps_h = ncomps_d.GetHostMirror();
  auto swarm_offset_h = swarm_offset_d.GetHostMirror();
  auto total_size_h = total_size_d.GetHostMirror();

  int swarm_offset = 0;
  int max_sparse_id = 0;
  std::vector<std::map<int, int>> sparse_cumulative_offsets;
  for (int f = 0; f < n_fields; f++) {
    const auto &field_name = sample_fields[f];
    bool is_sparse = false;
    int ncomps = 1;
    int total_size = 1;
    std::map<int, int> cumulative_offset;

    // Check if this is a sparse field by searching in all packages
    for (const auto &pkg_pair : packages.AllPackages()) {
      const auto &pkg = pkg_pair.second;
      if (pkg->SparseBaseNamePresent(field_name)) {
        is_sparse = true;

        // Get sparse pool and build cumulative offset map
        const auto &pool = pkg->GetSparsePool(field_name);
        total_size = 0;
        int max_ncomps = 0;
        for (const auto &[sparse_id, metadata] : pool.pool()) {
          // Tracer sampling reconstructs about the cell center on the cell-centered
          // stencil. Sampling of non-cell-centered (face/edge/node) fields is not yet
          // supported (planned for a future update).
          PARTHENON_REQUIRE(
              parthenon::GetTopologicalType(metadata) == parthenon::TopologicalType::Cell,
              "Tracer sample_fields entry '" + field_name +
                  "' is not cell-centered; sampling of non-cell-centered fields is not "
                  "yet supported.");
          const auto &shape = metadata.Shape();
          const int this_ncomps =
              std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

          cumulative_offset[sparse_id] = total_size;
          total_size += this_ncomps;
          max_ncomps = std::max(max_ncomps, this_ncomps);
          max_sparse_id = std::max(max_sparse_id, sparse_id);
        }
        ncomps = max_ncomps;

        // Add swarm variable with flat allocation
        std::string swarm_var_name = "particle.sample." + field_name;
        std::vector<int> shape_flat = {total_size};
        parthenon::Metadata m_vec = parthenon::Metadata({Metadata::Real}, shape_flat);
        tracers->AddSwarmValue(swarm_var_name, swarm_name, m_vec);
        break;
      }
    }

    if (!is_sparse) {
      // Non-sparse field - get number of components
      for (const auto &pkg_pair : packages.AllPackages()) {
        const auto &pkg = pkg_pair.second;
        if (pkg->FieldPresent(field_name)) {
          const auto &all_pkg_fields = pkg->AllFields();
          parthenon::VarID vid(field_name);
          if (all_pkg_fields.count(vid) > 0) {
            const auto &field_metadata = all_pkg_fields.at(vid);
            // Tracer sampling reconstructs about the cell center on the cell-centered
            // stencil. Sampling of non-cell-centered (face/edge/node) fields is not yet
            // supported (planned for a future update).
            PARTHENON_REQUIRE(
                parthenon::GetTopologicalType(field_metadata) ==
                    parthenon::TopologicalType::Cell,
                "Tracer sample_fields entry '" + field_name +
                    "' is not cell-centered; sampling of non-cell-centered fields is not "
                    "yet supported.");
            const auto &shape = field_metadata.Shape();
            ncomps =
                std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
          }
          break;
        }
      }

      // Add swarm variable
      total_size = ncomps;
      std::string swarm_var_name = "particle.sample." + field_name;
      if (ncomps == 1) {
        parthenon::Metadata m_real = parthenon::Metadata({Metadata::Real});
        tracers->AddSwarmValue(swarm_var_name, swarm_name, m_real);
      } else {
        std::vector<int> shape_vec = {ncomps};
        parthenon::Metadata m_vec = parthenon::Metadata({Metadata::Real}, shape_vec);
        tracers->AddSwarmValue(swarm_var_name, swarm_name, m_vec);
      }
    }

    // Fill output arrays
    all_fields.push_back(field_name);
    all_swarm_vars.push_back("particle.sample." + field_name);
    sparse_cumulative_offsets.push_back(cumulative_offset);
    is_sparse_h(f) = is_sparse;
    ncomps_h(f) = ncomps;
    swarm_offset_h(f) = swarm_offset;
    total_size_h(f) = total_size;
    swarm_offset += total_size;
  }

  // Copy 1D arrays to device
  is_sparse_d.DeepCopy(is_sparse_h);
  ncomps_d.DeepCopy(ncomps_h);
  swarm_offset_d.DeepCopy(swarm_offset_h);
  total_size_d.DeepCopy(total_size_h);

  // Create and fill 2D sparse offset array
  parthenon::ParArray2D<int> sparse_cumulative_offset_d(
      "sparse_cumulative_offset_" + swarm_name, n_fields, max_sparse_id + 1);
  auto sparse_cumulative_offset_h = sparse_cumulative_offset_d.GetHostMirror();
  for (int f = 0; f < n_fields; f++) {
    for (int s = 0; s <= max_sparse_id; s++) {
      sparse_cumulative_offset_h(f, s) = -1;
    }
    for (const auto &[sparse_id, offset] : sparse_cumulative_offsets[f]) {
      sparse_cumulative_offset_h(f, sparse_id) = offset;
    }
  }
  sparse_cumulative_offset_d.DeepCopy(sparse_cumulative_offset_h);
  sparse_cumulative_offset_d.DeepCopy(sparse_cumulative_offset_h);

  params.Add(swarm_name + "_all_fields", all_fields);
  params.Add(swarm_name + "_all_swarm_vars", all_swarm_vars);
  params.Add(swarm_name + "_n_fields", n_fields);
  params.Add(swarm_name + "_is_sparse_d", is_sparse_d);
  params.Add(swarm_name + "_ncomps_d", ncomps_d);
  params.Add(swarm_name + "_swarm_offset_d", swarm_offset_d);
  params.Add(swarm_name + "_total_size_d", total_size_d);
  params.Add(swarm_name + "_sparse_cumulative_offset_d", sparse_cumulative_offset_d);
}

//----------------------------------------------------------------------------------------
//! \fn  void ReconstructVelocity
//  \brief Reconstruct velocity at particle position using piecewise-linear reconstruction
//         with MC limiting
template <typename Pack>
KOKKOS_INLINE_FUNCTION void
ReconstructVelocity(const Pack &pack_vel, const int b, const int k, const int j,
                    const int i, const Real x1, const Real x2, const Real x3,
                    const auto &coords, const bool multi_d, const bool three_d, Real &v1,
                    Real &v2, Real &v3) {
  using RiotReconstruction::ComputeMCSlope;
  namespace ccbulk = cell_variables::cell_averaged::bulk;

  // Cell center position
  const Real x1c = coords.template Xc<X1DIR>(k, j, i);
  const Real x2c = coords.template Xc<X2DIR>(k, j, i);
  const Real x3c = coords.template Xc<X3DIR>(k, j, i);

  // Cell spacing (minus = from neighbor-1 to current, plus = from current to neighbor+1)
  const Real dx1m = coords.Dxc(X1DIR, k, j, i);
  const Real dx1p = coords.Dxc(X1DIR, k, j, i + 1);
  const Real dx2m = coords.Dxc(X2DIR, k, j, i);
  const Real dx2p = coords.Dxc(X2DIR, k, j + multi_d, i);
  const Real dx3m = coords.Dxc(X3DIR, k, j, i);
  const Real dx3p = coords.Dxc(X3DIR, k + three_d, j, i);

  // Displacement from cell center
  const std::array<Real, 3> delta = {x1 - x1c, x2 - x2c, x3 - x3c};

  // Cell-centered velocities (reused in slope computation and final reconstruction)
  const std::array<Real, 3> vi = {pack_vel(b, ccbulk::velocity(0), k, j, i),
                                  pack_vel(b, ccbulk::velocity(1), k, j, i),
                                  pack_vel(b, ccbulk::velocity(2), k, j, i)};

  // Compute x1-direction slopes
  const std::array<Real, 3> slope_x1 = {
      ComputeMCSlope(pack_vel(b, ccbulk::velocity(0), k, j, i - 1), vi[0],
                     pack_vel(b, ccbulk::velocity(0), k, j, i + 1), dx1m, dx1p),
      ComputeMCSlope(pack_vel(b, ccbulk::velocity(1), k, j, i - 1), vi[1],
                     pack_vel(b, ccbulk::velocity(1), k, j, i + 1), dx1m, dx1p),
      ComputeMCSlope(pack_vel(b, ccbulk::velocity(2), k, j, i - 1), vi[2],
                     pack_vel(b, ccbulk::velocity(2), k, j, i + 1), dx1m, dx1p)};

  // Compute x2-direction slopes
  std::array<Real, 3> slope_x2 = {0.0, 0.0, 0.0};
  if (multi_d) {
    slope_x2 = {ComputeMCSlope(pack_vel(b, ccbulk::velocity(0), k, j - 1, i), vi[0],
                               pack_vel(b, ccbulk::velocity(0), k, j + 1, i), dx2m, dx2p),
                ComputeMCSlope(pack_vel(b, ccbulk::velocity(1), k, j - 1, i), vi[1],
                               pack_vel(b, ccbulk::velocity(1), k, j + 1, i), dx2m, dx2p),
                ComputeMCSlope(pack_vel(b, ccbulk::velocity(2), k, j - 1, i), vi[2],
                               pack_vel(b, ccbulk::velocity(2), k, j + 1, i), dx2m,
                               dx2p)};
  }

  // Compute x3-direction slopes
  std::array<Real, 3> slope_x3 = {0.0, 0.0, 0.0};
  if (three_d) {
    slope_x3 = {ComputeMCSlope(pack_vel(b, ccbulk::velocity(0), k - 1, j, i), vi[0],
                               pack_vel(b, ccbulk::velocity(0), k + 1, j, i), dx3m, dx3p),
                ComputeMCSlope(pack_vel(b, ccbulk::velocity(1), k - 1, j, i), vi[1],
                               pack_vel(b, ccbulk::velocity(1), k + 1, j, i), dx3m, dx3p),
                ComputeMCSlope(pack_vel(b, ccbulk::velocity(2), k - 1, j, i), vi[2],
                               pack_vel(b, ccbulk::velocity(2), k + 1, j, i), dx3m,
                               dx3p)};
  }

  // Reconstruct velocities at particle position
  v1 = vi[0] + slope_x1[0] * delta[0] + slope_x2[0] * delta[1] + slope_x3[0] * delta[2];
  v2 = vi[1] + slope_x1[1] * delta[0] + slope_x2[1] * delta[1] + slope_x3[1] * delta[2];
  v3 = vi[2] + slope_x1[2] * delta[0] + slope_x2[2] * delta[1] + slope_x3[2] * delta[2];
}

} // namespace Tracers

#endif // TRACERS_TRACERS_HPP_

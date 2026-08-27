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

// C++ includes
#include <numeric>
#include <string>
#include <vector>

// Parthenon includes
#include "parthenon/driver.hpp"
#include <basic_types.hpp>
#include <interface/var_id.hpp>
#include <parthenon/package.hpp>

// Riot includes
#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "tracers.hpp"
#include "variables.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon::driver::prelude;

namespace Tracers {

//----------------------------------------------------------------------------------------
//! \fn  std::shared_ptr<StateDescriptor> Tracers::Initialize
//  \brief Initialize tracer particle package and create swarm variables
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t &packages) {
  auto tracers = std::make_shared<StateDescriptor>("tracers");
  Params &params = tracers->AllParams();

  // Read list of swarm names
  std::vector<std::string> swarm_names;
  const std::string prefix = "tracers/";
  auto blocks = pin->GetBlockNamesWithPrefix(prefix);
  for (const auto &block_name : blocks) {
    const std::string swarm_name = block_name.substr(prefix.size());
    swarm_names.push_back(swarm_name);
  }
  params.Add("swarm_names", swarm_names);

  // Set up swarms
  Metadata swarm_metadata({Metadata::Provides, Metadata::None});
  for (const auto &swarm_name : swarm_names) {
    // Construct block name for this swarm's configuration
    const std::string block_name = "tracers/" + swarm_name;

    // Read advection control flag for this swarm
    const bool advect = pin->GetOrAddBoolean(block_name, "advect", true);
    params.Add(swarm_name + "_advect", advect);

    // Read list of fields to sample at particle locations for this swarm
    const auto sample_fields =
        pin->GetOrAddVector<std::string>(block_name, "sample_fields", {});

    // Add swarm
    tracers->AddSwarm(swarm_name, swarm_metadata);

    // Register swarm sampling variables and create indexing arrays
    AddSamplingSwarmValues(sample_fields, packages, tracers.get(), swarm_name, params);
  }

  tracers->PostInitializationBlock = PostAMRInitialization;
  return tracers;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection Tracers::PushTracers
//  \brief Build task collection for tracer particle transport and field sampling
TaskCollection PushTracers(Mesh *pmesh, parthenon::SimTime &tm, Real dt) {
  TaskCollection tc;
  TaskID none(0);

  // Get list of swarms
  const auto &tracers_pkg = pmesh->packages.Get("tracers");
  const auto &swarm_names = tracers_pkg->Param<std::vector<std::string>>("swarm_names");

  const int num_partitions = pmesh->DefaultNumPartitions();
  auto &tr = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = tr[i];
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);

    // Transport all swarms
    TaskID transport = none;
    for (const auto &swarm_name : swarm_names) {
      auto transport_swarm =
          tl.AddTask(none, TransportParticles, base.get(), swarm_name, dt);
      transport = transport | transport_swarm;
    }

    // Swarm communication (once for all swarms)
    auto reset_comms =
        tl.AddTask(transport, parthenon::ResetSwarmsCommunicationMesh, base);
    auto send = tl.AddTask(reset_comms | transport, parthenon::SendSwarmsMesh, base);
    auto receive =
        tl.AddTask(send | reset_comms | transport, parthenon::ReceiveSwarmsMesh, base);

    // Sample all swarms (each depends on receive)
    for (const auto &swarm_name : swarm_names) {
      auto sample = tl.AddTask(receive, SampleFieldsAtTracers<MeshData<Real>>, base.get(),
                               swarm_name);
    }
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Tracers::TransportParticles
//  \brief Transport tracer particles using RK2 with MC-limited velocity reconstruction
TaskStatus TransportParticles(MeshData<Real> *md, const std::string &swarm_name,
                              const Real dt) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Extract tracers package
  const auto &tracers_pkg = pm->packages.Get("tracers");

  // If tracers are fixed in space, skip transport
  const bool advect = tracers_pkg->Param<bool>(swarm_name + "_advect");
  if (!advect) return TaskStatus::complete;

  // Pack positions
  // NOTE(@pdmullen): Cannot use static when swarm_name varies per call
  auto desc_pos = parthenon::MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y,
                                                     swarm_position::z>(swarm_name);
  auto pack_pos = desc_pos.GetPack(md);

  // Pack bulk velocity fields
  static auto desc_vel =
      parthenon::MakePackDescriptor<ccbulk::velocity>(resolved_pkgs.get());
  auto pack_vel = desc_vel.GetPack(md);

  // Check dimensionality
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);

  auto pidx_space = RiotUtils::GetParticleIndexSpace(pack_pos);
  RiotParticleLoop::flat(
      pidx_space, KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = pack_pos.GetBlockParticleIndices(idx);
        const auto swarm_d = pack_pos.GetContext(b);

        if (swarm_d.IsActive(n)) {
          const auto &coords = pack_vel.GetCoordinates(b);
          Real &x1 = pack_pos(b, swarm_position::x(), n);
          Real &x2 = pack_pos(b, swarm_position::y(), n);
          Real &x3 = pack_pos(b, swarm_position::z(), n);

          // Compute cell indices
          int ip, jp, kp;
          swarm_d.Xtoijk(x1, x2, x3, ip, jp, kp);

          // Reconstruct velocity at initial position
          Real v1, v2, v3;
          ReconstructVelocity(pack_vel, b, kp, jp, ip, x1, x2, x3, coords, multi_d,
                              three_d, v1, v2, v3);

          // Midpoint position
          const Real x1_tmp = x1 + 0.5 * dt * v1;
          const Real x2_tmp = x2 + 0.5 * dt * v2;
          const Real x3_tmp = x3 + 0.5 * dt * v3;

          // Update cell indices for midpoint
          swarm_d.Xtoijk(x1_tmp, x2_tmp, x3_tmp, ip, jp, kp);

          // Reconstruct velocity at midpoint
          ReconstructVelocity(pack_vel, b, kp, jp, ip, x1_tmp, x2_tmp, x3_tmp, coords,
                              multi_d, three_d, v1, v2, v3);

          // Final position update using midpoint velocity (RK2)
          x1 += dt * v1;
          x2 += dt * v2;
          x3 += dt * v3;
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Tracers::SampleFieldsAtTracers
//  \brief Sample requested fields at tracer particle locations (template implementation)
template <typename T>
TaskStatus SampleFieldsAtTracers(T *data, const std::string &swarm_name) {
  using RiotReconstruction::ComputeMCSlope;

  auto pm = data->GetParentPointer();
  auto pmesh = data->GetMeshPointer();
  const auto &tracers_pkg = pm->packages.Get("tracers");
  const auto &resolved_pkgs = pm->resolved_packages;

  // Get precomputed metadata from Initialize
  const auto &all_fields =
      tracers_pkg->template Param<std::vector<std::string>>(swarm_name + "_all_fields");
  const auto &all_swarm_vars = tracers_pkg->template Param<std::vector<std::string>>(
      swarm_name + "_all_swarm_vars");
  const int n_fields = tracers_pkg->template Param<int>(swarm_name + "_n_fields");

  // Get precomputed device arrays
  const auto &is_sparse_d = tracers_pkg->template Param<parthenon::ParArray1D<bool>>(
      swarm_name + "_is_sparse_d");
  const auto &ncomps_d =
      tracers_pkg->template Param<parthenon::ParArray1D<int>>(swarm_name + "_ncomps_d");
  const auto &swarm_offset_d = tracers_pkg->template Param<parthenon::ParArray1D<int>>(
      swarm_name + "_swarm_offset_d");
  const auto &total_size_d = tracers_pkg->template Param<parthenon::ParArray1D<int>>(
      swarm_name + "_total_size_d");
  const auto &sparse_cumulative_offset_d =
      tracers_pkg->template Param<parthenon::ParArray2D<int>>(
          swarm_name + "_sparse_cumulative_offset_d");

  // Create packs
  auto desc_pos = parthenon::MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y,
                                                     swarm_position::z>(swarm_name);
  auto pack_pos = desc_pos.GetPack(data);

  auto desc_swarm = parthenon::MakeSwarmPackDescriptor<Real>(swarm_name, all_swarm_vars);
  auto pack_swarm = desc_swarm.GetPack(data);

  auto desc_field = parthenon::MakePackDescriptor<>(resolved_pkgs.get(), all_fields);
  auto pack_field = desc_field.GetPack(data);

  // Build per-block, per-field pack bounds (depends on current sparse allocation)
  const int nblocks = pack_field.GetNBlocks();
  parthenon::ParArray2D<int> pack_lo_d("pack_lo", nblocks, n_fields);
  parthenon::ParArray2D<int> pack_hi_d("pack_hi", nblocks, n_fields);
  auto pack_lo_h = pack_lo_d.GetHostMirror();
  auto pack_hi_h = pack_hi_d.GetHostMirror();
  for (int b = 0; b < nblocks; b++) {
    for (int f = 0; f < n_fields; f++) {
      parthenon::PackIdx pidx(f);
      pack_lo_h(b, f) = pack_field.GetLowerBoundHost(b, pidx);
      pack_hi_h(b, f) = pack_field.GetUpperBoundHost(b, pidx);
    }
  }
  pack_lo_d.DeepCopy(pack_lo_h);
  pack_hi_d.DeepCopy(pack_hi_h);

  // Get bounds for reconstruction
  const bool multi_d = (pmesh->ndim > 1);
  const bool three_d = (pmesh->ndim > 2);

  auto pidx_space = RiotUtils::GetParticleIndexSpace(pack_swarm);
  RiotParticleLoop::flat(
      pidx_space, KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = pack_swarm.GetBlockParticleIndices(idx);
        const auto swarm_d = pack_swarm.GetContext(b);

        if (swarm_d.IsActive(n)) {
          // Get particle position and compute cell indices
          const Real x1 = pack_pos(b, swarm_position::x(), n);
          const Real x2 = pack_pos(b, swarm_position::y(), n);
          const Real x3 = pack_pos(b, swarm_position::z(), n);
          int ip, jp, kp;
          swarm_d.Xtoijk(x1, x2, x3, ip, jp, kp);

          // Cell center position
          const auto &coords = pack_field.GetCoordinates(b);
          const Real x1c = coords.template Xc<X1DIR>(kp, jp, ip);
          const Real x2c = coords.template Xc<X2DIR>(kp, jp, ip);
          const Real x3c = coords.template Xc<X3DIR>(kp, jp, ip);

          // Cell spacing (minus = from neighbor-1 to current, plus = from current to
          // neighbor+1)
          const Real dx1m = coords.Dxc(X1DIR, kp, jp, ip);
          const Real dx1p = coords.Dxc(X1DIR, kp, jp, ip + 1);
          const Real dx2m = coords.Dxc(X2DIR, kp, jp, ip);
          const Real dx2p = coords.Dxc(X2DIR, kp, jp + multi_d, ip);
          const Real dx3m = coords.Dxc(X3DIR, kp, jp, ip);
          const Real dx3p = coords.Dxc(X3DIR, kp + three_d, jp, ip);

          // Displacement from cell center
          const Real delta1 = x1 - x1c;
          const Real delta2 = x2 - x2c;
          const Real delta3 = x3 - x3c;

          // Lambda for second-order reconstruction
          auto reconstruct = [&](const int pack_idx) {
            const Real u_i = pack_field(b, pack_idx, kp, jp, ip);

            // Compute slopes in each direction
            const Real u_im1 = pack_field(b, pack_idx, kp, jp, ip - 1);
            const Real u_ip1 = pack_field(b, pack_idx, kp, jp, ip + 1);
            const Real slope1 = ComputeMCSlope(u_im1, u_i, u_ip1, dx1m, dx1p);

            Real slope2 = 0.0;
            if (multi_d) {
              const Real u_jm1 = pack_field(b, pack_idx, kp, jp - 1, ip);
              const Real u_jp1 = pack_field(b, pack_idx, kp, jp + 1, ip);
              slope2 = ComputeMCSlope(u_jm1, u_i, u_jp1, dx2m, dx2p);
            }

            Real slope3 = 0.0;
            if (three_d) {
              const Real u_km1 = pack_field(b, pack_idx, kp - 1, jp, ip);
              const Real u_kp1 = pack_field(b, pack_idx, kp + 1, jp, ip);
              slope3 = ComputeMCSlope(u_km1, u_i, u_kp1, dx3m, dx3p);
            }

            return u_i + slope1 * delta1 + slope2 * delta2 + slope3 * delta3;
          };

          // Sample each field
          for (int field_idx = 0; field_idx < n_fields; field_idx++) {
            const int nc = ncomps_d(field_idx);
            const int offset = swarm_offset_d(field_idx);
            const int total_size = total_size_d(field_idx);
            const int lo = pack_lo_d(b, field_idx);
            const int hi = pack_hi_d(b, field_idx);

            if (!is_sparse_d(field_idx)) {
              for (int c = 0; c < nc; c++) {
                pack_swarm(b, offset + c, n) = reconstruct(lo + c);
              }
            } else {
              for (int idx = 0; idx < total_size; idx++) {
                pack_swarm(b, offset + idx, n) = 0.0;
              }

              // Sample allocated sparse components using cumulative offsets
              for (int pack_idx = lo; pack_idx <= hi; pack_idx++) {
                const int sparse_id = pack_field(b, pack_idx).sparse_id;
                const int t = pack_field(b, pack_idx).t;
                const int u = pack_field(b, pack_idx).u;
                const int v = pack_field(b, pack_idx).v;
                const auto &shape = pack_field(b, pack_idx).tensor_shape;
                const int component = t * shape[1] * shape[0] + u * shape[0] + v;
                const int cumul_offset = sparse_cumulative_offset_d(field_idx, sparse_id);
                const int swarm_idx = offset + cumul_offset + component;

                pack_swarm(b, swarm_idx, n) = reconstruct(pack_idx);
              }
            }
          }
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Tracers::PostAMRInitialization
//  \brief Seed tracer particles on mesh blocks from input file positions
void PostAMRInitialization(MeshBlock *pmb, ParameterInput *pin) {
  // Get list of swarms to initialize
  const auto &tracers_pkg = pmb->packages.Get("tracers");
  const auto &swarm_names = tracers_pkg->Param<std::vector<std::string>>("swarm_names");

  // Seed particles for each swarm
  for (const auto &swarm_name : swarm_names) {
    auto &swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get(swarm_name);

    // Construct block name for this swarm's configuration
    const std::string block_name = "tracers/" + swarm_name;

    // Read particle positions from input file
    const auto x1_list = pin->GetOrAddVector<Real>(block_name, "x1", {});
    const auto x2_list = pin->GetOrAddVector<Real>(block_name, "x2", {});
    const auto x3_list = pin->GetOrAddVector<Real>(block_name, "x3", {});

    // Validate particle position lists
    const int n_particles_total = x1_list.size();
    PARTHENON_REQUIRE(n_particles_total > 0,
                      block_name + "/x1 list must have at least one particle");
    PARTHENON_REQUIRE(x2_list.size() == n_particles_total,
                      block_name + "/x2 list length does not match x1");
    PARTHENON_REQUIRE(x3_list.size() == n_particles_total,
                      block_name + "/x3 list length does not match x1");

    // Extract particles that belong to this block
    std::vector<uint64_t> id;
    std::vector<Real> x1, x2, x3;
    const auto &reg = pmb->block_size;
    for (int n = 0; n < x1_list.size(); n++) {
      if (x1_list[n] >= reg.xmin_[0] && x1_list[n] <= reg.xmax_[0] &&
          x2_list[n] >= reg.xmin_[1] && x2_list[n] <= reg.xmax_[1] &&
          x3_list[n] >= reg.xmin_[2] && x3_list[n] <= reg.xmax_[2]) {
        id.push_back(n);
        x1.push_back(x1_list[n]);
        x2.push_back(x2_list[n]);
        x3.push_back(x3_list[n]);
      }
    }

    // Skip if no particles on this block for this swarm
    const int n_particles_this_block = id.size();
    if (n_particles_this_block == 0) continue;

    auto id_d = parthenon::ParArray1D<uint64_t>("id", n_particles_this_block);
    auto x1_d = parthenon::ParArray1D<Real>("x1", n_particles_this_block);
    auto x2_d = parthenon::ParArray1D<Real>("x2", n_particles_this_block);
    auto x3_d = parthenon::ParArray1D<Real>("x3", n_particles_this_block);
    auto id_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), id_d);
    auto x1_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), x1_d);
    auto x2_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), x2_d);
    auto x3_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), x3_d);
    for (int n = 0; n < n_particles_this_block; n++) {
      id_h(n) = id[n];
      x1_h(n) = x1[n];
      x2_h(n) = x2[n];
      x3_h(n) = x3[n];
    }
    Kokkos::deep_copy(id_d, id_h);
    Kokkos::deep_copy(x1_d, x1_h);
    Kokkos::deep_copy(x2_d, x2_h);
    Kokkos::deep_copy(x3_d, x3_h);

    // Add particles that belong to this block
    auto new_particles_context = swarm->AddEmptyParticles(n_particles_this_block);
    auto &swarm_id = swarm->Get<uint64_t>(swarm_position::id::name()).Get();
    auto &swarm_x1 = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &swarm_x2 = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &swarm_x3 = swarm->Get<Real>(swarm_position::z::name()).Get();
    pmb->par_for(
        PARTHENON_AUTO_LABEL, 0, new_particles_context.GetNewParticlesMaxIndex(),
        KOKKOS_LAMBDA(const int new_n) {
          const int n = new_particles_context.GetNewParticleIndex(new_n);
          swarm_id(n) = id_d(new_n);
          swarm_x1(n) = x1_d(new_n);
          swarm_x2(n) = x2_d(new_n);
          swarm_x3(n) = x3_d(new_n);
        });
  }

  // Sample fields at initial particle positions
  for (const auto &swarm_name : swarm_names) {
    SampleFieldsAtTracers(pmb->meshblock_data.Get().get(), swarm_name);
  }
}

//----------------------------------------------------------------------------------------
//! template instantiations
template TaskStatus SampleFieldsAtTracers<MeshData<Real>>(MeshData<Real> *data,
                                                          const std::string &swarm_name);
template TaskStatus
SampleFieldsAtTracers<MeshBlockData<Real>>(MeshBlockData<Real> *data,
                                           const std::string &swarm_name);

} // namespace Tracers

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
#include <array>
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
//  \brief Initialize tracer particle package and create lukewarm fields
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t &packages) {
  auto tracers = std::make_shared<StateDescriptor>("tracers");
  Params &params = tracers->AllParams();

  // Read list of lukewarm names
  std::vector<std::string> lukewarm_names;
  const std::string prefix = "tracers/";
  auto blocks = pin->GetBlockNamesWithPrefix(prefix);
  for (const auto &block_name : blocks) {
    const std::string lukewarm_name = block_name.substr(prefix.size());
    lukewarm_names.push_back(lukewarm_name);
  }
  params.Add("lukewarm_names", lukewarm_names);

  // Set up lukewarms
  Metadata lukewarm_metadata({Metadata::Provides});
  for (const auto &lukewarm_name : lukewarm_names) {
    // Construct block name for this lukewarm's configuration
    const std::string block_name = "tracers/" + lukewarm_name;

    // Read advection control flag for this lukewarm
    const bool advect = pin->GetOrAddBoolean(block_name, "advect", true);
    params.Add(lukewarm_name + "_advect", advect);

    // Read list of fields to sample at particle locations for this lukewarm
    const auto sample_fields =
        pin->GetOrAddVector<std::string>(block_name, "sample_fields", {});

    // Add lukewarm
    tracers->AddLukewarm(lukewarm_name, lukewarm_metadata);

    // Register lukewarm sampling fields and create indexing arrays
    AddSamplingLukewarmFields(sample_fields, packages, tracers.get(), lukewarm_name,
                              params);
  }

  tracers->PostInitializationMesh = SeedTracers;
  return tracers;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection Tracers::PushTracers
//  \brief Build task collection for tracer particle transport and field sampling
TaskCollection PushTracers(Mesh *pmesh, parthenon::SimTime &tm, Real dt) {
  TaskCollection tc;
  TaskID none(0);

  // Get list of lukewarms
  const auto &tracers_pkg = pmesh->packages.Get("tracers");
  const auto &lukewarm_names =
      tracers_pkg->Param<std::vector<std::string>>("lukewarm_names");

  const int num_partitions = pmesh->DefaultNumPartitions();
  auto &tr = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = tr[i];
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);

    // Transport each lukewarm, hand particles that left this MeshData to their new
    // owners, then sample
    for (const auto &lukewarm_name : lukewarm_names) {
      auto transport =
          tl.AddTask(none, TransportParticles, base.get(), lukewarm_name, dt);
      auto exchange =
          parthenon::AddLukewarmExchangeTasks(transport, tl, base.get(), lukewarm_name);
      auto sample =
          tl.AddTask(exchange, SampleFieldsAtTracers, base.get(), lukewarm_name);
    }
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Tracers::TransportParticles
//  \brief Transport tracer particles using RK2 with MC-limited velocity reconstruction
TaskStatus TransportParticles(MeshData<Real> *md, const std::string &lukewarm_name,
                              const Real dt) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Extract tracers package
  const auto &tracers_pkg = pm->packages.Get("tracers");

  // If tracers are fixed in space, skip transport
  const bool advect = tracers_pkg->Param<bool>(lukewarm_name + "_advect");
  if (!advect) return TaskStatus::complete;

  // Pack positions
  // NOTE(@pdmullen): Cannot use static when lukewarm_name varies per call
  auto desc_pos =
      parthenon::MakeLukewarmPackDescriptor<lukewarm::x, lukewarm::y, lukewarm::z>(
          md, lukewarm_name);
  auto pack_pos = desc_pos.GetPack(md);

  // Pack bulk velocity fields
  static auto desc_vel =
      parthenon::MakePackDescriptor<ccbulk::velocity>(resolved_pkgs.get());
  auto pack_vel = desc_vel.GetPack(md);

  // Check dimensionality
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);

  // First interior cell, for indexing the midpoint
  const int is = md->GetBoundsI(IndexDomain::interior).s;
  const int js = md->GetBoundsJ(IndexDomain::interior).s;
  const int ks = md->GetBoundsK(IndexDomain::interior).s;

  // Storage is dense, so every particle is active
  RiotParticleLoop::flat(
      {0, pack_pos.GetNParticles() - 1}, KOKKOS_LAMBDA(const int n) {
        const int b = pack_pos.GetBlockIndex(n);
        const auto &geometry = pack_pos.GetGeometry();
        const auto &coords = pack_vel.GetCoordinates(b);
        Real &x1 = pack_pos(lukewarm::x(), n);
        Real &x2 = pack_pos(lukewarm::y(), n);
        Real &x3 = pack_pos(lukewarm::z(), n);

        // Compute cell indices
        const auto [kp, jp, ip] = pack_pos.GetCellIndices(b, n);

        // Reconstruct velocity at initial position
        Real v1, v2, v3;
        ReconstructVelocity(pack_vel, b, kp, jp, ip, x1, x2, x3, coords, multi_d, three_d,
                            v1, v2, v3);

        // Midpoint position
        const Real x1_tmp = x1 + 0.5 * dt * v1;
        const Real x2_tmp = x2 + 0.5 * dt * v2;
        const Real x3_tmp = x3 + 0.5 * dt * v3;

        // Cell indices for midpoint. It may be outside block b, so unlike
        // GetCellIndices() these are not clamped to the interior, and the velocity there
        // is reconstructed from the ghost cells of block b.
        const int im = is + static_cast<int>(Kokkos::floor(
                                (x1_tmp - geometry.Xmin(b, X1DIR)) / coords.Dx<X1DIR>()));
        const int jm =
            multi_d ? js + static_cast<int>(Kokkos::floor(
                               (x2_tmp - geometry.Xmin(b, X2DIR)) / coords.Dx<X2DIR>()))
                    : js;
        const int km =
            three_d ? ks + static_cast<int>(Kokkos::floor(
                               (x3_tmp - geometry.Xmin(b, X3DIR)) / coords.Dx<X3DIR>()))
                    : ks;

        // Reconstruct velocity at midpoint
        ReconstructVelocity(pack_vel, b, km, jm, im, x1_tmp, x2_tmp, x3_tmp, coords,
                            multi_d, three_d, v1, v2, v3);

        // Final position update using midpoint velocity (RK2)
        x1 += dt * v1;
        x2 += dt * v2;
        x3 += dt * v3;
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Tracers::SampleFieldsAtTracers
//  \brief Sample requested fields at tracer particle locations
TaskStatus SampleFieldsAtTracers(MeshData<Real> *md, const std::string &lukewarm_name) {
  using RiotReconstruction::ComputeMCSlope;

  auto pm = md->GetParentPointer();
  const auto &tracers_pkg = pm->packages.Get("tracers");
  const auto &resolved_pkgs = pm->resolved_packages;

  // Get precomputed metadata from Initialize
  const auto &all_fields =
      tracers_pkg->Param<std::vector<std::string>>(lukewarm_name + "_all_fields");
  const auto &all_sample_vars =
      tracers_pkg->Param<std::vector<std::string>>(lukewarm_name + "_all_sample_vars");
  const int n_fields = tracers_pkg->Param<int>(lukewarm_name + "_n_fields");
  if (n_fields == 0) return TaskStatus::complete;

  // Get precomputed device arrays
  const auto &is_sparse_d =
      tracers_pkg->Param<parthenon::ParArray1D<bool>>(lukewarm_name + "_is_sparse_d");
  const auto &ncomps_d =
      tracers_pkg->Param<parthenon::ParArray1D<int>>(lukewarm_name + "_ncomps_d");
  const auto &total_size_d =
      tracers_pkg->Param<parthenon::ParArray1D<int>>(lukewarm_name + "_total_size_d");
  const auto &sparse_cumulative_offset_d = tracers_pkg->Param<parthenon::ParArray2D<int>>(
      lukewarm_name + "_sparse_cumulative_offset_d");

  // Create packs
  auto desc_pos =
      parthenon::MakeLukewarmPackDescriptor<lukewarm::x, lukewarm::y, lukewarm::z>(
          md, lukewarm_name);
  auto pack_pos = desc_pos.GetPack(md);

  auto desc_field = parthenon::MakePackDescriptor<>(resolved_pkgs.get(), all_fields);
  auto pack_field = desc_field.GetPack(md);

  // The sampled fields are only known at runtime, so they are written straight to the
  // lukewarm's Real storage, where component c of field f is row sample_row(f) + c
  const auto &lukewarm = md->GetLukewarm(lukewarm_name);
  const auto samples = lukewarm.GetData<Real>();
  parthenon::ParArray1D<int> sample_row_d("sample_row", n_fields);
  auto sample_row_h = sample_row_d.GetHostMirror();
  for (int f = 0; f < n_fields; f++) {
    sample_row_h(f) = lukewarm.Get<Real>(all_sample_vars[f]).GetOffset();
  }
  sample_row_d.DeepCopy(sample_row_h);

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
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);

  // Storage is dense, so every particle is active
  RiotParticleLoop::flat(
      {0, pack_pos.GetNParticles() - 1}, KOKKOS_LAMBDA(const int n) {
        // Get particle position and compute cell indices
        const int b = pack_pos.GetBlockIndex(n);
        const Real x1 = pack_pos(lukewarm::x(), n);
        const Real x2 = pack_pos(lukewarm::y(), n);
        const Real x3 = pack_pos(lukewarm::z(), n);
        const auto [kp, jp, ip] = pack_pos.GetCellIndices(b, n);

        // Cell center position
        const auto &coords = pack_field.GetCoordinates(b);
        const Real x1c = coords.Xc<X1DIR>(kp, jp, ip);
        const Real x2c = coords.Xc<X2DIR>(kp, jp, ip);
        const Real x3c = coords.Xc<X3DIR>(kp, jp, ip);

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
          const int row = sample_row_d(field_idx);
          const int total_size = total_size_d(field_idx);
          const int lo = pack_lo_d(b, field_idx);
          const int hi = pack_hi_d(b, field_idx);

          if (!is_sparse_d(field_idx)) {
            for (int c = 0; c < nc; c++) {
              samples(row + c, n) = reconstruct(lo + c);
            }
          } else {
            for (int idx = 0; idx < total_size; idx++) {
              samples(row + idx, n) = 0.0;
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

              samples(row + cumul_offset + component, n) = reconstruct(pack_idx);
            }
          }
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Tracers::SeedTracers
//  \brief Seed tracer particles on the blocks of a MeshData from input file positions
void SeedTracers(Mesh *pm, ParameterInput *pin, MeshData<Real> *md) {
  // Get list of lukewarms to initialize
  const auto &tracers_pkg = pm->packages.Get("tracers");
  const auto &lukewarm_names =
      tracers_pkg->Param<std::vector<std::string>>("lukewarm_names");
  const int nblocks = md->NumBlocks();

  // Seed particles for each lukewarm
  for (const auto &lukewarm_name : lukewarm_names) {
    auto &lukewarm = md->GetLukewarm(lukewarm_name);

    // Construct block name for this lukewarm's configuration
    const std::string block_name = "tracers/" + lukewarm_name;

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

    // Extract particles that belong to the blocks of this MeshData. Blocks share faces,
    // so each block owns its lower faces only, as Locate() does, and every particle is
    // seeded once. Inactive directions are ignored.
    std::vector<uint64_t> id;
    std::vector<Real> x1, x2, x3;
    for (int n = 0; n < n_particles_total; n++) {
      const std::array<Real, 3> x = {x1_list[n], x2_list[n], x3_list[n]};
      for (int b = 0; b < nblocks; b++) {
        const auto &reg = md->GetBlockData(b)->GetBlockPointer()->block_size;
        bool inside = true;
        for (int d = 0; d < pm->ndim; d++) {
          inside = inside && x[d] >= reg.xmin_[d] && x[d] < reg.xmax_[d];
        }
        if (inside) {
          id.push_back(n);
          x1.push_back(x[0]);
          x2.push_back(x[1]);
          x3.push_back(x[2]);
          break;
        }
      }
    }

    // Skip if no particles on this MeshData for this lukewarm
    const int n_particles_here = id.size();
    if (n_particles_here == 0) continue;

    auto id_d = parthenon::ParArray1D<uint64_t>("id", n_particles_here);
    auto x1_d = parthenon::ParArray1D<Real>("x1", n_particles_here);
    auto x2_d = parthenon::ParArray1D<Real>("x2", n_particles_here);
    auto x3_d = parthenon::ParArray1D<Real>("x3", n_particles_here);
    auto id_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), id_d);
    auto x1_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), x1_d);
    auto x2_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), x2_d);
    auto x3_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), x3_d);
    for (int n = 0; n < n_particles_here; n++) {
      id_h(n) = id[n];
      x1_h(n) = x1[n];
      x2_h(n) = x2[n];
      x3_h(n) = x3[n];
    }
    Kokkos::deep_copy(id_d, id_h);
    Kokkos::deep_copy(x1_d, x1_h);
    Kokkos::deep_copy(x2_d, x2_h);
    Kokkos::deep_copy(x3_d, x3_h);

    // Add the particles, keeping their index in the input lists as their id, and locate
    // each in its block. A pack is a snapshot, so get it after creating them.
    const auto range = lukewarm.Create(n_particles_here);
    auto desc =
        parthenon::MakeLukewarmPackDescriptor<lukewarm::x, lukewarm::y, lukewarm::z,
                                              lukewarm::id>(md, lukewarm_name);
    auto pack = desc.GetPack(md);
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, range.s, range.e, KOKKOS_LAMBDA(const int n) {
          const int new_n = n - range.s;
          pack(lukewarm::id(), n) = id_d(new_n);
          pack(lukewarm::x(), n) = x1_d(new_n);
          pack(lukewarm::y(), n) = x2_d(new_n);
          pack(lukewarm::z(), n) = x3_d(new_n);
          const auto loc = pack.Locate(n);
          PARTHENON_REQUIRE(loc.InMeshData(), "Tracer seeded outside of its MeshData");
        });
  }

  // Sample fields at initial particle positions
  for (const auto &lukewarm_name : lukewarm_names) {
    SampleFieldsAtTracers(md, lukewarm_name);
  }
}

} // namespace Tracers

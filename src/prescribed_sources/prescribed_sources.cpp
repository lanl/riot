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

#include <algorithm>
#include <limits>
#include <ranges>
#include <string>
#include <vector>

#include <utils/error_checking.hpp>
#include <utils/string_utils.hpp>

#include <ports-of-call/robust_utils.hpp>

#include "diagnostics/masses.hpp"
#include "materials/materials.hpp"
#include "riot_driver.hpp"
#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "riot_utils/table_utils.hpp"
#include "variables.hpp"

#include "prescribed_sources/prescribed_sources.hpp"

namespace PrescribedSources {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;

  auto physics = std::make_shared<StateDescriptor>("prescribed_sources");
  Params &params = physics->AllParams();

  // Masses per material required to uniformly distribute energies by mass
  int nmat = Materials::CountMaterials(pin);
  parthenon::ParArray1D<Real> material_masses("material_mass_sums", nmat);
  // TODO(JMM): Probably doesn't need to be restart. Probably overkill.
  physics->AddParam("material_mass_sums", material_masses,
                    parthenon::Params::Mutability::Restart);

  std::vector<RiotTables::Uniform1D> cumulative_energies_h;
  std::vector<RiotTables::Uniform1D> cumulative_energies_d_vec;
  std::vector<SourceMode> mode;
  std::vector<int> matids_h;
  // JMM: Spiner::DataBox::range() is not marked for device execution, so
  // we cannot query the table time bounds inside device kernels. Cache
  // tmin/tmax per source on host here and copy them to device below.
  // TODO(JMM): Remove once Spiner marks range() portable.
  std::vector<Real> tmin_h;
  std::vector<Real> tmax_h;

  const std::string prefix = "energy_source";
  for (const std::string &block : pin->GetBlockNamesWithPrefix(prefix)) {
    // Simple way to disable a given energy source
    const bool active = pin->GetOrAddBoolean(block, "active", true);
    if (!active) continue;

    // TODO(JMM): Any reason to be able to save as hdf5 and reload?
    // Or should we try and make it transparently work with restarts?
    const int matid = pin->GetInteger(block, "material", "material for this source");
    const std::string ratefile = pin->GetString(
        block, "cumulative_energies",
        "path to file containing a time series of total energy to add "
        "to a given material. First column is time, second is energy added. "
        "energies are cumulative and total, integrated over the material. "
        "sources mustbe monotone and positive.");
    auto ratedata_host = RiotTables::UniformlyResampleTimeSeries(
        parthenon::string_utils::ParseAsciiTable<Real>(ratefile));
    auto ratedata_device = ratedata_host.getOnDevice();

    // TODO(JMM): We COULD inject 0 as an initial time point but this
    // is ambiguous because when should hte first zero be placed?
    // JMM: Use single precision epsilon as the bound here to account
    // for users using different precision when writing rate files.
    if (std::abs(ratedata_host(0)) > 10 * std::numeric_limits<float>::epsilon()) {
      printf("Ratedata first row = %.14e\n", ratedata_host(0));
      PARTHENON_THROW("A rate data file must specify 0 as the first energy point.");
    }

    matids_h.push_back(matid);
    cumulative_energies_h.push_back(ratedata_host);
    cumulative_energies_d_vec.push_back(ratedata_device);
    tmin_h.push_back(ratedata_host.range(0).min());
    tmax_h.push_back(ratedata_host.range(0).max());
  }

  std::vector<int> source_from_matid_h(nmat, -1); // negative value means no source
  for (int i = 0; i < matids_h.size(); ++i) {
    source_from_matid_h[matids_h[i]] = i;
  }

  auto cumulative_energies_d =
      RiotUtils::VectorToDevice(cumulative_energies_d_vec, "cumulative_energies");
  auto source_from_matid =
      RiotUtils::VectorToDevice(source_from_matid_h, "energy_source_from_matid");
  auto tmin_d = RiotUtils::VectorToDevice(tmin_h, "cumulative_energies_tmin");
  auto tmax_d = RiotUtils::VectorToDevice(tmax_h, "cumulative_energies_tmax");

  params.Add("energy_source_from_matid", source_from_matid);
  params.Add("sourced_materials", matids_h);
  params.Add("d.cumulative_energies", cumulative_energies_d);
  params.Add("h.cumulative_energies", cumulative_energies_h);
  params.Add("d.cumulative_energies_tmin", tmin_d);
  params.Add("d.cumulative_energies_tmax", tmax_d);

  const Real dt_safety =
      pin->GetOrAddReal("prescribed_sources", "dt_safety", 0.9,
                        "Safety factor on prescribed sources timestep control");
  params.Add("dt_safety", dt_safety);

  // Most recent time used for update. Needed for estimate timestep so
  // we know where we are in the table.
  Real tlast = pin->GetOrAddReal("parthenon/time", "start_time", 0.0,
                                 "physical time at which to start the simulation");
  params.Add("tlast", tlast, parthenon::Params::Mutability::Restart);

  physics->EstimateTimestepMesh = EstimateTimestepMesh;

  // Compute mass sums so we have them available
  physics->PreStepDiagnosticsMesh = [](parthenon::SimTime const &, MeshData<Real> *md) {
    ReduceMasses(md);
  };
  // TODO(JMM): This one is necessary so the diagnostic is in the
  // first output/restart
  physics->UserWorkBeforeLoopMesh = [](Mesh *pmesh, ParameterInput *,
                                       parthenon::SimTime &) {
    MeshData<Real> *md = pmesh->mesh_data.Get().get();
    ReduceMasses(md);
  };

  return physics;
}

/* JMM: Note a tricky subtlety with energy sources and mass sums:
 *
 * Mass sums and energy sources are per-MATERIAL, not per-PHASE. But
 * of course densities, internal energies, and specific heats are per
 * phase. This is ok because sources are treated as SPECIFIC and thus
 * we are assuming that deps/dt is constant accross phases and equal
 * to the deps/dt for a material.
 */

TaskStatus EnergySources(MeshData<Real> *md, parthenon::SimTime &tm, const Real dt) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using parthenon::ParArray1D;
  using PortsOfCall::Robust::ratio;
  using RiotTables::Uniform1D;

  auto pm = md->GetParentPointer();

  auto &sources_pkg = pm->packages.Get("prescribed_sources");

  const auto &src_from_matid =
      sources_pkg->Param<ParArray1D<int>>("energy_source_from_matid");
  const auto &sourced_materials =
      sources_pkg->Param<std::vector<int>>("sourced_materials");
  const auto &cumulative_energies =
      sources_pkg->Param<ParArray1D<Uniform1D>>("d.cumulative_energies");
  const auto &cumulative_energies_tmin =
      sources_pkg->Param<ParArray1D<Real>>("d.cumulative_energies_tmin");
  const auto &cumulative_energies_tmax =
      sources_pkg->Param<ParArray1D<Real>>("d.cumulative_energies_tmax");
  Real *ptlast = sources_pkg->MutableParam<Real>("tlast");
  const Real t = tm.time;

  if (sourced_materials.size() == 0) {
    return TaskStatus::complete;
  }

  // assume this is available
  const auto &mass_sum =
      sources_pkg->Param<parthenon::ParArray1D<Real>>("material_mass_sums");

  // Pack material densities (sparse) and the bulk energy this package sources into
  auto v = riot::MakePack<ccmat::rho, ccbulk::total_material_energy>(md);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(), md,
                                     parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        const int nmat = v.GetSize(b, ccmat::rho());
        for (int m = 0; m < nmat; ++m) {
          auto pv_m = RiotLoop::make_sparse_pack_view(idx_range, v, m);
          const int matid = v(b, ccmat::rho(m)).sparse_id;
          const int mm = src_from_matid(matid);
          Real dE = 0.0;
          if ((mm >= 0) && (mass_sum(matid) > 0)) {
            auto &tab_esum = cumulative_energies(mm);
            const Real tmin = cumulative_energies_tmin(mm); // constant extrapolation
            const Real tmax = cumulative_energies_tmax(mm);
            const Real tnow = std::max(tmin, std::min(tmax, t));
            const Real tnext = std::max(tmin, std::min(tmax, t + dt));
            const Real enow = tab_esum.interpToReal(tnow);
            const Real enext = tab_esum.interpToReal(tnext);
            dE = ratio(std::max(enext - enow, 0.0), mass_sum(matid));
          }
          RiotLoop::inner(idx_range, [&](const auto kji) {
            pv(ccbulk::total_material_energy(), kji) += pv_m(ccmat::rho(), kji) * dE;
          });
          idx_range.TeamBarrier();
        }
      });

  *ptlast = t + dt;

  return TaskStatus::complete;
}

parthenon::TaskCollection Step(Mesh *pm, parthenon::SimTime &tm, const Real dt) {
  parthenon::TaskCollection tc;
  parthenon::TaskID none(0);
  namespace mdname = riot::container_names;

  auto &options = pm->packages.Get("prescribed_sources");
  const auto &sourced_materials = options->Param<std::vector<int>>("sourced_materials");
  if (sourced_materials.size() == 0) return tc;

  const int num_partitions = pm->DefaultNumPartitions();
  parthenon::TaskRegion &tr = tc.AddRegion(num_partitions);

  for (int i = 0; i < num_partitions; i++) {
    auto &tl = tr[i];
    auto &mdpart = pm->mesh_data.GetOrAdd("base", i);
    // update bulk volumetric energy (conserved quantity)
    auto add_energies = tl.AddTask(none, EnergySources, mdpart.get(), tm, dt);
    // Recompute internal energies, call PTE
    auto int_derived =
        tl.AddTask(add_energies, parthenon::Update::PreCommFillDerived<MeshData<Real>>,
                   mdpart.get());
    // comms
    bool multilevel = pm->multilevel;
    auto set_bc = parthenon::AddBoundaryExchangeTasks<parthenon::BoundaryType::any>(
        int_derived, tl, mdpart, multilevel);
    // pot-comms fill derived after PTE
    auto derive =
        tl.AddTask(set_bc, parthenon::Update::FillDerived<MeshData<Real>>, mdpart.get());
  }

  return tc;
}

Real EstimateTimestepMesh(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace cm = cell_variables::material_averaged;
  using parthenon::MakePackDescriptor;
  using parthenon::ParArray1D;
  using PortsOfCall::Robust::ratio;
  using RiotTables::Uniform1D;

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  auto &materials = pm->packages.Get("materials");
  auto &sources_pkg = pm->packages.Get("prescribed_sources");
  const int nummat_max = materials->Param<int>("max_array_size");

  const auto &src_from_matid =
      sources_pkg->Param<ParArray1D<int>>("energy_source_from_matid");
  const auto &sourced_materials =
      sources_pkg->Param<std::vector<int>>("sourced_materials");
  const auto &cumulative_energies =
      sources_pkg->Param<ParArray1D<Uniform1D>>("d.cumulative_energies");
  const auto &cumulative_energies_tmin =
      sources_pkg->Param<ParArray1D<Real>>("d.cumulative_energies_tmin");
  const auto &cumulative_energies_tmax =
      sources_pkg->Param<ParArray1D<Real>>("d.cumulative_energies_tmax");
  const auto &mass_sum =
      sources_pkg->Param<parthenon::ParArray1D<Real>>("material_mass_sums");
  const Real tlast = sources_pkg->Param<Real>("tlast");
  const Real dt_safety = sources_pkg->Param<Real>("dt_safety");

  if (sourced_materials.size() == 0) {
    return 1.e-3 * std::numeric_limits<Real>::max();
  }

  static auto desc =
      MakePackDescriptor<ccbulk::temperature, cm::specific_heat>(resolved_pkgs.get());
  auto v = desc.GetPack(md);

  Real min_dt;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "Hydro::EstimateTimestep", DevExecSpace(), 0,
      v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &ldt) {
        const int nmat = v.GetSize(b, cm::specific_heat());
        const Real T = v(b, ccbulk::temperature(), k, j, i);

        for (int m = 0; m < nmat; ++m) {
          const int matid = v(b, cm::specific_heat(m)).sparse_id;
          const int mm = src_from_matid(matid);
          if (mass_sum(matid) <= 0) continue; // this material isn't present
          if (mm >= 0) {
            const Real Cv = v(b, cm::specific_heat(m), k, j, i); // per phase
            if (Cv <= 0) continue; // this material isn't present

            auto &tab_esum = cumulative_energies(mm);
            const Real tmin = cumulative_energies_tmin(mm); // constant extrapolation
            const Real tmax = cumulative_energies_tmax(mm);

            // per material d total energy dT. dEdt / mass_sum is deps/dt
            const Real dEdt = ((tmin < tlast) && (tlast < tmax))
                                  ? RiotTables::GetFDDerivative(tab_esum, tlast)
                                  : 0.0;
            // min per phase of the linearized time derivative of phase temperature
            // TODO(JMM): Note that this is pretty heuristic. We should revisit.
            ldt = std::min(ldt, ratio(Cv * T * mass_sum(matid), dEdt));
          }
        }
      },
      Kokkos::Min<Real>(min_dt));

  return dt_safety * min_dt;
}

void ReduceMasses(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto &sources_pkg = pm->packages.Get("prescribed_sources");
  auto masses_local = masses::ReduceMassesLocal(md);
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, masses_local.data(), masses_local.size(), MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD);
#endif
  auto *pmass_sum =
      sources_pkg->MutableParam<parthenon::ParArray1D<Real>>("material_mass_sums");
  RiotUtils::DeepCopyVectorToDevice(*pmass_sum, masses_local);
}

} // namespace PrescribedSources

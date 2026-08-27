//========================================================================================
// (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
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

#include "laser.hpp"
#include "riot_utils/riot_utils.hpp"
#include "variables.hpp"

namespace Laser {

using LaserFacePts_t = std::array<std::array<std::vector<Real>, sample::nvalues>, 6>;

void AddLaser(ParameterInput *pbase, const std::string &laser_name, LaserInfo &laser_info,
              const int laser_id, const int ndim) {
  // allow for a separate file containing info for this laser
  ParameterInput plaser;
  ParameterInput *pin = pbase;
  std::string name = laser_name;
  if (pin->DoesParameterExist(name, "laser_file")) {
    plaser = ParameterInput(pin->GetString(name, "laser_file"));
    if (pin->DoesParameterExist(name, "name")) name = pin->GetString(name, "name");
    pin = &plaser;
  }

  // get power vs time for this laser
  auto time = pin->GetVector<Real>(name, "time_ns");
  auto power = pin->GetVector<Real>(name, "power_watts");
  PARTHENON_REQUIRE_THROWS(time.size() == power.size(),
                           "laser time and power must be same size.");
  PARTHENON_REQUIRE_THROWS(time.size() > 1,
                           "laser time/power must have at least two values.");
  laser_info.energy.emplace_back(LaserEnergy(time, power));

  // get it's wavelength
  auto wavelength = 1.e-7 * pin->GetOrAddReal(name, "wavelength_nm", 351.0);
  laser_info.wavelength.push_back(wavelength);

  // build the local grid to span the laser profile
  LaserGrid lg(pin, name);

  // now get laser pointing
  auto lens_x = pin->GetVector<Real>(name, "lens_x");
  auto target_x = pin->GetVector<Real>(name, "target_x");
  Real target_size_ratio = pin->GetReal(name, "target_size_ratio");
  Real phit = pin->GetReal(name, "phi");
  auto pax = pin->GetString(name, "phi_axis");
  // permute coordinates to enable different reference axes
  int ix, iy, iz;
  std::array<int, 3> inv;
  if (pax == "x" || pax == "X") {
    ix = 1;
    inv[0] = 2;
    iy = 2;
    inv[1] = 0;
    iz = 0;
    inv[2] = 1;
  } else if (pax == "y" || pax == "Y") {
    ix = 2;
    inv[0] = 1;
    iy = 0;
    inv[1] = 2;
    iz = 1;
    inv[2] = 0;
  } else if (pax == "z" || pax == "Z") {
    ix = 0;
    inv[0] = 0;
    iy = 1;
    inv[1] = 1;
    iz = 2;
    inv[2] = 2;
  } else {
    PARTHENON_FAIL(pax + " is an invalid selection for phi_axis in " + name);
  }

  // build local 2d coordinate system orthogonal to the beam centerline
  std::vector<Real> d{lens_x[0] - target_x[0], lens_x[1] - target_x[1],
                      lens_x[2] - target_x[2]};
  Real dmag = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  Real dnorm = std::sqrt(d[ix] * d[ix] + d[iy] * d[iy]);
  PARTHENON_REQUIRE(dmag * dnorm > 0.0,
                    "lens and target are colocated or else their difference vector "
                    "points exactly along phi_axis, so change phi_axis");
  std::array<Real, 3> n1{
      -1.0 / dnorm * (d[ix] * d[iz] / dmag * std::cos(phit) + d[iy] * std::sin(phit)),
      1.0 / dnorm * (d[ix] * std::sin(phit) - d[ix] * d[iz] / dmag * std::cos(phit)),
      dnorm / dmag * std::cos(phit)};
  std::array<Real, 3> n2{
      1.0 / dnorm * (d[ix] * d[iz] / dmag * std::sin(phit) - d[iy] * std::cos(phit)),
      1.0 / dnorm * (d[iy] * d[iz] / dmag * std::sin(phit) + d[ix] * std::cos(phit)),
      -dnorm / dmag * std::sin(phit)};

  // permute lens and target locations to deal with different phi_axis specs
  std::array<Real, 3> lp{lens_x[ix], lens_x[iy], lens_x[iz]};
  std::array<Real, 3> tp{target_x[ix], target_x[iy], target_x[iz]};

  // get domain boundaries so we can initialize the particles there
  std::array<Real, 3> xmin, xmax;
  xmin[0] = pin->GetReal("parthenon/mesh", "x1min");
  xmax[0] = pin->GetReal("parthenon/mesh", "x1max");
  xmin[1] = pin->GetReal("parthenon/mesh", "x2min");
  xmax[1] = pin->GetReal("parthenon/mesh", "x2max");
  xmin[2] = pin->GetReal("parthenon/mesh", "x3min");
  xmax[2] = pin->GetReal("parthenon/mesh", "x3max");

  // semi-major/minor axes of the lens and target
  auto [lens_a, lens_b] = lg.Axes();
  auto target_a = target_size_ratio * lens_a;
  auto target_b = target_size_ratio * lens_b;

  // now actually fill in the laser_info object with initial particle positions and
  // weights
  auto n = lg.NumSamples();
  std::array<Real, 3> xl, xt, nray, lambda, trial;
  for (int i = 0; i < n; i++) {
    auto [r, th, w] = lg.LocationAndWeight(i);
    auto sth = std::sin(th);
    auto cth = std::cos(th);
    for (int j = 0; j < 3; j++) {
      xl[j] = lp[inv[j]] + r * lens_a * n1[inv[j]] * cth + r * lens_b * n2[inv[j]] * sth;
      xt[j] =
          tp[inv[j]] + r * target_a * n1[inv[j]] * cth + r * target_b * n2[inv[j]] * sth;
      nray[j] = xt[j] - xl[j];
    }
    auto nmag = std::sqrt(nray[0] * nray[0] + nray[1] * nray[1] + nray[2] * nray[2]);
    for (auto &nr : nray)
      nr /= nmag;

    // find step size lambda along nray that puts ray at entry point into [xmin, xmax]
    int face_dir;
    Real lambda_use = 1.e300;
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      for (int j = 0; j < ndim; j++) {
        if (nray[j] > 0.0) {
          // coming in from the left/bottom
          lambda[j] = (xmin[j] - xl[j]) / nray[j];
        } else if (nray[j] < 0.0) {
          // coming in from the right/top
          lambda[j] = (xmax[j] - xl[j]) / nray[j];
        } else {
          // orthogonal so never crosses
          lambda[j] = std::numeric_limits<Real>::max();
        }
      }
      // figure out which lambda puts the ray on the multi-d domain boundary
      for (int j = 0; j < ndim; j++) {
        bool works = true;
        for (int k = 0; k < ndim; k++) {
          trial[k] = xl[k] + lambda[j] * nray[k];
          if (trial[k] < xmin[k] - 1.e-12 || trial[k] > xmax[k] + 1.e-12) {
            works = false;
            break;
          }
        }
        if (works) {
          face_dir = j;
          lambda_use = lambda[j];
        }
      }
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      auto r_l = std::sqrt(xl[0] * xl[0] + xl[1] * xl[1]);
      auto a = nray[0] * nray[0] + nray[1] * nray[1];
      auto b = 2 * (xl[0] * nray[0] + xl[1] * nray[1]);
      auto c = xl[0] * xl[0] + xl[1] * xl[1] - xmax[0] * xmax[0];
      auto det = b * b - 4.0 * a * c;
      if (det > 0.0) {
        det = std::sqrt(det);
        auto lam1 = (-b - det) / (2.0 * a);
        auto lam2 = (-b + det) / (2.0 * a);
        // which one is entry?
        for (int j = 0; j < 2; j++) {
          trial[j] = xl[j] + lam1 * nray[j];
        }
        auto vr = trial[0] * nray[0] + trial[1] * nray[1];
        if (vr < 0) {
          // at this lambda, the ray is entering since \hat{r}\cdot\hat{n} < 0
          lambda[0] = lam1;
        } else {
          lambda[0] = lam2;
        }
        if (nray[2] > 0.0) {
          lambda[1] = (xmin[1] - xl[2]) / nray[2];
        } else if (nray[2] < 0.0) {
          lambda[1] = (xmax[1] - xl[2]) / nray[2];
        } else {
          lambda[1] = std::numeric_limits<Real>::max();
        }

        for (int j = 0; j < 2; j++) {
          for (int k = 0; k < 3; k++) {
            trial[k] = xl[k] + lambda[j] * nray[k];
          }
          auto rt = std::sqrt(trial[0] * trial[0] + trial[1] * trial[1]);
          if (rt < xmax[0] + 1.e-12 &&
              (trial[2] > xmin[1] - 1e-12 && trial[2] < xmax[1] + 1.e-12)) {
            face_dir = j;
            lambda_use = lambda[j];
          }
        }
      } else {
        lambda_use = std::numeric_limits<Real>::max();
      }
    } else {
      PARTHENON_THROW("Unsupported coordinate system in laser.");
    }
    if (lambda_use > 0.5 * std::numeric_limits<Real>::max()) {
      PARTHENON_WARN("Laser ray does not intersect domain, skipping.");
      continue;
    }
    // set coordinates = ndim and up
    for (int j = ndim; j < 3; j++)
      trial[j] = xl[j] + lambda_use * nray[j];
    int face_id = 0;
    // reset position so it falls *exactly* on boundary
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      trial[face_dir] = (nray[face_dir] > 0.0 ? xmin[face_dir] : xmax[face_dir]);
      face_id = 2 * face_dir + (nray[face_dir] < 0);
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      if (face_dir == 1 && nray[2] > 0) {
        trial[2] = xmin[1];
        face_id = 2;
      } else if (face_dir == 1) {
        trial[2] = xmax[1];
        face_id = 3;
      } else {
        auto r = std::sqrt(trial[0] * trial[0] + trial[1] * trial[1]);
        trial[0] *= xmax[0] / r;
        trial[1] *= xmax[0] / r;
        face_id = 1;
      }
    }
    laser_info.face_pts[face_id][sample::x].push_back(trial[0]);
    laser_info.face_pts[face_id][sample::y].push_back(trial[1]);
    laser_info.face_pts[face_id][sample::z].push_back(trial[2]);
    laser_info.face_pts[face_id][sample::nx].push_back(nray[0]);
    laser_info.face_pts[face_id][sample::ny].push_back(nray[1]);
    laser_info.face_pts[face_id][sample::nz].push_back(nray[2]);
    laser_info.face_pts[face_id][sample::wgt].push_back(w);
    laser_info.id[face_id].push_back(laser_id);
  }
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace nv = node_variables;
  namespace pl = particles::laser;
  auto laser = std::make_shared<StateDescriptor>("laser");
  Params &params = laser->AllParams();

  int ndim = 1 + (pin->GetInteger("parthenon/mesh", "nx2") > 1) +
             (pin->GetInteger("parthenon/mesh", "nx3") > 1);

  std::string base_name("laser");
  int numlaser = 0;
  LaserFacePts_t pts;
  LaserInfo laser_info;
  for (;;) {
    std::string name = base_name + std::to_string(numlaser);
    if (!pin->DoesBlockExist(name)) {
      break;
    }

    AddLaser(pin, name, laser_info, numlaser, ndim);

    numlaser++;
  }

  params.Add("coulomb_log", 7.0);
  params.Add("node_interp_order", pin->GetOrAddInteger("laser", "node_interp_order", 2));
  params.Add("enable_laser_deposition",
             static_cast<Real>(pin->GetOrAddBoolean("laser", "enable_deposition", true)));
  params.Add("laser_dt_safety", pin->GetOrAddReal("laser", "dt_safety", 0.95));
  params.Add("laser_dt_edot_floor", pin->GetOrAddReal("laser", "dt_edot_floor", 0.2));
  params.Add("laser_dt_tau_cutoff", pin->GetOrAddReal("laser", "dt_tau_cutoff", 0.2));

  params.Add("laser_info", laser_info);

  auto lambda_d =
      parthenon::ParArray1D<Real>("laser wavelength", laser_info.wavelength.size());
  auto lambda_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), lambda_d);
  for (int i = 0; i < laser_info.wavelength.size(); i++) {
    lambda_h(i) = laser_info.wavelength[i];
  }
  Kokkos::deep_copy(lambda_d, lambda_h);
  params.Add("laser_wavelength", lambda_d);

  params.Add("num_active", AllReduce<int>(), true);
  params.Add("laser_dt", 1.e300, true);

  Metadata m({Metadata::Cell, Metadata::OneCopy});
  laser->AddField<ccbulk::laser_deposition>(m);
  laser->AddField<ccbulk::laser_energy_density>(m);
  laser->AddField<ccbulk::laser_tau_max>(m);
  laser->AddField<ccbulk::ionization_zbar>(m);
  m = Metadata(std::vector<parthenon::MetadataFlag>{Metadata::Node, Metadata::OneCopy,
                                                    Metadata::CellMemAligned});
  laser->AddField<nv::electron_number_density>(m);

  Metadata mswarm({Metadata::Provides, Metadata::None});
  laser->AddSwarm(pl::particles::name(), mswarm);
  Metadata m_real({Metadata::Real});
  laser->AddSwarmValue<pl::t, pl::particles>(m_real);
  laser->AddSwarmValue<pl::energy, pl::particles>(m_real);
  laser->AddSwarmValue<pl::vx, pl::particles>(m_real);
  laser->AddSwarmValue<pl::vy, pl::particles>(m_real);
  laser->AddSwarmValue<pl::vz, pl::particles>(m_real);
  laser->AddSwarmValue<pl::wavelength, pl::particles>(m_real);

  laser->EstimateTimestepMesh = EstimateTimestepMesh;

  laser->RegisterMeshDataSubset("dudt", RiotUtils::MakePackageDudtRequirements(
                                            {ccbulk::total_material_energy::name(),
                                             ccbulk::electron_internal_energy::name()}));

  return laser;
}

Real EstimateTimestepMesh(MeshData<Real> *md) {
  auto *pm = md->GetMeshPointer();
  return *pm->packages.Get("laser")->MutableParam<Real>("laser_dt");
}

bool CheckDt(Mesh *pm, Real *dt) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  auto md = pm->mesh_data.Get("base").get();
  auto v = riot::MakePack<ccbulk::electron_internal_energy, ccbulk::laser_deposition,
                          ccbulk::laser_tau_max, ccbulk::velocity, ccbulk::bulk_modulus,
                          ccbulk::rho, ccbulk::electron_number_density>(md);

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  auto fixed_fluid = pm->packages.Get("riot")->Param<bool>("fixed_fluid");
  auto edot_floor = pm->packages.Get("laser")->Param<Real>("laser_dt_edot_floor");
  auto tau_cutoff = pm->packages.Get("laser")->Param<Real>("laser_dt_tau_cutoff");
  const int ndim = pm->ndim;
  const Real dt0 = *dt;

  Real min_dt;
  parthenon::par_reduce(
      parthenon::loop_pattern_flatrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &dtmin) {
        // time step necessary to ensure thermal waves propagate at the right speed
        Real tau = v(b, ccbulk::laser_tau_max(), k, j, i);
        Real tau_2_3 = std::pow(tau, 2.0 / 3.0);
        // Real tau_fact = tau > tau_fact0 ? (tau_2_3 - 1.0): (tau_fact0 * (tau0/tau));
        Real tau_fact = std::max(std::max(tau_2_3, 1.0) - 1.0, edot_floor);
        tau_fact *= (tau > tau_cutoff ? 1.0 : 1.e100);
        Real u_e = v(b, ccbulk::electron_internal_energy(), k, j, i);
        Real dt_prop =
            tau_fact * u_e / (v(b, ccbulk::laser_deposition(), k, j, i) + 1.e-12 * u_e);
        // now time step necessary to avoid cfl violations in the subsequent hydro step
        Real cmax = 0.0;
        auto &coord = v.GetCoordinates(b);
        for (int d = 0; d < ndim; d++) {
          cmax = std::max(cmax, coord.Dx(d + 1) / dt0 -
                                    std::abs(v(b, ccbulk::velocity(d), k, j, i)));
        }
        const Real c0sq =
            v(b, ccbulk::bulk_modulus(), k, j, i) / v(b, ccbulk::rho(), k, j, i);
        Real dt_cfl =
            fixed_fluid ? 1.e300
                        : (cmax * cmax / c0sq - 1.0) * u_e /
                              (v(b, ccbulk::laser_deposition(), k, j, i) + 1.e-12 * u_e);
        dtmin = std::min(dtmin, std::min(dt_prop, dt_cfl));
      },
      Kokkos::Min<Real>(min_dt));

  auto *laser_dt = pm->packages.Get("laser")->MutableParam<Real>("laser_dt");
  MPI_Allreduce(&min_dt, laser_dt, 1, MPI_PARTHENON_REAL, MPI_MIN, MPI_COMM_WORLD);

  auto laser_dt_safety = pm->packages.Get("laser")->Param<Real>("laser_dt_safety");
  if (*laser_dt < (*dt)) {
    *dt = laser_dt_safety * *laser_dt;
    return true;
  }
  return false;
}

TaskCollection LaserUpdateTasks(Mesh *pm, const Real t0, const Real dt) {
  namespace PL = particles::laser;
  using namespace ::parthenon::Update;
  using TQ = TaskQualifier;
  TaskCollection tc;
  TaskID none;

  auto fixed_fluid = pm->packages.Get("riot")->Param<bool>("fixed_fluid");
  auto laser = pm->packages.Get("laser").get();
  auto num_active = laser->MutableParam<AllReduce<int>>("num_active");
  num_active->val = 0;

  const int num_partitions = pm->DefaultNumPartitions();
  auto &reg = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = reg[i];
    auto md_sp = pm->mesh_data.GetOrAdd("base", i);
    auto md = md_sp.get();
    auto md_dudt = laser->GetOrAddMeshDataSubset(pm, "dudt", i).get();

    auto ne_gradne = tl.AddTask(none, SetElectrons, md);
    auto init_particles =
        tl.AddTask(ne_gradne, InitializeLaserSweep, md, md_dudt, t0, dt);

    // keep pushing particles until there are none left
    auto [itl, push] = tl.AddSublist(init_particles | ne_gradne, {1, 100000});
    auto transport =
        itl.AddTask(TQ::completion | TQ::global_sync, none, Update, md, md_dudt, t0, dt);
    auto reset_comms = itl.AddTask(none, parthenon::ResetSwarmsCommunicationMesh, md_sp);
    auto send = itl.AddTask(reset_comms, parthenon::SendSwarmsMesh, md_sp);
    auto recv = itl.AddTask(reset_comms, parthenon::ReceiveSwarmsMesh, md_sp);
  }
  return tc;
}

TaskCollection LaserDepositionTasks(Mesh *pm, const Real dt) {
  using namespace ::parthenon::Update;
  TaskID none;
  const int num_partitions = pm->DefaultNumPartitions();
  TaskCollection tc;
  auto &reg = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = reg[i];
    auto &md_sp = pm->mesh_data.GetOrAdd("base", i);
    auto md = md_sp.get();
    auto update_mat_energies = tl.AddTask(none, UpdateMatEnergy, md, dt);
    auto int_derived =
        tl.AddTask(update_mat_energies, PreCommFillDerived<MeshData<Real>>, md);
    // probably don't need to send/recv _everything_
    auto comms =
        parthenon::AddBoundaryExchangeTasks(int_derived, tl, md_sp, pm->multilevel);
    auto derive = tl.AddTask(comms, FillDerived<MeshData<Real>>, md);
  }
  return tc;
}

TaskStatus SetElectrons(MeshData<Real> *md) {
  auto node_interp_order =
      md->GetMeshPointer()->packages.Get("laser")->Param<int>("node_interp_order");
  if (node_interp_order == 2) {
    return SetElectronsImpl<2>(md);
  } else if (node_interp_order == 4) {
    return SetElectronsImpl<4>(md);
  } else {
    PARTHENON_FAIL("Specified laser/node_interp_order not supported.");
  }
}

TaskStatus UpdateMatEnergy(MeshData<Real> *md, const Real dt) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetMeshPointer();
  auto v = riot::MakePack<ccbulk::total_material_energy, ccbulk::electron_internal_energy,
                          ccbulk::laser_deposition>(md);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;

  using lt = RiotUtils::LoopType<>;
  auto idx_space =
      lt::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(), md, TE::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto ue =
            RiotLoop::make_var_view(idx_range, v, ccbulk::electron_internal_energy());
        auto Etot =
            RiotLoop::make_var_view(idx_range, v, ccbulk::total_material_energy());
        auto laser_de = RiotLoop::make_var_view(idx_range, v, ccbulk::laser_deposition());
        RiotLoop::inner(idx_range, [&](auto kji) {
          ue(kji) += dt * laser_de(kji);
          Etot(kji) += dt * laser_de(kji);
        });
      });

  return TaskStatus::complete;
}

TaskStatus InitializeLaserSweep(MeshData<Real> *md, MeshData<Real> *md_dudt,
                                const Real t0, const Real dt) {
  namespace PL = particles::laser;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using ne_t = ccbulk::electron_number_density;
  using nen_t = node_variables::electron_number_density;

  auto &laser = md->GetMeshPointer()->packages.Get("laser");
  const auto &li = laser->Param<LaserInfo>("laser_info");
  auto &lenergy = li.energy;
  parthenon::ParArray1D<Real> laser_energy("laser energy", lenergy.size());
  auto leng_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), laser_energy);
  bool adding_laser_energy = false;
  for (int i = 0; i < lenergy.size(); i++) {
    leng_host(i) = lenergy[i].GetEnergy(t0, t0 + dt);
    if (leng_host(i) > 0.0) adding_laser_energy = true;
  }
  if (adding_laser_energy) {
    Kokkos::deep_copy(laser_energy, leng_host);
  }

  auto laser_wavelength = laser->Param<parthenon::ParArray1D<Real>>("laser_wavelength");

  const int ndim = md->GetMeshPointer()->ndim;
  int dj = ndim > 1 ? 1 : 0;
  int dk = ndim > 2 ? 1 : 0;

  auto v = riot::MakePack<ne_t, nen_t>(md);

  for (int b = 0; b < md->NumBlocks(); b++) {
    auto &mbd = md->GetBlockData(b);
    auto &mbd_dudt = md_dudt->GetBlockData(b);
    auto *pmb = mbd->GetBlockPointer();

    const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    auto ldep_tot = mbd_dudt->Get(ccbulk::total_material_energy::name()).data;
    auto ldep_ue = mbd_dudt->Get(ccbulk::electron_internal_energy::name()).data;

    auto edens = mbd->Get(ccbulk::laser_energy_density::name()).data;
    auto ldep_src = mbd->Get(ccbulk::laser_deposition::name()).data;
    auto tau_max = mbd->Get(ccbulk::laser_tau_max::name()).data;
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), kb.s, kb.e, jb.s,
        jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          ldep_tot(k, j, i) = 0.0;
          ldep_ue(k, j, i) = 0.0;
          edens(k, j, i) = 0.0;
          ldep_src(k, j, i) = 0.0;
          tau_max(k, j, i) = 0.0;
        });

    if (!adding_laser_energy) continue;

    // skip if not a boundary block
    bool is_boundary = false;
    std::array<bool, 6> bound_flag;
    for (int i = 0; i < 6; i++) {
      if (pmb->boundary_flag[i] != parthenon::BoundaryFlag::block) {
        is_boundary = true;
        bound_flag[i] = true;
      } else {
        bound_flag[i] = false;
      }
    }
    if (!is_boundary) continue;

    auto &coords = pmb->coords;
    const std::array<Real, 3> block_xmin(
        {coords.Xf<X1DIR>(ib.s), coords.Xf<X2DIR>(jb.s), coords.Xf<X3DIR>(kb.s)});
    // assume Dxf is constant
    const std::array<Real, 3> dx(
        {coords.Dxf<X1DIR>(0), coords.Dxf<X2DIR>(0), coords.Dxf<X3DIR>(0)});
    const std::array<int, 3> Ni({ib.e - ib.s + 1, jb.e - jb.s + 1, kb.e - kb.s + 1});
    std::array<Real, 3> block_xmax;
    for (int i = 0; i < 3; i++)
      block_xmax[i] = block_xmin[i] + Ni[i] * dx[i];

    // figure out which points are on this block boundary
    std::array<std::vector<int>, 6> ind;
    int total_number = 0;
    for (int i = 0; i < 2 * ndim; i++) {
      if (!bound_flag[i]) continue;
      for (int j = 0; j < li.face_pts[i][sample::x].size(); j++) {
        bool on_block = false;
        if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
          bool inside_x = (li.face_pts[i][sample::x][j] >= block_xmin[0] &&
                           li.face_pts[i][sample::x][j] <= block_xmax[0]);
          bool inside_y = ndim < 2 || (li.face_pts[i][sample::y][j] >= block_xmin[1] &&
                                       li.face_pts[i][sample::y][j] <= block_xmax[1]);
          bool inside_z = ndim < 3 || (li.face_pts[i][sample::z][j] >= block_xmin[2] &&
                                       li.face_pts[i][sample::z][j] <= block_xmax[2]);
          if (i < 2) {
            // x-boundary, check y and z
            on_block = inside_y && inside_z;
          } else if (i < 4) {
            // y-boundary, check x and z
            on_block = inside_x && inside_z;
          } else {
            on_block = inside_x && inside_y;
          }
        } else if (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
          auto x = li.face_pts[i][sample::x][j];
          auto y = li.face_pts[i][sample::y][j];
          auto z = li.face_pts[i][sample::z][j];
          auto r = std::sqrt(x * x + y * y);
          bool inside_r = (r >= block_xmin[0] && r <= block_xmax[0]);
          bool inside_z = (li.face_pts[i][sample::z][j] >= block_xmin[1] &&
                           li.face_pts[i][sample::z][j] <= block_xmax[1]);
          if (i < 2) {
            on_block = inside_z;
          } else if (i < 4) {
            on_block = inside_r;
          } else {
            PARTHENON_FAIL("Cylindrical does not support 3D");
          }
        }
        if (on_block) {
          ind[i].push_back(j);
        }
      }
      total_number += ind[i].size();
    }

    // now make a device side array that lists the included laser points
    parthenon::ParArray2D<Real> laser_pts("laser points", total_number,
                                          static_cast<int>(sample::nvalues));
    auto lhost = Kokkos::create_mirror_view(Kokkos::HostSpace(), laser_pts);
    parthenon::ParArray1D<int> laser_id("laser_id", total_number);
    auto lid_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), laser_id);
    int idx = 0;
    for (int i = 0; i < 2 * ndim; i++) {
      if (!bound_flag[i]) continue;
      for (int j = 0; j < ind[i].size(); j++) {
        int k = ind[i][j];
        for (int n = 0; n < sample::nvalues; n++) {
          lhost(idx, n) = li.face_pts[i][n][k];
        }
        lid_host(idx) = li.id[i][k];
        idx++;
      }
    }
    Kokkos::deep_copy(laser_pts, lhost);
    Kokkos::deep_copy(laser_id, lid_host);

    // make space for the particles
    auto *swarm = md->GetSwarmData(b)->Get(PL::particles::name()).get();
    parthenon::ParArray1D<parthenon::NewParticlesContext> new_context("New context", 1);
    auto new_context_h = new_context.GetHostMirror();
    new_context_h(0) = swarm->AddEmptyParticles(total_number);
    new_context.DeepCopy(new_context_h);
    Kokkos::fence();

    auto &t = swarm->Get<Real>(PL::t::name()).Get();
    auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
    auto &vx = swarm->Get<Real>(PL::vx::name()).Get();
    auto &vy = swarm->Get<Real>(PL::vy::name()).Get();
    auto &vz = swarm->Get<Real>(PL::vz::name()).Get();
    auto &energy = swarm->Get<Real>(PL::energy::name()).Get();
    auto &lambda = swarm->Get<Real>(PL::wavelength::name()).Get();
    const int nghost = parthenon::Globals::nghost;

    // set the particle properties
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, total_number - 1,
        KOKKOS_LAMBDA(const int n) {
          const int pidx = new_context(0).GetNewParticleIndex(n);
          t(pidx) = t0;
          x(pidx) = laser_pts(n, sample::x);
          y(pidx) = laser_pts(n, sample::y);
          z(pidx) = laser_pts(n, sample::z);
          vx(pidx) = laser_pts(n, sample::nx);
          vy(pidx) = laser_pts(n, sample::ny);
          vz(pidx) = laser_pts(n, sample::nz);
          lambda(pidx) = laser_wavelength(laser_id(n));

          auto &bcoords = v.GetCoordinates(b);
          CellInfo ci(v, b, x(pidx), y(pidx), z(pidx), vx(pidx), vy(pidx), vz(pidx), dj,
                      dk, nghost);

          // interpolate ne to the initial location to determine initial speed
          const Real ne = ci.ne(x(pidx), y(pidx), z(pidx));

          // compute the critical density
          Real ncrit = ne_crit(lambda(pidx));

          // don't allow NaNs when ne > ne_crit.  this particle will just immediately get
          // absorbed when transported
          const Real speed = (ne < ncrit ? pc::c * std::sqrt(1.0 - ne / ncrit) : 1.0);
          vx(pidx) *= speed;
          vy(pidx) *= speed;
          vz(pidx) *= speed;

          energy(pidx) = laser_pts(n, sample::wgt) * laser_energy(laser_id(n));
        });

  } // loop over blocks in MeshData
  return TaskStatus::complete;
}

TaskStatus Update(MeshData<Real> *md, MeshData<Real> *md_dudt, const Real t0,
                  const Real dt) {
  namespace PL = particles::laser;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using ne_t = ccbulk::electron_number_density;
  using nen_t = node_variables::electron_number_density;
  // using nen_d_t = node_variables::grad_ne;
  using TE = parthenon::TopologicalElement;

  auto pm = md->GetMeshPointer();
  auto &mesh_xmin = pm->mesh_size.xmin_;
  auto &mesh_xmax = pm->mesh_size.xmax_;
  const int ndim = pm->ndim;
  int dj = ndim > 1 ? 1 : 0;
  int dk = ndim > 2 ? 1 : 0;

  auto resolved_pkgs = pm->resolved_packages.get();
  auto v =
      riot::MakePack<ne_t, nen_t, ccbulk::electron_temperature, ccbulk::ionization_zbar,
                     ccbulk::laser_energy_density, ccbulk::total_material_energy,
                     ccbulk::laser_deposition, ccbulk::laser_tau_max>(md);

  auto dudt =
      riot::MakePack<ccbulk::total_material_energy, ccbulk::electron_internal_energy>(
          md_dudt);

  static auto desc_ps =
      parthenon::MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y,
                                         swarm_position::z, PL::t, PL::vx, PL::vy, PL::vz,
                                         PL::energy, PL::wavelength>(
          PL::particles::name());
  auto ps = desc_ps.GetPack(md);

  const auto enable_laser_deposition =
      pm->packages.Get("laser")->Param<Real>("enable_laser_deposition");

  auto clog = pm->packages.Get("laser")->Param<Real>("coulomb_log");
  Real ei_coll_coeff = (4.0 / 3.0) * std::sqrt(2.0 * M_PI / pc::me) *
                       std::pow(pc::qe, 4) * clog * std::pow(pc::kb, -1.5) / pc::c;

  // For now, we'll make the fast light approximation so lasers propagate across the whole
  // mesh every cycle setting tstop to a huge number ensures this
  Real tstop = 1.e300; // t0 + dt;

  const int nghost = parthenon::Globals::nghost;

  int num_not_done = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_flatrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      ps.GetMaxFlatIndex(),
      KOKKOS_LAMBDA(const int idx, int &num_unfinished) {
        auto [b, n] = ps.GetBlockParticleIndices(idx);
        const auto swarm_d = ps.GetContext(b);
        if (!swarm_d.IsActive(n)) return;

        LaserParticle p(ps, b, n);

        Real ncrit = ne_crit(p.lambda);

        bool on_current_block = true;
        while (p.t < tstop && on_current_block) {
          CellInfo ci(v, b, p.x[0], p.x[1], p.x[2], p.v[0], p.v[1], p.v[2], dj, dk,
                      nghost);

          // just deposit energy if ne > ne_crit and move on
          if (ci.ne() >= ncrit) {
            Kokkos::atomic_add(
                &dudt(b, ccbulk::total_material_energy(), ci.k, ci.j, ci.i),
                enable_laser_deposition * p.energy / (ci.volume() * dt));
            p.energy = 0.0;
            swarm_d.MarkParticleForRemoval(n);
            break;
          }

          ParticlePusher pusher(p, ci);

          // push to cell boundary
          auto [path_length, speed] = pusher.step_to_boundary(tstop);

          // now deposit energy in the cell
          // TODO(JCD): Cache this on the mesh.  Use atomics.
          Real kappa =
              ci.ne() * (ci.ne() / ncrit) *
              v(b, ccbulk::ionization_zbar(), ci.k, ci.j, ci.i) * ei_coll_coeff *
              std::pow(
                  std::max(v(b, ccbulk::electron_temperature(), ci.k, ci.j, ci.i), 1.0),
                  -1.5) /
              std::sqrt(1.0 - ci.ne() / ncrit);
          Real tau = kappa * path_length;
          Kokkos::atomic_max(&v(b, ccbulk::laser_tau_max(), ci.k, ci.j, ci.i), tau);
          Real delta_energy = enable_laser_deposition * p.energy *
                              (tau < 1.e-10 ? tau : (1.0 - std::exp(-tau)));
          p.energy -= delta_energy;
          Kokkos::atomic_add(&dudt(b, ccbulk::total_material_energy(), ci.k, ci.j, ci.i),
                             delta_energy / (ci.volume() * dt));
          Kokkos::atomic_add(
              &dudt(b, ccbulk::electron_internal_energy(), ci.k, ci.j, ci.i),
              delta_energy / (ci.volume() * dt));
          Kokkos::atomic_add(&v(b, ccbulk::laser_deposition(), ci.k, ci.j, ci.i),
                             delta_energy / (ci.volume() * dt));

          Kokkos::atomic_add(&v(b, ccbulk::laser_energy_density(), ci.k, ci.j, ci.i),
                             (p.energy + 0.5 * delta_energy) / (speed * dt) *
                                 path_length / ci.volume());

          swarm_d.GetNeighborBlockIndex(n, p.x[0], p.x[1], p.x[2], on_current_block);
        }

        bool outside_domain = false;
        for (int d = 0; d < ndim; d++) {
          if (p.x[d] < mesh_xmin[d] || p.x[d] > mesh_xmax[d]) {
            outside_domain = true;
            break;
          }
        }
        if (outside_domain) {
          swarm_d.MarkParticleForRemoval(n);
        } else if (p.energy > 0) {
          num_unfinished += (p.t < tstop);
        }

        // copy back out updated state
        p.apply_update(ps, b, n);
      },
      Kokkos::Sum<int>(num_not_done));

  for (int b = 0; b < md->NumBlocks(); b++) {
    md->GetSwarmData(b)->Get(PL::particles::name())->RemoveMarkedParticles();
  }

  if (num_not_done) return TaskStatus::iterate;
  return TaskStatus::complete;
}

} // namespace Laser

//========================================================================================
// (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
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

#include "regions.hpp"

#include <array>

namespace region_pgen {

std::tuple<Real, Real, Real> to_cartesian(const Real x0, const Real x1, const Real x2) {
  if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
    // TODO(JCD): do this right if we ever want to support spherical beyond 1d
    return {x0, 0.0, 0.0};
  }
  if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
    // TODO(JCD): same as above for 3d cylindrical
    return {x0, 0.0, x1};
  }
  return {x0, x1, x2};
}

//----------------------------------------------------------------------------------------
//! \fn  Region::Region
//! \brief
Region::Region(ParameterInput *pin, const std::string &block, const int id,
               python_region_t &default_py)
    : py(pin, block), region_id(id), region_block(block) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  /*****************************************************************
   * Set the mask func which determines the geometry of the region *
   *****************************************************************/
  auto mask_type = pin->GetString(block, "mask_type");
  if (mask_type == "python") {
    mask = py.make_mask();
  } else {
    PARTHENON_REQUIRE(region_primitives::region_mask_map.count(mask_type) > 0,
                      mask_type + " is not a valid mask_type");
    mask = region_primitives::region_mask_map.at(mask_type)(pin, block);
  }

  /******
   * 3T *
   ******/
  auto do_3t = pin->GetBoolean("physics", "ionization");

  /*************************
   * Material related init *
   *************************/
  matid = pin->GetVector<int>(block, "matid", "materials in this region");
  // build map of variable name to initialization function
  set_mat_init_funcs<ccmat::volume_fraction, cm::rho, cm::pressure, cm::temperature,
                     cm::sie, cm::phase_fraction>(mat_state, pin, block, py, default_py,
                                                  matid);
  bool has_electron_temp = false;
  if (do_3t) {
    set_bulk_init_funcs<ccbulk::electron_temperature>(bulk_state, pin, block, py,
                                                      default_py);
    if (bulk_state.count(var_name<ccbulk::electron_temperature>()))
      has_electron_temp = true;
  }

  // now build the map of material id to list of variables that will be set in init
  // along the way construct the unique string that maps to the correct InitType for each
  // mat
  std::unordered_map<int, std::string> init_string;
  // we always need volume fraction since we don't assume pure regions
  for (auto mid : matid)
    mat_init_fields[mid].push_back(var_name<ccmat::volume_fraction>({mid}));
  set_mat_init_vars<cm::rho, cm::pressure, cm::temperature, cm::sie>(
      mat_init_fields, init_string, mat_state, matid);
  if (do_3t) {
    // if we aren't setting electron temperature explicitly, make Te = Ti, i.e.
    // equilibrium
    for (auto mid : matid) {
      if (has_electron_temp)
        init_string[mid] += "c_c_bulk_electron_temperature";
      else
        init_string[mid] += "Equil";
    }
  }

  set_bulk_init_funcs<ccbulk::velocity>(bulk_state, pin, block, py, default_py);

  // and set the init_type using the map
  for (auto mid : matid) {
    try {
      init_type[mid] = init_type_map.at(init_string[mid]);
    } catch (...) {
      PARTHENON_FAIL("Invalid setup for material " + std::to_string(mid) + ": " +
                     init_string[mid] + " in region " + std::to_string(id));
    }
  }
  // volume fraction and phase fraction *must* exist, so provide defaults (if reasonable)
  for (auto m : matid) {
    auto name = var_name<ccmat::volume_fraction>({m});
    if (mat_state.count(name) == 0) {
      PARTHENON_REQUIRE(
          matid.size() == 1,
          "Must provide volume fraction init for region with more than one mat.");
      mat_state[name] = make_const_func(std::vector<Real>{1.0});
    }

    name = var_name<cm::phase_fraction>({m});
    if (mat_state.count(name) == 0) {
      int nphase = pin->GetInteger("material" + std::to_string(m), "nphase");
      std::vector<Real> pfrac(nphase, 0.0);
      // default to the first phase being everything
      pfrac[0] = 1.0;
      mat_state[name] = make_const_func(pfrac);
    }
  }

  /************
   * Strength *
   ************/
  const bool do_strength = pin->GetBoolean("physics", "strength");
  if (do_strength) {
    std::vector<int> strength_mats_in_region;
    for (auto mid : matid) {
      std::string mat_block = "material" + std::to_string(mid);
      bool strong = pin->GetBoolean(mat_block, "strong");
      if (strong) strength_mats_in_region.push_back(mid);
    }
    StrengthMapEntries::for_each_key_value<set_specific_mat_init_funcs>(
        specific_mat_state, strength_mats_in_region, pin, block, py, default_py);
  }

  /*******
   * Mix *
   *******/
  auto do_mix = pin->GetBoolean("physics", "mix");
  if (do_mix) {
    BHRMapEntries::for_each_key_value<set_specific_bulk_init_funcs>(
        specific_bulk_state, pin, block, py, default_py);
  }

  /************
   * Isotopes *
   ************/
  // build the list of isotopes for each material in the region
  std::vector<int> iso_mats_in_region;
  for (auto mid : matid) {
    int icnt = 0;
    const std::string block = "material" + std::to_string(mid);
    std::vector<Real> default_iso_fractions;
    for (;;) {
      std::string iso = "isotope" + std::to_string(icnt);
      if (!pin->DoesParameterExist(block, iso)) break;
      iso += "_mfrac";
      if (pin->DoesParameterExist(block, iso))
        default_iso_fractions.push_back(pin->GetReal(block, iso));
      else
        default_iso_fractions.push_back(0.0);
      icnt++;
    }
    if (icnt) {
      iso_mats_in_region.push_back(mid);
      auto cname = var_name<ccmat::iso>({mid});
      specific_mat_state[mid][cname] = make_const_func(default_iso_fractions);
    }
  }
  IsoMapEntries::for_each_key_value<set_specific_mat_init_funcs>(
      specific_mat_state, iso_mats_in_region, pin, block, py, default_py);

  /*********************
   * Bulk related init *
   *********************/
  set_bulk_init_funcs<ccbulk::velocity>(bulk_state, pin, block, py, default_py);

  /***********
   * Scalars *
   ***********/
  // scalars are a bit wierd because they're not typed variables, so we have
  // to handle whatever names we're given in the inputs.  also, there's no
  // init function we're looking for because we just want them to tag
  // a region/mat with a one, otherwise they're zero
  const bool do_scalars = pin->GetBoolean("physics", "scalars");
  if (do_scalars && (pin->DoesParameterExist(block, "passive_scalars"))) {
    auto scalar_names = pin->GetVector<std::string>(block, "passive_scalars");
    for (auto &sname : scalar_names) {
      auto sblock = find_scalar_block(pin, sname);
      if (pin->DoesParameterExist(sblock, "matid")) {
        auto mid = pin->GetInteger(sblock, "matid");
        mat_tied_scalars[mid] = parthenon::MakeVarLabel(sname, mid);
      } else {
        bulk_tied_scalars.push_back(sname);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn  BlockInitData::BlockInitData
//! \brief
BlockInitData::BlockInitData(ParameterInput *pin)
    : regions(pin), nghost(pin->GetInteger("parthenon/mesh", "nghost")),
      nx1((pin->DoesParameterExist("parthenon/meshblock", "nx1")
               ? pin->GetInteger("parthenon/meshblock", "nx1")
               : pin->GetInteger("parthenon/mesh", "nx1"))),
      nx2((pin->DoesParameterExist("parthenon/meshblock", "nx2")
               ? pin->GetInteger("parthenon/meshblock", "nx2")
               : pin->GetInteger("parthenon/mesh", "nx2"))),
      nx3((pin->DoesParameterExist("parthenon/meshblock", "nx3")
               ? pin->GetInteger("parthenon/meshblock", "nx3")
               : pin->GetInteger("parthenon/mesh", "nx3"))),
      region_id("region_id",
                nx3 + 2 * nghost * (pin->GetInteger("parthenon/mesh", "nx3") > 1),
                nx2 + 2 * nghost * (pin->GetInteger("parthenon/mesh", "nx2") > 1),
                nx1 + 2 * nghost),
      vol_frac_sum("vol_frac_sum",
                   nx3 + 2 * nghost * (pin->GetInteger("parthenon/mesh", "nx3") > 1),
                   nx2 + 2 * nghost * (pin->GetInteger("parthenon/mesh", "nx2") > 1),
                   nx1 + 2 * nghost),
      ndim(pin->GetInteger("parthenon/mesh", "nx3") > 1   ? 3
           : pin->GetInteger("parthenon/mesh", "nx2") > 1 ? 2
                                                          : 1) {

  ib[state].s = nghost;
  ib[state].e = nghost + nx1 - 1;
  jb[state].s = nghost * (ndim > 1);
  jb[state].e = (nghost + nx2 - 1) * (ndim > 1);
  kb[state].s = nghost * (ndim > 2);
  kb[state].e = (nghost + nx3 - 1) * (ndim > 2);
  ib[mask].s = ib[state].s - 1;
  ib[mask].e = ib[state].e + 1;
  jb[mask].s = jb[state].s - (ndim > 1);
  jb[mask].e = jb[state].e + (ndim > 1);
  kb[mask].s = kb[state].s - (ndim > 2);
  kb[mask].e = kb[state].e + (ndim > 2);
  num_mask = (ib[mask].e - ib[mask].s + 1) * (jb[mask].e - jb[mask].s + 1) *
             (kb[mask].e - kb[mask].s + 1);
  num_state = (ib[state].e - ib[state].s + 1) * (jb[state].e - jb[state].s + 1) *
              (kb[state].e - kb[state].s + 1);
  do_3t = pin->GetBoolean("physics", "ionization");
  if (do_3t) fully_ionized = pin->GetBoolean("ionization", "fully_ionized");
}

//----------------------------------------------------------------------------------------
//! \fn  void BlockInitData::InitBlock
//! \brief
void BlockInitData::InitBlock(MeshBlock *pmb, ParameterInput *pin) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  auto rc = pmb->meshblock_data.Get();
  auto pmesh = pmb->pmy_mesh;
  ndim = pmesh->ndim;

  coords = &pmb->coords;

  auto mats = pmb->packages.Get("materials");
  const int nummat_max = pmb->packages.Get("materials")->Param<int>("nummat");
  auto eos_host = mats->Param<std::vector<RiotEOS::EOS>>("h.h.EOS");
  auto eos_from_matid = mats->Param<std::vector<int>>("h.EOS_from_matid");
  std::vector<RiotEOS::EOS> electron_eos_host;
  if (do_3t)
    electron_eos_host = mats->Param<std::vector<RiotEOS::EOS>>("h.h.electron_EOS");
  auto matids = mats->Param<std::vector<int>>("matids");
  std::map<std::string, std::set<int>> scalar_vars;
  if (pin->GetBoolean("physics", "scalars")) {
    auto scalars = pmb->packages.Get("scalars");
    scalar_vars = scalars->Param<std::map<std::string, std::set<int>>>("scalar_tie_map");
  }
  if (!pin->GetBoolean("physics", "ionization")) {
    independent_init_vars.erase(ccmat::ionization_zbar::name());
    independent_init_vars.erase(ccmat::electron_internal_energy::name());
  }

  block_loop<mask>([&](const int k, const int j, const int i) {
    region_id(k, j, i) = -1;
    vol_frac_sum(k, j, i) = 0.0;
  });

  // build maps to Variables and host mirrors of all variables that need initializing
  using host_mirror_t = decltype(rc->GetVariableVector()[0]->data.GetHostMirror());
  std::map<std::string, std::shared_ptr<Variable<Real>>> var_map;
  std::map<std::string, host_mirror_t> var_mirror_map;
  for (auto &var : rc->GetVariableVector()) {

    if (independent_init_vars.count(var->base_name())) {
      var_map[var->label()] = rc->GetVarPtr(var->label());
      // only get host mirrors of non-sparse fields since we'll deallocate all of them
      if (!var->IsSparse()) {
        var_mirror_map[var->label()] = var_map[var->label()]->data.GetHostMirror();
      }
    } else if (var->label() == ccbulk::velocity::name()) {
      var_map[var->label()] = rc->GetVarPtr(var->label());
    }
  }

  // deal with scalars whose name is only known at runtime
  for (auto &sv : scalar_vars) {
    if (sv.second.size() > 0) { // sparse-tied
      for (auto mid : sv.second) {
        const std::string sname = parthenon::MakeVarLabel(sv.first, mid);
        var_map[sname] = rc->GetVarPtr(sname);
      }
    } else { // bulk-tied
      var_map[sv.first] = rc->GetVarPtr(sv.first);
      var_mirror_map[sv.first] = var_map[sv.first]->data.GetHostMirror();
    }
  }

  // now zero out host mirrors of all non-sparse fields
  for (auto &var : var_mirror_map) {
    auto pv = var_map[var.first];
    if (!pv->IsSparse()) {
      auto data = var.second;
      for (int n = 0; n < pv->GetDim(6); n++) {
        for (int m = 0; m < pv->GetDim(5); m++) {
          for (int l = 0; l < pv->GetDim(4); l++) {
            block_loop<state>([&](const int k, const int j, const int i) {
              data(n, m, l, k, j, i) = 0.0;
            });
          }
        }
      }
    }
  }

  // build a list for each matid of sparse fields we need host copies of after allocating
  // also make sure all sparse fields are unallocated
  std::map<int, std::vector<std::string>> mat_tied_mirror_vars;
  for (auto mid : matids) {
    const std::string rho_mat_name = var_name<ccmat::rho>({mid});
    auto all_controlled_vars =
        pmesh->resolved_packages->GetControlledVariables(rho_mat_name);
    for (auto &cvar : all_controlled_vars) {
      if (var_map.count(cvar)) {
        mat_tied_mirror_vars[mid].push_back(cvar);
      }
    }
    pmb->DeallocateSparse(rho_mat_name);
  }

  // mark cells that lie cleanly within a single region
  auto region_cnts = set_clean_mask();
  // a bit of a hack
  auto &elec_eos = (do_3t ? electron_eos_host : eos_host);
  // initialize the state in all cells marked as clean
  set_clean_state(pmb, eos_host, elec_eos, eos_from_matid, region_cnts, var_map,
                  var_mirror_map, mat_tied_mirror_vars);
  // initialize the state in all cells not marked as clean
  auto dirty_map = set_dirty_state(pmb, eos_host, elec_eos, eos_from_matid, var_map,
                                   var_mirror_map, mat_tied_mirror_vars, matids);

  // face-centered magnetic field (also adds the magnetic energy to the total energy)
  if (pin->GetOrAddBoolean("physics", "mhd", false)) {
    init_magnetic_field(pmb, rc.get(),
                        var_mirror_map.at(ccbulk::total_material_energy::name()));
  }

  var_mirror_map.erase("does_not_exist");
  for (auto v : var_mirror_map) {
    if (var_map[v.first]->IsAllocated()) {
      var_map[v.first]->data.DeepCopy(v.second);
    }
  }
  var_mirror_map.clear();

  Multiphysics::FillInteriorBlockDerived(rc);

  std::map<std::string, host_mirror_t> readonly_mirror_map;

  for (auto v : var_mirror_map) {
    // make sure the variable is in var_map.  it should be if it's a field that needs
    // initialization
    PARTHENON_REQUIRE(var_map.count(v.first) > 0, v.first + " is not in var_map");
    var_map[v.first]->data.DeepCopy(v.second);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void BlockInitData::init_magnetic_field
//! \brief Set the face-centered magnetic field from the optional methods of the default
//!         python <regions> class:
//!           f_b_vector_potential(self, pos, out): A sampled at edge centers; B is its
//!             discrete curl, so div B = 0 to round-off (preferred).
//!           f_b_bfield(self, pos, out): B sampled at face centers; only the normal
//!             component of each face is used. Init throws unless the discrete div B
//!             vanishes to round-off.
//!         out has shape (n, 3). Without either method B = 0. The cell-centered magnetic
//!         energy is added to the total energy host mirror E over the interior.
template <typename host_mirror_t>
void BlockInitData::init_magnetic_field(MeshBlock *pmb, MeshBlockData<Real> *rc,
                                        host_mirror_t &E) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ffbulk = face_variables::face_averaged::bulk;
  using TE = parthenon::TopologicalElement;
  auto &pyreg = regions.default_python_region;
  const bool have_A = pyreg.HasMethod("f_b_vector_potential");
  const bool have_B = pyreg.HasMethod("f_b_bfield");
  PARTHENON_REQUIRE_THROWS(!(have_A && have_B),
                           "Provide only one of f_b_vector_potential and f_b_bfield in "
                           "the <regions> python class.");

  auto &cb = pmb->cellbounds;
  // Counterpart of block_loop for element te (face or edge) over the entire domain
  auto element_loop = [&](const TE te, auto &&f) {
    const auto ibe = cb.GetBoundsI(IndexDomain::entire, te);
    const auto jbe = cb.GetBoundsJ(IndexDomain::entire, te);
    const auto kbe = cb.GetBoundsK(IndexDomain::entire, te);
    for (int k = kbe.s; k <= kbe.e; k++) {
      for (int j = jbe.s; j <= jbe.e; j++) {
        for (int i = ibe.s; i <= ibe.e; i++) {
          f(k, j, i);
        }
      }
    }
  };
  constexpr TE faces[3] = {TE::F1, TE::F2, TE::F3};
  constexpr TE edges[3] = {TE::E1, TE::E2, TE::E3};
  auto measure = [&](const TE te, const int k, const int j, const int i) {
    return coords->Volume(parthenon::CellLevel::same, te, k, j, i);
  };

  auto bvar = rc->GetVarPtr(ffbulk::bfield::name());
  auto bh = bvar->data.GetHostMirror();
  for (int d = 0; d < 3; d++) {
    element_loop(faces[d], [&](const int k, const int j, const int i) {
      bh(d, 0, 0, k, j, i) = 0.0;
    });
  }

  // Sample the python function name at the centers of element te (face or edge) over
  // the entire domain and keep coordinate component comp; off[d] marks that te sits on
  // the lower cell face in direction d. As for the cell state, positions and vector
  // components seen by python are Cartesian (the x-z plane for cylindrical coordinates).
  auto sample = [&](const std::string &name, const TE te, const std::array<bool, 3> off,
                    const int comp, auto &&store) {
    // python component holding coordinate component comp (cf. coordinate_velocities)
    constexpr bool swap_yz = parthenon::IsCoord<parthenon::UniformCylindrical>() ||
                             parthenon::IsCoord<parthenon::UniformSpherical>();
    const int py_comp = (swap_yz && comp > 0) ? 3 - comp : comp;
    int n = 0;
    element_loop(te, [&](const int k, const int j, const int i) { n++; });
    sample_positions_t xs(n);
    n = 0;
    element_loop(te, [&](const int k, const int j, const int i) {
      const auto [xx, yy, zz] = to_cartesian(
          (off[0] && ndim > 0) ? coords->Xf<X1DIR>(i) : coords->Xc<X1DIR>(i),
          (off[1] && ndim > 1) ? coords->Xf<X2DIR>(j) : coords->Xc<X2DIR>(j),
          (off[2] && ndim > 2) ? coords->Xf<X3DIR>(k) : coords->Xc<X3DIR>(k));
      xs(n, 0) = xx;
      xs(n, 1) = yy;
      xs(n, 2) = zz;
      n++;
    });
    field_data_t out(n, 3);
    pyreg.make_state(name)(xs, out);
    n = 0;
    element_loop(te, [&](const int k, const int j, const int i) {
      store(k, j, i, out(n, py_comp));
      n++;
    });
  };
  if (have_B) {
    for (int d = 0; d < 3; d++) {
      std::array<bool, 3> off{d == 0, d == 1, d == 2};
      sample("f_b_bfield", faces[d], off, d,
             [&](const int k, const int j, const int i, const Real b) {
               bh(d, 0, 0, k, j, i) = b;
             });
    }
    // CT preserves div B but cannot remove it, so reject a sampled field that is not
    // discretely divergence free: max |div B| dx_min must be round-off relative to |B|.
    Real max_divb = 0.0;
    Real max_b = 0.0;
    for (int k = cb.ks(IndexDomain::interior); k <= cb.ke(IndexDomain::interior); k++) {
      for (int j = cb.js(IndexDomain::interior); j <= cb.je(IndexDomain::interior); j++) {
        for (int i = cb.is(IndexDomain::interior); i <= cb.ie(IndexDomain::interior);
             i++) {
          const Real dx[3] = {coords->CellWidth<X1DIR>(k, j, i),
                              coords->CellWidth<X2DIR>(k, j, i),
                              coords->CellWidth<X3DIR>(k, j, i)};
          Real flux = 0.0;
          Real dxmin = dx[0];
          for (int d = 0; d < ndim; d++) {
            const int kd = k + (d == 2), jd = j + (d == 1), id = i + (d == 0);
            flux += measure(faces[d], kd, jd, id) * bh(d, 0, 0, kd, jd, id) -
                    measure(faces[d], k, j, i) * bh(d, 0, 0, k, j, i);
            dxmin = std::min(dxmin, dx[d]);
            max_b = std::max(max_b, std::abs(bh(d, 0, 0, k, j, i)));
          }
          max_divb =
              std::max(max_divb, std::abs(flux) / measure(TE::CC, k, j, i) * dxmin);
        }
      }
    }
    PARTHENON_REQUIRE_THROWS(
        max_divb <= 1.0e-10 * max_b,
        "f_b_bfield gives max |div B| dx = " + std::to_string(max_divb) +
            "; use a divergence-free field or f_b_vector_potential");
  } else if (have_A) {
    // edge-centered vector potential, A_d on edges E_d over the entire domain
    std::array<parthenon::HostArray3D<Real>, 3> A;
    for (int d = 0; d < 3; d++) {
      const TE te = edges[d];
      A[d] =
          parthenon::HostArray3D<Real>("A", cb.GetBoundsK(IndexDomain::entire, te).e + 1,
                                       cb.GetBoundsJ(IndexDomain::entire, te).e + 1,
                                       cb.GetBoundsI(IndexDomain::entire, te).e + 1);
      std::array<bool, 3> off{d != 0, d != 1, d != 2};
      sample("f_b_vector_potential", te, off, d,
             [&](const int k, const int j, const int i, const Real a) {
               A[d](k, j, i) = a;
             });
    }
    // B_d = (curl A)_d: by Stokes, the circulation of A around the F_d face divided by
    // the face area, [(L A_q)|_{+p} - (L A_q)|_{0} - (L A_p)|_{+q} + (L A_p)|_{0}] / A_d
    // with p = (d+1)%3, q = (d+2)%3 and L the edge lengths (matches
    // Hydro::UpdateFaceFields)
    for (int d = 0; d < 3; d++) {
      const int p = (d + 1) % 3;
      const int q = (d + 2) % 3;
      const bool p_active = (p < ndim);
      const bool q_active = (q < ndim);
      element_loop(faces[d], [&](const int k, const int j, const int i) {
        Real circ = 0.0;
        if (p_active) {
          const int kp = k + (p == 2), jp = j + (p == 1), ip = i + (p == 0);
          if (kp < A[q].extent_int(0) && jp < A[q].extent_int(1) &&
              ip < A[q].extent_int(2)) {
            circ += measure(edges[q], kp, jp, ip) * A[q](kp, jp, ip) -
                    measure(edges[q], k, j, i) * A[q](k, j, i);
          }
        }
        if (q_active) {
          const int kq = k + (q == 2), jq = j + (q == 1), iq = i + (q == 0);
          if (kq < A[p].extent_int(0) && jq < A[p].extent_int(1) &&
              iq < A[p].extent_int(2)) {
            circ -= measure(edges[p], kq, jq, iq) * A[p](kq, jq, iq) -
                    measure(edges[p], k, j, i) * A[p](k, j, i);
          }
        }
        // radial faces on the axis have zero area (and circulation)
        const Real area = measure(faces[d], k, j, i);
        bh(d, 0, 0, k, j, i) = (area > 0.0) ? circ / area : 0.0;
      });
    }
  }
  bvar->data.DeepCopy(bh);

  // add the magnetic energy of the cell-centered field to the total energy; the cell
  // field is interpolated as in Hydro::FaceToCellB
  const bool curvilinear = !parthenon::IsCoord<parthenon::UniformCartesian>();
  block_loop<state>([&](const int k, const int j, const int i) {
    Real b2 = 0.0;
    for (int d = 0; d < 3; d++) {
      const Real blo = bh(d, 0, 0, k, j, i);
      const Real bhi =
          (d < ndim) ? bh(d, 0, 0, k + (d == 2), j + (d == 1), i + (d == 0)) : blo;
      Real whi = 0.5;
      if (d == 0 && curvilinear) {
        const Real xlo = coords->Xf<X1DIR>(i);
        whi = (coords->Xc<X1DIR>(i) - xlo) / (coords->Xf<X1DIR>(i + 1) - xlo);
      }
      b2 += SQR((1.0 - whi) * blo + whi * bhi);
    }
    E(k, j, i) += 0.5 * b2;
  });
}

//----------------------------------------------------------------------------------------
//! \fn  std::vector<int> BlockInitData::set_mask
//! \brief
std::vector<int> BlockInitData::set_mask(sample_positions_t &x) {
  sample_positions_t xs(x.size());
  std::vector<int> reg_id(x.size(), -1);
  for (int r = regions.size() - 1; r >= 0; r--) {
    int n = 0;
    xs.resize(x.size());
    for (int i = 0; i < x.size(); i++) {
      if (reg_id[i] == -1) {
        auto [xx, yy, zz] = to_cartesian(x(i, 0), x(i, 1), x(i, 2));
        xs(n, 0) = xx;
        xs(n, 1) = yy;
        xs(n, 2) = zz;
        n++;
      }
    }
    xs.resize(n);
    auto m = regions[r].mask(xs);
    n = 0;
    for (int i = 0; i < x.size(); i++) {
      if (reg_id[i] == -1) {
        if (m[n]) {
          reg_id[i] = r;
        }
        n++;
      }
    }
  }
  return reg_id;
}

//----------------------------------------------------------------------------------------
//! \fn  bool BlockInitData::near_boundary
//! \brief
bool BlockInitData::near_boundary(const int k0, const int j0, const int i0,
                                  const int reg_id) const {
  const int kl = k0 - (ndim > 2);
  const int kh = k0 + (ndim > 2);
  const int jl = j0 - (ndim > 1);
  const int jh = j0 + (ndim > 1);
  int sum = 0;
  int sum_all = 0;
  for (int k = kl; k <= kh; k++) {
    for (int j = jl; j <= jh; j++) {
      for (int i = i0 - 1; i <= i0 + 1; i++) {
        if (region_id(k, j, i) == reg_id) sum++;
        sum_all++;
      }
    }
  }
  return (sum > 0 && sum != sum_all);
}

//----------------------------------------------------------------------------------------
//! \fn  std::vector<int> BlockInitData::set_clean_mask
//! \brief set region_id according to highest mask_func = true at cell centers
std::vector<int> BlockInitData::set_clean_mask() {
  sample_positions_t xs(num_mask);
  std::vector<bool> dirty(num_mask, false);
  std::vector<int> cnt(regions.size(), 0);
  for (int r = regions.size() - 1; r >= 0; r--) {
    int n = 0;
    xs.resize(num_mask);
    block_loop<mask>([&](const int k, const int j, const int i) {
      if (region_id(k, j, i) == -1) {
        auto [xx, yy, zz] =
            to_cartesian(coords->Xc<X1DIR>(k, j, i), coords->Xc<X2DIR>(k, j, i),
                         coords->Xc<X3DIR>(k, j, i));
        xs(n, 0) = xx;
        xs(n, 1) = yy;
        xs(n, 2) = zz;
        n++;
      }
    });
    xs.resize(n);
    auto block_mask = regions[r].mask(xs);
    n = 0;
    block_loop<mask>([&](const int k, const int j, const int i) {
      if (region_id(k, j, i) == -1) {
        if (block_mask[n]) region_id(k, j, i) = r;
        n++;
      }
    });
  }
  if (regions.nlev_max > 0) {
    // now mark dirty flag
    int n = 0;
    block_loop<state>([&](const int k, const int j, const int i) {
      dirty[n] = near_boundary(k, j, i, region_id(k, j, i));
      n++;
    });
  }
  // and reset region_id accordingly
  int n = 0;
  block_loop<state>([&](const int k, const int j, const int i) {
    if (dirty[n])
      region_id(k, j, i) = -1;
    else {
      cnt[region_id(k, j, i)]++;
    }
    n++;
  });
  return cnt;
}

//----------------------------------------------------------------------------------------
//! \fn  auto get_subcell_volume
//! \brief
auto get_subcell_volume(std::array<Real, 3> &xlo, std::array<Real, 3> &xhi, Real *dx,
                        const int ndim) {
  if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
    return 4.0 / 3.0 * M_PI * (std::pow(xhi[0], 3) - std::pow(xlo[0], 3));
  }
  if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
    return M_PI * (std::pow(xhi[0], 2) - std::pow(xlo[0], 2)) * (ndim > 1 ? dx[1] : 1);
  }
  return dx[0] * (ndim > 1 ? dx[1] : 1) * (ndim > 2 ? dx[2] : 1);
}

// non-class utilities

//----------------------------------------------------------------------------------------
//! \fn  field_func_t set_init_func
//! \brief
field_func_t set_init_func(const std::string &name, ParameterInput *pin,
                           const std::string &block, python_region_t &pyreg,
                           python_region_t &default_pyreg) {
  auto &py = pyreg.py_obj;
  auto &default_py = default_pyreg.py_obj;
  bool has_const = pin->DoesParameterExist(block, name);
  bool default_has_const = pin->DoesParameterExist("regions", name);
#ifdef RIOT_ENABLE_PYTHON
  bool has_python = py.exists(name.c_str());
  bool default_has_python = default_py.exists(name.c_str());
#else
  bool has_python = false;
  bool default_has_python = false;
#endif

  bool prefer_python = get_or_use_default(pin, block, "prefer_python_init", false);

  bool any_const = has_const || default_has_const;
  bool any_python = has_python || default_has_python;

  // if (we prefer python or don't have any const state defined) and have a python
  // funcition
  if ((prefer_python || !any_const) && any_python) {
#ifdef RIOT_ENABLE_PYTHON
    // return in priority order
    if (has_python) {
      if (pcall::get_method_arity(py, name.c_str()).min_positional == 2)
        return pyreg.make_state(name);
      else
        PARTHENON_FAIL("Unexpected number of arguments required from python function " +
                       name);
    } else if (default_has_python) {
      if (pcall::get_method_arity(default_py, name.c_str()).min_positional == 2)
        return default_pyreg.make_state(name);
      else
        PARTHENON_FAIL("Unexpected number of arguments required from python function " +
                       name);
    }
#else
    PARTHENON_FAIL("Riot was not built with python support.");
#endif
  }

  // if not python, maybe const
  if (has_const) {
    auto vals = pin->GetVector<Real>(block, name);
    return make_const_func(vals);
  } else if (default_has_const) {
    auto vals = pin->GetVector<Real>("regions", name);
    return make_const_func(vals);
  }

  return field_func_t();
}

} // namespace region_pgen

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
    PARTHENON_REQUIRE(region_mask_map.count(mask_type) > 0,
                      mask_type + " is not a valid mask_type");
    mask = region_mask_map.at(mask_type)(pin, block);
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

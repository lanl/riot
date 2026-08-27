//========================================================================================
// (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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
#ifndef RIOT_PGEN_REGIONS_HPP_
#define RIOT_PGEN_REGIONS_HPP_
// This file was made in part with generative AI.

#include <cctype>
#include <map>
#include <set>
#include <string>
#include <vector>

// Parthenon includes
#include <kokkos_abstraction.hpp>
#include <pack/sparse_pack/sparse_pack.hpp>
#include <utils/constants.hpp>
#include <utils/error_checking.hpp>

#include "ionization/ionization.hpp"
#include "microphysics/eos_riot.hpp"
#include "multiphysics/fill_shared_derived.hpp"
#include "region_primitives.hpp"
#include "region_python.hpp"
#include "riot_utils/type_maps.hpp"
#include "variables.hpp"

namespace region_pgen {

#define FOREACH_REGION                                                                   \
  REG(background)                                                                        \
  REG(inside_sphere)                                                                     \
  REG(inside_spherical_shell)                                                            \
  REG(inside_cylinder)                                                                   \
  REG(inside_cylindrical_shell)                                                          \
  REG(inside_ellipsoid)                                                                  \
  REG(inside_ellipsoidal_shell)                                                          \
  REG(inside_rectangle)                                                                  \
  REG(cad)

// build a map of region name to mask generator
#define REG(name) {#name, name},
static std::map<std::string, mask_generator_t> region_mask_map({FOREACH_REGION});
#undef REG

enum class InitType {
  PT,
  rhoT,
  rhoe,
  rhoP,
  rhoTiEquil,
  rhoTiTe,
  rhoPTe,
  rhoPEquil,
  PTiTe,
  PTiEquil
};
static std::map<std::string, InitType> init_type_map{
    {"c_m_pressurec_m_temperature", InitType::PT},
    {"c_m_rhoc_m_temperature", InitType::rhoT},
    {"c_m_rhoc_m_sie", InitType::rhoe},
    {"c_m_rhoc_m_pressure", InitType::rhoP},
    {"c_m_rhoc_m_temperatureEquil", InitType::rhoTiEquil},
    {"c_m_rhoc_m_temperaturec_c_bulk_electron_temperature", InitType::rhoTiTe},
    {"c_m_rhoc_m_pressurec_c_bulk_electron_temperature", InitType::rhoPTe},
    {"c_m_rhoc_m_pressureEquil", InitType::rhoPEquil},
    {"c_m_pressurec_m_temperaturec_c_bulk_electron_temperature", InitType::PTiTe},
    {"c_m_pressurec_m_temperatureEquil", InitType::PTiEquil}};

inline std::string make_safe(std::string input) {
  std::replace(input.begin(), input.end(), '.', '_');
  return input;
}

template <typename T>
std::string safe_name(const std::vector<int> &id = {}) {
  std::string name(make_safe(T::name()));
  for (auto i : id)
    name += "_" + std::to_string(i);
  return name;
}

template <typename T>
std::string var_name(const std::vector<int> &id = {}) {
  std::string name(T::name());
  for (auto i : id)
    name += "_" + std::to_string(i);
  return name;
}

template <typename T, typename... Vars>
static void make_set(std::unordered_set<std::string> &s) {
  s.insert(T::name());
  if constexpr (sizeof...(Vars) > 0) make_set<Vars...>(s);
}

template <typename... Vars>
static std::unordered_set<std::string> make_set() {
  std::unordered_set<std::string> s;
  make_set<Vars...>(s);
  return s;
}

#define ccbulk cell_variables::cell_averaged::bulk
#define ccmat cell_variables::cell_averaged::mat
#define cm cell_variables::material_averaged

static std::unordered_set<std::string> independent_init_vars =
    make_set<ccmat::rho, ccmat::volume_fraction, ccbulk::momentum,
             ccbulk::total_material_energy, ccbulk::rho_reynolds_stress,
             ccbulk::rho_bhr_a, ccbulk::rho_bhr_b, ccbulk::rho_bhr_ST, ccbulk::rho_bhr_SD,
             ccmat::iso, ccmat::deviatoric_stress, ccmat::equivalent_plastic_strain,
             ccmat::ionization_zbar, ccbulk::electron_internal_energy>();

template <typename T>
using StrengthMap = TypeMaps::TypeMap<
    T, TypeMaps::GenericTypeMap<
           TypeMaps::TypePair<ccmat::deviatoric_stress, cm::deviatoric_stress>,
           TypeMaps::TypePair<ccmat::equivalent_plastic_strain,
                              cm::equivalent_plastic_strain>>>;
using StrengthMapEntries = StrengthMap<void>;

template <typename T>
using BHRMap = TypeMaps::TypeMap<
    T, TypeMaps::GenericTypeMap<
           TypeMaps::TypePair<ccbulk::rho_reynolds_stress, ccbulk::reynolds_stress>,
           TypeMaps::TypePair<ccbulk::rho_bhr_a, ccbulk::bhr_a>,
           TypeMaps::TypePair<ccbulk::rho_bhr_b, ccbulk::bhr_b>,
           TypeMaps::TypePair<ccbulk::rho_bhr_ST, ccbulk::bhr_ST>,
           TypeMaps::TypePair<ccbulk::rho_bhr_SD, ccbulk::bhr_SD>>>;
using BHRMapEntries = BHRMap<void>;

template <typename T>
using IsoMap =
    TypeMaps::TypeMap<T,
                      TypeMaps::GenericTypeMap<TypeMaps::TypePair<ccmat::iso, cm::iso>>>;
using IsoMapEntries = IsoMap<void>;

#undef cm
#undef ccmat
#undef ccbulk

inline std::string find_scalar_block(ParameterInput *pin, const std::string &label) {
  int scalar_id = 0;
  std::string block_name;
  for (;;) {
    block_name = "scalars" + std::to_string(scalar_id);
    if (!pin->DoesBlockExist(block_name))
      PARTHENON_FAIL("Did not find input block for requested scalar" + label);
    if (pin->GetString(block_name, "label") == label) break;
    scalar_id++;
  }
  return block_name;
}

field_func_t set_init_func(const std::string &name, ParameterInput *pin,
                           const std::string &block, python_region_t &pyreg,
                           python_region_t &default_pyreg);

template <typename T>
T get_or_use_default(ParameterInput *pin, const std::string &block,
                     const std::string &param, T default_val) {
  if (pin->DoesParameterExist(block, param))
    return pin->Get<T>(block, param);
  else if (pin->DoesParameterExist("regions", param))
    return pin->Get<T>("regions", param);
  return default_val;
}

inline field_func_t make_const_func(std::vector<Real> vals) {
  return [=](const sample_positions_t &x, VectorOfArrays &s) mutable {
    for (int i = 0; i < s.nstructs; i++) {
      for (int j = 0; j < s.nelem; j++) {
        s(i, j) = vals[j];
      }
    }
  };
}

template <typename Var>
field_func_t default_init() {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace cm = cell_variables::material_averaged;
  if constexpr (std::is_same<Var, cm::deviatoric_stress>::value)
    return make_const_func(std::vector<Real>(6, 0.0));
  if constexpr (std::is_same<Var, cm::equivalent_plastic_strain>::value)
    return make_const_func({0.0});
  if constexpr (std::is_same<Var, ccbulk::velocity>::value)
    return make_const_func(std::vector<Real>(3, 0.0));
  if constexpr (std::is_same<Var, ccbulk::bhr_a>::value)
    return make_const_func(std::vector<Real>(3, 0.0));
  return field_func_t();
}

template <typename Var>
void check_if_required() {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  const std::string error("No valid initialization of " + Var::name() + " provided.");
  if constexpr (std::is_same<Var, ccbulk::reynolds_stress>::value) PARTHENON_FAIL(error);
  if constexpr (std::is_same<Var, ccbulk::bhr_b>::value) PARTHENON_FAIL(error);
}

template <typename T, typename... Vars>
void set_mat_init_funcs(std::unordered_map<std::string, field_func_t> &mat_state,
                        ParameterInput *pin, const std::string &block,
                        python_region_t &pyreg, python_region_t &default_pyreg,
                        const std::vector<int> &matid) {
  for (auto mid : matid) {
    std::string mat_name = var_name<T>({mid});
    std::string mat_name_safe = safe_name<T>({mid});
    mat_state[mat_name] = set_init_func(mat_name_safe, pin, block, pyreg, default_pyreg);
    if (!mat_state[mat_name]) {
      std::string mat_block = "material" + std::to_string(mid);
      if (pin->DoesParameterExist(mat_block, "name")) {
        auto label = pin->GetString(mat_block, "name");
        mat_name_safe = safe_name<T>() + "_" + label;
        mat_state[mat_name] =
            set_init_func(mat_name_safe, pin, block, pyreg, default_pyreg);
      }
    }
    if (!mat_state[mat_name]) {
      // if only one mat, look for init without the "_matid" tag
      if (matid.size() == 1) {
        mat_name_safe = safe_name<T>();
        mat_state[mat_name] =
            set_init_func(mat_name_safe, pin, block, pyreg, default_pyreg);
        if (!mat_state[mat_name]) {
          mat_state[mat_name] = default_init<T>();
          if (!mat_state[mat_name]) {
            check_if_required<T>();
            mat_state.erase(mat_name);
          }
        }
      } else {
        if (!mat_state[mat_name]) {
          mat_state[mat_name] = default_init<T>();
          if (!mat_state[mat_name]) {
            check_if_required<T>();
            mat_state.erase(mat_name);
          }
        }
      }
    }
  }
  if constexpr (sizeof...(Vars) > 0)
    set_mat_init_funcs<Vars...>(mat_state, pin, block, pyreg, default_pyreg, matid);
}

template <typename T, typename... Vars>
void set_mat_init_vars(std::unordered_map<int, std::vector<std::string>> &init_fields,
                       std::unordered_map<int, std::string> &init_type_string,
                       std::unordered_map<std::string, field_func_t> &mat_state,
                       const std::vector<int> &matid) {
  static std::vector<std::string> init_strings{"c_m_rho", "c_m_pressure",
                                               "c_m_temperature", "c_m_sie"};
  for (auto mid : matid) {
    const auto name = var_name<T>({mid});
    const auto sname = safe_name<T>({mid});
    if (mat_state.count(name)) {
      init_fields[mid].push_back(name);
      for (auto &s : init_strings) {
        if (sname.find(s) != std::string::npos) init_type_string[mid] += s;
      }
    }
  }
  if constexpr (sizeof...(Vars) > 0)
    set_mat_init_vars<Vars...>(init_fields, init_type_string, mat_state, matid);
}

template <typename T, typename... Vars>
void set_bulk_init_funcs(std::unordered_map<std::string, field_func_t> &bulk_state,
                         ParameterInput *pin, const std::string &block,
                         python_region_t &pyreg, python_region_t &default_pyreg) {
  std::string name = var_name<T>();
  std::string sname = safe_name<T>();
  bulk_state[name] = set_init_func(sname, pin, block, pyreg, default_pyreg);
  if (!bulk_state[name]) {
    bulk_state[name] = default_init<T>();
    if (!bulk_state[name]) {
      check_if_required<T>();
      bulk_state.erase(name);
    }
  }
  if constexpr (sizeof...(Vars) > 0)
    set_bulk_init_funcs<Vars...>(bulk_state, pin, block, pyreg, default_pyreg);
}

template <typename Conserved_t, typename Primitive_t>
struct set_specific_bulk_init_funcs {
  template <typename... Args>
  static void apply(std::unordered_map<std::string, field_func_t> &sp_bulk_state,
                    Args &&...args) {
    namespace ccbulk = cell_variables::cell_averaged::bulk;
    sp_bulk_state[Conserved_t::name()] =
        set_init_func(safe_name<Primitive_t>(), std::forward<Args>(args)...);
    if (!sp_bulk_state[Conserved_t::name()]) {
      sp_bulk_state[Conserved_t::name()] = default_init<Primitive_t>();
      if (!sp_bulk_state[Conserved_t::name()]) {
        check_if_required<Primitive_t>();
        sp_bulk_state.erase(Conserved_t::name());
      }
    }
  }
};

template <typename Conserved_t, typename Primitive_t>
struct set_specific_mat_init_funcs {
  template <typename... Args>
  static void apply(std::unordered_map<int, std::unordered_map<std::string, field_func_t>>
                        &sp_mat_state,
                    const std::vector<int> &matid, ParameterInput *pin, Args &&...args) {
    for (auto mid : matid) {
      auto cname = var_name<Conserved_t>({mid});
      auto pname = safe_name<Primitive_t>({mid});
      auto func = set_init_func(pname, pin, std::forward<Args>(args)...);
      if (!func) {
        std::string mat_block = "material" + std::to_string(mid);
        if (pin->DoesParameterExist(mat_block, "name")) {
          auto label = pin->GetString(mat_block, "name");
          pname = safe_name<Primitive_t>() + "_" + label;
          func = set_init_func(pname, pin, std::forward<Args>(args)...);
        }
      }
      if (!func) {
        if (matid.size() == 1) {
          pname = safe_name<Primitive_t>();
          func = set_init_func(pname, pin, std::forward<Args>(args)...);
          if (!func) {
            func = default_init<Primitive_t>();
            if (!func) {
              check_if_required<Primitive_t>();
            }
          }
        } else {
          func = default_init<Primitive_t>();
          if (!func) {
            check_if_required<Primitive_t>();
          }
        }
      }
      if (func) sp_mat_state[mid][cname] = func;
    }
  }
};

class Region {
 private:
  python_region_t py;

 public:
  Region() = default;
  Region(ParameterInput *pin, const std::string &block, const int id,
         python_region_t &default_py);
  bool operator<(const Region &reg) const { return region_id < reg.region_id; }

  int region_id = std::numeric_limits<int>::lowest();
  std::string region_block;
  mask_func_t mask;
  std::vector<int> matid;
  std::unordered_map<int, std::vector<std::string>> mat_init_fields;
  std::unordered_map<std::string, field_func_t> mat_state;
  std::unordered_map<std::string, field_func_t> bulk_state;
  std::unordered_map<int, std::unordered_map<std::string, field_func_t>>
      specific_mat_state;
  std::unordered_map<std::string, field_func_t> specific_bulk_state;
  std::map<int, std::string> mat_tied_scalars;
  std::vector<std::string> bulk_tied_scalars;
  std::unordered_map<int, InitType> init_type;
  std::unordered_map<std::string, bool> flags;
};

struct Regions {
  Regions(ParameterInput *pin)
      : nlev_min(
            pin->GetOrAddInteger("regions", "nlev_min", 0,
                                 "In cells where two regions overlap, minimum number of "
                                 "refinements to compute volume fractions")),
        nlev_max(
            pin->GetOrAddInteger("regions", "nlev_max", 0,
                                 "In cells where two regions overlap, maximum number of "
                                 "refinements to compute volume fractions")),
        default_python_region(pin, "regions") {
    namespace ccbulk = cell_variables::cell_averaged::bulk;
    namespace cm = cell_variables::material_averaged;
    int nregions = 0;
    auto blocks = pin->GetBlockNamesWithPrefix("region");
    for (const auto &block_name : blocks) {
      if (block_name.length() > 6) {
        // TODO(jcd): insert more error checkings
        auto suffix = block_name[6];
        if (std::isdigit(suffix)) {
          auto id = atoi(block_name.substr(6).c_str());
          PARTHENON_REQUIRE_THROWS(reg_ids.count(id) == 0,
                                   "Two region input blocks are numbered identically.");
          reg_ids.insert(id);
          reg.emplace_back(
              std::make_unique<Region>(pin, block_name, id, default_python_region));
          nregions++;
        }
      }
    }

    idx.resize(nregions);
    std::iota(idx.begin(), idx.end(), 0);

    // Sort indices based on keys
    std::sort(idx.begin(), idx.end(), [&](size_t i, size_t j) {
      return (*reg[i]).region_id < (*reg[j]).region_id;
    });
  }

  Region &operator[](const int i) { return *reg[idx[i]]; }

  auto size() const { return reg.size(); }

  std::set<int> reg_ids;
  std::vector<std::unique_ptr<Region>> reg;
  std::vector<int> idx;
  int nlev_min, nlev_max;
  bool have_default_region = false;
  python_region_t default_python_region;
};

namespace detail {
template <int ndim>
int index(const int k, const int j, const int i) {
  int n = i;
  if constexpr (ndim > 1) n += 3 * j;
  if constexpr (ndim > 2) n += 9 * k;
  return n;
}

template <typename T, std::size_t N>
std::array<T, N> operator+(const std::array<T, N> &arr1, const std::array<T, N> &arr2) {
  std::array<T, N> result;
  for (std::size_t i = 0; i < N; ++i) {
    result[i] = arr1[i] + arr2[i];
  }
  return result;
}
} // namespace detail

std::tuple<Real, Real, Real> to_cartesian(const Real x0, const Real x1, const Real x2);

struct TreeCell {
  TreeCell(const std::tuple<int, int, int> &cell_id, const int numdim,
           const int my_stride, const int my_level)
      : cell(cell_id), ndim(numdim), stride(my_stride), level(my_level) {}
  std::tuple<int, int, int> cell;
  const int ndim, stride, level;
  std::vector<int> corner;
  std::vector<TreeCell> child;

  int refine_to(const int nlev_target) {
    if (level < nlev_target) {
      auto [k, j, i] = cell;
      auto cc = std::make_tuple(k, j, i);
      int child_stride = stride / 2;
      child.emplace_back(cc, ndim, child_stride, level + 1);
      child.back().refine_to(nlev_target);
      cc = std::make_tuple(k, j, i + child_stride);
      child.emplace_back(cc, ndim, child_stride, level + 1);
      child.back().refine_to(nlev_target);
      if (ndim > 1) {
        cc = std::make_tuple(k, j + child_stride, i);
        child.emplace_back(cc, ndim, child_stride, level + 1);
        child.back().refine_to(nlev_target);
        cc = std::make_tuple(k, j + child_stride, i + child_stride);
        child.emplace_back(cc, ndim, child_stride, level + 1);
        child.back().refine_to(nlev_target);
      }
      if (ndim > 2) {
        cc = std::make_tuple(k + child_stride, j, i);
        child.emplace_back(cc, ndim, child_stride, level + 1);
        child.back().refine_to(nlev_target);
        cc = std::make_tuple(k + child_stride, j, i + child_stride);
        child.emplace_back(cc, ndim, child_stride, level + 1);
        child.back().refine_to(nlev_target);
        cc = std::make_tuple(k + child_stride, j + child_stride, i);
        child.emplace_back(cc, ndim, child_stride, level + 1);
        child.back().refine_to(nlev_target);
        cc = std::make_tuple(k + child_stride, j + child_stride, i + child_stride);
        child.emplace_back(cc, ndim, child_stride, level + 1);
        child.back().refine_to(nlev_target);
      }
    }
    return child.size();
  }

  template <typename F>
  void fill_leaf_corners(std::unordered_set<int> &new_corners,
                         std::unordered_set<int> &all_corners, F &flatten) {
    if (child.size() > 0) {
      for (auto &c : child) {
        c.fill_leaf_corners(new_corners, all_corners, flatten);
      }
    }
    if (child.size() == 0 && corner.size() == 0) {
      auto [k, j, i] = cell;
      corner.push_back(flatten(k, j, i));
      auto isnew = all_corners.insert(corner.back());
      if (isnew.second) new_corners.insert(corner.back());
      corner.push_back(flatten(k, j, i + stride));
      isnew = all_corners.insert(corner.back());
      if (isnew.second) new_corners.insert(corner.back());
      if (ndim > 1) {
        corner.push_back(flatten(k, j + stride, i));
        isnew = all_corners.insert(corner.back());
        if (isnew.second) new_corners.insert(corner.back());
        corner.push_back(flatten(k, j + stride, i + stride));
        isnew = all_corners.insert(corner.back());
        if (isnew.second) new_corners.insert(corner.back());
      }
      if (ndim > 2) {
        corner.push_back(flatten(k + stride, j, i));
        isnew = all_corners.insert(corner.back());
        if (isnew.second) new_corners.insert(corner.back());
        corner.push_back(flatten(k + stride, j, i + stride));
        isnew = all_corners.insert(corner.back());
        if (isnew.second) new_corners.insert(corner.back());
        corner.push_back(flatten(k + stride, j + stride, i));
        isnew = all_corners.insert(corner.back());
        if (isnew.second) new_corners.insert(corner.back());
        corner.push_back(flatten(k + stride, j + stride, i + stride));
        isnew = all_corners.insert(corner.back());
        if (isnew.second) new_corners.insert(corner.back());
      }
    }
  }

  template <typename T>
  int refine(const std::unordered_map<int, T> &pid_to_reg, const int level_max) {
    if (level == level_max) return 0;
    if (child.size() > 0) {
      int sum_new = 0;
      for (int i = 0; i < child.size(); i++)
        sum_new += child[i].refine(pid_to_reg, level_max);
      return sum_new;
    }
    int r0 = pid_to_reg.at(corner[0]);
    for (int i = 1; i < corner.size(); i++) {
      if (pid_to_reg.at(corner[i]) != r0) {
        return refine_to(level + 1);
      }
    }
    return 0;
  }

  template <typename F>
  void integrate(const std::unordered_map<int, int> &pid_to_reg, F &id_to_xyz,
                 std::vector<std::array<Real, 4>> &vol_moments) {
    integrate_impl(pid_to_reg, id_to_xyz, vol_moments);
    // now normalize by volume
    std::array<Real, 3> xlo, xhi;
    set_bounds(id_to_xyz, xlo, xhi);
    auto moments = get_subcell_moments(xlo, xhi);
    auto norm = 1.0 / moments[0];
    for (int r = 0; r < vol_moments.size(); r++) {
      for (int m = 0; m < 4; m++)
        vol_moments[r][m] *= norm;
    }
  }

 private:
  template <typename F>
  void integrate_impl(const std::unordered_map<int, int> &pid_to_reg, F &id_to_xyz,
                      std::vector<std::array<Real, 4>> &vol_moments) {
    if (child.size() == 0) {
      // get bounds of current subcell
      std::array<Real, 3> xlo, xhi;
      set_bounds(id_to_xyz, xlo, xhi);
      // integrate over subcell
      auto moments = get_subcell_moments(xlo, xhi);
      // and add to region integrals
      Real wgt = 1.0 / (1 << ndim);
      for (auto id : corner) {
        int r = pid_to_reg.at(id);
        for (int m = 0; m < 4; m++)
          vol_moments[r][m] += wgt * moments[m];
      }
    } else {
      for (auto &c : child) {
        c.integrate_impl(pid_to_reg, id_to_xyz, vol_moments);
      }
    }
  }

  std::array<Real, 4> get_subcell_moments(std::array<Real, 3> &xlo,
                                          std::array<Real, 3> &xhi) {
    std::array<Real, 4> moments;
    if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      Real r0sq = xlo[0] * xlo[0];
      Real r0cb = xlo[0] * r0sq;
      Real r1sq = xhi[0] * xhi[0];
      Real r1cb = xhi[0] * r1sq;
      moments[0] = 4.0 / 3.0 * M_PI * (r1cb - r0cb);
      moments[1] = 3.0 * M_PI * (xhi[0] + r0cb / (r0sq + xlo[0] * xhi[0] + r1sq));
      moments[2] = 0.0;
      moments[3] = 0.0;
      return moments;
    }
    if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      Real r0sq = xlo[0] * xlo[0];
      Real r0cb = xlo[0] * r0sq;
      Real r1sq = xhi[0] * xhi[0];
      Real r1cb = xhi[0] * r1sq;
      Real dz = xhi[1] - xlo[1];
      moments[0] = M_PI * (r1sq - r0sq) * dz;
      moments[1] = 2.0 * M_PI / 3.0 * (r1cb - r0cb) * dz;
      moments[2] = 0.5 * moments[0] * (xlo[1] + xhi[1]);
      moments[3] = 0.0;
      return moments;
    }
    moments[0] = 1.0;
    for (int d = 0; d < ndim; d++) {
      moments[0] *= (xhi[d] - xlo[d]);
    }
    moments[1] = moments[0] * 0.5 * (xlo[0] + xhi[0]);
    moments[2] = ndim > 1 ? moments[0] * 0.5 * (xlo[1] + xhi[1]) : 0.0;
    moments[3] = ndim > 2 ? moments[0] * 0.5 * (xlo[2] + xhi[2]) : 0.0;
    return moments;
  }

  template <typename F>
  void set_bounds(F &id_to_xyz, std::array<Real, 3> &xlo, std::array<Real, 3> &xhi) {
    for (int d = 0; d < 3; d++) {
      xlo[d] = 1.e300;
      xhi[d] = -1e300;
    }
    for (auto &id : corner) {
      auto [x1, x2, x3] = id_to_xyz(id);
      xlo[0] = std::min(xlo[0], x1);
      xhi[0] = std::max(xhi[0], x1);
      xlo[1] = std::min(xlo[1], x2);
      xhi[1] = std::max(xhi[1], x2);
      xlo[2] = std::min(xlo[2], x3);
      xhi[2] = std::max(xhi[2], x3);
    }
  }
};

class BlockInitData {
  enum region_loop_bounds { state = 0, mask = 1 };

 public:
  BlockInitData(ParameterInput *pin);
  void InitBlock(MeshBlock *pmb, ParameterInput *pin);

  Regions regions;
  int nghost, nx1, nx2, nx3;
  parthenon::HostArray3D<int> region_id;
  parthenon::HostArray3D<Real> vol_frac_sum;
  IndexRange ib[2], jb[2], kb[2];
  std::vector<std::vector<int>> cells_in_region;

 private:
  int ndim, nlev_min, nlev_max;
  int num_mask, num_state;
  parthenon::Coordinates_t *coords;
  bool do_3t, fully_ionized;

  inline auto coordinate_velocities(Real *v) {
    if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      // convert cartesian vector components to cylindrical
      // we'll assume we're on the x-z plane at y = 0, so theta = 0
      return std::make_tuple(v[0], v[2], v[1]);
    } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
      // same deal for spherical
      return std::make_tuple(v[0], v[2], v[1]);
    }
    return std::make_tuple(v[0], v[1], v[2]);
  }

  template <typename EOS_t>
  void set_cell_conserved(InitType inputs, Real region_frac, Real *v, EOS_t &eos,
                          Real &vfrac, Real &density, Real &mx, Real &my, Real &mz,
                          Real &energy) {
    enum InputIndex { ivfrac, iu0, iu1, ivx, ivy, ivz };
    Real rho, sie;
    if (inputs == InitType::rhoT) {
      rho = v[iu0];
      sie = eos.InternalEnergyFromDensityTemperature(rho, v[iu1]);
    } else if (inputs == InitType::rhoe) {
      rho = v[iu0];
      sie = v[iu1];
    } else if (inputs == InitType::rhoP) {
      rho = v[iu0];
      sie = eos.InternalEnergyFromDensityTemperature(rho, 300.0); // just a guess
      sie = RiotEOS::energy_from_rho_P(eos, rho, v[iu1], sie) / rho;
    } else if (inputs == InitType::PT) {
      rho = RiotEOS::rho_from_P_T(eos, v[iu0], v[iu1], 20.0);
      sie = eos.InternalEnergyFromDensityTemperature(rho, v[iu1]);
    } else {
      PARTHENON_THROW("Invalid InitType for region");
    }

    const Real rho_cont = region_frac * v[ivfrac] * rho;
    vfrac += region_frac * v[ivfrac];
    density += rho_cont;
    auto [v1, v2, v3] = coordinate_velocities(&v[ivx]);
    mx += rho_cont * v1;
    my += rho_cont * v2;
    mz += rho_cont * v3;
    const Real ske = 0.5 * (v[ivx] * v[ivx] + v[ivy] * v[ivy] + v[ivz] * v[ivz]);
    energy += rho_cont * (sie + ske);
  }

  template <typename ion_EOS_t, typename electron_EOS_t>
  void set_cell_conserved_3t(InitType inputs, Real region_frac, Real *v,
                             ion_EOS_t &ion_eos, electron_EOS_t &electron_eos,
                             Real &vfrac, Real &density, Real &mx, Real &my, Real &mz,
                             Real &energy, Real &electron_energy, Real &rhozbar) {
    using namespace RiotEOS;
    enum InputIndex { ivfrac, iu0, iu1, iu2, ivx, ivy, ivz };
    Real rho, sie, sie_e, zbar;
    Real anuc = ion_eos.MeanAtomicMass();
    Real znuc = ion_eos.MeanAtomicNumber();
    if (fully_ionized) {
      zbar = znuc;
    };
    if (inputs == InitType::rhoTiEquil)
      v[iu2] = v[iu1]; // set electron temperature to ion temperature
    if (inputs == InitType::rhoTiTe || inputs == InitType::rhoTiEquil) {
      rho = v[iu0];
      if (!fully_ionized) {
        Ionization::ComputeIonizationState(anuc, znuc, rho, v[iu2], zbar, fully_ionized);
      }
      LambdaIndexerSimple lambda(zbar);
      sie = ion_eos.InternalEnergyFromDensityTemperature(rho, v[iu1], lambda);
      sie_e = electron_eos.InternalEnergyFromDensityTemperature(rho, v[iu2], lambda);
    } else if (inputs == InitType::rhoPEquil) {
      rho = v[iu0];
      auto [temp, zb] = temperature_zbar_from_rho_P(ion_eos, electron_eos, v[iu0], v[iu1],
                                                    300.0, fully_ionized);
      zbar = zb;
      LambdaIndexerSimple lambda(zbar);
      sie = ion_eos.InternalEnergyFromDensityTemperature(rho, temp, lambda);
      sie_e = electron_eos.InternalEnergyFromDensityTemperature(rho, temp, lambda);
    } else if (inputs == InitType::rhoPTe) {
      rho = v[iu0];
      auto [temp, zb] = temperature_zbar_from_rho_P_Te(
          ion_eos, electron_eos, v[iu0], v[iu1], v[iu2], 300.0, fully_ionized);
      zbar = zb;
      LambdaIndexerSimple lambda(zbar);
      sie = ion_eos.InternalEnergyFromDensityTemperature(rho, temp, lambda);
      sie_e = electron_eos.InternalEnergyFromDensityTemperature(rho, v[iu2], lambda);
    } else if (inputs == InitType::PTiEquil) {
      auto [dens, zb] = density_zbar_from_P_temperature(ion_eos, electron_eos, v[iu0],
                                                        v[iu1], 1.0, fully_ionized);
      rho = dens;
      zbar = zb;
      LambdaIndexerSimple lambda(zbar);
      sie = ion_eos.InternalEnergyFromDensityTemperature(rho, v[iu1], lambda);
      sie_e = electron_eos.InternalEnergyFromDensityTemperature(rho, v[iu1], lambda);
    } else if (inputs == InitType::PTiTe) {
      auto [dens, zb] = density_zbar_from_P_Ti_Te(ion_eos, electron_eos, v[iu0], v[iu1],
                                                  v[iu2], 1.0, fully_ionized);
      rho = dens;
      zbar = zb;
      LambdaIndexerSimple lambda(zbar);
      sie = ion_eos.InternalEnergyFromDensityTemperature(rho, v[iu1], lambda);
      sie_e = electron_eos.InternalEnergyFromDensityTemperature(rho, v[iu2], lambda);
    } else {
      PARTHENON_THROW("Invalid InitType for region");
    }

    const Real rho_cont = region_frac * v[ivfrac] * rho;
    vfrac += region_frac * v[ivfrac];
    density += rho_cont;
    auto [v1, v2, v3] = coordinate_velocities(&v[ivx]);
    mx += rho_cont * v1;
    my += rho_cont * v2;
    mz += rho_cont * v3;
    const Real ske = 0.5 * (v[ivx] * v[ivx] + v[ivy] * v[ivy] + v[ivz] * v[ivz]);
    energy += rho_cont * (sie + sie_e + ske);
    electron_energy += rho_cont * sie_e;
    rhozbar += rho_cont * zbar;
  }

  template <int bound_id, typename func_t>
  void block_loop(func_t &&f) {
    for (int k = kb[bound_id].s; k <= kb[bound_id].e; k++) {
      for (int j = jb[bound_id].s; j <= jb[bound_id].e; j++) {
        for (int i = ib[bound_id].s; i <= ib[bound_id].e; i++) {
          f(k, j, i);
        }
      }
    }
  }

  std::vector<int> set_mask(sample_positions_t &x);
  bool near_boundary(const int k0, const int j0, const int i0, const int reg_id) const;
  std::vector<int> set_clean_mask();

  template <typename host_mirror_t>
  void set_clean_state(MeshBlock *pmb, std::vector<RiotEOS::EOS> &eos_v,
                       std::vector<RiotEOS::EOS> &electron_eos_v,
                       std::vector<int> &eos_from_matid, std::vector<int> &reg_cnt,
                       std::map<std::string, std::shared_ptr<Variable<Real>>> &var_map,
                       std::map<std::string, host_mirror_t> &var_mirror_map,
                       std::map<int, std::vector<std::string>> &mat_tied_mirror_vars) {
    namespace ccbulk = cell_variables::cell_averaged::bulk;
    namespace ccmat = cell_variables::cell_averaged::mat;
    namespace cm = cell_variables::material_averaged;
    sample_positions_t xs(num_state);
    std::vector<VectorOfArrays> ms;
    VectorOfArrays vs, Te;
    std::unordered_map<std::string, VectorOfArrays> ss;
    VectorOfArrays phase_frac;

    for (int r = 0; r < regions.size(); r++) {
      if (reg_cnt[r] == 0) continue;
      int n = 0;
      xs.resize(num_state);
      block_loop<state>([&](const int k, const int j, const int i) {
        if (region_id(k, j, i) == r) {
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
      vs.reshape(n, 1);
      // fill in bulk tied scalars, just = 1 if in the region
      for (const auto &scalar : regions[r].bulk_tied_scalars) {
        auto scl = var_mirror_map[scalar];
        block_loop<state>([&](const int k, const int j, const int i) {
          scl(k, j, i) = (region_id(k, j, i) == r);
        });
      }

      for (const auto &bulk : regions[r].bulk_state) {
        // skip fields that are only used as an intermediary in init
        if (var_mirror_map.count(bulk.first) == 0) continue;
        auto q = var_mirror_map[bulk.first];
        int ncomp = var_map[bulk.first]->NumComponents();
        vs.reshape(n, ncomp);
        bulk.second(xs, vs);
        n = 0;
        block_loop<state>([&](const int k, const int j, const int i) {
          if (region_id(k, j, i) == r) {
            for (int l = 0; l < ncomp; l++) {
              q(l, k, j, i) = vs(n, l);
            }
            n++;
          }
        });
      }
      VectorOfArrays temp(xs.size(), 1);
      for (auto &sp_bulk : regions[r].specific_bulk_state) {
        if (var_mirror_map.count(sp_bulk.first) == 0) continue;
        int ncomp = var_map[sp_bulk.first]->NumComponents();
        ss[sp_bulk.first].reshape(xs.size(), ncomp);
        sp_bulk.second(xs, ss[sp_bulk.first]);
      }

      // special little call here for velocity.  this must be last so vs is the velocity
      // when needed below
      int ncomp = var_map[var_name<ccbulk::velocity>()]->NumComponents();
      vs.reshape(xs.size(), ncomp);
      regions[r].bulk_state[var_name<ccbulk::velocity>()](xs, vs);

      // special call here for electron temperature if 3t is on and Te was provided
      if (do_3t &&
          regions[r].bulk_state.count(var_name<ccbulk::electron_temperature>())) {
        Te.reshape(xs.size(), 1);
        regions[r].bulk_state[var_name<ccbulk::electron_temperature>()](xs, Te);
      } else {
        Te.reshape(0, 0);
      }

      for (int mat_id : regions[r].matid) {
        const int num_mat_init_fields = regions[r].mat_init_fields[mat_id].size();
        ms.resize(num_mat_init_fields);
        for (int i = 0; i < num_mat_init_fields; i++) {
          // assume all mat quantities are scalars for now
          ms[i].reshape(xs.size(), 1);
          regions[r].mat_state[regions[r].mat_init_fields[mat_id][i]](xs, ms[i]);
        }
        std::vector<Real> local_state(num_mat_init_fields + 3 + do_3t);
        const std::string rho_mat_name = var_name<ccmat::rho>({mat_id});
        if (!var_map[rho_mat_name]->IsAllocated()) {
          pmb->AllocateSparse(rho_mat_name);
          var_mirror_map[rho_mat_name] = var_map[rho_mat_name]->data.GetHostMirror();
          for (auto &mv : mat_tied_mirror_vars[mat_id]) {
            var_mirror_map[mv] = var_map.at(mv)->data.GetHostMirror();
          }
          // must zero out rho and vfrac everywhere on the block
          auto rho = var_mirror_map[rho_mat_name];
          auto vfrac = var_mirror_map[var_name<ccmat::volume_fraction>({mat_id})];
          int nphase = var_map[rho_mat_name]->GetDim(4);
          for (int p = 0; p < nphase; p++) {
            block_loop<state>([&](const int k, const int j, const int i) {
              rho(p, k, j, i) = 0.0;
              vfrac(p, k, j, i) = 0.0;
            });
          }
        }
        auto rho = var_mirror_map[rho_mat_name];
        auto vfrac = var_mirror_map[var_name<ccmat::volume_fraction>({mat_id})];
        auto nphases = var_map[rho_mat_name]->GetDim(4);
        phase_frac.reshape(xs.size(), nphases);
        regions[r].mat_state[var_name<cm::phase_fraction>({mat_id})](xs, phase_frac);

        for (auto &sp_mat : regions[r].specific_mat_state[mat_id]) {
          if (var_mirror_map.count(sp_mat.first) == 0) continue;
          int ncomp = var_map[sp_mat.first]->NumComponents();
          ss[sp_mat.first].reshape(xs.size(), ncomp);
          sp_mat.second(xs, ss[sp_mat.first]);
        }

        auto rhov = var_mirror_map[var_name<ccbulk::momentum>()];
        auto E = var_mirror_map[var_name<ccbulk::total_material_energy>()];
        auto u_e = (do_3t ? var_mirror_map[var_name<ccbulk::electron_internal_energy>()]
                          : var_mirror_map["does_not_exist"]);
        auto rhozbar = (do_3t ? var_mirror_map[var_name<ccmat::ionization_zbar>({mat_id})]
                              : var_mirror_map["does_not_exist"]);
        for (int p = 0; p < nphases; p++) {
          auto &eos = eos_v[eos_from_matid[mat_id] + p];
          auto &electron_eos = electron_eos_v[eos_from_matid[mat_id] + p];
          n = 0;
          block_loop<state>([&](const int k, const int j, const int i) {
            if (region_id(k, j, i) == r) {
              // copy material state
              for (int l = 0; l < num_mat_init_fields; l++)
                local_state[l] = ms[l](n, 0);
              // throw in electron temperature
              if (do_3t) {
                if (Te.size() > 0)
                  local_state[num_mat_init_fields] = Te(n, 0);
                else
                  local_state[num_mat_init_fields] =
                      0.0; // electron temp must be in equilibrium
              }
              // copy velocities
              for (int l = num_mat_init_fields + do_3t;
                   l < num_mat_init_fields + 3 + do_3t; l++) {
                local_state[l] = vs(n, l - num_mat_init_fields - do_3t);
              }
              if (do_3t) {
                set_cell_conserved_3t(
                    regions[r].init_type[mat_id], phase_frac(n, p), local_state.data(),
                    eos, electron_eos, vfrac(p, k, j, i), rho(p, k, j, i),
                    rhov(0, k, j, i), rhov(1, k, j, i), rhov(2, k, j, i), E(k, j, i),
                    u_e(k, j, i), rhozbar(p, k, j, i));
              } else {
                set_cell_conserved(regions[r].init_type[mat_id], phase_frac(n, p),
                                   local_state.data(), eos, vfrac(p, k, j, i),
                                   rho(p, k, j, i), rhov(0, k, j, i), rhov(1, k, j, i),
                                   rhov(2, k, j, i), E(k, j, i));
              }
              if (auto it = regions[r].mat_tied_scalars.find(mat_id);
                  it != regions[r].mat_tied_scalars.end()) {
                var_mirror_map[it->second](k, j, i) =
                    (p > 0 ? var_mirror_map[it->second](k, j, i) : 0) + rho(p, k, j, i);
              }
              for (auto &sp_bulk : regions[r].specific_bulk_state) {
                if (var_mirror_map.count(sp_bulk.first) == 0) continue;
                int ncomp = var_map[sp_bulk.first]->NumComponents();
                for (int l = 0; l < ncomp; l++) {
                  var_mirror_map[sp_bulk.first](l, k, j, i) +=
                      rho(p, k, j, i) * ss[sp_bulk.first](n, l);
                }
              }
              for (auto &sp_mat : regions[r].specific_mat_state[mat_id]) {
                if (var_mirror_map.count(sp_mat.first) == 0) continue;
                int ncomp = var_map[sp_mat.first]->NumComponents();
                for (int l = 0; l < ncomp; l++) {
                  var_mirror_map[sp_mat.first](l, k, j, i) =
                      (p > 0 ? var_mirror_map[sp_mat.first](l, k, j, i) : 0) +
                      rho(p, k, j, i) * ss[sp_mat.first](n, l);
                }
              }
              n++;
            }
          });
        }
      }
    }
  }

  using dirty_map_t =
      std::unordered_map<std::tuple<int, int, int>, std::unordered_map<int, Real>>;
  template <typename host_mirror_t>
  dirty_map_t
  set_dirty_state(MeshBlock *pmb, std::vector<RiotEOS::EOS> &eos_v,
                  std::vector<RiotEOS::EOS> &electron_eos_v,
                  std::vector<int> &eos_from_matid,
                  std::map<std::string, std::shared_ptr<Variable<Real>>> &var_map,
                  std::map<std::string, host_mirror_t> &var_mirror_map,
                  std::map<int, std::vector<std::string>> &mat_tied_mirror_vars,
                  std::vector<int> &all_matids) {
    namespace ccbulk = cell_variables::cell_averaged::bulk;
    namespace ccmat = cell_variables::cell_averaged::mat;
    namespace cm = cell_variables::material_averaged;
    std::vector<std::array<Real, 4>> vol_moments(regions.size());
    std::array<Real, 3> xlo, dx;
    VectorOfArrays vs(1, 1);
    VectorOfArrays phase_frac;
    VectorOfArrays Te;
    std::vector<VectorOfArrays> ms;
    std::unordered_map<std::string, VectorOfArrays> ss;
    std::set<int> mats_to_remove;
    dirty_map_t dirty_map;
    std::vector<Real> local_state;
    std::vector<TreeCell> dirty_cell;
    std::unordered_set<int> dirty_id;

    for (auto mat_id : all_matids) {
      const std::string rho_mat_name = var_name<ccmat::rho>({mat_id});
      if (!var_map[rho_mat_name]->IsAllocated()) {
        pmb->AllocateSparse(rho_mat_name);
        var_mirror_map[rho_mat_name] = var_map[rho_mat_name]->data.GetHostMirror();
        for (auto &cvar : mat_tied_mirror_vars[mat_id]) {
          var_mirror_map[cvar] = var_map[cvar]->data.GetHostMirror();
        }
        auto rho = var_mirror_map[rho_mat_name];
        auto vfrac = var_mirror_map[var_name<ccmat::volume_fraction>({mat_id})];
        int nphase = var_map[rho_mat_name]->GetDim(4);
        for (int p = 0; p < nphase; p++) {
          block_loop<state>([&](const int k, const int j, const int i) {
            rho(p, k, j, i) = 0.0;
            vfrac(p, k, j, i) = 0.0;
          });
        }
      }
      mats_to_remove.insert(mat_id);
    }

    const int nmax = regions.nlev_max;
    const int nref = 1 << nmax;
    // max number of faces in the fully refined mesh
    const int n1 = (nx1 + 2 * nghost) * nref + 1;
    const int n2 = ndim > 1 ? (nx2 + 2 * nghost) * nref + 1 : 1;
    xlo[0] = coords->Xf<X1DIR>(0, 0, 0);
    xlo[1] = coords->Xf<X2DIR>(0, 0, 0);
    xlo[2] = coords->Xf<X3DIR>(0, 0, 0);
    // assume dx is constant on the block
    dx[0] = coords->Dxf<X1DIR>(0) / nref;
    dx[1] = coords->Dxf<X2DIR>(0) / nref;
    dx[2] = coords->Dxf<X3DIR>(0) / nref;

    // Find the dirty cells and start building the tree
    block_loop<state>([&](const int k, const int j, const int i) {
      if (region_id(k, j, i) == -1) {
        dirty_cell.emplace_back(std::make_tuple(k * nref, j * nref, i * nref), ndim, nref,
                                0);
        dirty_cell.back().refine_to(regions.nlev_min);
      } else {
        for (auto mat_id : regions[region_id(k, j, i)].matid) {
          mats_to_remove.erase(mat_id);
        }
      }
    });

    auto flatten = [&](const int k, const int j, const int i) {
      return i + n1 * (j + n2 * k);
    };
    std::unordered_set<int> new_points;
    std::unordered_set<int> all_points;
    for (auto &c : dirty_cell) {
      c.fill_leaf_corners(new_points, all_points, flatten);
    }

    auto id_to_xyz = [&](const int id) {
      const int k = id / (n1 * n2);
      const int j = (id - k * n1 * n2) / n1;
      const int i = id - n1 * (j + n2 * k);
      return std::make_tuple(xlo[0] + i * dx[0], xlo[1] + j * dx[1], xlo[2] + k * dx[2]);
    };
    sample_positions_t xs(new_points.size());
    int n = 0;
    for (auto pid : new_points) {
      auto [x, y, z] = id_to_xyz(pid);
      xs(n, 0) = x;
      xs(n, 1) = y;
      xs(n, 2) = z;
      n++;
    }

    std::vector<int> reg_id;
    if (new_points.size() > 0) reg_id = set_mask(xs);
    std::unordered_map<int, int> pid_to_reg;
    n = 0;
    for (auto pid : new_points) {
      pid_to_reg[pid] = reg_id[n++];
    }

    for (int lev = 0; lev <= regions.nlev_max; lev++) {
      for (auto &c : dirty_cell) {
        c.refine(pid_to_reg, regions.nlev_max);
      }

      new_points.clear();
      for (auto &c : dirty_cell) {
        c.fill_leaf_corners(new_points, all_points, flatten);
      }
      if (new_points.size() == 0) break;

      xs.resize(new_points.size());
      n = 0;
      for (auto pid : new_points) {
        auto [x, y, z] = id_to_xyz(pid);
        xs(n, 0) = x;
        xs(n, 1) = y;
        xs(n, 2) = z;
        n++;
      }
      reg_id = set_mask(xs);
      n = 0;
      for (auto pid : new_points) {
        pid_to_reg[pid] = reg_id[n++];
      }
    }

    xs.resize(1);
    for (auto &c : dirty_cell) {

      auto [kfine, jfine, ifine] = c.cell;
      int k = kfine / nref;
      int j = jfine / nref;
      int i = ifine / nref;

      for (int r = 0; r < regions.size(); r++) {
        vol_moments[r][0] = 0.0;
        vol_moments[r][1] = 0.0;
        vol_moments[r][2] = 0.0;
        vol_moments[r][3] = 0.0;
      }
      c.integrate(pid_to_reg, id_to_xyz, vol_moments);

      for (int r = 0; r < regions.size(); r++) {
        if (vol_moments[r][0] == 0.0) continue; // region does not overlap cell
        dirty_map[std::make_tuple(k, j, i)][r] = vol_moments[r][0];
        xs(0, 0) = vol_moments[r][1];
        xs(0, 1) = vol_moments[r][2];
        xs(0, 2) = vol_moments[r][3];

        // set bulk tied scalars
        for (auto &scalar : regions[r].bulk_tied_scalars) {
          var_mirror_map[scalar](k, j, i) = vol_moments[r][0];
        }
        // set bulk state
        for (auto &bulk : regions[r].bulk_state) {
          if (var_mirror_map.count(bulk.first) == 0) continue;
          bulk.second(xs, vs);
          var_mirror_map[bulk.first](k, j, i) += vol_moments[r][0] * vs(0, 0);
        }
        // bulk state with specific (1/rho) initialization
        for (auto &sp_bulk : regions[r].specific_bulk_state) {
          if (var_mirror_map.count(sp_bulk.first) == 0) continue;
          int ncomp = var_map[sp_bulk.first]->NumComponents();
          ss[sp_bulk.first].reshape(xs.size(), ncomp);
          sp_bulk.second(xs, ss[sp_bulk.first]);
        }

        // special little call here for velocity.  this must be last so vs is the
        // velocity when needed below
        int ncomp = var_map[var_name<ccbulk::velocity>()]->NumComponents();
        vs.reshape(xs.size(), ncomp);
        regions[r].bulk_state[var_name<ccbulk::velocity>()](xs, vs);

        // special call here for electron temperature if 3t is on and Te was provided
        if (do_3t &&
            regions[r].bulk_state.count(var_name<ccbulk::electron_temperature>())) {
          Te.reshape(xs.size(), 1);
          regions[r].bulk_state[var_name<ccbulk::electron_temperature>()](xs, Te);
        } else {
          Te.reshape(0, 0);
        }

        for (auto mat_id : regions[r].matid) {
          const int num_mat_init_fields = regions[r].mat_init_fields[mat_id].size();
          ms.resize(num_mat_init_fields);
          for (int l = 0; l < num_mat_init_fields; l++)
            ms[l].reshape(1, 1);
          for (int l = 0; l < num_mat_init_fields; l++) {
            regions[r].mat_state[regions[r].mat_init_fields[mat_id][l]](xs, ms[l]);
          }
          local_state.resize(num_mat_init_fields + 3 + do_3t);
          // copy material state
          for (int l = 0; l < num_mat_init_fields; l++)
            local_state[l] = ms[l](0, 0);
          if (do_3t) {
            if (Te.size() > 0)
              local_state[num_mat_init_fields] = Te(0, 0);
            else
              local_state[num_mat_init_fields] = 0.0; // electrons in equilibrium
          }
          // copy velocities
          for (int l = num_mat_init_fields + do_3t; l < num_mat_init_fields + 3 + do_3t;
               l++) {
            local_state[l] = vs(0, l - num_mat_init_fields - do_3t);
          }

          for (auto &sp_mat : regions[r].specific_mat_state[mat_id]) {
            if (var_mirror_map.count(sp_mat.first) == 0) continue;
            int ncomp = var_map[sp_mat.first]->NumComponents();
            ss[sp_mat.first].reshape(xs.size(), ncomp);
            sp_mat.second(xs, ss[sp_mat.first]);
          }

          const std::string rho_mat_name = var_name<ccmat::rho>({mat_id});
          int nphases = var_map[rho_mat_name]->GetDim(4);
          phase_frac.reshape(1, nphases);
          regions[r].mat_state[var_name<cm::phase_fraction>({mat_id})](xs, phase_frac);

          for (int p = 0; p < nphases; p++) {
            auto &eos = eos_v[eos_from_matid[mat_id] + p];
            auto &electron_eos = electron_eos_v[eos_from_matid[mat_id] + p];
            if (do_3t) {
              set_cell_conserved_3t(
                  regions[r].init_type[mat_id], vol_moments[r][0] * phase_frac(0, p),
                  local_state.data(), eos, electron_eos,
                  var_mirror_map[var_name<ccmat::volume_fraction>({mat_id})](p, k, j, i),
                  var_mirror_map[rho_mat_name](p, k, j, i),
                  var_mirror_map[ccbulk::momentum::name()](0, k, j, i),
                  var_mirror_map[ccbulk::momentum::name()](1, k, j, i),
                  var_mirror_map[ccbulk::momentum::name()](2, k, j, i),
                  var_mirror_map[ccbulk::total_material_energy::name()](k, j, i),
                  var_mirror_map[var_name<ccbulk::electron_internal_energy>()](k, j, i),
                  var_mirror_map[var_name<ccmat::ionization_zbar>({mat_id})](p, k, j, i));
            } else {
              set_cell_conserved(
                  regions[r].init_type[mat_id], vol_moments[r][0] * phase_frac(0, p),
                  local_state.data(), eos,
                  var_mirror_map[var_name<ccmat::volume_fraction>({mat_id})](p, k, j, i),
                  var_mirror_map[rho_mat_name](p, k, j, i),
                  var_mirror_map[ccbulk::momentum::name()](0, k, j, i),
                  var_mirror_map[ccbulk::momentum::name()](1, k, j, i),
                  var_mirror_map[ccbulk::momentum::name()](2, k, j, i),
                  var_mirror_map[ccbulk::total_material_energy::name()](k, j, i));
            }
            // if there's some mass, remove the material from the list of materials to
            // remove. note the double negative above.  if there's mass, we keep the
            // material
            Real rho_added =
                vol_moments[r][0] * phase_frac(0, p) * local_state[0] * local_state[1];
            if (rho_added > 0.0) mats_to_remove.erase(mat_id);
            // look for a scalar tied to this mat in this region
            if (auto it = regions[r].mat_tied_scalars.find(mat_id);
                it != regions[r].mat_tied_scalars.end()) {
              var_mirror_map[it->second](k, j, i) =
                  (p > 0 ? var_mirror_map[it->second](k, j, i) : 0) +
                  var_mirror_map[rho_mat_name](p, k, j,
                                               i); // conserved material density
            }
            for (auto &sp_bulk : regions[r].specific_bulk_state) {
              if (var_mirror_map.count(sp_bulk.first) == 0) continue;
              int ncomp = var_map[sp_bulk.first]->NumComponents();
              for (int l = 0; l < ncomp; l++) {
                var_mirror_map[sp_bulk.first](l, k, j, i) +=
                    rho_added * ss[sp_bulk.first](0, l);
              }
            }
            for (auto &sp_mat : regions[r].specific_mat_state[mat_id]) {
              if (var_mirror_map.count(sp_mat.first) == 0) continue;
              int ncomp = var_map[sp_mat.first]->NumComponents();
              for (int l = 0; l < ncomp; l++) {
                var_mirror_map[sp_mat.first](l, k, j, i) =
                    (p > 0 ? var_mirror_map[sp_mat.first](l, k, j, i) : 0) +
                    rho_added * ss[sp_mat.first](0, l);
              }
            }
          }
        }
      }
    }

    for (auto mat_id : all_matids) {
      if (mats_to_remove.count(mat_id)) {
        const std::string rho_mat_name = var_name<ccmat::rho>({mat_id});
        pmb->DeallocateSparse(rho_mat_name);
      }
    }
    return dirty_map;
  }
};

} // namespace region_pgen

#endif

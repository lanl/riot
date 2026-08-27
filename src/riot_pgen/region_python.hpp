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
#ifndef RIOT_PGEN_REGION_PYTHON_HPP_
#define RIOT_PGEN_REGION_PYTHON_HPP_
// This file was made in part with generative AI.

#include <string>
#include <unordered_map>

#include <basic_types.hpp>

#include "region_primitives.hpp"
#ifdef RIOT_ENABLE_PYTHON
#include "riot_utils/py_caller.hpp"

template <typename T>
pcall::ArrayViewND<Real> make_real_view(T &arg) {
  return pcall::view_nd(arg.vec, arg.shape(), true);
}
template <typename T>
pcall::ArrayViewND<const Real> make_real_view(const T &arg) {
  return pcall::view_nd(arg.vec, arg.shape(), true);
}
#endif

std::string to_lower(std::string input);
bool is_string(const std::string &val);
bool is_int(const std::string &val);
bool is_float(const std::string &val);
bool is_bool(std::string val);

template <typename T>
void add_to_param_map(std::unordered_map<std::string, T> &m, const std::string &name,
                      const std::string &value) {
  if constexpr (std::is_same<T, std::string>::value)
    if (is_string(value)) m[name] = value;
  if constexpr (std::is_same<T, int>::value)
    if (is_int(value)) m[name] = std::stoi(value);
  if constexpr (std::is_same<T, Real>::value)
    if (is_float(value)) m[name] = std::stod(value);
  if constexpr (std::is_same<T, bool>::value) {
    auto s = to_lower(value);
    if (s == "true") m[name] = true;
    if (s == "false") m[name] = false;
  }
}

std::tuple<std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, int>, std::unordered_map<std::string, Real>,
           std::unordered_map<std::string, bool>>
build_param_map(parthenon::ParameterInput *pin, const std::string &block);

struct python_region_t {
  python_region_t() = default;
  python_region_t(parthenon::ParameterInput *pin, const std::string &block_name);

  bool IsInitialized() const {
#ifdef RIOT_ENABLE_PYTHON
    return static_cast<bool>(py_obj);
#else
    return false;
#endif
  }

  mask_func_t make_mask() {
#ifdef RIOT_ENABLE_PYTHON
    return [this](const sample_positions_t &x) {
      return py_obj.call_method<std::vector<bool>>("mask", make_real_view(x));
    };
#else
    return mask_func_t{
        [](const sample_positions_t &x) { return std::vector<bool>(x.size(), true); }};
#endif
  }

  field_func_t make_state(const std::string &name) {
#ifdef RIOT_ENABLE_PYTHON
    PARTHENON_REQUIRE(py_obj.exists(name.c_str()),
                      "Missing function in provided python class.");
    return [name, this](const sample_positions_t &x, field_data_t &s) mutable {
      return py_obj.call_method<void>(name.c_str(), make_real_view(x), make_real_view(s));
    };
#else
    return [=](const sample_positions_t &x, field_data_t &s) { return field_func_t(); };
#endif
  }
#ifdef RIOT_ENABLE_PYTHON
  pcall::PyObjectRef py_obj;
#else
  char py_obj;
#endif
};

#endif // RIOT_PGEN_REGION_PYTHON_HPP_

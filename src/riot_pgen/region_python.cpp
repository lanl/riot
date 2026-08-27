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
// This file was made in part with generative AI.

#include "region_python.hpp"

#include <string>

#include <parameter_input.hpp>
using parthenon::ParameterInput;
#include <utils/error_checking.hpp>

#ifdef RIOT_ENABLE_PYTHON

python_region_t::python_region_t(ParameterInput *pin, const std::string &block_name)
    : py_obj(pin->DoesParameterExist(block_name, "file")
                 ? pcall::PyClass(
                       pin->GetString(
                           block_name, "file",
                           "Name of the python file defining the class for this region"),
                       pin->GetString(block_name, "name"), {"."})
                       .new_instance()
                 : (pin->DoesParameterExist("regions", "file")
                        ? pcall::PyClass(pin->GetString("regions", "file"),
                                         pin->GetString("regions", "name"), {"."})
                              .new_instance()
                        : nullptr)) {
  if (py_obj) {
    py_obj.set_attr("x", 0);
    py_obj.set_attr("y", 1);
    py_obj.set_attr("z", 2);
    std::string block;
    if (pin->DoesParameterExist(block_name, "file"))
      block = block_name;
    else
      block = "regions";

    auto name = pin->GetString(block, "name") + "/params";
    if (pin->DoesBlockExist(name)) {
      auto [smap, imap, rmap, bmap] = build_param_map(pin, name);
      for (auto &params : smap) {
        py_obj.set_attr(params.first.c_str(), params.second);
      }
      for (auto &params : imap) {
        py_obj.set_attr(params.first.c_str(), params.second);
      }
      for (auto &params : rmap) {
        py_obj.set_attr(params.first.c_str(), params.second);
      }
      for (auto &params : bmap) {
        py_obj.set_attr(params.first.c_str(), params.second);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn  std::tuple<std::unordered_map<std::string, std::string>,
//! std::unordered_map<std::string, int>, std::unordered_map<std::string, Real>,
//! std::unordered_map<std::string, bool>> build_param_map
//! \brief
std::tuple<std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, int>, std::unordered_map<std::string, Real>,
           std::unordered_map<std::string, bool>>
build_param_map(ParameterInput *pin, const std::string &block) {
  std::unordered_map<std::string, std::string> smap;
  std::unordered_map<std::string, int> imap;
  std::unordered_map<std::string, Real> rmap;
  std::unordered_map<std::string, bool> bmap;

  for (auto &pname : pin->GetParameterNames(block)) {
    std::string str_value = pin->GetAsUnresolvedString(block, pname);
    add_to_param_map(smap, pname, str_value);
    add_to_param_map(imap, pname, str_value);
    add_to_param_map(rmap, pname, str_value);
    add_to_param_map(bmap, pname, str_value);
  }
  return std::make_tuple(smap, imap, rmap, bmap);
}

//----------------------------------------------------------------------------------------
//! \fn  std::string to_lower
//! \brief
std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

//----------------------------------------------------------------------------------------
//! \fn  bool is_bool
//! \brief
bool is_bool(std::string s) {
  s = to_lower(s);
  if (s == "true" || s == "false") return true;
  return false;
}

//----------------------------------------------------------------------------------------
//! \fn  bool is_int
//! \brief
bool is_int(const std::string &s) {
  if (!is_bool(s)) {
    try {
      size_t pos;
      std::stoi(s, &pos);
      return pos == s.length();
    } catch (const std::invalid_argument &e) {
      return false;
    } catch (const std::out_of_range &e) {
      return false;
    }
  }
  return false;
}

//----------------------------------------------------------------------------------------
//! \fn  bool is_float
//! \brief
bool is_float(const std::string &s) {
  if (!is_int(s)) {
    try {
      size_t pos;
      std::stod(s, &pos);
      return pos == s.length();
    } catch (std::out_of_range &e) {
      return false;
    } catch (std::invalid_argument &e) {
      return false;
    }
  }
  return false;
}

//----------------------------------------------------------------------------------------
//! \fn  bool is_string
//! \brief
bool is_string(const std::string &s) {
  if (!is_float(s)) {
    return true;
  }
  return false;
}

#else // RIOT_ENABLE_PYTHON is not defined

python_region_t::python_region_t(ParameterInput *pin, const std::string &block_name) {
  PARTHENON_FAIL("Attempting to initialize a python region but riot is not built with "
                 "python support.");
}

#endif // RIOT_ENABLE_PYTHON

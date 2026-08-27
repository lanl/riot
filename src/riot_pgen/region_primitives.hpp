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
#ifndef RIOT_PGEN_REGION_PRIMITIVES_HPP_
#define RIOT_PGEN_REGION_PRIMITIVES_HPP_
// This file was made in part with generative AI.

#include <functional>

#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

struct VectorOfArrays {
  VectorOfArrays() = default;
  VectorOfArrays(const int n1, const int n2) : nstructs(n1), nelem(n2), vec(n1 * n2) {}
  Real &operator()(const int n, const int d) {
    assert(d < nelem);
    assert(nelem * n + d < vec.size());
    return vec[nelem * n + d];
  }
  Real operator()(const int n, const int d) const { return vec[nelem * n + d]; }
  size_t size() const { return (nelem > 0 ? vec.size() / nelem : 0); }
  virtual void reshape(const int n1, const int n2) {
    nstructs = n1;
    nelem = n2;
    vec.resize(n1 * n2);
  }
  void resize(const size_t new_size) {
    nstructs = new_size;
    vec.resize(nstructs * nelem);
  }
  std::vector<size_t> shape() const { return {nstructs, nelem}; }
  size_t nstructs, nelem;
  std::vector<Real> vec;
};

struct sample_positions_t : public VectorOfArrays {
  static constexpr size_t nelem = 3;
  sample_positions_t(const int n1, const int n2) = delete;
  sample_positions_t(const int n) : VectorOfArrays(n, nelem) {}
  void reshape(const int n1, const int n2) {
    assert(n2 == nelem);
    VectorOfArrays::reshape(n1, n2);
  }
};
using mask_t = std::vector<bool>;
using mask_func_t = std::function<mask_t(const sample_positions_t &)>;

template <typename T>
mask_func_t base_region_loop(T &&f) {
  return [=](const sample_positions_t &x) {
    mask_t mask(x.size());
    for (int i = 0; i < x.size(); i++) {
      mask[i] = f(x(i, 0), x(i, 1), x(i, 2));
    }
    return mask;
  };
}

using field_data_t = VectorOfArrays;
using field_func_t = std::function<void(const sample_positions_t &, field_data_t &)>;
using mask_generator_t =
    std::function<mask_func_t(ParameterInput *, const std::string &)>;

mask_func_t background(ParameterInput *pin, const std::string &block_name);
mask_func_t inside_sphere(ParameterInput *pin, const std::string &block_name);
mask_func_t inside_spherical_shell(ParameterInput *pin, const std::string &block_name);
mask_func_t inside_cylinder(ParameterInput *pin, const std::string &block_name);
mask_func_t inside_cylindrical_shell(ParameterInput *pin, const std::string &block_name);
mask_func_t inside_ellipsoid(ParameterInput *pin, const std::string &block_name);
mask_func_t inside_ellipsoidal_shell(ParameterInput *pin, const std::string &block_name);
mask_func_t inside_rectangle(ParameterInput *pin, const std::string &block_name);

#ifdef RIOT_ENABLE_CAD
mask_func_t cad(ParameterInput *pin, const std::string &block_name);
#else
inline mask_func_t cad(ParameterInput *pin, const std::string &block_name) {
  PARTHENON_FAIL("Riot not build with support for CAD initializiation.");
  return mask_func_t{
      [](const sample_positions_t &x) { return std::vector<bool>(x.size(), true); }};
}
#endif

#endif // REGION_PGEN_REGION_PRIMITIVES_HPP_

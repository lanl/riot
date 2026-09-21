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

#include <utils/type_list.hpp>

namespace region_primitives {

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

struct background_mask {
  bool invert;
  KOKKOS_INLINE_FUNCTION
  bool operator()(const Real x, const Real y, const Real z) const {
    return true ^ invert;
  }
};
inline auto background_lambda(ParameterInput *pin, const std::string &block_name) {
  bool invert = pin->GetOrAddBoolean(block_name, "invert", false);
  return background_mask{invert};
}

struct sphere_mask {
  Real x0, y0, z0, rsq;
  bool invert;
  KOKKOS_INLINE_FUNCTION
  bool operator()(const Real x, const Real y, const Real z) const {
    Real dx = x - x0;
    dx *= dx;
    Real dy = y - y0;
    dy *= dy;
    Real dz = z - z0;
    dz *= dz;
    return (dx + dy + dz < rsq) ^ invert;
  }
};
inline auto inside_sphere_lambda(ParameterInput *pin, const std::string &block_name) {
  Real x0 = pin->GetOrAddReal(block_name, "x0", 0.0, "x-coordinate of center of sphere");
  Real y0 = pin->GetOrAddReal(block_name, "y0", 0.0, "y-coordinate of center of sphere");
  Real z0 = pin->GetOrAddReal(block_name, "z0", 0.0, "z-coordinate of center of sphere");
  Real r = pin->GetOrAddReal(block_name, "radius", 1.0);
  bool invert = pin->GetOrAddBoolean(block_name, "invert", false);
  Real rsq = r * r;

  return sphere_mask{x0, y0, z0, rsq, invert};
}

struct spherical_shell_mask {
  Real x0, y0, z0, r0sq, r1sq;
  bool invert;
  KOKKOS_INLINE_FUNCTION
  bool operator()(const Real x, const Real y, const Real z) const {
    Real dx = x - x0;
    dx *= dx;
    Real dy = y - y0;
    dy *= dy;
    Real dz = z - z0;
    dz *= dz;
    const Real rsq = dx + dy + dz;
    return (rsq > r0sq && rsq < r1sq) ^ invert;
  }
};
inline auto inside_spherical_shell_lambda(ParameterInput *pin,
                                          const std::string &block_name) {
  Real x0 = pin->GetOrAddReal(block_name, "x0", 0.0,
                              "x-coordinate of center of spherical shell");
  Real y0 = pin->GetOrAddReal(block_name, "y0", 0.0,
                              "y-coordinate of center of spherical shell");
  Real z0 = pin->GetOrAddReal(block_name, "z0", 0.0,
                              "z-coordinate of center of spherical shell");
  Real r0 = pin->GetOrAddReal(block_name, "inner_radius", 0.0, "Radius of inner surface");
  Real r1 = pin->GetOrAddReal(block_name, "outer_radius", 1.0, "Radius of outer surface");
  bool invert = pin->GetOrAddBoolean(block_name, "invert", false);
  Real r0sq = r0 * r0;
  Real r1sq = r1 * r1;

  return spherical_shell_mask{x0, y0, z0, r0sq, r1sq, invert};
}

struct cylinder_mask {
  Real x0, y0, z0, x1, y1, z1, rsq, nx, ny, nz, dsq;
  bool invert;
  KOKKOS_INLINE_FUNCTION
  bool operator()(const Real x, const Real y, const Real z) const {
    // check if point is above bottom of cylinder
    Real dx0 = x - x0;
    Real dy0 = y - y0;
    Real dz0 = z - z0;
    Real dot = nx * dx0 + ny * dy0 + nz * dz0;
    if (dot < 0.0) return false ^ invert;
    // check if point is below top of cylinder
    Real dx1 = x - x1;
    Real dy1 = y - y1;
    Real dz1 = z - z1;
    dot = nx * dx1 + ny * dy1 + nz * dz1;
    if (dot > 0.0) return false ^ invert;
    // now check if point is inside of cylinder
    Real cx = dy0 * dz1 - dz0 * dy1;
    Real cy = dz0 * dx1 - dx0 * dz1;
    Real cz = dx0 * dy1 - dy0 * dx1;
    Real csq = cx * cx + cy * cy + cz * cz;
    return (csq / dsq < rsq) ^ invert;
  }
};
inline auto inside_cylinder_lambda(ParameterInput *pin, const std::string &block_name) {
  Real x0 = pin->GetOrAddReal(block_name, "x0", 0.0,
                              "x-coordinate of bottom of cylindrical volume");
  Real y0 = pin->GetOrAddReal(block_name, "y0", 0.0,
                              "y-coordinate of bottom of cylindrical volume");
  Real z0 = pin->GetOrAddReal(block_name, "z0", 0.0,
                              "z-coordinate of bottom of cylindrical volume");
  Real x1 = pin->GetOrAddReal(block_name, "x1", 0.0,
                              "x-coordinate of top of cylindrical volume");
  Real y1 = pin->GetOrAddReal(block_name, "y1", 0.0,
                              "y-coordinate of top of cylindrical volume");
  Real z1 = pin->GetOrAddReal(block_name, "z1", 1.0,
                              "z-coordinate of top of cylindrical volume");
  Real r = pin->GetOrAddReal(block_name, "radius", 1.0);
  bool invert = pin->GetOrAddBoolean(block_name, "invert", false);
  Real rsq = r * r;

  Real nx = x1 - x0;
  Real ny = y1 - y0;
  Real nz = z1 - z0;
  Real dsq = nx * nx + ny * ny + nz * nz;

  return cylinder_mask{x0, y0, z0, x1, y1, z1, rsq, nx, ny, nz, dsq, invert};
}

struct cylindrical_shell_mask {
  Real x0, y0, z0, x1, y1, z1, r0sq, r1sq, nx, ny, nz, dsq;
  bool invert;
  KOKKOS_INLINE_FUNCTION
  bool operator()(const Real x, const Real y, const Real z) const {
    // check if point is above bottom of cylinder
    Real dx0 = x - x0;
    Real dy0 = y - y0;
    Real dz0 = z - z0;
    Real dot = nx * dx0 + ny * dy0 + nz * dz0;
    if (dot < 0.0) return false ^ invert;
    // check if point is below top of cylinder
    Real dx1 = x - x1;
    Real dy1 = y - y1;
    Real dz1 = z - z1;
    dot = nx * dx1 + ny * dy1 + nz * dz1;
    if (dot > 0.0) return false ^ invert;
    // now check if point is inside of cylindrical shell
    Real cx = dy0 * dz1 - dz0 * dy1;
    Real cy = dz0 * dx1 - dx0 * dz1;
    Real cz = dx0 * dy1 - dy0 * dx1;
    Real csq = (cx * cx + cy * cy + cz * cz) / dsq;
    return (csq > r0sq && csq < r1sq) ^ invert;
  }
};
inline auto inside_cylindrical_shell_lambda(ParameterInput *pin,
                                            const std::string &block_name) {
  Real x0 = pin->GetOrAddReal(block_name, "x0", 0.0,
                              "x-coordinate of bottom of cylindrical shell");
  Real y0 = pin->GetOrAddReal(block_name, "y0", 0.0,
                              "y-coordinate of bottom of cylindrical shell");
  Real z0 = pin->GetOrAddReal(block_name, "z0", 0.0,
                              "z-coordinate of bottom of cylindrical shell");
  Real x1 = pin->GetOrAddReal(block_name, "x1", 0.0,
                              "x-coordinate of top of cylindrical shell");
  Real y1 = pin->GetOrAddReal(block_name, "y1", 0.0,
                              "y-coordinate of top of cylindrical shell");
  Real z1 = pin->GetOrAddReal(block_name, "z1", 1.0,
                              "z-coordinate of top of cylindrical shell");
  Real r0 = pin->GetOrAddReal(block_name, "inner_radius", 0.0, "Radius of inner surface");
  Real r1 = pin->GetOrAddReal(block_name, "outer_radius", 1.0, "Radius of outer surface");
  bool invert = pin->GetOrAddBoolean(block_name, "invert", false);
  Real r0sq = r0 * r0;
  Real r1sq = r1 * r1;

  Real nx = x1 - x0;
  Real ny = y1 - y0;
  Real nz = z1 - z0;
  Real dsq = nx * nx + ny * ny + nz * nz;

  return cylindrical_shell_mask{x0,   y0, z0, x1, y1,  z1,    r0sq,
                                r1sq, nx, ny, nz, dsq, invert};
}

struct ellipsoid_mask {
  Real x0, y0, z0, asq, bsq, csq;
  bool invert;
  KOKKOS_INLINE_FUNCTION
  bool operator()(const Real x, const Real y, const Real z) const {
    Real dxsq = x - x0;
    dxsq *= dxsq;
    Real dysq = y - y0;
    dysq *= dysq;
    Real dzsq = z - z0;
    dzsq *= dzsq;
    return (dxsq / asq + dysq / bsq + dzsq / csq < 1.0) ^ invert;
  }
};
inline auto inside_ellipsoid_lambda(ParameterInput *pin, const std::string &block_name) {
  Real x0 =
      pin->GetOrAddReal(block_name, "x0", 0.0, "x-coordinate of center of ellipsoid");
  Real y0 =
      pin->GetOrAddReal(block_name, "y0", 0.0, "y-coordinate of center of ellipsoid");
  Real z0 =
      pin->GetOrAddReal(block_name, "z0", 0.0, "z-coordinate of center of ellipsoid");
  Real ax = pin->GetOrAddReal(block_name, "ax", 1.0, "Eccentricity parameter in x");
  Real ay = pin->GetOrAddReal(block_name, "ay", 1.0, "Eccentricity parameter in y");
  Real az = pin->GetOrAddReal(block_name, "az", 1.0, "Eccentricity parameter in z");
  bool invert = pin->GetOrAddBoolean(block_name, "invert", false);
  Real asq = ax * ax;
  Real bsq = ay * ay;
  Real csq = az * az;

  return ellipsoid_mask{x0, y0, z0, asq, bsq, csq, invert};
}

struct ellipsoidal_shell_mask {
  Real x0, y0, z0, a0sq, b0sq, c0sq, a1sq, b1sq, c1sq;
  bool invert;
  KOKKOS_INLINE_FUNCTION
  bool operator()(const Real x, const Real y, const Real z) const {
    Real dxsq = x - x0;
    dxsq *= dxsq;
    Real dysq = y - y0;
    dysq *= dysq;
    Real dzsq = z - z0;
    dzsq *= dzsq;
    Real eq0 = dxsq / a0sq + dysq / b0sq + dzsq / c0sq;
    Real eq1 = dxsq / a1sq + dysq / b1sq + dzsq / c1sq;
    return (eq0 > 1.0 && eq1 < 1.0) ^ invert;
  }
};
inline auto inside_ellipsoidal_shell_lambda(ParameterInput *pin,
                                            const std::string &block_name) {
  Real x0 = pin->GetOrAddReal(block_name, "x0", 0.0,
                              "x-coordinate of center of ellipsoidal shell");
  Real y0 = pin->GetOrAddReal(block_name, "y0", 0.0,
                              "y-coordinate of center of ellipsoidal shell");
  Real z0 = pin->GetOrAddReal(block_name, "z0", 0.0,
                              "z-coordinate of center of ellipsoidal shell");
  Real ax0 = pin->GetOrAddReal(block_name, "inner_ax", 1.0,
                               "Eccentricity parameter of inner surface in x");
  Real ay0 = pin->GetOrAddReal(block_name, "inner_ay", 1.0,
                               "Eccentricity parameter of inner surface in y");
  Real az0 = pin->GetOrAddReal(block_name, "inner_az", 1.0,
                               "Eccentricity parameter of inner surface in z");
  Real ax1 = pin->GetOrAddReal(block_name, "outer_ax", 1.0,
                               "Eccentricity parameter of outer surface in x");
  Real ay1 = pin->GetOrAddReal(block_name, "outer_ay", 1.0,
                               "Eccentricity parameter of outer surface in y");
  Real az1 = pin->GetOrAddReal(block_name, "outer_az", 1.0,
                               "Eccentricity parameter of outer surface in z");
  bool invert = pin->GetOrAddBoolean(block_name, "invert", false);

  Real a0sq = ax0 * ax0;
  Real b0sq = ay0 * ay0;
  Real c0sq = az0 * az0;
  Real a1sq = ax1 * ax1;
  Real b1sq = ay1 * ay1;
  Real c1sq = az1 * az1;

  return ellipsoidal_shell_mask{x0, y0, z0, a0sq, b0sq, c0sq, a1sq, b1sq, c1sq, invert};
}

struct rectangle_mask {
  Real x0, y0, z0, x1, y1, z1;
  bool invert;
  KOKKOS_INLINE_FUNCTION
  bool operator()(const Real x, const Real y, const Real z) const {
    bool xin = (x >= x0 && x <= x1);
    bool yin = (y >= y0 && y <= y1);
    bool zin = (z >= z0 && z <= z1);
    return ((xin && yin) && zin) ^ invert;
  }
};
inline auto inside_rectangle_lambda(ParameterInput *pin, const std::string &block_name) {
  Real x0 =
      pin->GetOrAddReal(block_name, "x0", -1e300, "Left edge. Default is minus infinity");
  Real y0 = pin->GetOrAddReal(block_name, "y0", -1e300,
                              "Bottom edge. Default is minus infinity");
  Real z0 = pin->GetOrAddReal(block_name, "z0", -1e300,
                              "Vertical-bottom edge. Default is minus infinity");
  Real x1 =
      pin->GetOrAddReal(block_name, "x1", 1e300, "Right edge. Default is plus infinity");
  Real y1 =
      pin->GetOrAddReal(block_name, "y1", 1e300, "Top edge. Default is plus infinity");
  Real z1 = pin->GetOrAddReal(block_name, "z1", 1e300,
                              "Vertical-top edge. Default is plus infinity");
  bool invert = pin->GetOrAddBoolean(block_name, "invert", false);

  return rectangle_mask{x0, y0, z0, x1, y1, z1, invert};
}

#define FOREACH_REGION                                                                   \
  REG(background)                                                                        \
  REG(inside_sphere)                                                                     \
  REG(inside_spherical_shell)                                                            \
  REG(inside_cylinder)                                                                   \
  REG(inside_cylindrical_shell)                                                          \
  REG(inside_ellipsoid)                                                                  \
  REG(inside_ellipsoidal_shell)                                                          \
  REG(inside_rectangle)

#define REG(name) mask_func_t name(ParameterInput *pin, const std::string &block_name);
FOREACH_REGION
#undef REG

#define REG(name) using name##_t = decltype(name##_lambda(nullptr, ""));
FOREACH_REGION
#undef REG

#define REG(name) name##_t,
using mask_types =
    parthenon::TypeList<FOREACH_REGION bool // leave this bool alone, it's because of the
                                            // trailing comma and won't hurt anybody
                        >;
#undef REG

#ifdef RIOT_ENABLE_CAD
mask_func_t cad(ParameterInput *pin, const std::string &block_name);
#else
inline mask_func_t cad(ParameterInput *pin, const std::string &block_name) {
  PARTHENON_FAIL("Riot not build with support for CAD initializiation.");
  return mask_func_t{
      [](const sample_positions_t &x) { return std::vector<bool>(x.size(), true); }};
}
#endif

// build a map of region name to mask generator
#define REG(name) {#name, name},
inline std::map<std::string, mask_generator_t> region_mask_map({FOREACH_REGION
#ifdef RIOT_ENABLE_CAD
                                                                {"cad", cad}
#endif
});
#undef REG

// build an enum
#define REG(name) name,
enum class RegionID { FOREACH_REGION };
#undef REG
// and build a map for the enums
#define REG(name) {#name, RegionID::name},
inline std::map<std::string, RegionID> region_enum_map({FOREACH_REGION});
#undef REG

// Runtime -> compile-time dispatch over the mask tuple. Both the invoke path
// (device) and the build path (host) index the tuple by the RegionID that was
// resolved at runtime, so adding a region to FOREACH_REGION extends them
// automatically instead of requiring another hand-written case.

// invoke the mask stored in slot `id` of tuple `t`
KOKKOS_INLINE_FUNCTION
bool eval_mask(const mask_types::types &t, const int id, const Real x, const Real y,
               const Real z) {
  switch (id) {
#define REG(name)                                                                        \
  case static_cast<int>(RegionID::name):                                                 \
    return std::get<static_cast<std::size_t>(RegionID::name)>(t)(x, y, z);
    FOREACH_REGION
#undef REG
  default:
    return true;
  }
}

// construct the mask for `id` from input and store it in slot `id` of tuple `t`
inline void set_mask(mask_types::types &t, const int id, ParameterInput *pin,
                     const std::string &block_name) {
  switch (id) {
#define REG(name)                                                                        \
  case static_cast<int>(RegionID::name):                                                 \
    std::get<static_cast<std::size_t>(RegionID::name)>(t) =                              \
        name##_lambda(pin, block_name);                                                  \
    return;
    FOREACH_REGION
#undef REG
  default:
    PARTHENON_FAIL("Invalid mask type specified.");
  }
}

#undef FOREACH_REGION

} // namespace region_primitives

#endif // REGION_PGEN_REGION_PRIMITIVES_HPP_

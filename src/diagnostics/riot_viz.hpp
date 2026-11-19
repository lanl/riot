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
#ifndef DIAGNOSTICS_RIOT_VIZ_HPP_
#define DIAGNOSTICS_RIOT_VIZ_HPP_
// This file was made in part with generative AI.


#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;
#include <parthenon/driver.hpp>
using namespace parthenon::driver::prelude;

#include <kokkos_types.hpp>
#include <pack/pack_utils.hpp>

#include <spiner/databox.hpp>

#include "riot_pgen/region_primitives.hpp"
#include "riot_utils/ray_trace.hpp"
#include "riot_utils/riot_utils.hpp"

namespace riot_viz {

KOKKOS_INLINE_FUNCTION
std::uint8_t real_to_uint8_t(const Real val) {
  return static_cast<std::uint8_t>(std::clamp(val, 0.0, 1.0) * 255.0 + 0.5);
}

namespace sample {
enum RenderSample { x, y, z, nx, ny, nz, nvalues };
}

enum class VizType { volume, contour, slice, contour_slice };
inline const std::unordered_map<std::string, VizType> VizTypeMap{
    {"volume", VizType::volume},
    {"contour", VizType::contour},
    {"slice", VizType::slice},
    {"contour_slice", VizType::contour_slice}};
enum class GradType { none, mag, log_mag };
inline const std::unordered_map<std::string, GradType> GradTypeMap{
    {"none", GradType::none},
    {"magnitude", GradType::mag},
    {"log_magnitude", GradType::log_mag}};
enum class ScaleType { linear, log };
inline const std::unordered_map<std::string, ScaleType> ScaleTypeMap{
    {"linear", ScaleType::linear}, {"log", ScaleType::log}};

// The names of the options in a map, for the allowed values of an input parameter
template <typename T>
std::vector<std::string> MapKeys(const std::unordered_map<std::string, T> &map) {
  std::vector<std::string> keys;
  for (const auto &[k, v] : map)
    keys.push_back(k);
  return keys;
}

using DataBox = Spiner::DataBox<Real>;

// Ray-march robustness factors. These are deliberately distinct because they
// live at different scales and cannot be interchanged:
//   kMarchTiny     - tiny absolute guard against degenerate denominators and
//                    zero-length steps (also the alpha off-mesh sentinel and
//                    the dump-time comparison tolerance).
//   kAxisStepFrac  - largest step permitted near a coordinate axis, as a
//                    fraction of the cylindrical radius (but no less than the
//                    smallest cell).
//   kContourBisect - bisection iterations locating a contour crossing in a step.
//   kGeomTol       - relative tolerance for placing/testing points on the
//                    domain boundary during host-side camera setup.
//   kBoundaryBand  - relative band for detecting that a host point coincides
//                    with a block's outer radius.
constexpr Real kMarchTiny = 1.e-14;
constexpr Real kAxisStepFrac = 0.1;
constexpr int kContourBisect = 24;
constexpr Real kGeomTol = 1.e-12;
constexpr Real kBoundaryBand = 1.e-6;

inline DataBox MakeNormalizedDataBox(const std::vector<Real> &input) {
  DataBox output(input.size());
  for (int i = 0; i < input.size(); i++)
    output(i) = input[i];
  output.setRange(0, 0, 1, input.size());
  return output;
}

struct PixelVal {
  Real r, g, b, a;
};

// A mask of a viz layer, which is one of the region masks of the problem generator
struct VizMask {
  enum class Type {
    background,
    plane,
    inside_sphere,
    inside_spherical_shell,
    inside_cylinder,
    inside_cylindrical_shell,
    inside_ellipsoid,
    inside_ellipsoidal_shell,
    inside_rectangle
  };
  Type type;
  region_primitives::background_mask background;
  region_primitives::plane_mask plane;
  region_primitives::sphere_mask sphere;
  region_primitives::spherical_shell_mask spherical_shell;
  region_primitives::cylinder_mask cylinder;
  region_primitives::cylindrical_shell_mask cylindrical_shell;
  region_primitives::ellipsoid_mask ellipsoid;
  region_primitives::ellipsoidal_shell_mask ellipsoidal_shell;
  region_primitives::rectangle_mask rectangle;

  KOKKOS_INLINE_FUNCTION
  bool operator()(const Real x, const Real y, const Real z) const {
    switch (type) {
    case Type::background:
      return background(x, y, z);
    case Type::plane:
      return plane(x, y, z);
    case Type::inside_sphere:
      return sphere(x, y, z);
    case Type::inside_spherical_shell:
      return spherical_shell(x, y, z);
    case Type::inside_cylinder:
      return cylinder(x, y, z);
    case Type::inside_cylindrical_shell:
      return cylindrical_shell(x, y, z);
    case Type::inside_ellipsoid:
      return ellipsoid(x, y, z);
    case Type::inside_ellipsoidal_shell:
      return ellipsoidal_shell(x, y, z);
    case Type::inside_rectangle:
      return rectangle(x, y, z);
    }
    return true;
  }

  // The mask at (x, y, z), narrowing [tlo, thi] to the range of t over which it keeps
  // that value along the ray (x, y, z) + t (vx, vy, vz). Masks other than the background
  // and planes shrink it to nothing.
  KOKKOS_INLINE_FUNCTION
  bool span(const Real x, const Real y, const Real z, const Real vx, const Real vy,
            const Real vz, Real &tlo, Real &thi) const {
    if (type == Type::plane) {
      for (int i = 0; i < plane.nplanes; i++) {
        const Real nx = plane.pdata[4 * i];
        const Real ny = plane.pdata[4 * i + 1];
        const Real nz = plane.pdata[4 * i + 2];
        const Real offset = plane.pdata[4 * i + 3];
        const Real dot = nx * x + ny * y + nz * z + offset;
        const Real ndotv = nx * vx + ny * vy + nz * vz;
        if (ndotv == 0.0) continue;
        // the ray crosses the plane at tb, and the side dot >= 0 lies ahead of it when
        // ndotv > 0. Points within roundoff of the plane may land on either side of it,
        // as they do when the mask is evaluated there directly.
        const Real tb = -dot / ndotv;
        if ((dot >= 0.0) == (ndotv > 0.0)) {
          tlo = std::max(tlo, tb);
        } else {
          thi = std::min(thi, tb);
        }
      }
    } else if (type != Type::background) {
      tlo = 1.0;
      thi = -1.0;
    }
    return (*this)(x, y, z);
  }
};

inline VizMask MakeVizMask(ParameterInput *pin, const std::string &block) {
  namespace rpr = region_primitives;
  using Type = VizMask::Type;
  const auto mask_type = pin->GetString(block, "mask_type");
  VizMask m{};
  if (mask_type == "background") {
    m.type = Type::background;
    m.background = rpr::background_lambda(pin, block);
  } else if (mask_type == "plane") {
    m.type = Type::plane;
    m.plane = rpr::plane_lambda(pin, block);
  } else if (mask_type == "inside_sphere") {
    m.type = Type::inside_sphere;
    m.sphere = rpr::inside_sphere_lambda(pin, block);
  } else if (mask_type == "inside_spherical_shell") {
    m.type = Type::inside_spherical_shell;
    m.spherical_shell = rpr::inside_spherical_shell_lambda(pin, block);
  } else if (mask_type == "inside_cylinder") {
    m.type = Type::inside_cylinder;
    m.cylinder = rpr::inside_cylinder_lambda(pin, block);
  } else if (mask_type == "inside_cylindrical_shell") {
    m.type = Type::inside_cylindrical_shell;
    m.cylindrical_shell = rpr::inside_cylindrical_shell_lambda(pin, block);
  } else if (mask_type == "inside_ellipsoid") {
    m.type = Type::inside_ellipsoid;
    m.ellipsoid = rpr::inside_ellipsoid_lambda(pin, block);
  } else if (mask_type == "inside_ellipsoidal_shell") {
    m.type = Type::inside_ellipsoidal_shell;
    m.ellipsoidal_shell = rpr::inside_ellipsoidal_shell_lambda(pin, block);
  } else if (mask_type == "inside_rectangle") {
    m.type = Type::inside_rectangle;
    m.rectangle = rpr::inside_rectangle_lambda(pin, block);
  } else {
    PARTHENON_THROW("mask_type " + mask_type + " in " + block +
                    " is not supported by viz");
  }
  return m;
}

inline std::vector<VizMask> MakeMasks(ParameterInput *pin, const std::string &layer) {
  std::vector<VizMask> masks;
  for (auto &block : pin->GetOrAddVector<std::string>(layer, "masks", {}))
    masks.push_back(MakeVizMask(pin, block));
  return masks;
}
template <typename T>
using vec2d_t = std::vector<std::vector<T>>;
template <typename T>
using vec3d_t = std::vector<std::vector<std::vector<T>>>;

struct RenderingParamsVec {
  vec2d_t<VizType> type;
  vec2d_t<int> pack_idx, alpha_pack_idx;
  vec2d_t<ScaleType> field_scale;
  vec2d_t<Real> range_min, range_max, alpha_range_min, alpha_range_max;
  vec2d_t<GradType> use_grad, alpha_use_grad;
  std::vector<Real> alpha_tol;
  vec2d_t<DataBox> r, g, b, a;
  vec3d_t<Real> contours, rc, gc, bc, ac;
  vec3d_t<Real> slice_x, slice_n;
  vec2d_t<Real> slice_alpha;
  vec3d_t<bool> slice_transparent_outside;
  std::vector<Real> xl, yl, zl, ambient, diffuse;
  vec3d_t<VizMask> masks;

  // Start a camera with no layers in each of the per-layer parameters
  void NewCamera() {
    auto add = [](auto &...v) { (v.emplace_back(), ...); };
    add(type, pack_idx, alpha_pack_idx, field_scale, range_min, range_max,
        alpha_range_min, alpha_range_max, use_grad, alpha_use_grad, r, g, b, a, contours,
        rc, gc, bc, ac, slice_x, slice_n, slice_alpha, slice_transparent_outside, masks);
  }
};

template <typename T>
class VectorView3D {
 public:
  VectorView3D(std::vector<std::vector<std::vector<T>>> &h) {
    size_t outer_size = h.size();
    size_t inner_size = 0;
    size_t total_size = 0;
    for (int o = 0; o < h.size(); o++) {
      inner_size += h[o].size();
      for (int i = 0; i < h[o].size(); i++) {
        total_size += h[o][i].size();
      }
    }

    data = parthenon::ParArray1D<T>("flat3d", total_size);
    inner_offset = parthenon::ParArray1D<int>("vecview3d_inner", inner_size);
    outer_offset = parthenon::ParArray1D<int>("vecview3d_outer", outer_size);

    auto host_data = data.GetHostMirror();
    auto host_inner_offset = inner_offset.GetHostMirror();
    auto host_outer_offset = outer_offset.GetHostMirror();

    int idx = 0;
    int curr_out = 0;
    int curr_in = 0;
    for (int o = 0; o < h.size(); o++) {
      host_outer_offset[o] = curr_out;
      for (int i = 0; i < h[o].size(); i++) {
        host_inner_offset[curr_out + i] = curr_in;
        curr_in += h[o][i].size();
        for (int ii = 0; ii < h[o][i].size(); ii++)
          host_data(idx++) = h[o][i][ii];
      }
      curr_out += h[o].size();
    }

    Kokkos::deep_copy(data, host_data);
    Kokkos::deep_copy(inner_offset, host_inner_offset);
    Kokkos::deep_copy(outer_offset, host_outer_offset);
  }

  KOKKOS_INLINE_FUNCTION
  T &operator()(const int k, const int j, const int i) const {
    return data(inner_offset(outer_offset(k) + j) + i);
  }

 private:
  parthenon::ParArray1D<T> data;
  parthenon::ParArray1D<int> inner_offset;
  parthenon::ParArray1D<int> outer_offset;
};

template <typename T>
class VectorView2D {
 public:
  VectorView2D() = default;
  VectorView2D(std::vector<std::vector<T>> &h) {
    size_t total_size = 0;
    for (int i = 0; i < h.size(); i++) {
      total_size += h[i].size();
    }

    data = parthenon::ParArray1D<T>("vecview2d", total_size);
    offset = parthenon::ParArray1D<int>("vecview2d_offset", h.size());

    auto host_data = data.GetHostMirror();
    auto host_offset = offset.GetHostMirror();

    int idx = 0;
    int curr = 0;
    for (int i = 0; i < h.size(); i++) {
      host_offset(i) = curr;
      for (int j = 0; j < h[i].size(); j++)
        host_data(idx++) = h[i][j];
      curr += h[i].size();
    }

    Kokkos::deep_copy(data, host_data);
    Kokkos::deep_copy(offset, host_offset);
  }

  KOKKOS_INLINE_FUNCTION
  T &operator()(const int j, const int i) const { return data(offset(j) + i); }

 private:
  parthenon::ParArray1D<T> data;
  parthenon::ParArray1D<int> offset;
};

struct RenderingParams {

  explicit RenderingParams(RenderingParamsVec &rv)
      : type(rv.type), pack_idx(rv.pack_idx), alpha_pack_idx(rv.alpha_pack_idx),
        field_scale(rv.field_scale), range_min(rv.range_min), range_max(rv.range_max),
        alpha_range_min(rv.alpha_range_min), alpha_range_max(rv.alpha_range_max),
        use_grad(rv.use_grad), alpha_use_grad(rv.alpha_use_grad),
        alpha_tol(RiotUtils::VectorToDevice(rv.alpha_tol, "alpha_tol")), r(rv.r), g(rv.g),
        b(rv.b), a(rv.a), contours(rv.contours), rc(rv.rc), gc(rv.gc), bc(rv.bc),
        ac(rv.ac), slice_x(rv.slice_x), slice_n(rv.slice_n), slice_alpha(rv.slice_alpha),
        slice_transparent_outside(rv.slice_transparent_outside),
        xl(RiotUtils::VectorToDevice(rv.xl, "xl")),
        yl(RiotUtils::VectorToDevice(rv.yl, "yl")),
        zl(RiotUtils::VectorToDevice(rv.zl, "zl")),
        ambient(RiotUtils::VectorToDevice(rv.ambient, "ambient")),
        diffuse(RiotUtils::VectorToDevice(rv.diffuse, "diffuse")), mask(rv.masks) {
    // the number of layers of each camera, and of contours and masks of each layer
    std::vector<int> nl;
    std::vector<std::vector<int>> ncv, nmv;
    for (int cam = 0; cam < rv.type.size(); cam++) {
      nl.push_back(rv.type[cam].size());
      ncv.emplace_back();
      nmv.emplace_back();
      for (int ilay = 0; ilay < rv.type[cam].size(); ilay++) {
        ncv.back().push_back(rv.contours[cam][ilay].size());
        nmv.back().push_back(rv.masks[cam][ilay].size());
      }
    }
    nlayers = RiotUtils::VectorToDevice(nl, "nlayers");
    ncontours = VectorView2D<int>(ncv);
    nmasks = VectorView2D<int>(nmv);
  }

  // The color of val from the transfer function, with no opacity. A field drawn on a log
  // scale has none where it is not positive.
  KOKKOS_INLINE_FUNCTION
  PixelVal val_to_rgb(const int outer, const int inner, Real val) const {
    if (field_scale(outer, inner) == ScaleType::log) {
      if (val <= 0.0) return {0.0, 0.0, 0.0, 0.0};
      val = std::log10(val);
    }
    Real v = (val - range_min(outer, inner)) /
             (range_max(outer, inner) - range_min(outer, inner));
    v = std::clamp(v, 0.0, 1.0);
    return {r(outer, inner).interpToReal(v), g(outer, inner).interpToReal(v),
            b(outer, inner).interpToReal(v), 0.0};
  }

  // The color of val and the opacity of aval, which is transparent below its range
  KOKKOS_INLINE_FUNCTION
  PixelVal val_to_rgba(const int outer, const int inner, const Real val,
                       const Real aval) const {
    if (field_scale(outer, inner) == ScaleType::log && val <= 0.0)
      return {0.0, 0.0, 0.0, 0.0};
    if (aval < alpha_range_min(outer, inner)) return {0.0, 0.0, 0.0, 0.0};
    PixelVal p = val_to_rgb(outer, inner, val);
    Real av = (aval - alpha_range_min(outer, inner)) /
              (alpha_range_max(outer, inner) - alpha_range_min(outer, inner));
    av = std::clamp(av, 0.0, 1.0);
    p.a = a(outer, inner).interpToReal(av);
    return p;
  }

  // masks, and the range [tlo, thi] of t over which they keep their value along the ray
  // (x, y, z) + t (vx, vy, vz)
  KOKKOS_INLINE_FUNCTION
  bool masks_span(const int cam_id, const int layer_id, const Real x, const Real y,
                  const Real z, const Real vx, const Real vy, const Real vz, Real &tlo,
                  Real &thi) const {
    tlo = -std::numeric_limits<Real>::infinity();
    thi = std::numeric_limits<Real>::infinity();
    for (int m = 0; m < nmasks(cam_id, layer_id); m++) {
      Real lo = -std::numeric_limits<Real>::infinity();
      Real hi = std::numeric_limits<Real>::infinity();
      const bool in = mask(cam_id, layer_id, m).span(x, y, z, vx, vy, vz, lo, hi);
      if (!in) {
        // the layer is masked out for as long as this mask is
        tlo = lo;
        thi = hi;
        return false;
      }
      tlo = std::max(tlo, lo);
      thi = std::min(thi, hi);
    }
    return true;
  }

  VectorView2D<VizType> type;
  VectorView2D<int> pack_idx, alpha_pack_idx;
  VectorView2D<ScaleType> field_scale;
  VectorView2D<Real> range_min, range_max, alpha_range_min, alpha_range_max;
  VectorView2D<GradType> use_grad, alpha_use_grad;
  parthenon::ParArray1D<Real> alpha_tol;
  VectorView2D<DataBox> r, g, b, a;
  VectorView3D<Real> contours, rc, gc, bc, ac;
  VectorView3D<Real> slice_x, slice_n;
  VectorView2D<Real> slice_alpha;
  VectorView3D<bool> slice_transparent_outside;
  parthenon::ParArray1D<Real> xl, yl, zl, ambient, diffuse;
  VectorView3D<VizMask> mask;
  parthenon::ParArray1D<int> nlayers;
  VectorView2D<int> ncontours;
  VectorView2D<int> nmasks;
};

// The coefficients of a trilinear interpolant over a cell, in the fractions wx, wy, wz of
// the way across it: c[0] + c[1] wx + c[2] wy + c[3] wz + c[4] wx wy + c[5] wx wz +
// c[6] wy wz + c[7] wx wy wz
using Coeffs = std::array<Real, 8>;

// The coefficients from the values val(kk, jj, ii) at the corners. Those along an
// inactive direction vanish, since val does not step along it, so they are not gathered.
template <typename F>
KOKKOS_INLINE_FUNCTION void SetCoeffs(const F &val, Coeffs &c, const int dj,
                                      const int dk) {
  c[0] = val(0, 0, 0);
  c[1] = val(0, 0, 1) - c[0];
  if (dj) {
    c[2] = val(0, 1, 0) - c[0];
    c[4] = val(0, 1, 1) - val(0, 0, 1) - val(0, 1, 0) + c[0];
  } else {
    c[2] = 0.0;
    c[4] = 0.0;
  }
  if (dk) {
    c[3] = val(1, 0, 0) - c[0];
    c[5] = val(1, 0, 1) - val(0, 0, 1) - val(1, 0, 0) + c[0];
    c[6] = val(1, 1, 0) - val(0, 1, 0) - val(1, 0, 0) + c[0];
    c[7] = val(1, 1, 1) - val(0, 1, 1) - val(1, 0, 1) - val(1, 1, 0) + val(0, 0, 1) +
           val(0, 1, 0) + val(1, 0, 0) - c[0];
  } else {
    c[3] = 0.0;
    c[5] = 0.0;
    c[6] = 0.0;
    c[7] = 0.0;
  }
}

// The coefficients of the centered differences of val along each direction, in units of
// the cell. Those along an inactive direction vanish, and are left as they were filled.
template <typename F>
KOKKOS_INLINE_FUNCTION void SetGradCoeffs(const F &val, std::array<Coeffs, 3> &g,
                                          const int dj, const int dk) {
  auto diff = [&](const int d) {
    return [&, d](const int kk, const int jj, const int ii) {
      return 0.5 * (val(kk + (d == 2), jj + (d == 1), ii + (d == 0)) -
                    val(kk - (d == 2), jj - (d == 1), ii - (d == 0)));
    };
  };
  SetCoeffs(diff(0), g[0], dj, dk);
  if (dj) SetCoeffs(diff(1), g[1], dj, dk);
  if (dk) SetCoeffs(diff(2), g[2], dj, dk);
}

// The cell the interpolants are built on, and what they give at points on it
struct CellFrame {
  KOKKOS_DEFAULTED_FUNCTION
  CellFrame() = default;
  template <typename Integrator_t>
  KOKKOS_FUNCTION explicit CellFrame(const Integrator_t &composer)
      : inv_dx(1.0 / composer.template dx<X1DIR>()),
        inv_dy(1.0 / composer.template dx<X2DIR>()),
        inv_dz(1.0 / composer.template dx<X3DIR>()), xlo(composer.template xlo<X1DIR>()),
        ylo(composer.template xlo<X2DIR>()), zlo(composer.template xlo<X3DIR>()) {}

  KOKKOS_INLINE_FUNCTION
  Real interp(const Real xt, const Real yt, const Real zt, const Coeffs &c) const {
    const Real wx = (xt - xlo) * inv_dx;
    const Real wy = (yt - ylo) * inv_dy;
    const Real wz = (zt - zlo) * inv_dz;
    return c[0] + wx * c[1] + wy * (c[2] + wx * c[4]) +
           wz * (c[3] + wx * c[5] + wy * (c[6] + wx * c[7]));
  }
  // the gradient from the coefficients of SetGradCoeffs
  KOKKOS_INLINE_FUNCTION
  std::array<Real, 3> gradient(const Real xt, const Real yt, const Real zt,
                               const std::array<Coeffs, 3> &g) const {
    return {interp(xt, yt, zt, g[0]) * inv_dx, interp(xt, yt, zt, g[1]) * inv_dy,
            interp(xt, yt, zt, g[2]) * inv_dz};
  }
  // the magnitude of the gradient of the interpolant itself, over its magnitude
  KOKKOS_INLINE_FUNCTION
  Real log_grad_mag(const Real xt, const Real yt, const Real zt, const Coeffs &c) const {
    const Real wx = (xt - xlo) * inv_dx;
    const Real wy = (yt - ylo) * inv_dy;
    const Real wz = (zt - zlo) * inv_dz;
    const Real gx = c[1] + wy * c[4] + wz * (c[5] + wy * c[7]);
    const Real gy = c[2] + wx * c[4] + wz * (c[6] + wx * c[7]);
    const Real gz = c[3] + wx * c[5] + wy * (c[6] + wx * c[7]);
    const Real f = c[0] + wx * c[1] + wy * (c[2] + wx * c[4]) +
                   wz * (c[3] + wx * c[5] + wy * (c[6] + wx * c[7]));
    return std::sqrt(gx * gx + gy * gy + gz * gz) / std::abs(f);
  }

  Real inv_dx, inv_dy, inv_dz, xlo, ylo, zlo;
};

inline constexpr int max_layers = 20;
static_assert(max_layers <= 32, "Layers that could show in a block are kept as bits");

// Whether layer ilay of camera cam could show in a block with the given nodal bounds. A
// slice also depends on where the ray meets it, which is left to the Composer.
KOKKOS_INLINE_FUNCTION
bool LayerMayShow(const RenderingParams &rp, const int cam, const int ilay, const int b,
                  const parthenon::ParArray2D<Real> &max_value,
                  const parthenon::ParArray2D<Real> &min_value) {
  const Real min_v = min_value(b, rp.pack_idx(cam, ilay));
  const Real max_v = max_value(b, rp.pack_idx(cam, ilay));
  const Real min_av = min_value(b, rp.alpha_pack_idx(cam, ilay));
  const Real max_av = max_value(b, rp.alpha_pack_idx(cam, ilay));
  if (std::isnan(min_v) || std::isnan(min_av)) return false;
  // the bounds are of the fields, so they cannot rule out a layer showing a gradient
  const bool alpha_grad = rp.alpha_use_grad(cam, ilay) != GradType::none;
  if (rp.type(cam, ilay) == VizType::slice) {
    return alpha_grad || !(rp.slice_transparent_outside(cam, ilay, 0) &&
                           max_av < rp.alpha_range_min(cam, ilay));
  } else if (rp.type(cam, ilay) == VizType::contour ||
             rp.type(cam, ilay) == VizType::contour_slice) {
    if (rp.use_grad(cam, ilay) != GradType::none) return true;
    // check if min/max values bound any of the contour values
    for (int i = 0; i < rp.ncontours(cam, ilay); i++) {
      const Real val = rp.contours(cam, ilay, i);
      if ((val - min_v) * (max_v - val) >= 0) return true;
    }
    return false;
  } else if (rp.type(cam, ilay) == VizType::volume) {
    // check if max value is above min alpha (otherwise it's transparent)
    return alpha_grad || max_av > rp.alpha_range_min(cam, ilay);
  }
  PARTHENON_FAIL("Unsupported VizType\n");
  return true;
}

// Where, for each camera, any layer could show. The blocks where one may lie within
// box(cam, 2 * d) <= x_d <= box(cam, 2 * d + 1) in the mesh coordinates x_d, and so within
// the convex hull of that box in Cartesian space: the box itself on Cartesian meshes, the
// cylinder R <= box(cam, 1) between the z planes on cylindrical ones, and the ball
// r <= box(cam, 1) on spherical ones. A ray crosses the hull along one stretch, and before
// and after it there is nothing to draw.
struct ShowVolume {
  parthenon::ParArray2D<Real> box;
  // how far the hull is grown, which keeps its faces off those of the blocks
  Real margin = 0.0;

  // The stretch t0 <= t <= t1 of the ray x + t n in the hull grown by grow, or t0 > t1
  KOKKOS_INLINE_FUNCTION
  void Clip(const int cam, const Real x[3], const Real n[3], const Real grow, Real &t0,
            Real &t1) const {
    constexpr Real big = std::numeric_limits<Real>::max();
    t0 = -big;
    t1 = big;
    auto miss = [&]() {
      t0 = big;
      t1 = -big;
    };
    if (!(box(cam, 0) <= box(cam, 1))) return miss();
    auto slab = [&](const Real x0, const Real n0, const Real lo, const Real hi) {
      if (n0 == 0.0) {
        if (x0 < lo || x0 > hi) miss();
        return;
      }
      const Real ta = (lo - x0) / n0;
      const Real tb = (hi - x0) / n0;
      t0 = std::max(t0, std::min(ta, tb));
      t1 = std::min(t1, std::max(ta, tb));
    };
    // the ball of radius rmax in the first nd components
    auto ball = [&](const int nd, const Real rmax) {
      Real qa = 0.0, qb = 0.0, qc = -rmax * rmax;
      for (int d = 0; d < nd; d++) {
        qa += n[d] * n[d];
        qb += x[d] * n[d];
        qc += x[d] * x[d];
      }
      if (qa == 0.0) {
        if (qc > 0.0) miss();
        return;
      }
      const Real disc = qb * qb - qa * qc;
      if (disc < 0.0) return miss();
      const Real s = std::sqrt(disc);
      t0 = std::max(t0, (-qb - s) / qa);
      t1 = std::min(t1, (-qb + s) / qa);
    };
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      for (int d = 0; d < 3; d++)
        slab(x[d], n[d], box(cam, 2 * d) - grow, box(cam, 2 * d + 1) + grow);
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      ball(2, box(cam, 1) + grow);
      slab(x[2], n[2], box(cam, 2) - grow, box(cam, 3) + grow);
    } else {
      ball(3, box(cam, 1) + grow);
    }
  }

  // Whether the point x is in the hull grown by grow
  KOKKOS_INLINE_FUNCTION
  bool Contains(const int cam, const Real x[3], const Real grow) const {
    if (!(box(cam, 0) <= box(cam, 1))) return false;
    auto in = [&](const Real x0, const int d) {
      return x0 >= box(cam, 2 * d) - grow && x0 <= box(cam, 2 * d + 1) + grow;
    };
    const Real rmax = box(cam, 1) + grow;
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      return in(x[0], 0) && in(x[1], 1) && in(x[2], 2);
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      return x[0] * x[0] + x[1] * x[1] <= rmax * rmax && in(x[2], 1);
    } else {
      return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] <= rmax * rmax;
    }
  }
};

template <typename VarPack_t>
class Composer : public RayTrace::IntegratorBase<VarPack_t> {
 public:
  using Base = RayTrace::IntegratorBase<VarPack_t>;
  using Base::cart_to_coord;
  using Base::context;
  using Base::loop_3d;
  using Base::min_size;
  using Base::on_block;
  using Base::on_mesh;
  using Base::pidx;
  using Base::push_to_block_boundary;
  using Base::sync_cart_to_coord;
  using Base::transform_coord_to_cart;
  using Base::v;
  using Base::x;
  using Base::xc;

  template <typename SwarmPack_t, typename SwarmPackInt_t>
  KOKKOS_FUNCTION Composer(VarPack_t &vp, SwarmPack_t &ps, const int blk, const int pidx,
                           const int ndim, const parthenon::SwarmDeviceContext &context,
                           SwarmPackInt_t &ps_int, const RenderingParams &rp,
                           const parthenon::ParArray1D<parthenon::PackIdx> &idx_map,
                           const parthenon::ParArray2D<uint32_t> &may_show,
                           const ShowVolume &show_volume,
                           const parthenon::ParArray2D<uint8_t> &images)
      : Base(vp, ps, blk, pidx, ndim, context), rp(rp),
        cam_id(ps_int(blk, particles::riot_viz::camera_id(), pidx)),
        pix_id(ps_int(blk, particles::riot_viz::pixel_id(), pidx)),
        r(ps(blk, particles::riot_viz::r(), pidx)),
        g(ps(blk, particles::riot_viz::g(), pidx)),
        b(ps(blk, particles::riot_viz::b(), pidx)),
        a(ps(blk, particles::riot_viz::a(), pidx)), nlayers(rp.nlayers(cam_id)),
        alpha_tol(rp.alpha_tol(cam_id)), idx_map(idx_map), images(images) {
    PARTHENON_REQUIRE(nlayers <= max_layers,
                      "Too many viz layers requested. Must recompile after setting "
                      "max_layers large enough");
    vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    any_slo = std::numeric_limits<Real>::infinity();
    any_shi = -std::numeric_limits<Real>::infinity();
    // Rays step exactly from face to face of the cells, so a layer that cannot show in a
    // block cannot be met in a step through it either
    const uint32_t may = may_show(blk, cam_id);
    int ninteract = 0;
    for (int ilay = 0; ilay < nlayers; ilay++) {
      data_cell[ilay] = -1;
      mask_slo[ilay] = std::numeric_limits<Real>::infinity();
      mask_shi[ilay] = -std::numeric_limits<Real>::infinity();
      interact[ilay] = (may >> ilay) & 1;
      if (interact[ilay] && rp.type(cam_id, ilay) == VizType::slice) {
        // check if intersection of ray with plane is inside block
        interact[ilay] = false;
        const Real ht = slice_distance(ilay);
        if (ht > -kMarchTiny && a < rp.slice_alpha(cam_id, ilay)) {
          auto temp = along(ht);
          interact[ilay] = on_block(cart_to_coord(temp), RayTrace::kFaceEps);
        }
      }
      ninteract += interact[ilay];
      if (interact[ilay] && rp.type(cam_id, ilay) == VizType::volume) any_volume = true;
    }
    if (ninteract == 0) {
      // Rays start in the volume anything could show in, so once out of it they are
      // done, as if they had left the mesh
      if (!show_volume.Contains(cam_id, x.data(), 2.0 * show_volume.margin)) {
        flag_done();
        return;
      }
      push_to_block_boundary();
      check_and_flag_off_mesh();
    }
  }

  // The distance along the ray to the plane of slice ilay, or minus infinity where the
  // ray runs along it
  KOKKOS_INLINE_FUNCTION
  Real slice_distance(const int ilay) const {
    const Real n_dot_v = rp.slice_n(cam_id, ilay, 0) * v[0] +
                         rp.slice_n(cam_id, ilay, 1) * v[1] +
                         rp.slice_n(cam_id, ilay, 2) * v[2];
    if (std::abs(n_dot_v) <= kMarchTiny) return -std::numeric_limits<Real>::infinity();
    const Real xp = rp.slice_x(cam_id, ilay, 0) - x[0];
    const Real yp = rp.slice_x(cam_id, ilay, 1) - x[1];
    const Real zp = rp.slice_x(cam_id, ilay, 2) - x[2];
    const Real n_dot_p = rp.slice_n(cam_id, ilay, 0) * xp +
                         rp.slice_n(cam_id, ilay, 1) * yp +
                         rp.slice_n(cam_id, ilay, 2) * zp;
    return n_dot_p / n_dot_v;
  }

  // The light on a surface at x with normal (nx, ny, nz), which is lit from either side:
  // the ambient light, and the diffuse light falling off with the square of the distance
  // to its source
  KOKKOS_INLINE_FUNCTION
  Real lighting(const Real nx, const Real ny, const Real nz) const {
    Real dlx = rp.xl(cam_id) - x[0];
    Real dly = rp.yl(cam_id) - x[1];
    Real dlz = rp.zl(cam_id) - x[2];
    const Real inv_dist = 1.0 / std::sqrt(dlx * dlx + dly * dly + dlz * dlz);
    dlx *= inv_dist;
    dly *= inv_dist;
    dlz *= inv_dist;
    const Real ndotl = nx * dlx + ny * dly + nz * dlz;
    return rp.ambient(cam_id) +
           rp.diffuse(cam_id) * inv_dist * inv_dist * std::abs(ndotl);
  }

  // Whether the ray is done, flagged so or with nothing more showing through it
  KOKKOS_FORCEINLINE_FUNCTION
  bool composited() const { return a < 0.0 || (1 - a) < alpha_tol; }

  KOKKOS_FORCEINLINE_FUNCTION
  bool is_active() const { return Base::is_active() && !composited(); }

  KOKKOS_FORCEINLINE_FUNCTION
  bool stop() const { return Base::stop() || composited(); }

  KOKKOS_FORCEINLINE_FUNCTION
  void check_and_flag_off_mesh() {
    // this step pushed me off the mesh
    if (!on_mesh()) flag_done();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  void flag_done() {
    // make alpha negative as a sentinel
    // move particle back on mesh so it doesn't get removed
    a = -(a + kMarchTiny);
    this->angles_stale = false;
    xc[0] = 0.5 * (context.x_min_ + context.x_max_);
    xc[1] = 0.5 * (context.y_min_ + context.y_max_);
    xc[2] = 0.5 * (context.z_min_ + context.z_max_);
  }

  // The point x + t v along the ray
  KOKKOS_FORCEINLINE_FUNCTION
  std::array<Real, 3> along(const Real t) const {
    return {x[0] + t * v[0], x[1] + t * v[1], x[2] + t * v[2]};
  }
  // The coordinate x . v of the point xp along the ray
  KOKKOS_FORCEINLINE_FUNCTION
  Real ray_coord(const std::array<Real, 3> &xp) const {
    return xp[0] * v[0] + xp[1] * v[1] + xp[2] * v[2];
  }
  // The masks of layer ilay at the point xp on the ray. Each layer keeps the stretch of
  // the ray, in units of x . v, over which its masks were last found not to change, and
  // only evaluates them again outside of it.
  KOKKOS_FORCEINLINE_FUNCTION
  bool layer_mask(const int ilay, const std::array<Real, 3> &xp) {
    return layer_mask(ilay, xp, ray_coord(xp));
  }
  KOKKOS_FORCEINLINE_FUNCTION
  bool layer_mask(const int ilay, const std::array<Real, 3> &xp, const Real s) {
    if (s >= mask_slo[ilay] && s <= mask_shi[ilay]) return mask_val[ilay];
    Real tlo, thi;
    const bool in =
        rp.masks_span(cam_id, ilay, xp[0], xp[1], xp[2], v[0], v[1], v[2], tlo, thi);
    mask_val[ilay] = in;
    mask_slo[ilay] = s + tlo * vv;
    mask_shi[ilay] = s + thi * vv;
    return in;
  }

  // Whether the masks of any interacting layer let the point xp at s = ray_coord(xp) on
  // the ray through, kept with the stretch of the ray over which none of them change
  KOKKOS_FORCEINLINE_FUNCTION
  bool any_mask(const std::array<Real, 3> &xp, const Real s) {
    if (s >= any_slo && s <= any_shi) return any_val;
    any_val = false;
    any_slo = -std::numeric_limits<Real>::infinity();
    any_shi = std::numeric_limits<Real>::infinity();
    for (int ilay = 0; ilay < nlayers; ilay++) {
      if (!interact[ilay]) continue;
      any_val |= layer_mask(ilay, xp, s);
      any_slo = std::max(any_slo, mask_slo[ilay]);
      any_shi = std::min(any_shi, mask_shi[ilay]);
    }
    return any_val;
  }

  // The current cell, as a key for what has been gathered on it
  KOKKOS_FORCEINLINE_FUNCTION
  long cell_key() const {
    return (static_cast<long>(this->cell_k + 1) << 42) |
           (static_cast<long>(this->cell_j + 1) << 21) | (this->cell_i + 1);
  }

  // Gather the coefficients of the field at pack index var on the current cell with
  // set(val), from its values val(kk, jj, ii) at offsets from the cell. They are left as
  // they are where the field is not on the block.
  template <typename F>
  KOKKOS_FORCEINLINE_FUNCTION void gather(const int var, const F &set) const {
    const int idx = this->vp.GetLowerBound(this->cell_b, idx_map(var));
    if (idx < 0) return;
    set([&](const int kk, const int jj, const int ii) {
      return this->vp(this->cell_b, idx, this->cell_k + this->dk * kk,
                      this->cell_j + this->dj * jj, this->cell_i + ii);
    });
  }

  // Build the interpolants of layer ilay on the current cell, unless they already are
  KOKKOS_FORCEINLINE_FUNCTION
  void load_data(const int ilay) {
    const long cell = cell_key();
    if (frame_cell != cell) {
      frame = CellFrame(*this);
      frame_cell = cell;
    }
    if (data_cell[ilay] == cell) return;
    const int var = rp.pack_idx(cam_id, ilay);
    const int alpha_var = rp.alpha_pack_idx(cam_id, ilay);
    const int dj = this->dj, dk = this->dk;
    A[ilay].fill(0.0);
    gather(var, [&](const auto &val) { SetCoeffs(val, A[ilay], dj, dk); });
    if (alpha_var == var) {
      A_opac[ilay] = A[ilay];
    } else {
      A_opac[ilay].fill(0.0);
      gather(alpha_var, [&](const auto &val) { SetCoeffs(val, A_opac[ilay], dj, dk); });
    }
    data_cell[ilay] = cell;
  }

  // The gradient of the field at pack index var on the current cell, which takes a wider
  // stencil, so it is only gathered where it is used. The frame is that of the cell,
  // since a layer on it was loaded first.
  KOKKOS_FORCEINLINE_FUNCTION
  const std::array<Coeffs, 3> &load_grad(const int var) {
    const long cell = cell_key();
    if (grad_cell == cell && grad_var == var) return Agrad;
    for (auto &c : Agrad)
      c.fill(0.0);
    const int dj = this->dj, dk = this->dk;
    gather(var, [&](const auto &val) { SetGradCoeffs(val, Agrad, dj, dk); });
    grad_cell = cell;
    grad_var = var;
    return Agrad;
  }

  // The field at pack index var with interpolant c at (x1, x2, x3), or its gradient
  // magnitude as g asks
  KOKKOS_FORCEINLINE_FUNCTION
  Real sample(const GradType g, const int var, const Coeffs &c, const Real x1,
              const Real x2, const Real x3) {
    if (g == GradType::mag) {
      auto [gx, gy, gz] = frame.gradient(x1, x2, x3, load_grad(var));
      return std::sqrt(gx * gx + gy * gy + gz * gz);
    } else if (g == GradType::log_mag) {
      return frame.log_grad_mag(x1, x2, x3, c);
    }
    return frame.interp(x1, x2, x3, c);
  }

  // The value and the opacity field of layer ilay, which load_data has built on the cell
  KOKKOS_FORCEINLINE_FUNCTION
  Real get_value(const int ilay, const Real x1, const Real x2, const Real x3) {
    return sample(rp.use_grad(cam_id, ilay), rp.pack_idx(cam_id, ilay), A[ilay], x1, x2,
                  x3);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real get_value(const int ilay) { return get_value(ilay, xc[0], xc[1], xc[2]); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real get_alpha(const int ilay, const Real x1, const Real x2, const Real x3) {
    return sample(rp.alpha_use_grad(cam_id, ilay), rp.alpha_pack_idx(cam_id, ilay),
                  A_opac[ilay], x1, x2, x3);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real get_alpha(const int ilay) { return get_alpha(ilay, xc[0], xc[1], xc[2]); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real distance_to_face() const {
    Real h = std::numeric_limits<Real>::max();
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      loop_3d([&]<int d>() {
        auto ht = (Base::template xlo<d>() + (v[d - 1] > 0) * Base::template dx<d>() -
                   xc[d - 1]) /
                  v[d - 1];
        h = std::min(h, ht);
      });
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      // R(h)^2 = r2 + 2 bq h + aq h^2 along the straight ray
      const Real aq = v[0] * v[0] + v[1] * v[1];
      const Real bq = x[0] * v[0] + x[1] * v[1];
      const Real r2 = x[0] * x[0] + x[1] * x[1];
      if (aq > 0.0) {
        // the axis is not a face
        if (Base::template xlo<X1DIR>() > 0.0)
          h = std::min(h, Base::radial_in(aq, bq, r2, Base::template xlo<X1DIR>()));
        h = std::min(h, Base::radial_out(aq, bq, r2, Base::template xhi<X1DIR>()));
      }
      if (v[2] != 0.0) {
        h = std::min(h, (Base::template xlo<X2DIR>() +
                         (v[2] > 0) * Base::template dx<X2DIR>() - xc[1]) /
                            v[2]);
      }
      Real ht = std::max(kAxisStepFrac * xc[0], min_size());
      h = std::min(h, ht);
    } else {
      // r(h)^2 = r2 + 2 bq h + aq h^2 along the straight ray
      const Real aq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
      const Real bq = x[0] * v[0] + x[1] * v[1] + x[2] * v[2];
      const Real r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
      // the origin is not a face
      if (Base::template xlo<X1DIR>() > 0.0)
        h = std::min(h, Base::radial_in(aq, bq, r2, Base::template xlo<X1DIR>()));
      if (bq < 0.0) {
        // r is not monotonic through the point of closest approach, so stop there to
        // keep a contour the ray goes in and back out of from hiding inside one step.
        // The outer face is only met past it.
        h = std::min(h, -bq / aq);
      } else {
        h = std::min(h, Base::radial_out(aq, bq, r2, Base::template xhi<X1DIR>()));
      }
    }
    return h;
  }

  template <typename SwarmPack_t>
  KOKKOS_FORCEINLINE_FUNCTION void commit(SwarmPack_t &ps, const int blk) {
    Base::commit(ps, blk);
    namespace pvr = particles::riot_viz;
    ps(blk, pvr::r(), pidx) = r;
    ps(blk, pvr::g(), pidx) = g;
    ps(blk, pvr::b(), pidx) = b;
    ps(blk, pvr::a(), pidx) = a;
    if (!is_active()) {
      // the image is opaque, with the color composited along the ray
      images(cam_id, 4 * pix_id + 0) = real_to_uint8_t(r);
      images(cam_id, 4 * pix_id + 1) = real_to_uint8_t(g);
      images(cam_id, 4 * pix_id + 2) = real_to_uint8_t(b);
      images(cam_id, 4 * pix_id + 3) = real_to_uint8_t(1.0);
      context.MarkParticleForRemoval(pidx);
    }
  }

  KOKKOS_FUNCTION
  void advance() {
    const Real s0 = ray_coord(x);
    const bool initial_mask = any_mask(x, s0);
    // step size til next cell face
    auto h = distance_to_face();
    // std::max binds by reference, which device code cannot do to a host constexpr
    h = std::max(h, Real(kMarchTiny));
    if (!initial_mask) {
      // see if I can skip everything else
      auto temp = along(h);
      bool test = any_mask(temp, ray_coord(temp));

      if (!test) {
        // Where the masks are known to let nothing through past the end of the step,
        // nothing is drawn there either, so go on to where they might or out of the block
        if (any_slo <= s0 && any_shi > s0 + h * vv) {
          const Real hj = std::min(this->block_exit_distance(), (any_shi - s0) / vv);
          if (hj > h) temp = along(hj);
        }
        x = temp;
        sync_cart_to_coord();
        Base::snap_to_face();
        check_and_flag_off_mesh();
        return;
      }
    }

    // Loop over layers to initialize data and check for intersections with contours and
    // slices
    int layer_id = -1;
    int contour_id = -1;
    int slice_id = -1;
    for (int ilay = 0; ilay < nlayers; ilay++) {
      if (!interact[ilay]) continue;
      // first distance to slices OR construct the cell interpolant
      if (rp.type(cam_id, ilay) == VizType::slice) {
        const Real ht = slice_distance(ilay);
        if (ht <= h + kMarchTiny && ht > -kMarchTiny &&
            a < rp.slice_alpha(cam_id, ilay)) {
          load_data(ilay);
          auto temp = along(ht);
          Real alpha = 1.0;
          if (rp.slice_transparent_outside(cam_id, ilay, 0)) {
            auto tc = cart_to_coord(temp);
            Real val = get_alpha(ilay, tc[0], tc[1], tc[2]);
            if (val < rp.alpha_range_min(cam_id, ilay)) alpha = 0.0;
          }
          if (alpha > 0.0 && (layer_mask(ilay, x, s0) || layer_mask(ilay, temp))) {
            h = ht;
            slice_id = ilay;
            contour_id = -1;
          }
        }
      }
      // figure out distance to contours
      else if (rp.type(cam_id, ilay) == VizType::contour ||
               rp.type(cam_id, ilay) == VizType::contour_slice) {
        auto final_x = along(h);
        load_data(ilay);
        Real initial = get_value(ilay);
        auto final_coords = cart_to_coord(final_x);
        Real final = get_value(ilay, final_coords[0], final_coords[1], final_coords[2]);
        Real ht_min = 1.0;
        Real h0 = h;
        for (int i = 0; i < rp.ncontours(cam_id, ilay); i++) {
          const Real cval = rp.contours(cam_id, ilay, i);
          const Real f0 = initial - cval;
          if (f0 * (final - cval) >= 0.0) continue;
          Real tlo = 0.0;
          Real ht = 1.0;
          for (int it = 0; it < kContourBisect; it++) {
            const Real tm = 0.5 * (tlo + ht);
            auto xm = along(tm * h0);
            auto cm = cart_to_coord(xm);
            if ((get_value(ilay, cm[0], cm[1], cm[2]) - cval) * f0 > 0.0) {
              tlo = tm;
            } else {
              ht = tm;
            }
          }
          if (ht <= ht_min) {
            final_x[0] = x[0] + ht * h0 * v[0];
            final_x[1] = x[1] + ht * h0 * v[1];
            final_x[2] = x[2] + ht * h0 * v[2];
            if (layer_mask(ilay, final_x)) {
              h = h0 * ht;
              ht_min = ht;
              layer_id = ilay;
              contour_id = i;
              slice_id = -1;
            }
          }
        }
      }
    }
    h = std::max(h, RayTrace::kStepOvershoot * min_size());

    // half step, where only the volume layers are sampled
    x = along(0.5 * h);
    if (any_volume) sync_cart_to_coord();
    Real hint = h;
    if (!initial_mask) hint *= 0.5;

    // get the color and opacity for the step
    Real asum = 0.0;
    Real rsum = 0.0;
    Real gsum = 0.0;
    Real bsum = 0.0;
    for (int ilay = 0; ilay < nlayers; ilay++) {
      if (!interact[ilay]) continue;
      if (rp.type(cam_id, ilay) == VizType::volume) {
        if (layer_mask(ilay, x)) {
          load_data(ilay);
          auto val = get_value(ilay);
          auto alpha_val = get_alpha(ilay);
          auto p = rp.val_to_rgba(cam_id, ilay, val, alpha_val);
          asum += p.a;
          rsum += p.a * p.r;
          gsum += p.a * p.g;
          bsum += p.a * p.b;
        }
      }
    }

    // composite
    if (asum > 0.0) {
      rsum /= asum;
      gsum /= asum;
      bsum /= asum;
      const Real alpha = 1.0 - std::exp(-asum * hint);
      r += (1 - a) * alpha * rsum;
      g += (1 - a) * alpha * gsum;
      b += (1 - a) * alpha * bsum;
      a += (1 - a) * alpha;
    }

    // 2nd half step
    x = along(0.5 * h);
    sync_cart_to_coord();

    if (slice_id >= 0) {
      Real lum = lighting(rp.slice_n(cam_id, slice_id, 0),
                          rp.slice_n(cam_id, slice_id, 1),
                          rp.slice_n(cam_id, slice_id, 2));
      auto val = get_value(slice_id);
      auto alpha_val = get_alpha(slice_id);
      auto p = rp.val_to_rgba(cam_id, slice_id, val, alpha_val);
      Real alpha = rp.slice_alpha(cam_id, slice_id);
      if (alpha_val < rp.alpha_range_min(cam_id, slice_id) &&
          rp.slice_transparent_outside(cam_id, slice_id, 0))
        alpha = 0.0;
      if (alpha_val > rp.alpha_range_max(cam_id, slice_id) &&
          rp.slice_transparent_outside(cam_id, slice_id, 1))
        alpha = 0.0;
      Real max_lum = std::min(std::min(1.0 / p.r, 1.0 / p.g), 1.0 / p.b);
      lum = std::min(lum, max_lum);
      r += (1 - a) * alpha * p.r * lum;
      g += (1 - a) * alpha * p.g * lum;
      b += (1 - a) * alpha * p.b * lum;
      a += (1 - a) * alpha;
    } else if (contour_id >= 0) {
      // The surface is lit from either side, with its normal along the gradient. Where
      // the gradient vanishes it has no normal, and only the ambient light.
      const Real alpha = rp.ac(cam_id, layer_id, contour_id);
      load_data(layer_id);
      auto [dfdx1, dfdx2, dfdx3] =
          frame.gradient(xc[0], xc[1], xc[2], load_grad(rp.pack_idx(cam_id, layer_id)));
      auto [dfdx, dfdy, dfdz] = transform_coord_to_cart(dfdx1, dfdx2, dfdx3);
      const Real grad_mag = std::sqrt(dfdx * dfdx + dfdy * dfdy + dfdz * dfdz);
      const Real inv_grad_mag = grad_mag > 0.0 ? 1.0 / grad_mag : 0.0;
      dfdx *= inv_grad_mag;
      dfdy *= inv_grad_mag;
      dfdz *= inv_grad_mag;
      Real lum = lighting(dfdx, dfdy, dfdz);
      Real cr, cg, cb;
      if (rp.type(cam_id, layer_id) == VizType::contour) {
        cr = rp.rc(cam_id, layer_id, contour_id);
        cg = rp.gc(cam_id, layer_id, contour_id);
        cb = rp.bc(cam_id, layer_id, contour_id);
      } else {
        // colored by field_alpha through the transfer function
        auto p = rp.val_to_rgb(cam_id, layer_id, get_alpha(layer_id));
        cr = p.r;
        cg = p.g;
        cb = p.b;
        Real max_lum = std::min(std::min(1.0 / p.r, 1.0 / p.g), 1.0 / p.b);
        lum = std::min(lum, max_lum);
      }
      r += (1 - a) * alpha * cr * lum;
      g += (1 - a) * alpha * cg * lum;
      b += (1 - a) * alpha * cb * lum;
      a += (1 - a) * alpha;
    }
    // ensure it lands "on" the face in just the right way
    Base::snap_to_face();

    check_and_flag_off_mesh();
  }

  const RenderingParams &rp;
  int cam_id, pix_id;
  // the color along the ray, written back to the swarm in commit
  Real r, g, b, a;
  const int nlayers;
  const Real alpha_tol;
  // the interpolants of each layer's field and opacity field and the cell they were built
  // on, and the frame of the cell they are on
  std::array<Coeffs, max_layers> A, A_opac;
  std::array<long, max_layers> data_cell;
  CellFrame frame;
  long frame_cell = -1;
  // the gradient of the field at pack index grad_var on cell grad_cell
  std::array<Coeffs, 3> Agrad;
  long grad_cell = -1;
  int grad_var = -1;
  // each layer's masks, and the stretch of the ray, in units of x . v, they hold over,
  // and the same for whether any of them is let through
  std::array<bool, max_layers> mask_val;
  std::array<Real, max_layers> mask_slo, mask_shi;
  bool any_val;
  Real any_slo, any_shi;
  Real vv;
  std::array<bool, max_layers> interact;
  // whether any of them is a volume, which is sampled midway along each step
  bool any_volume = false;
  const parthenon::ParArray1D<parthenon::PackIdx> &idx_map;
  const parthenon::ParArray2D<uint8_t> &images;
};

namespace impl {
struct NodalBounds {
  // by block and camera, the layers that could show in the block, as bits
  parthenon::ParArray2D<uint32_t> may_show;
  // by camera, the box of the blocks where any may, laid out as ShowVolume::box
  std::vector<Real> show_box;
};
// Whether the ray of camera point p starts in block b: on the part of the mesh boundary
// the block covers, or in the block, where it starts inside the mesh. Points on a face
// shared by two blocks lie on both.
struct CameraPointOnBlock {
  // the bounds of each block, its xmin and xmax along each direction in turn
  parthenon::ParArray2D<Real> block_box;
  parthenon::ParArray2D<Real> pts;
  // camera id, pixel id, and mesh face of each point
  parthenon::ParArray2D<int> idx;
  // whether each face of each block is on the mesh boundary, from its boundary_flag
  parthenon::ParArray2D<bool> on_boundary;
  // whether each ray starts at its camera point (start_at_point), inside the mesh at
  // start(p, :) in mesh coordinates (start_inside), or not at all
  parthenon::ParArray1D<int> kind;
  parthenon::ParArray2D<Real> start;
  static constexpr int start_none = 0, start_at_point = 1, start_inside = 2;
  // the directions the mesh varies along: all three on Cartesian meshes, (R, z) on
  // cylindrical ones and r on spherical ones
  static constexpr int start_dims = parthenon::IsCoord<parthenon::UniformCartesian>() ? 3
                                    : parthenon::IsCoord<parthenon::UniformCylindrical>()
                                        ? 2
                                        : 1;

  KOKKOS_INLINE_FUNCTION
  bool Inside(const int b, const int d, const Real x) const {
    return x >= block_box(b, 2 * d) && x <= block_box(b, 2 * d + 1);
  }

  KOKKOS_INLINE_FUNCTION
  bool operator()(const int b, const int p) const {
    if (kind(p) == start_none) return false;
    if (kind(p) == start_inside) {
      for (int d = 0; d < start_dims; d++) {
        if (!Inside(b, d, start(p, d))) return false;
      }
      return true;
    }
    const int face = idx(p, 2);
    const int d = face / 2;
    // the block has to reach the mesh face
    if (!on_boundary(b, face)) return false;
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      const Real x[3] = {pts(p, sample::x), pts(p, sample::y), pts(p, sample::z)};
      for (int t = 0; t < 3; t++) {
        if (t != d && !Inside(b, t, x[t])) return false;
      }
      return true;
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      if (face < 2) return Inside(b, 1, pts(p, sample::z));
      if (face < 4) {
        const Real r = std::sqrt(pts(p, sample::x) * pts(p, sample::x) +
                                 pts(p, sample::y) * pts(p, sample::y));
        return Inside(b, 0, r);
      }
      return false;
    } else {
      return true;
    }
  }
};
// the pack descriptors of the real and integer ray variables
inline auto MakeRayDescriptor() {
  namespace prt = particles::ray_tracer;
  namespace pvr = particles::riot_viz;
  return parthenon::MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y,
                                            swarm_position::z, prt::vx, prt::vy, prt::vz,
                                            pvr::r, pvr::g, pvr::b, pvr::a>(
      pvr::particles::name());
}
inline auto MakeRayIntDescriptor() {
  namespace pvr = particles::riot_viz;
  return parthenon::MakeSwarmPackDescriptor<pvr::camera_id, pvr::pixel_id>(
      pvr::particles::name());
}

// The pack indices of fields, in order, on device
template <typename Descriptor>
auto PackIdxView(const Descriptor &desc, const std::vector<std::string> &fields,
                 const std::string &label) {
  auto map = desc.GetMap();
  std::vector<parthenon::PackIdx> pack_idx;
  for (auto &key : fields)
    pack_idx.emplace_back(map.at(key));
  return RiotUtils::VectorToDevice(pack_idx, label);
}

struct TracerInfo {
  using DescriptorType = decltype(MakeRayDescriptor());
  using IntDescriptorType = decltype(MakeRayIntDescriptor());

  TracerInfo(const std::vector<std::string> &fields, RenderingParams &&rp)
      : fields(fields), render_params(std::move(rp)), desc_ps(MakeRayDescriptor()),
        desc_ps_int(MakeRayIntDescriptor()) {}

  void initialize(MeshData<Real> *md) {
    desc.emplace(parthenon::MakePackDescriptor(md, fields));
    pack_idx_view = PackIdxView(*desc, fields, "pack_idx_view");
  }

  auto GetPacks(MeshData<Real> *md) const {
    return std::make_tuple(desc->GetPack(md),
                           desc_ps.GetPack(md),
                           desc_ps_int.GetPack(md));
  }

  std::vector<std::string> fields;
  RenderingParams render_params;
  std::optional<parthenon::SparsePack<>::Descriptor> desc;
  DescriptorType desc_ps;
  IntDescriptorType desc_ps_int;
  parthenon::ParArray1D<parthenon::PackIdx> pack_idx_view;
};
} // namespace impl

struct LayerInfo {
  Spiner::DataBox<Real> rcmap;
  Spiner::DataBox<Real> gcmap;
  Spiner::DataBox<Real> bcmap;
  Spiner::DataBox<Real> acmap;
  std::string label;
  bool colorbar;
  Real min_range, max_range;
  VizType type;
  ScaleType scale_type;
};

struct CameraInfo {
  int nwidth, nheight, colorbar_thickness;
  Real scale;
  std::string filename;
  std::vector<LayerInfo> layer;
  int ncolorbar{0};
};

struct SceneInfo {
  // entry points by domain face for all camera rays
  std::array<std::array<std::vector<Real>, sample::nvalues>, 6> face_pts;
  // corresponding camera and pixel ids to reassemble images from particles
  std::array<std::vector<int>, 6> camera_id, pixel_id;
  // all of the above on device, point by point: the position and direction, and the
  // camera, pixel, and face
  parthenon::ParArray2D<Real> camera_pts;
  parthenon::ParArray2D<int> camera_idx;
  // all the cameras filming my scene
  std::vector<CameraInfo> camera;
  // all the fields contributing to the imaging
  std::vector<std::string> field;
  // all the images from cameras, and their copy on host
  parthenon::ParArray2D<uint8_t> images;
  parthenon::ParArray2D<uint8_t>::HostMirror host_images;
  // vector copies of everything needed to construct the device side RenderingParams
  RenderingParamsVec rendering_params;
};

TaskCollection Render(Mesh *pm, Real time);
TaskStatus DumpImages(Mesh *pm, Real time);
bool TimeToRender(Mesh *pm, Real time);

} // namespace riot_viz

#endif // DIAGNOSTICS_RIOT_VIZ_HPP_

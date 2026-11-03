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

#include <functional>

#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;
#include <parthenon/driver.hpp>
using namespace parthenon::driver::prelude;

#include <kokkos_types.hpp>
#include <pack/pack_utils.hpp>

#include <utils/spiner/spiner/databox.hpp>

#include "riot_pgen/region_primitives.hpp"
#include "riot_utils/ray_trace.hpp"
#include <riot_utils/riot_utils.hpp>

namespace riot_viz {

KOKKOS_INLINE_FUNCTION
uint64_t real_to_uint8_t(const Real val) {
  return static_cast<std::uint8_t>(std::clamp(val, 0.0, 1.0) * 255.0 + 0.5);
}

namespace sample {
enum RenderSample { x, y, z, nx, ny, nz, nvalues };
}

enum class VizType { volume, contour, slice, contour_slice };
inline std::unordered_map<std::string, VizType> VizTypeMap{
    {"volume", VizType::volume},
    {"contour", VizType::contour},
    {"slice", VizType::slice},
    {"contour_slice", VizType::contour_slice}};
enum class GradType { none, mag, log_mag };
inline std::unordered_map<std::string, GradType> GradTypeMap{
    {"none", GradType::none},
    {"magnitude", GradType::mag},
    {"log_magnitude", GradType::log_mag}};
enum class ScaleType { linear, log };
inline std::unordered_map<std::string, ScaleType> ScaleTypeMap{
    {"linear", ScaleType::linear}, {"log", ScaleType::log}};

template <typename T>
T GetType(const std::string &key, const std::unordered_map<std::string, T> &map) {
  auto it = map.find(key);
  if (it != map.end()) return it->second;

  if (parthenon::Globals::my_rank == 0) {
    std::stringstream ss;
    if constexpr (std::is_same_v<T, VizType>) {
      ss << "Invalid key for VizTypeMap.  Available options are:" << std::endl;
    } else if constexpr (std::is_same_v<T, GradType>) {
      ss << "Invalid key for GradTypeMap.  Available options are:" << std::endl;
    } else {
      PARTHENON_FAIL("Unexpected map type.");
    }
    for (const auto &[key, value] : map) {
      ss << "  " << key << std::endl;
    }
    PARTHENON_THROW(ss.str());
  }
  return T();
}

template <typename T>
KOKKOS_INLINE_FUNCTION T clamp(const T &val, const T &min_val, const T &max_val) {
  return std::min(std::max(val, min_val), max_val);
}

namespace vr = particles::riot_viz;
using DataBox = Spiner::DataBox<Real>;

// Ray-march robustness factors. These are deliberately distinct because they
// live at different scales and cannot be interchanged:
//   kMarchTiny     - tiny absolute guard against degenerate denominators and
//                    zero-length steps (also the alpha off-mesh sentinel and
//                    the dump-time comparison tolerance).
//   kAxisStepFrac  - largest step permitted near a coordinate axis, as a
//                    fraction of the cylindrical/spherical radius.
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

inline auto MakeNormalizedDeviceDataBox(const std::vector<Real> &input) {
  return MakeNormalizedDataBox(input).getOnDevice();
}

struct PixelVal {
  Real r, g, b, a;
};

constexpr int region_index(region_primitives::RegionID e) { return static_cast<int>(e); }

using mask_tuple_t = region_primitives::mask_types::types;
inline auto MakeMasks(ParameterInput *pin, const std::string &cam_block) {
  // a std::tuple of functions that generate lambdas
  std::vector<mask_tuple_t> masks_vec;
  auto mask_blocks = pin->GetOrAddVector<std::string>(cam_block, "masks", {});
  std::vector<int> mask_ids;
  for (auto &mask : mask_blocks) {
    masks_vec.push_back(mask_tuple_t());
    auto &masks = masks_vec.back();
    auto mask_type = pin->GetString(mask, "mask_type");
    int id = region_index(region_primitives::region_enum_map.at(mask_type));
    mask_ids.push_back(id);
    region_primitives::set_mask(masks, id, pin, mask);
  }
  return std::make_tuple(masks_vec, mask_ids);
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
  vec3d_t<bool> slice_transparent_outside;
  std::vector<Real> xl, yl, zl, ambient, diffuse;
  vec3d_t<mask_tuple_t> masks;
  vec3d_t<int> mask_id;
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

template <typename T>
class VectorView1D {
 public:
  VectorView1D(std::vector<T> &h) : data("vecview1d", h.size()) {
    auto host_data = data.GetHostMirror();
    for (int i = 0; i < h.size(); i++)
      host_data(i) = h[i];
    Kokkos::deep_copy(data, host_data);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator()(const int i) const { return data(i); }

 private:
  parthenon::ParArray1D<T> data;
};

struct RenderingParams {

  RenderingParams(vec2d_t<VizType> &t, vec2d_t<int> &pack_idx,
                  vec2d_t<int> &alpha_pack_idx, vec2d_t<ScaleType> &field_scale,
                  vec2d_t<Real> &range_min, vec2d_t<Real> &range_max,
                  vec2d_t<Real> &alpha_range_min, vec2d_t<Real> &alpha_range_max,
                  vec2d_t<GradType> &use_grad, vec2d_t<GradType> &alpha_use_grad,
                  std::vector<Real> &alpha_tol, vec2d_t<DataBox> &r, vec2d_t<DataBox> &g,
                  vec2d_t<DataBox> &b, vec2d_t<DataBox> &a, vec3d_t<Real> &contours,
                  vec3d_t<Real> &rc, vec3d_t<Real> &gc, vec3d_t<Real> &bc,
                  vec3d_t<Real> &ac, vec3d_t<Real> &slice_x, vec3d_t<Real> &slice_n,
                  vec3d_t<bool> &slice_transparent_outside, std::vector<Real> &xl,
                  std::vector<Real> &yl, std::vector<Real> &zl,
                  std::vector<Real> &ambient, std::vector<Real> &diffuse,
                  vec3d_t<mask_tuple_t> &mask, vec3d_t<int> &mask_id)
      : type(t), pack_idx(pack_idx), alpha_pack_idx(alpha_pack_idx),
        field_scale(field_scale), range_min(range_min), range_max(range_max),
        alpha_range_min(alpha_range_min), alpha_range_max(alpha_range_max),
        use_grad(use_grad), alpha_use_grad(alpha_use_grad), alpha_tol(alpha_tol), r(r),
        g(g), b(b), a(a), contours(contours), rc(rc), gc(gc), bc(bc), ac(ac),
        slice_x(slice_x), slice_n(slice_n),
        slice_transparent_outside(slice_transparent_outside), xl(xl), yl(yl), zl(zl),
        ambient(ambient), diffuse(diffuse), mask(mask), mask_id(mask_id) {
    nlayers = parthenon::ParArray1D<int>("nlayers", t.size());
    auto host_nlayers = nlayers.GetHostMirror();
    for (int i = 0; i < t.size(); i++)
      host_nlayers(i) = t[i].size();
    Kokkos::deep_copy(nlayers, host_nlayers);

    std::vector<std::vector<int>> ncv;
    for (int cam = 0; cam < t.size(); cam++) {
      std::vector<int> nc;
      for (int ilay = 0; ilay < t[cam].size(); ilay++) {
        nc.push_back(contours[cam][ilay].size());
      }
      ncv.push_back(nc);
    }
    ncontours = VectorView2D<int>(ncv);

    std::vector<std::vector<int>> nmv;
    for (int cam = 0; cam < t.size(); cam++) {
      std::vector<int> nm;
      for (int ilay = 0; ilay < t[cam].size(); ilay++) {
        nm.push_back(mask_id[cam][ilay].size());
      }
      nmv.push_back(nm);
    }
    nmasks = VectorView2D<int>(nmv);
  }

  explicit RenderingParams(RenderingParamsVec &rv)
      : RenderingParams(rv.type, rv.pack_idx, rv.alpha_pack_idx, rv.field_scale,
                        rv.range_min, rv.range_max, rv.alpha_range_min,
                        rv.alpha_range_max, rv.use_grad, rv.alpha_use_grad, rv.alpha_tol,
                        rv.r, rv.g, rv.b, rv.a, rv.contours, rv.rc, rv.gc, rv.bc, rv.ac,
                        rv.slice_x, rv.slice_n, rv.slice_transparent_outside, rv.xl,
                        rv.yl, rv.zl, rv.ambient, rv.diffuse, rv.masks, rv.mask_id) {}

  KOKKOS_INLINE_FUNCTION
  PixelVal val_to_rgba(const int outer, const int inner, Real val,
                       const Real aval) const {
    if (field_scale(outer, inner) == ScaleType::log) {
      if (val <= 0.0) return {0.0, 0.0, 0.0, 0.0};
      val = std::log10(val);
    }
    if (aval < alpha_range_min(outer, inner)) return {0.0, 0.0, 0.0, 0.0};
    Real v = (val - range_min(outer, inner)) /
             (range_max(outer, inner) - range_min(outer, inner));
    Real av = (aval - alpha_range_min(outer, inner)) /
              (alpha_range_max(outer, inner) - alpha_range_min(outer, inner));
    v = clamp(v, 0.0, 1.0);
    av = clamp(av, 0.0, 1.0);
    return {r(outer, inner).interpToReal(v), g(outer, inner).interpToReal(v),
            b(outer, inner).interpToReal(v), a(outer, inner).interpToReal(av)};
  }

  KOKKOS_INLINE_FUNCTION
  bool masks(const int cam_id, const int layer_id, const Real x, const Real y,
             const Real z) const {
    bool in = true;
    for (int m = 0; m < nmasks(cam_id, layer_id); m++) {
      auto &t = mask(cam_id, layer_id, m);
      in &= region_primitives::eval_mask(t, mask_id(cam_id, layer_id, m), x, y, z);
      if (!in) return false;
    }
    return true;
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
      const bool in = region_primitives::eval_mask_span(
          mask(cam_id, layer_id, m), mask_id(cam_id, layer_id, m), x, y, z, vx, vy, vz,
          lo, hi);
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
  VectorView1D<Real> alpha_tol;
  VectorView2D<DataBox> r, g, b, a;
  VectorView3D<Real> contours, rc, gc, bc, ac;
  VectorView3D<Real> slice_x, slice_n;
  VectorView3D<bool> slice_transparent_outside;
  VectorView1D<Real> xl, yl, zl, ambient, diffuse;
  VectorView3D<mask_tuple_t> mask;
  VectorView3D<int> mask_id;
  parthenon::ParArray1D<int> nlayers;
  VectorView2D<int> ncontours;
  VectorView2D<int> nmasks;
};

class VizData {
 public:
  KOKKOS_DEFAULTED_FUNCTION
  VizData() = default;
  template <typename Integrator_t>
  KOKKOS_FUNCTION VizData(const RenderingParams &r, const int ilay, Integrator_t composer,
                          const bool with_grad)
      : inv_dx(1.0 / composer->template dx<X1DIR>()),
        inv_dy(1.0 / composer->template dx<X2DIR>()),
        inv_dz(1.0 / composer->template dx<X3DIR>()),
        xlo(composer->template xlo<X1DIR>()), ylo(composer->template xlo<X2DIR>()),
        zlo(composer->template xlo<X3DIR>()) {
    A.fill(0.0);
    A_opac.fill(0.0);
    if (with_grad) {
      for (auto &c : Agrad)
        c.fill(0.0);
    }
    if (ilay < r.nlayers(composer->cam_id)) {
      auto vidx = composer->idx_map(r.pack_idx(composer->cam_id, ilay));
      int idx = composer->vp.GetLowerBound(composer->cell_b, vidx);
      if (idx >= 0) {
        auto val = [&](const int kk, const int jj, const int ii) {
          return composer->vp(composer->cell_b, idx, composer->cell_k + composer->dk * kk,
                              composer->cell_j + composer->dj * jj,
                              composer->cell_i + ii);
        };
        set_coeffs(val, A, composer->dj, composer->dk);
        // the gradient takes a wider stencil, so it is only gathered when it will be used
        if (with_grad) set_grad_coeffs(val);
      }
      if (r.alpha_pack_idx(composer->cam_id, ilay) ==
          r.pack_idx(composer->cam_id, ilay)) {
        A_opac = A;
      } else {
        vidx = composer->idx_map(r.alpha_pack_idx(composer->cam_id, ilay));
        idx = composer->vp.GetLowerBound(composer->cell_b, vidx);
        if (idx >= 0) {
          auto val = [&](const int kk, const int jj, const int ii) {
            return composer->vp(
                composer->cell_b, idx, composer->cell_k + composer->dk * kk,
                composer->cell_j + composer->dj * jj, composer->cell_i + ii);
          };
          set_coeffs(val, A_opac, composer->dj, composer->dk);
        }
      }
    }
  }
  // sample the color field (A) or opacity field (A_opac), applying the
  // requested gradient transform. All three GradType branches live in
  // sample_(), so value() and alpha() stay in sync by construction.
  KOKKOS_INLINE_FUNCTION
  Real value(const GradType g, const Real xt, const Real yt, const Real zt) const {
    return sample_(g, xt, yt, zt, A);
  }
  KOKKOS_INLINE_FUNCTION
  Real alpha(const GradType g, const Real xt, const Real yt, const Real zt) const {
    return sample_(g, xt, yt, zt, A_opac);
  }
  KOKKOS_INLINE_FUNCTION
  auto grad_val(const Real xt, const Real yt, const Real zt) const {
    return gradient_(xt, yt, zt);
  }

 private:
  // left uninitialized by default, since a Composer holds max_layers of these and only
  // fills the ones it samples
  std::array<Real, 8> A; // 0 A0, 1 Ax, 2 Ay, 3 Az, 4 Axy, 5 Axz, 6 Ayz, 7 Axyz;
  std::array<std::array<Real, 8>, 3> Agrad;
  std::array<Real, 8> A_opac;
  KOKKOS_INLINE_FUNCTION
  Real sample_(const GradType g, const Real xt, const Real yt, const Real zt,
               const std::array<Real, 8> &c) const {
    if (g == GradType::mag) {
      auto [gx, gy, gz] = gradient_(xt, yt, zt);
      return std::sqrt(gx * gx + gy * gy + gz * gz);
    } else if (g == GradType::log_mag) {
      return log_grad_mag_(xt, yt, zt, c);
    }
    return interp_(xt, yt, zt, c);
  }
  Real inv_dx, inv_dy, inv_dz, xlo, ylo, zlo;
  // The coefficients along an inactive direction vanish, since val does not step along
  // it, so they are not gathered
  template <typename F>
  KOKKOS_INLINE_FUNCTION void set_coeffs(const F &val, std::array<Real, 8> &c,
                                         const int dj, const int dk) {
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
  template <typename F>
  KOKKOS_INLINE_FUNCTION void set_grad_coeffs(const F &val) {
    // d/dx
    Real v000 = 0.5 * (val(0, 0, 1) - val(0, 0, -1));
    Real v001 = 0.5 * (val(0, 0, 2) - val(0, 0, 0));
    Real v010 = 0.5 * (val(0, 1, 1) - val(0, 1, -1));
    Real v100 = 0.5 * (val(1, 0, 1) - val(1, 0, -1));
    Real v011 = 0.5 * (val(0, 1, 2) - val(0, 1, 0));
    Real v101 = 0.5 * (val(1, 0, 2) - val(1, 0, 0));
    Real v110 = 0.5 * (val(1, 1, 1) - val(1, 1, -1));
    Real v111 = 0.5 * (val(1, 1, 2) - val(1, 1, 0));
    Agrad[0][0] = v000;
    Agrad[0][1] = v001 - v000;
    Agrad[0][2] = v010 - v000;
    Agrad[0][3] = v100 - v000;
    Agrad[0][4] = v011 - v001 - v010 + v000;
    Agrad[0][5] = v101 - v001 - v100 + v000;
    Agrad[0][6] = v110 - v010 - v100 + v000;
    Agrad[0][7] = v111 - v011 - v101 - v110 + v001 + v010 + v100 - v000;
    // d/dy
    v000 = 0.5 * (val(0, 1, 0) - val(0, -1, 0));
    v001 = 0.5 * (val(0, 1, 1) - val(0, -1, 1));
    v010 = 0.5 * (val(0, 2, 0) - val(0, 0, 0));
    v100 = 0.5 * (val(1, 1, 0) - val(1, -1, 0));
    v011 = 0.5 * (val(0, 2, 1) - val(0, 0, 1));
    v101 = 0.5 * (val(1, 1, 1) - val(1, -1, 1));
    v110 = 0.5 * (val(1, 2, 0) - val(1, 0, 0));
    v111 = 0.5 * (val(1, 2, 1) - val(1, 0, 1));
    Agrad[1][0] = v000;
    Agrad[1][1] = v001 - v000;
    Agrad[1][2] = v010 - v000;
    Agrad[1][3] = v100 - v000;
    Agrad[1][4] = v011 - v001 - v010 + v000;
    Agrad[1][5] = v101 - v001 - v100 + v000;
    Agrad[1][6] = v110 - v010 - v100 + v000;
    Agrad[1][7] = v111 - v011 - v101 - v110 + v001 + v010 + v100 - v000;
    // d/dz
    v000 = 0.5 * (val(1, 0, 0) - val(-1, 0, 0));
    v001 = 0.5 * (val(1, 0, 1) - val(-1, 0, 1));
    v010 = 0.5 * (val(1, 1, 0) - val(-1, 1, 0));
    v100 = 0.5 * (val(2, 0, 0) - val(0, 0, 0));
    v011 = 0.5 * (val(1, 1, 1) - val(-1, 1, 1));
    v101 = 0.5 * (val(2, 0, 1) - val(0, 0, 1));
    v110 = 0.5 * (val(2, 1, 0) - val(0, 1, 0));
    v111 = 0.5 * (val(2, 1, 1) - val(0, 1, 1));
    Agrad[2][0] = v000;
    Agrad[2][1] = v001 - v000;
    Agrad[2][2] = v010 - v000;
    Agrad[2][3] = v100 - v000;
    Agrad[2][4] = v011 - v001 - v010 + v000;
    Agrad[2][5] = v101 - v001 - v100 + v000;
    Agrad[2][6] = v110 - v010 - v100 + v000;
    Agrad[2][7] = v111 - v011 - v101 - v110 + v001 + v010 + v100 - v000;
  }
  KOKKOS_INLINE_FUNCTION
  Real interp_(const Real xt, const Real yt, const Real zt,
               const std::array<Real, 8> &c) const {
    const Real wx = (xt - xlo) * inv_dx;
    const Real wy = (yt - ylo) * inv_dy;
    const Real wz = (zt - zlo) * inv_dz;
    return c[0] + wx * c[1] + wy * (c[2] + wx * c[4]) +
           wz * (c[3] + wx * c[5] + wy * (c[6] + wx * c[7]));
  };
  KOKKOS_INLINE_FUNCTION
  std::array<Real, 3> gradient_(const Real xt, const Real yt, const Real zt) const {
    return {interp_(xt, yt, zt, Agrad[0]) * inv_dx,
            interp_(xt, yt, zt, Agrad[1]) * inv_dy,
            interp_(xt, yt, zt, Agrad[2]) * inv_dz};
    // return {(c[1] + wy * c[4] + wz * c[5]) * inv_dx,
    //(c[2] + wx * c[4] + wz * c[6]) * inv_dy,
    //(c[3] + wx * c[5] + wy * c[6]) * inv_dz};
    // return {(c[1] + wy * c[4] + wz * (c[5] + wy * c[7])) * inv_dx,
    //(c[2] + wx * c[4] + wz * (c[6] + wx * c[7])) * inv_dy,
    //(c[3] + wx * c[5] + wy * (c[6] + wx * c[7])) * inv_dz};
  };
  KOKKOS_INLINE_FUNCTION
  Real log_grad_mag_(const Real xt, const Real yt, const Real zt,
                     const std::array<Real, 8> &c) const {
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
  if (rp.type(cam, ilay) == VizType::slice) {
    return !(rp.slice_transparent_outside(cam, ilay, 0) &&
             max_av < rp.alpha_range_min(cam, ilay));
  } else if (rp.type(cam, ilay) == VizType::contour ||
             rp.type(cam, ilay) == VizType::contour_slice) {
    // check if min/max values bound any of the contour values
    for (int i = 0; i < rp.ncontours(cam, ilay); i++) {
      const Real val = rp.contours(cam, ilay, i);
      if ((val - min_v) * (max_v - val) >= 0) return true;
    }
    return false;
  } else if (rp.type(cam, ilay) == VizType::volume) {
    // check if max value is above min alpha (otherwise it's transparent)
    return max_av > rp.alpha_range_min(cam, ilay);
  }
  PARTHENON_FAIL("Unsupported VizType\n");
  return true;
}

// Rays step exactly from face to face of Cartesian cells, of 2D cylindrical (R, z) cells
// and of 1D spherical shells, so a layer that cannot show in a block cannot be met in a
// step through it either
KOKKOS_INLINE_FUNCTION
bool CullsBlocks(const int ndim) {
  return parthenon::IsCoord<parthenon::UniformCartesian>() ||
         (parthenon::IsCoord<parthenon::UniformCylindrical>() && ndim == 2) ||
         (parthenon::IsCoord<parthenon::UniformSpherical>() && ndim == 1);
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
  using RayTrace::IntegratorBase<VarPack_t>::loop_3d;
  using RayTrace::IntegratorBase<VarPack_t>::loop_2d;
  using RayTrace::IntegratorBase<VarPack_t>::xlo;
  using RayTrace::IntegratorBase<VarPack_t>::dx;
  using RayTrace::IntegratorBase<VarPack_t>::min_size;
  using RayTrace::IntegratorBase<VarPack_t>::x;
  using RayTrace::IntegratorBase<VarPack_t>::xc;
  using RayTrace::IntegratorBase<VarPack_t>::v;
  using RayTrace::IntegratorBase<VarPack_t>::on_block;
  using RayTrace::IntegratorBase<VarPack_t>::on_mesh;
  using RayTrace::IntegratorBase<VarPack_t>::cart_to_coord;
  using RayTrace::IntegratorBase<VarPack_t>::coord_to_cart;
  using RayTrace::IntegratorBase<VarPack_t>::sync_cart_to_coord;
  using RayTrace::IntegratorBase<VarPack_t>::transform_coord_to_cart;
  using RayTrace::IntegratorBase<VarPack_t>::bounds;
  using RayTrace::IntegratorBase<VarPack_t>::pidx;
  using RayTrace::IntegratorBase<VarPack_t>::push_to_block_boundary;

  template <typename Pack_t>
  KOKKOS_INLINE_FUNCTION static bool
  Filter(Pack_t &ps, const int pidx, const RenderingParams &rp,
         const parthenon::ParArray1D<parthenon::PackIdx> &idx_map,
         const parthenon::ParArray2D<uint32_t> &may_show) {
    const Real a = ps(vr::a(), pidx);
    if (a < 0.0) return false;
    int camera_id = ps(vr::camera_id(), pidx);
    if ((1 - a) < rp.alpha_tol(camera_id)) return false;
    return true;
  }

  template <typename Pack_t>
  KOKKOS_FUNCTION Composer(VarPack_t &vp, Pack_t &ps, const int blk, const int pidx,
                           const int ndim, const RayTrace::BlockBounds &bounds,
                           const std::array<Real, 3> *xcart, const RenderingParams &rp,
                           const parthenon::ParArray1D<parthenon::PackIdx> &idx_map,
                           const parthenon::ParArray2D<uint32_t> &may_show,
                           const ShowVolume &show_volume,
                           const parthenon::ParArrayND<uint8_t> &images)
      : RayTrace::IntegratorBase<VarPack_t>(vp, ps, blk, pidx, ndim, bounds, xcart),
        rp(rp),
        cam_id(ps(vr::camera_id(), pidx)), pix_id(ps(vr::pixel_id(), pidx)),
        r(ps(vr::r(), pidx)), g(ps(vr::g(), pidx)), b(ps(vr::b(), pidx)),
        a(ps(vr::a(), pidx)), idx_map(idx_map), images(images) {
    PARTHENON_REQUIRE(rp.nlayers(cam_id) <= max_layers,
                      "Too many viz layers requested. Must recompile after setting "
                      "max_layers large enough");
    // if (!is_active()) return;
    vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    any_slo = std::numeric_limits<Real>::infinity();
    any_shi = -std::numeric_limits<Real>::infinity();
    const bool cull = CullsBlocks(ndim);
    const uint32_t may = cull ? may_show(blk, cam_id) : ~uint32_t(0);
    int ninteract = 0;
    for (int ilay = 0; ilay < rp.nlayers(cam_id); ilay++) {
      data_cell[ilay] = -1;
      mask_slo[ilay] = std::numeric_limits<Real>::infinity();
      mask_shi[ilay] = -std::numeric_limits<Real>::infinity();
      interact[ilay] = (may >> ilay) & 1;
      if (cull && interact[ilay] && rp.type(cam_id, ilay) == VizType::slice) {
        // check if intersection of ray with plane is inside block
        interact[ilay] = false;
        Real n_dot_v = rp.slice_n(cam_id, ilay, 0) * v[0] +
                       rp.slice_n(cam_id, ilay, 1) * v[1] +
                       rp.slice_n(cam_id, ilay, 2) * v[2];
        if (std::abs(n_dot_v) > kMarchTiny) {
          Real xp = rp.slice_x(cam_id, ilay, 0) - x[0];
          Real yp = rp.slice_x(cam_id, ilay, 1) - x[1];
          Real zp = rp.slice_x(cam_id, ilay, 2) - x[2];
          Real n_dot_p = rp.slice_n(cam_id, ilay, 0) * xp +
                         rp.slice_n(cam_id, ilay, 1) * yp +
                         rp.slice_n(cam_id, ilay, 2) * zp;
          Real ht = n_dot_p / n_dot_v;
          if (ht > -kMarchTiny && a < rp.ac(cam_id, ilay, 0)) {
            std::array<Real, 3> temp;
            for (int d = 0; d < 3; d++)
              temp[d] = x[d] + ht * v[d];
            interact[ilay] = on_block_fuzzy(temp);
          }
        }
      }
      ninteract += interact[ilay];
    }
    if (cull) {
      if (ninteract == 0) {
        // Rays start in the volume anything could show in, so once out of it they are
        // done, as if they had left the mesh
        if (!show_volume.Contains(cam_id, x.data(), 2.0 * show_volume.margin)) {
          flag_done();
          return;
        }
        push_to_block_boundary();
        check_and_flag_off_mesh();
        // commit(ps, blk);
      }
    }
  }

  // Whether the Cartesian point xp is in the block, to within kFaceEps, in the
  // directions on_block checks
  KOKKOS_INLINE_FUNCTION
  bool on_block_fuzzy(std::array<Real, 3> &xp) {
    constexpr Real tol = RayTrace::kFaceEps;
    const auto xb = cart_to_coord(xp);
    if (xb[0] < bounds.x_min - tol || xb[0] > bounds.x_max + tol) return false;
    if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) return true;
    if (xb[1] < bounds.y_min - tol || xb[1] > bounds.y_max + tol) return false;
    if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) return true;
    if (xb[2] < bounds.z_min - tol || xb[2] > bounds.z_max + tol) return false;
    return true;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  bool is_active() const {
    if (!RayTrace::IntegratorBase<VarPack_t>::is_active()) return false;
    if (a < 0.0) return false;
    if ((1 - a) < rp.alpha_tol(cam_id)) return false;
    return true;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  bool stop() const {
    if (RayTrace::IntegratorBase<VarPack_t>::stop()) return true;
    if (a < 0.0) return true;
    if ((1 - a) < rp.alpha_tol(cam_id)) return true;
    return false;
  }

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
    xc[0] = 0.5 * (bounds.x_min + bounds.x_max);
    xc[1] = 0.5 * (bounds.y_min + bounds.y_max);
    xc[2] = 0.5 * (bounds.z_min + bounds.z_max);
  }

  // The masks of layer ilay at the point xp on the ray. Each layer keeps the stretch of
  // the ray, in units of x . v, over which its masks were last found not to change, and
  // only evaluates them again outside of it.
  KOKKOS_FORCEINLINE_FUNCTION
  Real ray_coord(const std::array<Real, 3> &xp) const {
    return xp[0] * v[0] + xp[1] * v[1] + xp[2] * v[2];
  }
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
    for (int ilay = 0; ilay < rp.nlayers(cam_id); ilay++) {
      if (!interact[ilay]) continue;
      any_val |= layer_mask(ilay, xp, s);
      any_slo = std::max(any_slo, mask_slo[ilay]);
      any_shi = std::min(any_shi, mask_shi[ilay]);
    }
    return any_val;
  }

  // Build the interpolant of layer ilay on the current cell, unless it already is. The
  // gradient is gathered when sampling needs it or with_grad asks for it.
  KOKKOS_FORCEINLINE_FUNCTION
  void load_data(const int ilay, const bool with_grad = false) {
    const long cell = (static_cast<long>(this->cell_k + 1) << 42) |
                      (static_cast<long>(this->cell_j + 1) << 21) | (this->cell_i + 1);
    if (data_cell[ilay] == cell && (data_grad[ilay] || !with_grad)) return;
    const bool grad = with_grad || rp.use_grad(cam_id, ilay) == GradType::mag ||
                      rp.alpha_use_grad(cam_id, ilay) == GradType::mag;
    data[ilay] = VizData(rp, ilay, this, grad);
    data_cell[ilay] = cell;
    data_grad[ilay] = grad;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Real get_value(const int ilay, const Real x1, const Real x2, const Real x3) const {
    return data[ilay].value(rp.use_grad(cam_id, ilay), x1, x2, x3);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real get_value(const int ilay) const { return get_value(ilay, xc[0], xc[1], xc[2]); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real get_alpha(const int ilay, const Real x1, const Real x2, const Real x3) const {
    return data[ilay].alpha(rp.alpha_use_grad(cam_id, ilay), x1, x2, x3);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real get_alpha(const int ilay) const { return get_alpha(ilay, xc[0], xc[1], xc[2]); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real distance_to_face() const {
    Real h = std::numeric_limits<Real>::max();
    if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
      loop_3d([&]<int d>() {
        auto ht =
            (RayTrace::IntegratorBase<VarPack_t>::template xlo<d>() +
             (v[d - 1] > 0) * RayTrace::IntegratorBase<VarPack_t>::template dx<d>() -
             xc[d - 1]) /
            v[d - 1];
        h = std::min(h, ht);
      });
    } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
      using Base = RayTrace::IntegratorBase<VarPack_t>;
      // R(h)^2 = r2 + bq h + aq h^2 along the straight ray
      const Real aq = v[0] * v[0] + v[1] * v[1];
      const Real bq = 2.0 * (x[0] * v[0] + x[1] * v[1]);
      const Real r2 = x[0] * x[0] + x[1] * x[1];
      if (aq > 0.0) {
        const Real rlo = Base::template xlo<X1DIR>();
        const Real rhi = Base::template xhi<X1DIR>();
        const Real disc_lo = bq * bq - 4.0 * aq * (r2 - rlo * rlo);
        if (bq < 0.0 && disc_lo >= 0.0) {
          h = std::min(h, 2.0 * (r2 - rlo * rlo) / (-bq + std::sqrt(disc_lo)));
        }
        const Real sq_hi =
            std::sqrt(std::max(bq * bq - 4.0 * aq * (r2 - rhi * rhi), 0.0));
        h = std::min(h, bq > 0.0 ? 2.0 * (rhi * rhi - r2) / (bq + sq_hi)
                                 : (sq_hi - bq) / (2.0 * aq));
      }
      if (v[2] != 0.0) {
        h = std::min(h, (Base::template xlo<X2DIR>() +
                         (v[2] > 0) * Base::template dx<X2DIR>() - xc[1]) /
                            v[2]);
      }
      Real ht = std::max(kAxisStepFrac * xc[0], min_size());
      h = std::min(h, ht);
    } else {
      using Base = RayTrace::IntegratorBase<VarPack_t>;
      // r(h)^2 = r2 + bq h + aq h^2 along the straight ray
      const Real aq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
      const Real bq = 2.0 * (x[0] * v[0] + x[1] * v[1] + x[2] * v[2]);
      const Real r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
      const Real rlo = Base::template xlo<X1DIR>();
      const Real rhi = Base::template xhi<X1DIR>();
      const Real disc_lo = bq * bq - 4.0 * aq * (r2 - rlo * rlo);
      if (bq < 0.0 && disc_lo >= 0.0) {
        h = std::min(h, 2.0 * (r2 - rlo * rlo) / (-bq + std::sqrt(disc_lo)));
      }
      if (this->ndim == 1) {
        if (bq < 0.0) {
          // r is not monotonic through the point of closest approach, so stop there to
          // keep a contour the ray goes in and back out of from hiding inside one step.
          // The outer face is only met past it.
          h = std::min(h, -0.5 * bq / aq);
        } else {
          const Real sq_hi =
              std::sqrt(std::max(bq * bq - 4.0 * aq * (r2 - rhi * rhi), 0.0));
          h = std::min(h, bq > 0.0 ? 2.0 * (rhi * rhi - r2) / (bq + sq_hi)
                                   : (sq_hi - bq) / (2.0 * aq));
        }
      } else {
        const Real sq_hi = std::sqrt(std::max(bq * bq - 4.0 * aq * (r2 - rhi * rhi), 0.0));
        h = std::min(h, bq > 0.0 ? 2.0 * (rhi * rhi - r2) / (bq + sq_hi)
                                 : (sq_hi - bq) / (2.0 * aq));
        // the angular faces are not tracked, so keep the steps short near the axis
        const Real rcyl = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        h = std::min(h, std::max(kAxisStepFrac * rcyl, min_size()));
      }
    }
    return h;
  }

  template <typename Pack_t>
  KOKKOS_FORCEINLINE_FUNCTION void commit(Pack_t &ps, const bool full = true) {
    RayTrace::IntegratorBase<VarPack_t>::commit(ps, full);
    if (!is_active()) {
      Real alpha = std::abs(a);
      // replace this if check with rp.flatten
      if (true) alpha = 1.0;
      Real red = (alpha > 0.0) ? r / alpha : 0.0;
      Real green = (alpha > 0.0) ? g / alpha : 0.0;
      Real blue = (alpha > 0.0) ? b / alpha : 0.0;

      images(cam_id, 4 * pix_id + 0) = real_to_uint8_t(red);
      images(cam_id, 4 * pix_id + 1) = real_to_uint8_t(green);
      images(cam_id, 4 * pix_id + 2) = real_to_uint8_t(blue);
      images(cam_id, 4 * pix_id + 3) = real_to_uint8_t(alpha);
      RayTrace::IntegratorBase<VarPack_t>::remove(ps);
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
      std::array<Real, 3> temp;
      for (int d = 0; d < 3; d++)
        temp[d] = x[d] + h * v[d];
      bool test = any_mask(temp, ray_coord(temp));

      if (!test) {
        // Where the masks are known to let nothing through past the end of the step,
        // nothing is drawn there either, so go on to where they might or out of the block
        if (any_slo <= s0 && any_shi > s0 + h * vv) {
          const Real hb = this->block_exit_distance();
          if (hb > h) {
            const Real hj = std::min(hb, (any_shi - s0) / vv);
            if (hj > h) {
              for (int d = 0; d < 3; d++)
                temp[d] = x[d] + hj * v[d];
            }
          }
        }
        for (int d = 0; d < 3; d++)
          x[d] = temp[d];
        sync_cart_to_coord();
        RayTrace::IntegratorBase<VarPack_t>::snap_to_face();
        check_and_flag_off_mesh();
        return;
      }
    }

    // Loop over layers to initialize data and check for intersections with contours and
    // slices
    int layer_id = -1;
    int contour_id = -1;
    int slice_id = -1;
    for (int ilay = 0; ilay < rp.nlayers(cam_id); ilay++) {
      if (!interact[ilay]) continue;
      // first distance to slices OR construct the cell interpolant
      if (rp.type(cam_id, ilay) == VizType::slice) {
        Real n_dot_v = rp.slice_n(cam_id, ilay, 0) * v[0] +
                       rp.slice_n(cam_id, ilay, 1) * v[1] +
                       rp.slice_n(cam_id, ilay, 2) * v[2];
        if (std::abs(n_dot_v) > kMarchTiny) {
          Real xp = rp.slice_x(cam_id, ilay, 0) - x[0];
          Real yp = rp.slice_x(cam_id, ilay, 1) - x[1];
          Real zp = rp.slice_x(cam_id, ilay, 2) - x[2];
          Real n_dot_p = rp.slice_n(cam_id, ilay, 0) * xp +
                         rp.slice_n(cam_id, ilay, 1) * yp +
                         rp.slice_n(cam_id, ilay, 2) * zp;
          Real ht = n_dot_p / n_dot_v;
          // if (ht > 0.0 && ht <= h + kMarchTiny) {
          if (ht <= h + kMarchTiny && ht > -kMarchTiny && a < rp.ac(cam_id, ilay, 0)) {
            load_data(ilay);
            std::array<Real, 3> temp;
            for (int d = 0; d < 3; d++)
              temp[d] = x[d] + ht * v[d];
            Real alpha = 1.0; // rp.ac(cam_id, ilay, 0);
            if (rp.slice_transparent_outside(cam_id, ilay, 0)) {
              auto tc = cart_to_coord(temp);
              Real val = get_alpha(ilay, tc[0], tc[1], tc[2]);
              if (val < rp.alpha_range_min(cam_id, ilay)) alpha = 0.0;
            }
            if (alpha > 0.0 &&
                (layer_mask(ilay, x, s0) || layer_mask(ilay, temp))) {
              h = ht;
              slice_id = ilay;
              contour_id = -1;
            }
          }
        }
      }
      // figure out distance to contours
      else if (rp.type(cam_id, ilay) == VizType::contour ||
               rp.type(cam_id, ilay) == VizType::contour_slice) {
        std::array<Real, 3> final_x{x[0] + h * v[0], x[1] + h * v[1], x[2] + h * v[2]};
        // if (!mask[ilay]) {
        //   if (!rp.masks(cam_id, ilay, final_x[0], final_x[1], final_x[2])) continue;
        // }
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
            std::array<Real, 3> xm{x[0] + tm * h0 * v[0], x[1] + tm * h0 * v[1],
                                   x[2] + tm * h0 * v[2]};
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
    // h += RayTrace::kStepOvershoot * min_size();

    // half step
    for (int d = 0; d < 3; d++)
      x[d] += 0.5 * h * v[d];
    sync_cart_to_coord();
    Real hint = h;
    if (!initial_mask) hint *= 0.5;

    // get the color and opacity for the step
    Real asum = 0.0;
    Real rsum = 0.0;
    Real gsum = 0.0;
    Real bsum = 0.0;
    for (int ilay = 0; ilay < rp.nlayers(cam_id); ilay++) {
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
    Real asafe = (asum > 0.0 ? asum : 1.0);
    rsum /= asafe;
    gsum /= asafe;
    bsum /= asafe;

    // composite
    Real alpha = 1.0 - std::exp(-asum * hint);
    r += (1 - a) * alpha * rsum;
    g += (1 - a) * alpha * gsum;
    b += (1 - a) * alpha * bsum;
    a += (1 - a) * alpha;

    // 2nd half step
    for (int d = 0; d < 3; d++)
      x[d] += 0.5 * h * v[d];
    sync_cart_to_coord();

    if (slice_id >= 0) {
      Real dlx = rp.xl(cam_id) - x[0];
      Real dly = rp.yl(cam_id) - x[1];
      Real dlz = rp.zl(cam_id) - x[2];
      Real dist = std::sqrt(dlx * dlx + dly * dly + dlz * dlz);
      dlx /= dist;
      dly /= dist;
      dlz /= dist;
      Real ndotl = rp.slice_n(cam_id, slice_id, 0) * dlx +
                   rp.slice_n(cam_id, slice_id, 1) * dly +
                   rp.slice_n(cam_id, slice_id, 2) * dlz;
      Real lum = rp.ambient(cam_id) +
                 rp.diffuse(cam_id) / (dist * dist) * std::max(0.0, std::abs(ndotl));
      auto val = get_value(slice_id);
      auto alpha_val = get_alpha(slice_id);
      auto p = rp.val_to_rgba(cam_id, slice_id, val, alpha_val);
      alpha = rp.ac(cam_id, slice_id, 0);
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
      if (rp.type(cam_id, layer_id) == VizType::contour) {
        alpha = rp.ac(cam_id, layer_id, contour_id);
        Real dlx = rp.xl(cam_id) - x[0];
        Real dly = rp.yl(cam_id) - x[1];
        Real dlz = rp.zl(cam_id) - x[2];
        Real inv_dist = 1.0 / std::sqrt(dlx * dlx + dly * dly + dlz * dlz);
        dlx *= inv_dist;
        dly *= inv_dist;
        dlz *= inv_dist;
        load_data(layer_id, true);
        auto [dfdx1, dfdx2, dfdx3] = data[layer_id].grad_val(xc[0], xc[1], xc[2]);
        auto [dfdx, dfdy, dfdz] = transform_coord_to_cart(dfdx1, dfdx2, dfdx3);
        auto inv_grad_mag = 1.0 / std::sqrt(dfdx * dfdx + dfdy * dfdy + dfdz * dfdz);
        dfdx *= inv_grad_mag;
        dfdy *= inv_grad_mag;
        dfdz *= inv_grad_mag;
        Real gdotl = dfdx * dlx + dfdy * dly + dfdz * dlz;
        Real lum = rp.ambient(cam_id) + rp.diffuse(cam_id) * inv_dist * inv_dist *
                                            std::max(0.0, std::abs(gdotl));
        r += (1 - a) * alpha * rp.rc(cam_id, layer_id, contour_id) * lum;
        g += (1 - a) * alpha * rp.gc(cam_id, layer_id, contour_id) * lum;
        b += (1 - a) * alpha * rp.bc(cam_id, layer_id, contour_id) * lum;
        a += (1 - a) * alpha;
      } else {
        auto color_val = get_alpha(layer_id);
        // we'll ignore alpha, so just send in zero
        auto p = rp.val_to_rgba(cam_id, layer_id, color_val, 0.0);
        alpha = rp.ac(cam_id, layer_id, contour_id);
        Real dlx = rp.xl(cam_id) - x[0];
        Real dly = rp.yl(cam_id) - x[1];
        Real dlz = rp.zl(cam_id) - x[2];
        Real dist = std::sqrt(dlx * dlx + dly * dly + dlz * dlz);
        dlx /= dist;
        dly /= dist;
        dlz /= dist;
        load_data(layer_id, true);
        auto [dfdx1, dfdx2, dfdx3] = data[layer_id].grad_val(xc[0], xc[1], xc[2]);
        auto [dfdx, dfdy, dfdz] = transform_coord_to_cart(dfdx1, dfdx2, dfdx3);
        Real gdotl = dfdx * dlx + dfdy * dly + dfdz * dlz;
        Real lum = rp.ambient(cam_id) +
                   rp.diffuse(cam_id) / (dist * dist) * std::max(0.0, -gdotl);
        Real max_lum = std::min(std::min(1.0 / p.r, 1.0 / p.g), 1.0 / p.b);
        lum = std::min(lum, max_lum);
        r += (1 - a) * alpha * p.r * lum;
        g += (1 - a) * alpha * p.g * lum;
        b += (1 - a) * alpha * p.b * lum;
        a += (1 - a) * alpha;
      }
    } // else {
    // ensure it lands "on" the face in just the right way
    RayTrace::IntegratorBase<VarPack_t>::snap_to_face();

    //}

    check_and_flag_off_mesh();
  }

  const RenderingParams &rp;
  int cam_id, pix_id;
  Real &r, &g, &b, &a;
  std::array<VizData, max_layers> data;
  // the cell each layer's interpolant was built on, and whether it has the gradient
  std::array<long, max_layers> data_cell;
  std::array<bool, max_layers> data_grad;
  // each layer's masks, and the stretch of the ray, in units of x . v, they hold over,
  // and the same for whether any of them is let through
  std::array<bool, max_layers> mask_val;
  std::array<Real, max_layers> mask_slo, mask_shi;
  bool any_val;
  Real any_slo, any_shi;
  Real vv;
  std::array<bool, max_layers> interact;
  const parthenon::ParArray1D<parthenon::PackIdx> &idx_map;
  const parthenon::ParArrayND<uint8_t> &images;
};

namespace impl {
namespace rt = particles::ray_tracer;
namespace vr = particles::riot_viz;
struct NodalBounds {
  parthenon::ParArray2D<Real> max_value, min_value;
  // by block and camera, the layers that could show in the block, as bits
  parthenon::ParArray2D<uint32_t> may_show;
  // by camera, the box of the blocks where any may, laid out as ShowVolume::box
  std::vector<Real> show_box;
};
// Whether the ray of camera point p starts in block b: on the part of the mesh boundary
// the block covers, or in the block, where it starts inside the mesh. Points on a face
// shared by two blocks lie on both.
struct CameraPointOnBlock {
  parthenon::LukewarmGeometry geom;
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
  // the directions rays are traced along, as in CullsBlocks()
  static constexpr int start_dims = parthenon::IsCoord<parthenon::UniformCartesian>() ? 3
                                    : parthenon::IsCoord<parthenon::UniformCylindrical>()
                                        ? 2
                                        : 1;

  KOKKOS_INLINE_FUNCTION
  bool Inside(const int b, const int d, const Real x) const {
    const auto dir = static_cast<parthenon::CoordinateDirection>(d + 1);
    return x >= geom.Xmin(b, dir) && x <= geom.Xmax(b, dir);
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
struct TracerInfo {
  using DescriptorType =
      parthenon::LukewarmPack<lukewarm::x, lukewarm::y, lukewarm::z, rt::vx, rt::vy,
                              rt::vz, rt::source_block, vr::r, vr::g, vr::b, vr::a,
                              vr::camera_id, vr::pixel_id>::Descriptor;

  TracerInfo(const std::vector<std::string> &fields, RenderingParams &&rp)
      : fields(fields), render_params(std::move(rp)), initialized(false) {}

  void initialize(MeshData<Real> *md) {
    desc.emplace(parthenon::MakePackDescriptor(md, fields));
    desc_ps.emplace(
        parthenon::MakeLukewarmPackDescriptor<
            lukewarm::x, lukewarm::y, lukewarm::z, rt::vx, rt::vy, rt::vz,
            rt::source_block, vr::r, vr::g, vr::b, vr::a, vr::camera_id, vr::pixel_id>(
            md, vr::particles::name()));
    auto pack_idx_map = desc->GetMap();
    std::vector<parthenon::PackIdx> pack_idx;
    for (auto &key : fields)
      pack_idx.emplace_back(pack_idx_map.at(key));
    pack_idx_view = RiotUtils::VectorToDevice(pack_idx, "pack_idx_view");

    initialized = true;
  }

  auto GetPacks(MeshData<Real> *md) const {
    return std::make_tuple(desc->GetPack(md), desc_ps->GetPack(md));
  }

  std::vector<std::string> fields;
  RenderingParams render_params;
  bool initialized;
  std::optional<parthenon::SparsePack<>::Descriptor> desc;
  std::optional<DescriptorType> desc_ps;
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
  // all the cameras filming my scene
  std::vector<CameraInfo> camera;
  // all the fields contributing to the imaging
  std::vector<std::string> field;
  // all the images from cameras
  ParArrayND<uint8_t> images;
  // vector copies of everything needed to construct the device side RenderingParams
  RenderingParamsVec rendering_params;
  // bookkeeping to know how to index into packs
  int max_var_index = -1;
};

TaskCollection Render(Mesh *pm, Real time);
TaskStatus DumpImages(Mesh *pm, Real time);
bool TimeToRender(Mesh *pm, Real time);

} // namespace riot_viz

#endif // DIAGNOSTICS_RIOT_VIZ_HPP_

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
    {"linear", ScaleType::linear},
    {"log", ScaleType::log}};

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
T clamp(const T& val, const T& min_val, const T& max_val) {
  return std::min(std::max(val, min_val), max_val);
}

namespace vr = particles::riot_viz;
using DataBox = Spiner::DataBox<Real>;

// Ray-march robustness factors. These are deliberately distinct because they
// live at different scales and cannot be interchanged:
//   kMarchTiny     - tiny absolute guard against degenerate denominators and
//                    zero-length steps (also the alpha off-mesh sentinel and
//                    the dump-time comparison tolerance).
//   kStepOvershoot - relative amount (times min cell size) a step is extended
//                    past a face so the crossing is unambiguous.
//   kAxisStepFrac  - largest step permitted near a coordinate axis, as a
//                    fraction of the cylindrical/spherical radius.
//   kGeomTol       - relative tolerance for placing/testing points on the
//                    domain boundary during host-side camera setup.
//   kBoundaryBand  - relative band for detecting that a host point coincides
//                    with a block's outer radius.
constexpr Real kMarchTiny = 1.e-14;
constexpr Real kStepOvershoot = 1.e-8;
constexpr Real kAxisStepFrac = 0.1;
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
    }

    Kokkos::deep_copy(data, host_data);
    Kokkos::deep_copy(inner_offset, host_inner_offset);
    Kokkos::deep_copy(outer_offset, host_outer_offset);
  }

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
  T &operator()(const int i) const { return data(i); }

 private:
  parthenon::ParArray1D<T> data;
};

struct RenderingParams {

  RenderingParams(vec2d_t<VizType> &t, vec2d_t<int> &pack_idx,
                  vec2d_t<int> &alpha_pack_idx, vec2d_t<ScaleType> &field_scale, vec2d_t<Real> &range_min,
                  vec2d_t<Real> &range_max, vec2d_t<Real> &alpha_range_min,
                  vec2d_t<Real> &alpha_range_max, vec2d_t<GradType> &use_grad,
                  vec2d_t<GradType> &alpha_use_grad, std::vector<Real> &alpha_tol,
                  vec2d_t<DataBox> &r, vec2d_t<DataBox> &g, vec2d_t<DataBox> &b,
                  vec2d_t<DataBox> &a, vec3d_t<Real> &contours, vec3d_t<Real> &rc,
                  vec3d_t<Real> &gc, vec3d_t<Real> &bc, vec3d_t<Real> &ac,
                  vec3d_t<Real> &slice_x, vec3d_t<Real> &slice_n, std::vector<Real> &xl,
                  std::vector<Real> &yl, std::vector<Real> &zl,
                  std::vector<Real> &ambient, std::vector<Real> &diffuse,
                  vec3d_t<mask_tuple_t> &mask, vec3d_t<int> &mask_id)
      : type(t), pack_idx(pack_idx), alpha_pack_idx(alpha_pack_idx), field_scale(field_scale), range_min(range_min),
        range_max(range_max), alpha_range_min(alpha_range_min),
        alpha_range_max(alpha_range_max), use_grad(use_grad),
        alpha_use_grad(alpha_use_grad), alpha_tol(alpha_tol), r(r), g(g), b(b), a(a),
        contours(contours), rc(rc), gc(gc), bc(bc), ac(ac), slice_x(slice_x),
        slice_n(slice_n), xl(xl), yl(yl), zl(zl), ambient(ambient), diffuse(diffuse),
        mask(mask), mask_id(mask_id) {
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
      : RenderingParams(rv.type, rv.pack_idx, rv.alpha_pack_idx, rv.field_scale, rv.range_min,
                        rv.range_max, rv.alpha_range_min, rv.alpha_range_max, rv.use_grad,
                        rv.alpha_use_grad, rv.alpha_tol, rv.r, rv.g, rv.b, rv.a,
                        rv.contours, rv.rc, rv.gc, rv.bc, rv.ac, rv.slice_x, rv.slice_n,
                        rv.xl, rv.yl, rv.zl, rv.ambient, rv.diffuse, rv.masks,
                        rv.mask_id) {}

  KOKKOS_INLINE_FUNCTION
  PixelVal val_to_rgba(const int outer, const int inner, Real val,
                       const Real aval) const {
    if (field_scale(outer, inner) == ScaleType::log) {
      if (val <= 0.0) return {0.0, 0.0, 0.0, 0.0};
      val = std::log10(val);
    }
    Real v = (val - range_min(outer, inner)) /
             (range_max(outer, inner) - range_min(outer, inner));
    Real av = (aval - alpha_range_min(outer, inner)) /
              (alpha_range_max(outer, inner) - alpha_range_min(outer, inner));
    v = clamp(v, 0.0, 1.0);
    av = clamp(av, 0.0, 1.0);
    return {r(outer, inner).interpToReal(v), g(outer, inner).interpToReal(v),
            b(outer, inner).interpToReal(v), a(outer, inner).interpToReal(av)};
  }

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

  VectorView2D<VizType> type;
  VectorView2D<int> pack_idx, alpha_pack_idx;
  VectorView2D<ScaleType> field_scale;
  VectorView2D<Real> range_min, range_max, alpha_range_min, alpha_range_max;
  VectorView2D<GradType> use_grad, alpha_use_grad;
  VectorView1D<Real> alpha_tol;
  VectorView2D<DataBox> r, g, b, a;
  VectorView3D<Real> contours, rc, gc, bc, ac;
  VectorView3D<Real> slice_x, slice_n;
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
  KOKKOS_FUNCTION VizData(const RenderingParams &r, const int ilay, Integrator_t composer)
      : inv_dx(1.0 / composer->template dx<X1DIR>()),
        inv_dy(1.0 / composer->template dx<X2DIR>()),
        inv_dz(1.0 / composer->template dx<X3DIR>()),
        xlo(composer->template xlo<X1DIR>()), ylo(composer->template xlo<X2DIR>()),
        zlo(composer->template xlo<X3DIR>()) {
    if (ilay < r.nlayers(composer->cam_id)) {
      {
        auto val = [&](const int kk, const int jj, const int ii) {
          return composer->vp(composer->cell_b, r.pack_idx(composer->cam_id, ilay),
                              composer->cell_k + composer->dk * kk,
                              composer->cell_j + composer->dj * jj,
                              composer->cell_i + ii);
        };
        set_coeffs(val, A);
      }
      if (r.alpha_pack_idx(composer->cam_id, ilay) ==
          r.pack_idx(composer->cam_id, ilay)) {
        A_opac = A;
      } else {
        auto val = [&](const int kk, const int jj, const int ii) {
          return composer->vp(composer->cell_b, r.alpha_pack_idx(composer->cam_id, ilay),
                              composer->cell_k + composer->dk * kk,
                              composer->cell_j + composer->dj * jj,
                              composer->cell_i + ii);
        };
        set_coeffs(val, A_opac);
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
    return gradient_(xt, yt, zt, A);
  }

 private:
  std::array<Real, 8> A; // 0 A0, 1 Ax, 2 Ay, 3 Az, 4 Axy, 5 Axz, 6 Ayz, 7 Axyz;
  std::array<Real, 8> A_opac;
  KOKKOS_INLINE_FUNCTION
  Real sample_(const GradType g, const Real xt, const Real yt, const Real zt,
               const std::array<Real, 8> &c) const {
    if (g == GradType::mag) {
      auto [gx, gy, gz] = gradient_(xt, yt, zt, c);
      return std::sqrt(gx * gx + gy * gy + gz * gz);
    } else if (g == GradType::log_mag) {
      return log_grad_mag_(xt, yt, zt, c);
    }
    return interp_(xt, yt, zt, c);
  }
  Real inv_dx, inv_dy, inv_dz, xlo, ylo, zlo;
  template <typename F>
  KOKKOS_INLINE_FUNCTION void set_coeffs(const F &val, std::array<Real, 8> &c) {
    c[0] = val(0, 0, 0);
    c[1] = val(0, 0, 1) - c[0];
    c[2] = val(0, 1, 0) - c[0];
    c[3] = val(1, 0, 0) - c[0];
    c[4] = val(0, 1, 1) - val(0, 0, 1) - val(0, 1, 0) + c[0];
    c[5] = val(1, 0, 1) - val(0, 0, 1) - val(1, 0, 0) + c[0];
    c[6] = val(1, 1, 0) - val(0, 1, 0) - val(1, 0, 0) + c[0];
    c[7] = val(1, 1, 1) - val(0, 1, 1) - val(1, 0, 1) - val(1, 1, 0) + val(0, 0, 1) +
           val(0, 1, 0) + val(1, 0, 0) - c[0];
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
  std::array<Real, 3> gradient_(const Real xt, const Real yt, const Real zt,
                                const std::array<Real, 8> &c) const {
    const Real wx = (xt - xlo) * inv_dx;
    const Real wy = (yt - ylo) * inv_dy;
    const Real wz = (zt - zlo) * inv_dz;
    return {(c[1] + wy * c[4] + wz * (c[5] + wy * c[7])) * inv_dx,
            (c[2] + wx * c[4] + wz * (c[6] + wx * c[7])) * inv_dy,
            (c[3] + wx * c[5] + wy * (c[6] + wx * c[7])) * inv_dz};
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

template <typename VarPack_t>
class Composer : public RayTrace::IntegratorBase<VarPack_t> {
 public:
  static constexpr int max_layers = 5;
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
  using RayTrace::IntegratorBase<VarPack_t>::context;
  using RayTrace::IntegratorBase<VarPack_t>::pidx;
  template <typename SwarmPack_t, typename SwarmPackInt_t>
  KOKKOS_FUNCTION Composer(VarPack_t &vp, SwarmPack_t &ps, const int b, const int pidx,
                           const int ndim, SwarmPackInt_t &ps_int,
                           const RenderingParams &rp)
      : RayTrace::IntegratorBase<VarPack_t>(vp, ps, b, pidx, ndim), rp(rp),
        cam_id(ps_int(b, vr::camera_id(), pidx)), r(ps(b, vr::r(), pidx)),
        g(ps(b, vr::g(), pidx)), b(ps(b, vr::b(), pidx)), a(ps(b, vr::a(), pidx)) {
    PARTHENON_REQUIRE(rp.nlayers(cam_id) < max_layers,
                      "Too many viz layers requested. Must recompile after setting "
                      "max_layers large enough");
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

  void check_and_flag_off_mesh() {
    if (!on_mesh()) {
      // this step pushed me off the mesh
      // make alpha negative as a sentinel
      // move particle back on mesh so it doesn't get removed
      a = -(a + kMarchTiny);
      xc[0] = 0.5 * (context.x_min_ + context.x_max_);
      xc[1] = 0.5 * (context.y_min_ + context.y_max_);
      xc[2] = 0.5 * (context.z_min_ + context.z_max_);
    }
  }

  bool all_mask(std::array<Real, 3> &x, std::array<bool, max_layers> &mask) {
    bool ret = false;
    for (int ilay = 0; ilay < rp.nlayers(cam_id); ilay++) {
      mask[ilay] = rp.masks(cam_id, ilay, x[0], x[1], x[2]);
      ret |= mask[ilay];
    }
    return ret;
  }

  bool all_mask(std::array<Real, 3> &x) {
    for (int ilay = 0; ilay < rp.nlayers(cam_id); ilay++) {
      if (rp.masks(cam_id, ilay, x[0], x[1], x[2])) return true;
    }
    return false;
  }

  Real get_value(const int ilay, const Real x1, const Real x2, const Real x3) const {
    return data[ilay].value(rp.use_grad(cam_id, ilay), x1, x2, x3);
  }
  Real get_value(const int ilay) const { return get_value(ilay, xc[0], xc[1], xc[2]); }
  Real get_alpha(const int ilay) const {
    return data[ilay].alpha(rp.alpha_use_grad(cam_id, ilay), xc[0], xc[1], xc[2]);
  }
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
      Real nc[2];
      nc[0] = (x[0] * v[0] + x[1] * v[1]) / xc[0];
      nc[1] = v[2];
      loop_2d([&]<int d>() {
        auto ht =
            (RayTrace::IntegratorBase<VarPack_t>::template xlo<d>() +
             (nc[d - 1] > 0) * RayTrace::IntegratorBase<VarPack_t>::template dx<d>() -
             xc[d - 1]) /
            nc[d - 1];
        h = std::min(h, ht);
      });
      Real ht = std::max(kAxisStepFrac * xc[0], min_size());
      h = std::min(h, ht);
    } else {
      const Real rcyl = std::sqrt(x[0] * x[0] + x[1] * x[1]);
      const Real nr = (x[0] * v[0] + x[1] * v[1] + x[2] * v[2]) / xc[0];
      auto ht =
          (RayTrace::IntegratorBase<VarPack_t>::template xlo<X1DIR>() +
           (nr > 0) * RayTrace::IntegratorBase<VarPack_t>::template dx<X1DIR>() - xc[0]) /
          nr;
      h = std::min(h, ht);
      ht = std::max(kAxisStepFrac * rcyl, min_size());
      h = std::min(h, ht);
    }
    return h;
  }

  void advance() {
    bool initial_mask = all_mask(x, mask);
    // step size til next cell face
    auto h = distance_to_face();
    h = std::max(h, kMarchTiny);
    if (!initial_mask) {
      // see if I can skip everything else
      std::array<Real, 3> temp;
      for (int d = 0; d < 3; d++)
        temp[d] = x[d] + h * v[d];
      bool test = all_mask(temp);

      if (!test) {
        for (int d = 0; d < 3; d++)
          x[d] = temp[d];
        sync_cart_to_coord();
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
          if (ht > 0.0 && ht <= h + kMarchTiny) {
            h = ht;
            slice_id = ilay;
            contour_id = -1;
          }
        }
      }
      // figure out distance to contours
      else if (rp.type(cam_id, ilay) == VizType::contour ||
               rp.type(cam_id, ilay) == VizType::contour_slice) {
        std::array<Real, 3> final_x{x[0] + h * v[0],
                                    x[1] + h * v[1],
                                    x[2] + h * v[2]};
        if (!mask[ilay]) {
          if (!rp.masks(cam_id, ilay, final_x[0], final_x[1], final_x[2])) continue;
        }
        data[ilay] = VizData(rp, ilay, this);
        Real initial = get_value(ilay);
        auto final_coords = cart_to_coord(final_x);
        Real final = get_value(ilay, final_coords[0], final_coords[1], final_coords[2]);
        Real ht_min = 1.0;
        for (int i = 0; i < rp.ncontours(cam_id, ilay); i++) {
          Real ht = (rp.contours(cam_id, ilay, i) - initial) / (final - initial);
          if (ht > 0.0 && ht < ht_min) {
            h *= ht;
            ht_min = ht;
            layer_id = ilay;
            contour_id = i;
            slice_id = -1;
          }
        }
      }
    }
    h += kStepOvershoot * min_size();

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
      if (rp.type(cam_id, ilay) == VizType::volume) {
        if (rp.masks(cam_id, ilay, x[0], x[1], x[2])) {
          data[ilay] = VizData(rp, ilay, this);
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
      if (rp.masks(cam_id, slice_id, x[0], x[1], x[2])) {
        data[slice_id] = VizData(rp, slice_id, this);
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
                   rp.diffuse(cam_id) / (dist * dist) * std::max(0.0, ndotl);
        auto val = get_value(slice_id);
        auto alpha_val = get_alpha(slice_id);
        auto p = rp.val_to_rgba(cam_id, slice_id, val, alpha_val);
        alpha = rp.ac(cam_id, slice_id, 0);
        Real max_lum = std::min(std::min(1.0 / p.r, 1.0 / p.g), 1.0 / p.b);
        lum = std::min(lum, max_lum);
        r += (1 - a) * alpha * p.r * lum;
        g += (1 - a) * alpha * p.g * lum;
        b += (1 - a) * alpha * p.b * lum;
        a += (1 - a) * alpha;
      }
    } else if (contour_id >= 0) {
      if (rp.masks(cam_id, layer_id, x[0], x[1], x[2])) {
        if (rp.type(cam_id, layer_id) == VizType::contour) {
          alpha = rp.ac(cam_id, layer_id, contour_id);
          Real dlx = rp.xl(cam_id) - x[0];
          Real dly = rp.yl(cam_id) - x[1];
          Real dlz = rp.zl(cam_id) - x[2];
          Real dist = std::sqrt(dlx * dlx + dly * dly + dlz * dlz);
          dlx /= dist;
          dly /= dist;
          dlz /= dist;
          auto [dfdx1, dfdx2, dfdx3] = data[layer_id].grad_val(xc[0], xc[1], xc[2]);
          auto [dfdx, dfdy, dfdz] = transform_coord_to_cart(dfdx1, dfdx2, dfdx3);
          Real gdotl = dfdx * dlx + dfdy * dly + dfdz * dlz;
          Real lum = rp.ambient(cam_id) +
                     rp.diffuse(cam_id) / (dist * dist) * std::max(0.0, -gdotl);
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
      } else {
        PARTHENON_FAIL("Should never get here.");
      }
    } else {
      // ensure it lands "on" the face in just the right way
      RayTrace::IntegratorBase<VarPack_t>::snap_to_face();
    }

    check_and_flag_off_mesh();
  }

  const RenderingParams &rp;
  int cam_id;
  Real &r, &g, &b, &a;
  std::array<VizData, max_layers> data;
  std::array<bool, max_layers> mask;
};

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

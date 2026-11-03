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
// This file was made in part with generative AI.
#include <algorithm>
//#include <format>
#include <iomanip>
#include <sstream>

#include "riot_viz.hpp"

#include <riot_utils/riot_loops.hpp>

// the stb implementations are compiled as C, in stb_impl.c
#include <stb_image_write.h>
#include <stb_image_resize2.h>

#include <font8x8.h>

namespace riot_viz {

static bool riot_viz_initialized = false;

void AddCamera(ParameterInput *pin, const int img_id, const int ndim, SceneInfo &scene) {
  const std::string cam_block = "riot_viz/camera" + std::to_string(img_id);
  const int cam_id = scene.camera.size();
  scene.camera.push_back(CameraInfo{});
  auto &camera = scene.camera.back();
  auto &rp = scene.rendering_params;

  // first build the vector of layers
  // some info is common to all layers but stored in layers for convenience
  auto light_x = pin->GetOrAdd<Real>(cam_block, "light_x", 0.0);
  auto light_y = pin->GetOrAdd<Real>(cam_block, "light_y", 0.0);
  auto light_z = pin->GetOrAdd<Real>(cam_block, "light_z", 0.0);
  auto ambient = pin->GetOrAdd<Real>(cam_block, "light_ambient", 1.0);
  auto diffuse = pin->GetOrAdd<Real>(cam_block, "light_diffuse", 0.0);
  auto opacity_threshold = pin->GetOrAdd<Real>(cam_block, "opacity_threshold", 1.e-2,
                                               "Opacity to stop ray trace");

  rp.xl.push_back(light_x);
  rp.yl.push_back(light_y);
  rp.zl.push_back(light_z);
  rp.ambient.push_back(ambient);
  rp.diffuse.push_back(diffuse);
  rp.alpha_tol.push_back(opacity_threshold);

  auto layers = pin->GetVector<std::string>(cam_block, "layers");
  int ncolorbars = 0;
  std::vector<VizType> layer_type;
  std::vector<int> layer_pack_idx, layer_alpha_pack_idx;
  std::vector<ScaleType> layer_field_scale;
  std::vector<Real> layer_range_min, layer_range_max, layer_alpha_range_min,
      layer_alpha_range_max;
  std::vector<GradType> layer_use_grad, layer_alpha_use_grad;
  std::vector<DataBox> layer_r, layer_g, layer_b, layer_a;
  std::vector<std::vector<Real>> layer_c, layer_rc, layer_gc, layer_bc, layer_ac;
  std::vector<std::vector<Real>> layer_slice_x, layer_slice_n;
  std::vector<std::vector<bool>> layer_slice_transparent_outside;
  std::vector<std::vector<mask_tuple_t>> layer_masks;
  std::vector<std::vector<int>> layer_mask_id;
  for (auto &layer : layers) {
    auto type = pin->Get<std::string>(layer, "type");
    auto field = pin->Get<std::string>(layer, "field");
    auto field_use_grad = pin->GetOrAdd<std::string>(layer, "field_use_grad", "none");
    auto field_scale = pin->GetOrAdd<std::string>(layer, "field_scale", "linear");
    auto field_alpha = pin->GetOrAdd<std::string>(layer, "field_alpha", field);
    auto field_alpha_use_grad =
        pin->GetOrAdd<std::string>(layer, "field_alpha_use_grad", "none");
    auto label = pin->GetOrAdd<std::string>(layer, "label", field);

    // volume render, slice, and contour_slice params
    auto colorbar = pin->GetOrAdd<bool>(layer, "colorbar", false);
    camera.ncolorbar += colorbar;
    auto red = pin->GetOrAddVector<Real>(layer, "red", {0.0, 0.0});
    auto green = pin->GetOrAddVector<Real>(layer, "green", {0.0, 0.0});
    auto blue = pin->GetOrAddVector<Real>(layer, "blue", {0.0, 0.0});
    auto alpha = pin->GetOrAddVector<Real>(layer, "alpha", {0.0, 0.0});
    auto cmin = pin->GetOrAdd<Real>(layer, "min_value", 0.0);
    auto cmax = pin->GetOrAdd<Real>(layer, "max_value", 1.0);
    auto amin = pin->GetOrAdd<Real>(layer, "min_alpha_value", cmin);
    auto amax = pin->GetOrAdd<Real>(layer, "max_alpha_value", cmax);

    // contour and contour_slice params
    auto contours = pin->GetOrAddVector<Real>(layer, "contours", {});
    auto contour_alpha = pin->GetOrAddVector<Real>(layer, "contour_alpha", {});

    // contour params
    auto contour_red = pin->GetOrAddVector<Real>(layer, "contour_red", {});
    auto contour_green = pin->GetOrAddVector<Real>(layer, "contour_green", {});
    auto contour_blue = pin->GetOrAddVector<Real>(layer, "contour_blue", {});

    // slice params
    auto slice_pos = pin->GetOrAddVector<Real>(layer, "slice_location", {});
    auto slice_normal = pin->GetOrAddVector<Real>(layer, "slice_normal", {});
    auto slice_alpha = pin->GetOrAdd<Real>(layer, "slice_alpha", 1.0);
    auto slice_transparent_outside = pin->GetOrAddVector<bool>(layer, "transparent_outside_range", std::vector<bool>(2, false));
    if (type == "slice") {
      // stick slice_alpha in contour_alpha
      slice_alpha = std::clamp(slice_alpha, 0.0, 1.0);
      if (contour_alpha.size() == 0)
        contour_alpha.push_back(slice_alpha);
      else
        contour_alpha[0] = slice_alpha;
    }

    // contour_slice params

    if (contours.size() > 0) {
      PARTHENON_REQUIRE_THROWS(
          contours.size() == contour_red.size(),
          "size of contour list must match across values and colors");
      PARTHENON_REQUIRE_THROWS(
          contours.size() == contour_green.size(),
          "size of contour list must match across values and colors");
      PARTHENON_REQUIRE_THROWS(
          contours.size() == contour_blue.size(),
          "size of contour list must match across values and colors");
      PARTHENON_REQUIRE_THROWS(
          contours.size() == contour_alpha.size(),
          "size of contour list must match across values and colors");
    }

    for (int i = 0; i < contour_alpha.size(); i++) {
      contour_alpha[i] = clamp(contour_alpha[i], 0.0, 1.0);
    }
    auto cont_dev = RiotUtils::VectorToDevice(contours, "contours");
    auto cont_red_dev = RiotUtils::VectorToDevice(contour_red, "contour red");
    auto cont_green_dev = RiotUtils::VectorToDevice(contour_green, "contour green");
    auto cont_blue_dev = RiotUtils::VectorToDevice(contour_blue, "contour blue");
    auto cont_alpha_dev = RiotUtils::VectorToDevice(contour_alpha, "contour alpha");

    std::vector<Real> slice_x{std::numeric_limits<Real>::max(),
                              std::numeric_limits<Real>::max(),
                              std::numeric_limits<Real>::max()};
    std::vector<Real> slice_n = {0.0, 0.0, 0.0};
    if ((slice_pos.size() == slice_normal.size()) && slice_pos.size() == 3) {
      slice_x = {slice_pos[0], slice_pos[1], slice_pos[2]};
      slice_n = {slice_normal[0], slice_normal[1], slice_normal[2]};
    }

    auto it = std::find(scene.field.begin(), scene.field.end(), field);
    int var_index;
    if (it != scene.field.end()) {
      var_index = it - scene.field.begin();
    } else {
      var_index = ++scene.max_var_index;
      scene.field.push_back(field);
    }
    auto alpha_var_index = var_index;
    if (field != field_alpha) {
      it = std::find(scene.field.begin(), scene.field.end(), field_alpha);
      if (it != scene.field.end()) {
        alpha_var_index = it - scene.field.begin();
      } else {
        alpha_var_index = ++scene.max_var_index;
        scene.field.push_back(field_alpha);
      }
    }

    auto [masks_vec, mask_id] = MakeMasks(pin, layer);

    auto scale_type = GetType(field_scale, ScaleTypeMap);
    auto scale = [&](const Real val) {
      if (scale_type == ScaleType::log) {
        PARTHENON_REQUIRE_THROWS(
            val > 0.0, "Trying to take the log of a negative number is a no-no.");
        return std::log10(val);
      }
      return val;
    };

    layer_type.push_back(GetType(type, VizTypeMap));
    layer_pack_idx.push_back(var_index);
    layer_alpha_pack_idx.push_back(alpha_var_index);
    layer_field_scale.push_back(scale_type);
    layer_range_min.push_back(scale(cmin));
    layer_range_max.push_back(scale(cmax));
    layer_alpha_range_min.push_back(amin);
    layer_alpha_range_max.push_back(amax);
    layer_use_grad.push_back(GetType(field_use_grad, GradTypeMap));
    layer_alpha_use_grad.push_back(GetType(field_alpha_use_grad, GradTypeMap));
    layer_r.push_back(MakeNormalizedDeviceDataBox(red));
    layer_g.push_back(MakeNormalizedDeviceDataBox(green));
    layer_b.push_back(MakeNormalizedDeviceDataBox(blue));
    layer_a.push_back(MakeNormalizedDeviceDataBox(alpha));
    layer_c.push_back(contours);
    layer_rc.push_back(contour_red);
    layer_gc.push_back(contour_green);
    layer_bc.push_back(contour_blue);
    layer_ac.push_back(contour_alpha);
    layer_slice_x.push_back(slice_x);
    layer_slice_n.push_back(slice_n);
    layer_slice_transparent_outside.push_back(slice_transparent_outside);
    layer_masks.push_back(masks_vec);
    layer_mask_id.push_back(mask_id);

    // add this layer to the camera
    camera.layer.emplace_back(MakeNormalizedDataBox(red), MakeNormalizedDataBox(green),
                              MakeNormalizedDataBox(blue), MakeNormalizedDataBox(alpha),
                              label, colorbar, scale(cmin), scale(cmax),
                              GetType(type, VizTypeMap), scale_type);
  }

  rp.type.push_back(layer_type);
  rp.pack_idx.push_back(layer_pack_idx);
  rp.alpha_pack_idx.push_back(layer_alpha_pack_idx);
  rp.field_scale.push_back(layer_field_scale);
  rp.range_min.push_back(layer_range_min);
  rp.range_max.push_back(layer_range_max);
  rp.alpha_range_min.push_back(layer_alpha_range_min);
  rp.alpha_range_max.push_back(layer_alpha_range_max);
  rp.use_grad.push_back(layer_use_grad);
  rp.alpha_use_grad.push_back(layer_alpha_use_grad);
  rp.r.push_back(layer_r);
  rp.g.push_back(layer_g);
  rp.b.push_back(layer_b);
  rp.a.push_back(layer_a);
  rp.contours.push_back(layer_c);
  rp.rc.push_back(layer_rc);
  rp.gc.push_back(layer_gc);
  rp.bc.push_back(layer_bc);
  rp.ac.push_back(layer_ac);
  rp.slice_x.push_back(layer_slice_x);
  rp.slice_n.push_back(layer_slice_n);
  rp.slice_transparent_outside.push_back(layer_slice_transparent_outside);
  rp.masks.push_back(layer_masks);
  rp.mask_id.push_back(layer_mask_id);

  // now set some info at the camera/image level
  camera.filename = pin->GetString(cam_block, "name");
  auto cloc = pin->GetVector<Real>(cam_block, "location");
  auto floc = pin->GetVector<Real>(cam_block, "focus");
  auto up = pin->GetOrAddVector<Real>(cam_block, "up", {0, 0, 1});
  auto width = pin->Get<Real>(cam_block, "target_width");
  auto height = pin->Get<Real>(cam_block, "target_height");
  camera.nwidth = pin->Get<int>(cam_block, "nwidth");
  camera.nheight = pin->Get<int>(cam_block, "nheight");
  camera.scale = pin->GetOrAdd<Real>(cam_block, "scale_output", 1.0);
  camera.colorbar_thickness = pin->GetOrAdd<int>(cam_block, "colorbar_thickness", 10);

  // now setup the rays this camera's contribution to the scene
  auto normalize = [&](std::vector<Real> &vec) {
    auto mag = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
    vec[0] /= mag;
    vec[1] /= mag;
    vec[2] /= mag;
  };
  auto cross = [&](std::vector<Real> &a, std::vector<Real> &b) {
    return std::vector<Real>({a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                              a[0] * b[1] - a[1] * b[0]});
  };

  std::vector<Real> nhat0({floc[0] - cloc[0], floc[1] - cloc[1], floc[2] - cloc[2]});
  Real cdist = sqrt(nhat0[0] * nhat0[0] + nhat0[1] * nhat0[1] + nhat0[2] * nhat0[2]);
  normalize(up);
  normalize(nhat0);
  // rotate the up vector in the plane shared by up and nhat0 to make them perpendicular
  auto ixhat = cross(nhat0, up);
  normalize(ixhat);
  auto iyhat = cross(ixhat, nhat0);

  std::array<Real, 3> xmin, xmax;
  xmin[0] = pin->GetReal("parthenon/mesh", "x1min");
  xmax[0] = pin->GetReal("parthenon/mesh", "x1max");
  xmin[1] = pin->GetReal("parthenon/mesh", "x2min");
  xmax[1] = pin->GetReal("parthenon/mesh", "x2max");
  xmin[2] = pin->GetReal("parthenon/mesh", "x3min");
  xmax[2] = pin->GetReal("parthenon/mesh", "x3max");

  Real dx = width / camera.nwidth;
  Real dy = height / camera.nheight;
  std::array<Real, 3> nray, lambda, trial;
  std::array<Real, 3> lambda_xmin, lambda_xmax;
  for (int j = 0; j < camera.nheight; j++) {
    Real ly = (-0.5 * height + (j + 0.5) * dy) / cdist;
    for (int i = 0; i < camera.nwidth; i++) {
      Real lx = (-0.5 * width + (i + 0.5) * dx) / cdist;
      for (int d = 0; d < 3; d++) {
        nray[d] = nhat0[d] + lx * ixhat[d] - ly * iyhat[d];
      }

      // find step size lambda along nray that puts ray at entry point into [xmin, xmax]
      // note: this is where we would add slicing, cutouts, etc
      int face_dir;
      Real lambda_use = std::numeric_limits<Real>::max();
      if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
        for (int d = 0; d < 3; d++) {
          lambda_xmin[d] = (xmin[d] - cloc[d]) / nray[d];
          lambda_xmax[d] = (xmax[d] - cloc[d]) / nray[d];
        }
        // figure out which lambda puts the ray on the multi-d domain boundary
        auto check_lambda = [&](const Real lam) {
          if (lam < 0.0) return false;
          for (int dd = 0; dd < 3; dd++) {
            const Real x = cloc[dd] + lam * nray[dd];
            if (x < xmin[dd] - kGeomTol || x > xmax[dd] + kGeomTol) return false;
          }
          return true;
        };
        for (int d = 0; d < 3; d++) {
          if (check_lambda(lambda_xmin[d]) && lambda_xmin[d] < lambda_use) {
            lambda_use = lambda_xmin[d];
            face_dir = d;
          }
          if (check_lambda(lambda_xmax[d]) && lambda_xmax[d] < lambda_use) {
            lambda_use = lambda_xmax[d];
            face_dir = d;
          }
        }
      } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
        auto a = nray[0] * nray[0] + nray[1] * nray[1];
        auto b = 2 * (cloc[0] * nray[0] + cloc[1] * nray[1]);
        auto c = cloc[0] * cloc[0] + cloc[1] * cloc[1] - xmax[0] * xmax[0];
        auto det = b * b - 4.0 * a * c;
        if (det > 0.0) {
          det = std::sqrt(det);
          auto lam1 = (-b - det) / (2.0 * a);
          auto lam2 = (-b + det) / (2.0 * a);
          // which one is entry?
          for (int d = 0; d < 2; d++) {
            trial[d] = cloc[d] + lam1 * nray[d];
          }
          auto vr = trial[0] * nray[0] + trial[1] * nray[1];
          if (vr < 0) {
            // at this lambda, the ray is entering since \hat{r}\cdot\hat{n} < 0
            lambda[0] = lam1;
          } else {
            lambda[0] = lam2;
          }
          if (nray[2] > 0.0) {
            lambda[1] = (xmin[1] - cloc[2]) / nray[2];
          } else if (nray[2] < 0.0) {
            lambda[1] = (xmax[1] - cloc[2]) / nray[2];
          } else {
            lambda[1] = std::numeric_limits<Real>::max();
          }

          for (int d = 0; d < 2; d++) {
            for (int dd = 0; dd < 3; dd++) {
              trial[dd] = cloc[dd] + lambda[d] * nray[dd];
            }
            auto rt = std::sqrt(trial[0] * trial[0] + trial[1] * trial[1]);
            if (rt < xmax[0] + kGeomTol &&
                (trial[2] > xmin[1] - kGeomTol && trial[2] < xmax[1] + kGeomTol)) {
              face_dir = d;
              lambda_use = lambda[d];
            }
          }
        } else {
          lambda_use = std::numeric_limits<Real>::max();
        }
      } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
        auto r_l_sq = cloc[0] * cloc[0] + cloc[1] * cloc[1] + cloc[2] * cloc[2];
        auto a = nray[0] * nray[0] + nray[1] * nray[1] + nray[2] * nray[2];
        auto b = 2 * (cloc[0] * nray[0] + cloc[1] * nray[1] + cloc[2] * nray[2]);
        auto c = r_l_sq - xmax[0] * xmax[0];
        auto det = b * b - 4.0 * a * c;
        if (det > 0.0) {
          det = std::sqrt(det);
          auto lam1 = (-b - det) / (2.0 * a);
          auto lam2 = (-b + det) / (2.0 * a);
          // which one is entry?
          for (int d = 0; d < 3; d++) {
            trial[d] = cloc[d] + lam1 * nray[d];
          }
          auto vr = trial[0] * nray[0] + trial[1] * nray[1] + trial[2] * nray[2];
          if (vr < 0) {
            // at this lambda, the ray is entering since \hat{r}\cdot\hat{n} < 0
            lambda[0] = lam1;
          } else {
            lambda[0] = lam2;
          }
          face_dir = 0;
          lambda_use = lambda[0];
        } else {
          lambda_use = std::numeric_limits<Real>::max();
        }
      } else {
        PARTHENON_THROW("Unsupported coordinate system in volume renderer.");
      }
      if (lambda_use > 0.5 * std::numeric_limits<Real>::max()) {
        // PARTHENON_WARN("Camera ray does not intersect domain, skipping.");
        continue;
      }

      // set coordinates
      for (int d = 0; d < 3; d++)
        trial[d] = cloc[d] + lambda_use * nray[d];
      int face_id;
      // reset position so it falls *exactly* on boundary
      if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
        trial[face_dir] =
            (nray[face_dir] > 0.0
                 ? xmin[face_dir] + (xmax[face_dir] - xmin[face_dir]) * kGeomTol
                 : xmax[face_dir] - (xmax[face_dir] - xmin[face_dir]) * kGeomTol);
        face_id = 2 * face_dir + (nray[face_dir] < 0);
      } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
        if (face_dir == 1 && nray[2] > 0) {
          trial[2] = xmin[1];
          face_id = 2;
        } else if (face_dir == 1) {
          trial[2] = xmax[1];
          face_id = 3;
        } else {
          auto r = std::sqrt(trial[0] * trial[0] + trial[1] * trial[1]);
          trial[0] *= xmax[0] / r;
          trial[1] *= xmax[0] / r;
          face_id = 1;
        }
      } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
        face_id = 1;
        auto r =
            std::sqrt(trial[0] * trial[0] + trial[1] * trial[1] + trial[2] * trial[2]);
        auto rb = xmax[0] * (1.0 - kGeomTol);
        trial[0] *= rb / r;
        trial[1] *= rb / r;
        trial[2] *= rb / r;
      }

      scene.face_pts[face_id][sample::x].push_back(trial[0]);
      scene.face_pts[face_id][sample::y].push_back(trial[1]);
      scene.face_pts[face_id][sample::z].push_back(trial[2]);
      scene.face_pts[face_id][sample::nx].push_back(nray[0]);
      scene.face_pts[face_id][sample::ny].push_back(nray[1]);
      scene.face_pts[face_id][sample::nz].push_back(nray[2]);
      scene.camera_id[face_id].push_back(cam_id);
      scene.pixel_id[face_id].push_back(i + camera.nwidth * j);
    }
  }
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  namespace pvr = particles::riot_viz;
  namespace prt = particles::ray_tracer;
  auto pkg = std::make_shared<StateDescriptor>("riot_viz");
  riot_viz_initialized = true;
  Params &params = pkg->AllParams();

  // figure out when viz should happen
  auto dt_viz = pin->GetOrAdd<Real>("riot_viz", "dt", -1.0);
  auto t_viz = pin->GetOrAddVector<Real>("riot_viz", "t", {});
  auto tlim = pin->Get<Real>("parthenon/time", "tlim");

  if (dt_viz > 0 && t_viz.size() > 0)
    PARTHENON_FAIL("Cannot set both dt and t for riot_viz");
  PARTHENON_REQUIRE(dt_viz > 0 || t_viz.size() > 0,
                    "Must set either dt or t in riot_viz");
  std::queue<Real> tdump;
  if (dt_viz > 0.0) {
    tdump.push(0.0);
    while (tdump.back() + dt_viz < tlim + kMarchTiny * dt_viz)
      tdump.push(tdump.back() + dt_viz);
    if (std::abs(tlim - tdump.back()) < kMarchTiny * dt_viz) tdump.back() = tlim;
  } else {
    for (auto &tnext : t_viz)
      tdump.push(tnext);
  }
  params.Add("tdump", tdump, true);
  int dump_id = 0;
  params.Add("dump_id", dump_id, parthenon::Params::Mutability::Restart);
  Real next_dump_time = tdump.front();
  params.Add("next_dump_time", next_dump_time, parthenon::Params::Mutability::Restart);
  params.Add("first_dump_time", tdump.front());
  params.Add("last_dump_time", tdump.back());
  bool timebar = pin->GetOrAdd<bool>("riot_viz", "timebar", false);
  params.Add("timebar", timebar);
  auto timebar_rgb =
      pin->GetOrAddVector<Real>("riot_viz", "timebar_rgb", {1.0, 1.0, 1.0});
  params.Add("timebar_rgb", timebar_rgb);

  int ndim = 1 + (pin->GetInteger("parthenon/mesh", "nx2") > 1) +
             (pin->GetInteger("parthenon/mesh", "nx3") > 1);

  const std::string block_base = "riot_viz/camera";
  auto blocks = pin->GetBlockNamesWithPrefix(block_base);
  std::set<int> cam_block_ids;
  int max_size = 0;
  int nimages = 0;
  SceneInfo scene;
  for (const auto &block_name : blocks) {
    if (block_name.length() > block_base.size()) {
      auto suffix = block_name[block_base.size()];
      if (std::isdigit(suffix)) {
        auto id = atoi(block_name.substr(block_base.size()).c_str());
        PARTHENON_REQUIRE_THROWS(cam_block_ids.count(id) == 0,
                                 "Two camera input blocks are numbered identically.");
        cam_block_ids.insert(id);
        AddCamera(pin, nimages, ndim, scene);
        auto &camera = scene.camera.back();
        max_size = std::max(max_size, camera.nwidth * camera.nheight +
                                          (camera.ncolorbar + timebar) *
                                              camera.colorbar_thickness * camera.nwidth);
        nimages++;
      }
    }
  }

  // don't forget the 4 because it's 4 channels (RGBA)
  scene.images =
      ParArrayND<uint8_t>("Volume Render images", scene.camera.size(), 4 * max_size);

  pkg->AddLukewarm(pvr::particles::name(),
                   Metadata({Metadata::Provides, Metadata::Analysis}));
  Metadata m_real({Metadata::Real});
  pkg->AddLukewarmField<prt::vx>(pvr::particles::name(), m_real);
  pkg->AddLukewarmField<prt::vy>(pvr::particles::name(), m_real);
  pkg->AddLukewarmField<prt::vz>(pvr::particles::name(), m_real);
  pkg->AddLukewarmField<pvr::r>(pvr::particles::name(), m_real);
  pkg->AddLukewarmField<pvr::g>(pvr::particles::name(), m_real);
  pkg->AddLukewarmField<pvr::b>(pvr::particles::name(), m_real);
  pkg->AddLukewarmField<pvr::a>(pvr::particles::name(), m_real);
  Metadata m_int({Metadata::Integer});
  pkg->AddLukewarmField<pvr::camera_id>(pvr::particles::name(), m_int);
  pkg->AddLukewarmField<pvr::pixel_id>(pvr::particles::name(), m_int);
  pkg->AddLukewarmField<prt::source_block>(pvr::particles::name(), m_int);

  const bool analysis =
      pin->GetOrAddBoolean("parthenon/job", "run_only_analysis", false);
  std::vector<parthenon::MetadataFlag> flags{Metadata::OneCopy, Metadata::Node,
                                             Metadata::Analysis};
  if (analysis) flags.push_back(Metadata::FillGhost);
  Metadata m(flags);
  std::vector<std::string> orig_fields = scene.field;
  std::string suffix = "_nodal_";
  for (auto &f : scene.field) {
    f += suffix;
    pkg->AddField(f, m);
  }
  params.Add("scene_info", scene, true);
  params.Add("orig_fields", orig_fields);
  params.Add("analysis", analysis);
  //RenderingParams rparams(scene.rendering_params);
  //params.Add("rendering_params", rparams);
  params.Add("tracer_info", impl::TracerInfo(scene.field, RenderingParams(scene.rendering_params)), true);

  // One per partition, for Render
  params.Add("nodal_bounds", std::vector<impl::NodalBounds>(), true);
  params.Add("show_volume", ShowVolume(), true);
  params.Add("trace_iterations", std::vector<int>(), true);

  return pkg;
}

bool TimeToRender(Mesh *pm, Real time) {
  if (!riot_viz_initialized) return false;
  auto pkg = pm->packages.Get("riot_viz");
  auto dump_times = pkg->MutableParam<std::queue<Real>>("tdump");
  auto &t_next_dump = *(pkg->MutableParam<Real>("next_dump_time"));
  bool render = false;
  while (time >= t_next_dump) {
    render = true;
    dump_times->pop();
    if (dump_times->size())
      t_next_dump = dump_times->front();
    else
      t_next_dump = std::numeric_limits<Real>::max();
  }
  return render;
}

TaskStatus InitializeCameras(MeshData<Real> *md) {
  namespace prt = particles::ray_tracer;
  namespace pvr = particles::riot_viz;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using ne_t = ccbulk::electron_number_density;
  using nen_t = node_variables::electron_number_density;

  auto &viz = md->GetMeshPointer()->packages.Get("riot_viz");
  const auto &si = viz->Param<SceneInfo>("scene_info");

  const int ndim = md->GetMeshPointer()->ndim;
  int dj = ndim > 1 ? 1 : 0;
  int dk = ndim > 2 ? 1 : 0;

  auto v = riot::MakePack<ne_t, nen_t>(md);

  // Every ray of the last render is done, so the rays created here are all there are
  auto &lukewarm = md->GetLukewarm(pvr::particles::name());
  lukewarm.Clear();

  // Every camera point, face by face, with the mesh face it lies on
  int npts = 0;
  for (int i = 0; i < 6; i++) {
    npts += si.face_pts[i][sample::x].size();
  }
  if (npts == 0) return TaskStatus::complete;
  parthenon::ParArray2D<Real> camera_pts("camera points", npts,
                                         static_cast<int>(sample::nvalues));
  parthenon::ParArray2D<int> camera_idx("camera indexing", npts, 3);
  auto chost = Kokkos::create_mirror_view(Kokkos::HostSpace(), camera_pts);
  auto cid_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), camera_idx);
  for (int i = 0, p = 0; i < 6; i++) {
    for (int j = 0; j < si.face_pts[i][sample::x].size(); j++, p++) {
      for (int n = 0; n < sample::nvalues; n++) {
        chost(p, n) = si.face_pts[i][n][j];
      }
      cid_host(p, 0) = si.camera_id[i][j];
      cid_host(p, 1) = si.pixel_id[i][j];
      cid_host(p, 2) = i;
    }
  }
  Kokkos::deep_copy(camera_pts, chost);
  Kokkos::deep_copy(camera_idx, cid_host);

  // A block face is on the mesh boundary unless it borders another block, and faces along
  // inactive directions always are
  const int nblocks = md->NumBlocks();
  parthenon::ParArray2D<bool> on_boundary("camera boundary faces", nblocks, 6);
  auto bhost = Kokkos::create_mirror_view(Kokkos::HostSpace(), on_boundary);
  for (int b = 0; b < nblocks; b++) {
    auto *pmb = md->GetBlockData(b)->GetBlockPointer();
    for (int f = 0; f < 6; f++) {
      bhost(b, f) =
          f >= 2 * ndim || pmb->boundary_flag[f] != parthenon::BoundaryFlag::block;
    }
  }
  Kokkos::deep_copy(on_boundary, bhost);

  // Rays start where they meet the volume anything could show in, or at their camera
  // point if it is in it already. Those that miss it are background.
  using OnBlock = impl::CameraPointOnBlock;
  const auto &show_volume = viz->Param<ShowVolume>("show_volume");
  const bool clip = CullsBlocks(ndim);
  const RayTrace::BlockBounds mesh_bounds(md->GetMeshPointer()->mesh_size);
  auto &images = si.images;
  parthenon::ParArray1D<int> start_kind("camera start kinds", npts);
  parthenon::ParArray2D<Real> start("camera starts", npts, 3);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, npts - 1,
      KOKKOS_LAMBDA(const int p) {
        start_kind(p) = OnBlock::start_at_point;
        if (!clip) return;
        const Real x[3] = {camera_pts(p, sample::x), camera_pts(p, sample::y),
                           camera_pts(p, sample::z)};
        const Real n[3] = {camera_pts(p, sample::nx), camera_pts(p, sample::ny),
                           camera_pts(p, sample::nz)};
        Real t0, t1;
        show_volume.Clip(camera_idx(p, 0), x, n, show_volume.margin, t0, t1);
        const Real nn = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        bool miss = t1 < std::max(t0, 0.0);
        if (!miss && t0 * nn > show_volume.margin) {
          Real xs[3];
          for (int d = 0; d < 3; d++)
            xs[d] = x[d] + t0 * n[d];
          // the mesh coordinates of the start, as the tracer has them
          Real xm[3] = {xs[0], xs[1], xs[2]};
          if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
            xm[0] = std::sqrt(xs[0] * xs[0] + xs[1] * xs[1]);
            xm[1] = xs[2];
            xm[2] = std::atan2(xs[1], xs[0]);
            xm[2] += (xm[2] < 0.0) * 2.0 * M_PI;
          } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
            xm[0] = std::sqrt(xs[0] * xs[0] + xs[1] * xs[1] + xs[2] * xs[2]);
            xm[1] = xm[0] > 0.0 ? std::acos(xs[2] / xm[0]) : 0.0;
            xm[2] = std::atan2(xs[1], xs[0]);
            xm[2] += (xm[2] < 0.0) * 2.0 * M_PI;
          }
          // the hull is grown past the mesh where the blocks that may show reach it, and
          // a ray that only meets it out there meets nothing that shows
          const int nd = OnBlock::start_dims;
          const Real lo[3] = {mesh_bounds.x_min_global, mesh_bounds.y_min_global,
                              mesh_bounds.z_min_global};
          const Real hi[3] = {mesh_bounds.x_max_global, mesh_bounds.y_max_global,
                              mesh_bounds.z_max_global};
          for (int d = 0; d < nd; d++)
            miss = miss || xm[d] < lo[d] || xm[d] > hi[d];
          for (int d = 0; d < 3; d++)
            start(p, d) = xm[d];
          start_kind(p) = OnBlock::start_inside;
        }
        if (miss) {
          // as a ray that meets nothing would leave it
          start_kind(p) = OnBlock::start_none;
          const int cam = camera_idx(p, 0);
          const int pix = camera_idx(p, 1);
          for (int c = 0; c < 3; c++)
            images(cam, 4 * pix + c) = real_to_uint8_t(0.0);
          images(cam, 4 * pix + 3) = real_to_uint8_t(1.0);
        }
      });

  const OnBlock on_block{md->GetLukewarmGeometry(), camera_pts, camera_idx, on_boundary,
                         start_kind, start};

  // Only the points that could start on this MeshData are tried on each of its blocks:
  // those inside the box of its blocks, or on a mesh face one of them is on
  Kokkos::Array<Real, 3> md_lo, md_hi;
  Kokkos::Array<bool, 6> md_face;
  for (int d = 0; d < 3; d++) {
    md_lo[d] = std::numeric_limits<Real>::max();
    md_hi[d] = -std::numeric_limits<Real>::max();
  }
  for (int f = 0; f < 6; f++)
    md_face[f] = false;
  for (int b = 0; b < nblocks; b++) {
    const auto &bs = md->GetBlockData(b)->GetBlockPointer()->block_size;
    for (int d = 0; d < 3; d++) {
      const auto dir = static_cast<parthenon::CoordinateDirection>(d + 1);
      md_lo[d] = std::min(md_lo[d], bs.xmin(dir));
      md_hi[d] = std::max(md_hi[d], bs.xmax(dir));
    }
    for (int f = 0; f < 6; f++)
      md_face[f] = md_face[f] || bhost(b, f);
  }
  parthenon::ParArray1D<int> cand("camera candidates", npts);
  int ncand = 0;
  Kokkos::parallel_scan(
      PARTHENON_AUTO_LABEL, Kokkos::RangePolicy<DevExecSpace>(0, npts),
      KOKKOS_LAMBDA(const int p, int &k, const bool final) {
        bool c = false;
        if (start_kind(p) == OnBlock::start_inside) {
          c = true;
          for (int d = 0; d < OnBlock::start_dims; d++)
            c = c && start(p, d) >= md_lo[d] && start(p, d) <= md_hi[d];
        } else if (start_kind(p) == OnBlock::start_at_point) {
          c = md_face[camera_idx(p, 2)];
        }
        if (final && c) cand(k) = p;
        k += c;
      },
      ncand);

  // Every block creates a ray at each camera point on its part of the mesh boundary, or
  // in it
  parthenon::ParArray1DRaw<int> counts("camera counts", nblocks);
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, 0, 0, 0, nblocks - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
        int count = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(member, ncand),
            [&](const int k, int &c) { c += on_block(b, cand(k)); }, count);
        Kokkos::single(Kokkos::PerTeam(member), [&]() { counts(b) = count; });
      });
  const auto offsets = lukewarm.Create(counts);

  // the pack is a snapshot, so get it after making space for the rays
  static auto desc =
      parthenon::MakeLukewarmPackDescriptor<lukewarm::x, lukewarm::y, lukewarm::z,
                                            prt::vx, prt::vy, prt::vz, prt::source_block,
                                            pvr::r, pvr::g, pvr::b, pvr::a,
                                            pvr::camera_id, pvr::pixel_id>(
          md, pvr::particles::name());
  auto pack = desc.GetPack(md);
  const auto geom = pack.GetGeometry();

  // set the particle properties
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, 0, 0, 0, nblocks - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
        Kokkos::parallel_scan(
            Kokkos::TeamThreadRange(member, ncand),
            [&](const int k, int &offset, const bool final) {
              const int n = cand(k);
              const bool on = on_block(b, n);
              if (final && on) {
                const int pidx = offsets(b) + offset;
                Real &x = pack(lukewarm::x(), pidx);
                Real &y = pack(lukewarm::y(), pidx);
                Real &z = pack(lukewarm::z(), pidx);
                // nvcc does not let a device lambda first capture a variable in an if
                // constexpr branch, so what the branches use is read before them
                const Real px = camera_pts(n, sample::x);
                const Real py = camera_pts(n, sample::y);
                const Real pz = camera_pts(n, sample::z);
                const Real block_xmax1 = geom.Xmax(b, X1DIR);
                if (start_kind(n) == OnBlock::start_inside) {
                  x = start(n, 0);
                  y = start(n, 1);
                  z = start(n, 2);
                } else if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
                  x = px;
                  y = py;
                  z = pz;
                } else if constexpr (parthenon::IsCoord<
                                         parthenon::UniformCylindrical>()) {
                  Real rcyl = std::sqrt(px * px + py * py);
                  if (std::abs(rcyl - block_xmax1) / block_xmax1 < kBoundaryBand)
                    rcyl = block_xmax1 - kGeomTol;
                  Real phi_cyl = std::atan2(py, px);
                  phi_cyl += (phi_cyl < 0.0) * 2.0 * M_PI;
                  x = rcyl;
                  y = pz;
                  z = phi_cyl;
                } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
                  const Real rsph = std::sqrt(px * px + py * py + pz * pz);
                  const Real th = std::acos(pz / rsph);
                  Real phi_sph = std::atan2(py, px);
                  phi_sph += (phi_sph < 0.0) * 2.0 * M_PI;
                  x = rsph;
                  y = th;
                  z = phi_sph;
                } else {
                  PARTHENON_FAIL("Unsupported coordinate system");
                }
                pack(prt::vx(), pidx) = camera_pts(n, sample::nx);
                pack(prt::vy(), pidx) = camera_pts(n, sample::ny);
                pack(prt::vz(), pidx) = camera_pts(n, sample::nz);
                // The ray may start on the upper face of its block, where it is not
                // located
                pack(prt::source_block(), pidx) = b;
                pack(pvr::r(), pidx) = 0.0;
                pack(pvr::g(), pidx) = 0.0;
                pack(pvr::b(), pidx) = 0.0;
                pack(pvr::a(), pidx) = 0.0;
                pack(pvr::camera_id(), pidx) = camera_idx(n, 0);
                pack(pvr::pixel_id(), pidx) = camera_idx(n, 1);
              }
              offset += on;
            });
      });

  return TaskStatus::complete;
}

TaskStatus Tracer(MeshData<Real> *md, const impl::NodalBounds *bounds, int *iteration) {
  namespace rt = particles::ray_tracer;
  namespace vr = particles::riot_viz;

  // First trace every ray the cameras created, and after that only the ones the last
  // exchange delivered, since every other ray is done
  const bool first = (*iteration)++ == 0;
  const IndexRange range =
      first ? IndexRange{0, md->GetLukewarm(vr::particles::name()).NumParticles() - 1}
            : parthenon::GetLastReceived(md, vr::particles::name());
  if (range.size() <= 0) return TaskStatus::complete;

  auto pm = md->GetParentPointer();
  auto viz = pm->packages.Get("riot_viz");
  auto &scene_info = viz->Param<SceneInfo>("scene_info");
  auto tinfo = viz->MutableParam<impl::TracerInfo>("tracer_info");

  if (!tinfo->initialized) tinfo->initialize(md);

  auto [v, ps] = tinfo->GetPacks(md);

  return RayTrace::Trace<Composer>(ps, v, md->GetParentPointer(), range, first,
                                   tinfo->render_params, tinfo->pack_idx_view,
                                   bounds->may_show,
                                   viz->Param<ShowVolume>("show_volume"),
                                   scene_info.images);
}

TaskStatus InitializeNodalValues(MeshData<Real> *md, impl::NodalBounds *bounds) {
  using TE = parthenon::TopologicalElement;

  auto pm = md->GetMeshPointer();
  auto viz = pm->packages.Get("riot_viz");
  auto orig_fields =
      viz->Param<std::vector<std::string>>("orig_fields");
  auto &scene_info = viz->Param<SceneInfo>("scene_info");
  auto node_fields = scene_info.field;
  auto desc_orig = parthenon::MakePackDescriptor(md, orig_fields);
  auto vorig = desc_orig.GetPack(md);

  auto pack_idx_map = desc_orig.GetMap();
  std::vector<parthenon::PackIdx> pack_idx;
  for (auto &key : orig_fields)
    pack_idx.emplace_back(pack_idx_map.at(key));
  auto pack_idx_view = RiotUtils::VectorToDevice(pack_idx, "pack_idx_view");
  const int nfields = pack_idx.size();

  auto desc_node = parthenon::MakePackDescriptor(md, node_fields);
  auto vnode = desc_node.GetPack(md);
  auto node_pack_idx_map = desc_node.GetMap();
  std::vector<parthenon::PackIdx> node_pack_idx;
  for (auto &key : scene_info.field)
    node_pack_idx.emplace_back(node_pack_idx_map.at(key));
  auto node_pack_idx_view = RiotUtils::VectorToDevice(node_pack_idx, "node_pack_idx_view");

  PARTHENON_REQUIRE_THROWS(vorig.GetNBlocks() == vnode.GetNBlocks(), "oops");
  PARTHENON_REQUIRE_THROWS(vorig.GetNBlocks() == md->NumBlocks(), "oops");

  PARTHENON_REQUIRE_THROWS(nfields <= 2 * max_layers,
                           "Too many distinct visualization fields.");
  auto max_value = parthenon::ParArray2D<Real>("max_nodal_value", md->NumBlocks(), 2*max_layers);
  auto min_value = parthenon::ParArray2D<Real>("min_nodal_value", md->NumBlocks(), 2*max_layers);
  parthenon::par_for(DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(),
    0, md->NumBlocks() - 1, 0, 2 * max_layers - 1,
    KOKKOS_LAMBDA(const int b, const int l) {
      max_value(b, l) = std::numeric_limits<Real>::lowest();
      min_value(b, l) = std::numeric_limits<Real>::max();
    });

  Kokkos::fence();

  int dj = pm->ndim > 1 ? 1 : 0;
  int dk = pm->ndim > 2 ? 1 : 0;
  // In analysis mode only the first ghost layer of file-backed fields is communicated
  const int ngc = viz->Param<bool>("analysis") ? 1 : parthenon::Globals::nghost;
  const auto ibi = md->GetBoundsI(IndexDomain::interior);
  const auto jbi = md->GetBoundsJ(IndexDomain::interior);
  const auto kbi = md->GetBoundsK(IndexDomain::interior);
  const int ics = ibi.s - ngc;
  const int jcs = jbi.s - ngc * dj;
  const int kcs = kbi.s - ngc * dk;
  const int ice = ibi.e + ngc;
  const int jce = jbi.e + ngc * dj;
  const int kce = kbi.e + ngc * dk;

  using lt = RiotUtils::LoopType<LoopConstraint::DifferentMemSpaces>;
  auto idx_space =
      lt::GetIndexSpace(IndexDomain::entire, 0, vorig.GetNBlocks(), md, TE::NN);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const auto dkl = dk;
        const auto djl = dj;

        for (int n = 0; n < nfields; n++) {
          const auto nn = vnode.GetLowerBound(b, node_pack_idx_view(n));
          const auto vlo = vorig.GetLowerBound(b, pack_idx_view(n));
          const auto vhi = vorig.GetUpperBound(b, pack_idx_view(n));
          if (vlo >= 0) {
            PARTHENON_REQUIRE(vhi == vlo, "Viz only supports scalars.");
            for (int iv = vlo; iv <= vhi; iv++) {
              auto cell = RiotLoop::make_var_view(idx_range, vorig, iv);
              RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
                Real nodal_val = 0.0;
                int cnt = 0;
                const int kl = std::clamp(k - dkl, kcs, kce);
                const int jl = std::clamp(j - djl, jcs, jce);
                const int il = std::clamp(i - 1, ics, ice);
                for (int kc = kl; kc <= std::clamp(k, kl, kce); kc++) {
                  for (int jc = jl; jc <= std::clamp(j, jl, jce); jc++) {
                    for (int ic = il; ic <= std::clamp(i, il, ice); ic++) {
                      nodal_val += cell(kc, jc, ic);
                      cnt++;
                    }
                  }
                }
                nodal_val /= cnt;
                vnode(b, nn + iv - vlo, k, j, i) = nodal_val;
                Kokkos::atomic_min(&min_value(b, n), nodal_val);
                Kokkos::atomic_max(&max_value(b, n), nodal_val);
              });
            }
          } else {
            for (int iv = nn; iv <= vnode.GetUpperBound(b, node_pack_idx_view(n)); iv++) {
              RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
                vnode(b, iv, k, j, i) = 0.0;
              });
              Kokkos::atomic_store(&min_value(b, n), std::numeric_limits<Real>::quiet_NaN());
              Kokkos::atomic_store(&max_value(b, n), std::numeric_limits<Real>::quiet_NaN());
            }
          }
        }
      });
  
  //parthenon::par_for_outer

  Kokkos::fence();

  // the layers that could show in each block, so rays need not check that per block
  const auto &rp = viz->Param<impl::TracerInfo>("tracer_info").render_params;
  const int ncam = scene_info.camera.size();
  auto may_show = parthenon::ParArray2D<uint32_t>("may_show", md->NumBlocks(), ncam);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, md->NumBlocks() - 1,
      0, ncam - 1, KOKKOS_LAMBDA(const int b, const int cam) {
        uint32_t may = 0;
        for (int ilay = 0; ilay < rp.nlayers(cam); ilay++) {
          if (LayerMayShow(rp, cam, ilay, b, max_value, min_value)) may |= 1u << ilay;
        }
        may_show(b, cam) = may;
      });
  Kokkos::fence();

  // and the box of the blocks where any may, for ReduceShowVolume
  auto may_show_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), may_show);
  constexpr Real big = std::numeric_limits<Real>::max();
  auto &box = bounds->show_box;
  box.assign(6 * ncam, 0.0);
  for (int cam = 0; cam < ncam; cam++) {
    for (int d = 0; d < 3; d++) {
      box[6 * cam + 2 * d] = big;
      box[6 * cam + 2 * d + 1] = -big;
    }
    for (int b = 0; b < md->NumBlocks(); b++) {
      if (may_show_h(b, cam) == 0) continue;
      const auto &bs = md->GetBlockData(b)->GetBlockPointer()->block_size;
      for (int d = 0; d < 3; d++) {
        const auto dir = static_cast<parthenon::CoordinateDirection>(d + 1);
        box[6 * cam + 2 * d] = std::min(box[6 * cam + 2 * d], bs.xmin(dir));
        box[6 * cam + 2 * d + 1] = std::max(box[6 * cam + 2 * d + 1], bs.xmax(dir));
      }
    }
  }

  bounds->max_value = std::move(max_value);
  bounds->min_value = std::move(min_value);
  bounds->may_show = std::move(may_show);

  return TaskStatus::complete;
}

// The volume anything could show in, from the boxes of every partition on every rank
TaskStatus ReduceShowVolume(Mesh *pm) {
  auto viz = pm->packages.Get("riot_viz");
  const auto &bounds = viz->Param<std::vector<impl::NodalBounds>>("nodal_bounds");
  auto show_volume = viz->MutableParam<ShowVolume>("show_volume");
  const int ncam = viz->Param<SceneInfo>("scene_info").camera.size();

  // lower bounds are negated, so that one MPI_MAX reduces them along with the upper ones
  std::vector<Real> box(6 * ncam, -std::numeric_limits<Real>::max());
  for (const auto &bd : bounds) {
    for (int i = 0; i < 6 * ncam; i++) {
      box[i] = std::max(box[i], i % 2 == 0 ? -bd.show_box[i] : bd.show_box[i]);
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, box.data(), box.size(), MPI_PARTHENON_REAL, MPI_MAX,
                MPI_COMM_WORLD);

  if (show_volume->box.extent_int(0) != ncam) {
    show_volume->box = parthenon::ParArray2D<Real>("show volume", ncam, 6);
  }
  auto box_h = show_volume->box.GetHostMirror();
  for (int cam = 0; cam < ncam; cam++) {
    for (int i = 0; i < 6; i++) {
      box_h(cam, i) = i % 2 == 0 ? -box[6 * cam + i] : box[6 * cam + i];
    }
  }
  Kokkos::deep_copy(show_volume->box, box_h);

  // well above roundoff, and well below any cell
  const auto &mesh_size = pm->mesh_size;
  Real extent = mesh_size.xmax(X1DIR) - mesh_size.xmin(X1DIR);
  if (!parthenon::IsCoord<parthenon::UniformSpherical>())
    extent = std::max(extent, mesh_size.xmax(X2DIR) - mesh_size.xmin(X2DIR));
  if (parthenon::IsCoord<parthenon::UniformCartesian>())
    extent = std::max(extent, mesh_size.xmax(X3DIR) - mesh_size.xmin(X3DIR));
  show_volume->margin = 1.e-6 * extent;
  return TaskStatus::complete;
}

TaskCollection Render(Mesh *pm, Real time) {
  namespace pvr = particles::riot_viz;
  TaskCollection tc;
  TaskID none;
  auto viz = pm->packages.Get("riot_viz");
  // Shared by the task lists, and kept by the last task so it lives as long as they do.
  // Freeing it is collective, so it is not kept in the package: ranks free packages in
  // different orders, which deadlocks against the reductions of other packages.
  auto num_active = std::make_shared<AllReduce<int>>();
  auto iterations = viz->MutableParam<std::vector<int>>("trace_iterations");
  auto bounds = viz->MutableParam<std::vector<impl::NodalBounds>>("nodal_bounds");

  const int num_partitions = pm->DefaultNumPartitions();
  iterations->assign(num_partitions, 0);
  bounds->resize(num_partitions);
  auto &reg = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = reg[i];
    auto md_sp = pm->mesh_data.GetOrAdd("base", i);
    auto md = md_sp.get();
    auto *md_bounds = &(*bounds)[i];

    auto set_nodal_vals = tl.AddTask(none, InitializeNodalValues, md, md_bounds);
    if (viz->Param<bool>("analysis")) {
      auto &md_nodes = pm->mesh_data.AddShallow(
          "riot_viz_nodes", md_sp, viz->Param<SceneInfo>("scene_info").field);
      parthenon::AddBoundaryExchangeTasks(
          set_nodal_vals, tl, md_nodes, pm->multilevel,
          [](TaskID id, TaskList *, std::shared_ptr<MeshData<Real>>, bool) { return id; });
    }
  }
  // The cameras start rays where they meet the volume anything could show in, which
  // takes the nodal values of every block
  auto &show_reg = tc.AddRegion(1);
  show_reg[0].AddTask(none, ReduceShowVolume, pm);
  auto &trace_reg = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = trace_reg[i];
    auto md = pm->mesh_data.GetOrAdd("base", i).get();
    auto *md_bounds = &(*bounds)[i];
    int *iteration = &(*iterations)[i];
    auto init_particles = tl.AddTask(none, InitializeCameras, md);

    // Trace every ray until it is done or leaves this MeshData, and exchange, until no
    // ray on any MeshData of any rank is left to trace
    auto [itl, push] = tl.AddSublist(init_particles, {1, 100000});
    auto transport = itl.AddTask(none, Tracer, md, md_bounds, iteration);
    auto exchange =
        parthenon::AddLukewarmExchangeTasks(transport, itl, md, pvr::particles::name());
    // Every ray an exchange delivers still has to be traced
    auto complete = parthenon::AddLukewarmCompletionTasks(
        exchange, itl, md, num_active.get(), [](MeshData<Real> *md) {
          return GetLastReceived(md, pvr::particles::name()).size();
        });
  }
  auto &dump_reg = tc.AddRegion(1);
  auto &tl = dump_reg[0];
  tl.AddTask(none, "DumpImages",
             [pm, time, num_active]() { return DumpImages(pm, time); });
  return tc;
}

void draw_char(uint8_t *rgba, int width, int height, int x0, int y0, char c,
               std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) {
  const auto *glyph = (font8x8_basic[static_cast<unsigned char>(c)]);

  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {

      if (!(glyph[y] & (1u << x))) continue;

      const int px = x0 + x;
      const int py = y0 + y;

      if (px < 0 || px >= width || py < 0 || py >= height) continue;

      const std::size_t i = 4 * (static_cast<std::size_t>(py) * width + px);

      rgba[i + 0] = r;
      rgba[i + 1] = g;
      rgba[i + 2] = b;
      rgba[i + 3] = a;
    }
  }
}

void draw_text(uint8_t *rgba, int width, int height, int x, int y,
               const std::string &text, std::uint8_t r = 255, std::uint8_t g = 255,
               std::uint8_t b = 255, std::uint8_t a = 255, bool centered = true) {
  int num_chars = text.size();
  int pixel_width = 8 * num_chars;
  int offset = centered ? pixel_width / 2 : 0;
  for (char c : text) {
    draw_char(rgba, width, height, x - offset, y, c, r, g, b, a);

    x += 8;
  }
}

std::string FormatReal(Real value) {
  std::ostringstream out;

  const Real magnitude = std::abs(value);
  if (magnitude != 0.0 && (magnitude < 1.0e-3 || magnitude >= 1.0e4)) {
    out << std::scientific;
  } else {
    out << std::fixed;
  }

  out << std::setprecision(2) << value;
  return out.str();
}

std::string FormatMinMax(double min_val, double max_val) {
  return FormatReal(min_val) + " / " + FormatReal(max_val);
}
std::vector<std::uint8_t>
resize_rgba(const void *src,
            int src_w, int src_h,
            int dst_w, int dst_h)
{
    std::vector<std::uint8_t> dst(
        static_cast<std::size_t>(dst_w) * dst_h * 4);

    auto filter = STBIR_FILTER_CATMULLROM;
    stbir_resize(
        src,
        src_w,
        src_h,
        0,                  // tightly packed source stride
        dst.data(),
        dst_w,
        dst_h,
        0,                  // tightly packed destination stride
        STBIR_RGBA,
        STBIR_TYPE_UINT8,
        STBIR_EDGE_CLAMP,
        filter
    );

    return dst;
}

int write_png(char const *filename, int w, int h, int comp,
              const void *data, int stride_in_bytes, Real scale) {
  if (scale == 1.0) return stbi_write_png(filename, w, h, comp, data, stride_in_bytes);

  int nw = scale * w;
  int nh = scale * h;
  auto rescaled = resize_rgba(data, w, h, nw, nh);
  return stbi_write_png(filename, nw, nh, comp, rescaled.data(), nw * 4);
}

TaskStatus DumpImages(Mesh *pm, Real time) {
  namespace vr = particles::riot_viz;

  auto viz = pm->packages.Get("riot_viz");
  auto &dump_id = *(viz->MutableParam<int>("dump_id"));
  bool timebar = viz->Param<bool>("timebar");
  auto &time_rgb = viz->Param<std::vector<Real>>("timebar_rgb");
  auto time_start = viz->Param<Real>("first_dump_time");
  auto time_stop = viz->Param<Real>("last_dump_time");
  auto scene_info = viz->MutableParam<SceneInfo>("scene_info");

  auto &images = scene_info->images;
  auto md = pm->mesh_data.Get("base").get();

  Kokkos::fence();
  auto host_images = images.GetHostMirrorAndCopy();
  if (parthenon::Globals::my_rank == 0)
    MPI_Reduce(MPI_IN_PLACE, host_images.data(), host_images.size(), MPI_UINT8_T, MPI_MAX,
               0, MPI_COMM_WORLD);
  else
    MPI_Reduce(host_images.data(), NULL, host_images.size(), MPI_UINT8_T, MPI_MAX, 0,
               MPI_COMM_WORLD);
  if (parthenon::Globals::my_rank == 0) {
    auto &camera = scene_info->camera;
    for (int i = 0; i < camera.size(); i++) {
      size_t max_label_width = 0;
      size_t max_range_width = 0;
      for (auto &layer : camera[i].layer) {
        if (!layer.colorbar) continue;
        auto layer_label = layer.scale_type == ScaleType::log
                               ? "log10(" + layer.label + ")"
                               : layer.label;
        max_label_width = std::max(max_label_width, layer_label.size());
        auto range_str = FormatMinMax(layer.min_range, layer.max_range);
        max_range_width = std::max(max_range_width, range_str.size());
      }
      max_label_width = 8 * max_label_width + 2;
      max_range_width = 8 * max_range_width + 2;
      int w = camera[i].nwidth;
      int h = camera[i].nheight;
      Real scale = camera[i].scale;
      // set the colormap pixels
      int t = camera[i].colorbar_thickness;
      int total_h = h + (camera[i].ncolorbar + timebar) * t;
      uint8_t alpha_max = 0;
      //for (int j = 3; j < 4 * w * total_h; j+=4) {
        //if (host_images(i, j) > alpha_max) alpha_max = host_images(i, j);
      //}
      //for (int j = 3; j < 4 * w * total_h; j+=4) {
        //host_images(i, j) *= (255.0 / alpha_max);
      //}
      int ilabel = 0;
      if (timebar) {
        int jstart = 4 * w * h;
        for (int j = jstart; j < jstart + 4 * w * t; j += 4) {
          int ix = ((j - jstart) % (4 * w)) / 4;
          Real x = (time - time_start) / (time_stop - time_start);
          int ixt = x * w;
          if (ix <= ixt) {
            host_images(i, j + 0) = real_to_uint8_t(time_rgb[0]);
            host_images(i, j + 1) = real_to_uint8_t(time_rgb[1]);
            host_images(i, j + 2) = real_to_uint8_t(time_rgb[2]);
            host_images(i, j + 3) = 255;
          } else {
            host_images(i, j + 0) = real_to_uint8_t(0.7 * time_rgb[0]);
            host_images(i, j + 1) = real_to_uint8_t(0.7 * time_rgb[1]);
            host_images(i, j + 2) = real_to_uint8_t(0.7 * time_rgb[2]);
            host_images(i, j + 3) = 255;
          }
        }
        draw_text(&host_images(i, 0), w, total_h, w / 2, h + 1,
                  "Time = " + FormatReal(time), real_to_uint8_t(1.0 - time_rgb[0]),
                  real_to_uint8_t(1.0 - time_rgb[1]), real_to_uint8_t(1.0 - time_rgb[2]));
      }
      for (auto &layer : camera[i].layer) {
        if (!layer.colorbar) continue;
        Real amax = layer.acmap.max();
        int jstart = 4 * w * h + 4 * (ilabel + timebar) * w * t;
        for (int j = jstart; j < jstart + 4 * w * t; j += 4) {
          int ix = ((j - jstart) % (4 * w)) / 4;
          if (ix < max_label_width || (w - ix - 1) < max_range_width) {
            host_images(i, j + 0) = 0;
            host_images(i, j + 1) = 0;
            host_images(i, j + 2) = 0;
            host_images(i, j + 3) = 255;
            continue;
          }
          Real x =
              (1.0 * (ix - max_label_width)) / (w - max_label_width - max_range_width);
          int iy = t - (j - jstart) / (4 * w) - 1;
          Real a = layer.acmap.interpToReal(x) / amax;
          int iyl = a * (t - 2) + 1;
          if (iy == 0) {
            host_images(i, j + 0) = 0;
            host_images(i, j + 1) = 0;
            host_images(i, j + 2) = 0;
            host_images(i, j + 3) = 255;
          } else {
            if (iy == iyl && layer.type == VizType::volume) {
              host_images(i, j + 0) = 255 * (ix % 2);
              host_images(i, j + 1) = 255 * (ix % 2);
              host_images(i, j + 2) = 255 * (ix % 2);
              host_images(i, j + 3) = 255;
            } else {
              Real darken = (iy > iyl && layer.type == VizType::volume) ? 0.5 : 1.0;
              host_images(i, j + 0) = real_to_uint8_t(darken * layer.rcmap.interpToReal(x));
              host_images(i, j + 1) = real_to_uint8_t(darken * layer.gcmap.interpToReal(x));
              host_images(i, j + 2) = real_to_uint8_t(darken * layer.bcmap.interpToReal(x));
              host_images(i, j + 3) = 255;
            }
          }
        }
        auto layer_label = layer.scale_type == ScaleType::log
                               ? "log10(" + layer.label + ")"
                               : layer.label;
        draw_text(&host_images(i, 0), w, total_h, max_label_width / 2,
                  h + (ilabel + 1 + timebar) * t - 9, layer_label, 255, 255, 255);
        draw_text(&host_images(i, 0), w, total_h, w - max_range_width / 2,
                  h + (ilabel + 1 + timebar) * t - 9,
                  FormatMinMax(layer.min_range, layer.max_range), 255, 255, 255);
        ilabel++;
      }

      // Move to this commented out line when we go to C++20 and add include <format>
      //std::string name = camera[i].filename + std::format("{:04d}", dump_id) + ".png";
      std::ostringstream ss;
      ss << camera[i].filename << std::setfill('0') << std::setw(4) << dump_id << ".png";
      std::string name = ss.str();
      // stbi_write_png
      if (!write_png(name.c_str(), w, total_h,
                     4, // RGBA
                     &host_images(i, 0), w * 4,
                     scale)) {
        throw std::runtime_error("Failed to write PNG");
      }
    }
  }
  dump_id++;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, images.GetDim(2) - 1,
      0, images.GetDim(1) - 1,
      KOKKOS_LAMBDA(const int j, const int i) { images(j, i) = 0; });

  return TaskStatus::complete;
}

} // namespace riot_viz

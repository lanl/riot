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
#include <format>
#include <iomanip>
#include <sstream>

#include "riot_viz.hpp"

#include <riot_utils/riot_loops.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

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
  std::vector<Real> layer_range_min, layer_range_max, layer_alpha_range_min,
      layer_alpha_range_max;
  std::vector<GradType> layer_use_grad, layer_alpha_use_grad;
  std::vector<DataBox> layer_r, layer_g, layer_b, layer_a;
  std::vector<std::vector<Real>> layer_c, layer_rc, layer_gc, layer_bc, layer_ac;
  std::vector<std::vector<Real>> layer_slice_x, layer_slice_n;
  std::vector<std::vector<mask_tuple_t>> layer_masks;
  std::vector<std::vector<int>> layer_mask_id;
  for (auto &layer : layers) {
    auto type = pin->Get<std::string>(layer, "type");
    auto field = pin->Get<std::string>(layer, "field");
    auto field_use_grad = pin->GetOrAdd<std::string>(layer, "field_use_grad", "none");
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
      contour_alpha[i] = std::clamp(contour_alpha[i], 0.0, 1.0);
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

    layer_type.push_back(VizTypeMap.at(type));
    layer_pack_idx.push_back(var_index);
    layer_alpha_pack_idx.push_back(alpha_var_index);
    layer_range_min.push_back(cmin);
    layer_range_max.push_back(cmax);
    layer_alpha_range_min.push_back(amin);
    layer_alpha_range_max.push_back(amax);
    layer_use_grad.push_back(GradTypeMap.at(field_use_grad));
    layer_alpha_use_grad.push_back(GradTypeMap.at(field_alpha_use_grad));
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
    layer_masks.push_back(masks_vec);
    layer_mask_id.push_back(mask_id);

    // add this layer to the camera
    camera.layer.emplace_back(MakeNormalizedDataBox(red), MakeNormalizedDataBox(green),
                              MakeNormalizedDataBox(blue), MakeNormalizedDataBox(alpha),
                              label, colorbar, cmin, cmax, VizTypeMap.at(type));
  }

  rp.type.push_back(layer_type);
  rp.pack_idx.push_back(layer_pack_idx);
  rp.alpha_pack_idx.push_back(layer_alpha_pack_idx);
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
            if (x < xmin[dd] - 1.e-12 || x > xmax[dd] + 1.e-12) return false;
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
            if (rt < xmax[0] + 1.e-12 &&
                (trial[2] > xmin[1] - 1e-12 && trial[2] < xmax[1] + 1.e-12)) {
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
                 ? xmin[face_dir] + (xmax[face_dir] - xmin[face_dir]) * 1.e-12
                 : xmax[face_dir] - (xmax[face_dir] - xmin[face_dir]) * 1.e-12);
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
        auto rb = xmax[0] * (1.0 - 1.e-14);
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
    while (tdump.back() + dt_viz < tlim + 1.e-14 * dt_viz)
      tdump.push(tdump.back() + dt_viz);
    if (std::abs(tlim - tdump.back()) < 1.e-14 * dt_viz) tdump.back() = tlim;
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
                                          camera.ncolorbar * camera.colorbar_thickness *
                                              camera.nwidth +
                                          timebar * 10 * camera.nwidth);
        nimages++;
      }
    }
  }

  // don't forget the 4 because it's 4 channels (RGBA)
  scene.images =
      ParArrayND<uint8_t>("Volume Render images", scene.camera.size(), 4 * max_size);

  RenderingParams rparams(scene.rendering_params);
  params.Add("rendering_params", rparams);

  Metadata mswarm({Metadata::Provides, Metadata::None});
  pkg->AddSwarm(pvr::particles::name(), mswarm);
  Metadata m_real({Metadata::Real});
  pkg->AddSwarmValue<prt::vx, pvr::particles>(m_real);
  pkg->AddSwarmValue<prt::vy, pvr::particles>(m_real);
  pkg->AddSwarmValue<prt::vz, pvr::particles>(m_real);
  pkg->AddSwarmValue<pvr::r, pvr::particles>(m_real);
  pkg->AddSwarmValue<pvr::g, pvr::particles>(m_real);
  pkg->AddSwarmValue<pvr::b, pvr::particles>(m_real);
  pkg->AddSwarmValue<pvr::a, pvr::particles>(m_real);
  Metadata m_int({Metadata::Integer});
  pkg->AddSwarmValue<pvr::camera_id, pvr::particles>(m_int);
  pkg->AddSwarmValue<pvr::pixel_id, pvr::particles>(m_int);

  Metadata m({Metadata::OneCopy, Metadata::Node, Metadata::CellMemAligned});
  std::vector<std::string> orig_fields = scene.field;
  std::string suffix = "_nodal_";
  for (auto &f : scene.field) {
    f += suffix;
    pkg->AddField(f, m);
  }
  params.Add("scene_info", scene, true);
  params.Add("orig_fields", orig_fields);

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

  auto &vrend = md->GetMeshPointer()->packages.Get("riot_viz");
  const auto &si = vrend->Param<SceneInfo>("scene_info");

  const int ndim = md->GetMeshPointer()->ndim;
  int dj = ndim > 1 ? 1 : 0;
  int dk = ndim > 2 ? 1 : 0;

  auto v = riot::MakePack<ne_t, nen_t>(md);
  for (int blk = 0; blk < md->NumBlocks(); blk++) {
    auto &mbd = md->GetBlockData(blk);
    auto *pmb = mbd->GetBlockPointer();

    const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    // skip if not a boundary block
    // note: this is where we would add slicing, cutouts, etc
    bool is_boundary = false;
    std::array<bool, 6> bound_flag;
    for (int i = 0; i < 2 * ndim; i++) {
      if (pmb->boundary_flag[i] != parthenon::BoundaryFlag::block) {
        is_boundary = true;
        bound_flag[i] = true;
      } else {
        bound_flag[i] = false;
      }
    }
    for (int i = 2 * ndim; i < 6; i++) {
      bound_flag[i] = true;
      is_boundary = true;
    }
    if (!is_boundary) continue;

    auto &coords = pmb->coords;
    const std::array<Real, 3> block_xmin(
        {coords.Xf<X1DIR>(ib.s), coords.Xf<X2DIR>(jb.s), coords.Xf<X3DIR>(kb.s)});
    // assume Dxf is constant
    const std::array<Real, 3> dx(
        {coords.Dxf<X1DIR>(0), coords.Dxf<X2DIR>(0), coords.Dxf<X3DIR>(0)});
    const std::array<int, 3> Ni({ib.e - ib.s + 1, jb.e - jb.s + 1, kb.e - kb.s + 1});
    std::array<Real, 3> block_xmax;
    for (int i = 0; i < 3; i++)
      block_xmax[i] = block_xmin[i] + Ni[i] * dx[i];

    // figure out which points are on this block boundary
    std::array<std::vector<int>, 6> ind;
    int total_number = 0;
    for (int i = 0; i < 6; i++) {
      if (!bound_flag[i]) continue;
      for (int j = 0; j < si.face_pts[i][sample::x].size(); j++) {
        bool on_block = false;
        if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
          bool inside_x = (si.face_pts[i][sample::x][j] >= block_xmin[0] &&
                           si.face_pts[i][sample::x][j] <= block_xmax[0]);
          bool inside_y = (si.face_pts[i][sample::y][j] >= block_xmin[1] &&
                           si.face_pts[i][sample::y][j] <= block_xmax[1]);
          bool inside_z = (si.face_pts[i][sample::z][j] >= block_xmin[2] &&
                           si.face_pts[i][sample::z][j] <= block_xmax[2]);
          if (i < 2) {
            // x-boundary, check y and z
            on_block = inside_y && inside_z;
          } else if (i < 4) {
            // y-boundary, check x and z
            on_block = inside_x && inside_z;
          } else {
            on_block = inside_x && inside_y;
          }
        } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
          auto x = si.face_pts[i][sample::x][j];
          auto y = si.face_pts[i][sample::y][j];
          auto z = si.face_pts[i][sample::z][j];
          auto r = std::sqrt(x * x + y * y);
          bool inside_r = (r >= block_xmin[0] && r <= block_xmax[0]);
          bool inside_z = (si.face_pts[i][sample::z][j] >= block_xmin[1] &&
                           si.face_pts[i][sample::z][j] <= block_xmax[1]);
          if (i < 2) {
            on_block = inside_z;
          } else if (i < 4) {
            on_block = inside_r;
          } else {
            PARTHENON_FAIL("Cylindrical does not support 3D");
          }
        } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
          on_block = true;
        }
        if (on_block) {
          ind[i].push_back(j);
        }
      }
      total_number += ind[i].size();
    }

    // now make a device side array that lists the included points
    parthenon::ParArray2D<Real> camera_pts("camera points", total_number,
                                           static_cast<int>(sample::nvalues));
    parthenon::ParArray2D<int> camera_idx("camera indexing", total_number, 2);
    auto chost = Kokkos::create_mirror_view(Kokkos::HostSpace(), camera_pts);
    auto cid_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), camera_idx);
    int idx = 0;
    for (int i = 0; i < 6; i++) {
      if (!bound_flag[i]) continue;
      for (int j = 0; j < ind[i].size(); j++) {
        int k = ind[i][j];
        for (int n = 0; n < sample::nvalues; n++) {
          chost(idx, n) = si.face_pts[i][n][k];
        }
        cid_host(idx, 0) = si.camera_id[i][k];
        cid_host(idx, 1) = si.pixel_id[i][k];
        idx++;
      }
    }
    Kokkos::deep_copy(camera_pts, chost);
    Kokkos::deep_copy(camera_idx, cid_host);

    // make space for the particles
    auto *swarm = md->GetSwarmData(blk)->Get(pvr::particles::name()).get();
    parthenon::ParArray1D<parthenon::NewParticlesContext> new_context("New context", 1);
    auto new_context_h = new_context.GetHostMirror();
    new_context_h(0) = swarm->AddEmptyParticles(total_number);
    new_context.DeepCopy(new_context_h);
    Kokkos::fence();

    auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
    auto &vx = swarm->Get<Real>(prt::vx::name()).Get();
    auto &vy = swarm->Get<Real>(prt::vy::name()).Get();
    auto &vz = swarm->Get<Real>(prt::vz::name()).Get();
    auto &r = swarm->Get<Real>(pvr::r::name()).Get();
    auto &g = swarm->Get<Real>(pvr::g::name()).Get();
    auto &b = swarm->Get<Real>(pvr::b::name()).Get();
    auto &a = swarm->Get<Real>(pvr::a::name()).Get();
    auto &cam_id = swarm->Get<int>(pvr::camera_id::name()).Get();
    auto &pix_id = swarm->Get<int>(pvr::pixel_id::name()).Get();

    // set the particle properties
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, total_number - 1,
        KOKKOS_LAMBDA(const int n) {
          const int pidx = new_context(0).GetNewParticleIndex(n);
          if constexpr (parthenon::IsCoord<parthenon::UniformCartesian>()) {
            x(pidx) = camera_pts(n, sample::x);
            y(pidx) = camera_pts(n, sample::y);
            z(pidx) = camera_pts(n, sample::z);
          } else if constexpr (parthenon::IsCoord<parthenon::UniformCylindrical>()) {
            Real rcyl = std::sqrt(camera_pts(n, sample::x) * camera_pts(n, sample::x) +
                                  camera_pts(n, sample::y) * camera_pts(n, sample::y));
            if (std::abs(rcyl - block_xmax[0]) / block_xmax[0] < 1.e-6)
              rcyl = block_xmax[0] - 1.e-12;
            const Real zcyl = camera_pts(n, sample::z);
            Real phi_cyl = std::atan2(camera_pts(n, sample::y), camera_pts(n, sample::x));
            phi_cyl += (phi_cyl < 0.0) * 2.0 * M_PI;
            x(pidx) = rcyl;
            y(pidx) = zcyl;
            z(pidx) = phi_cyl;
          } else if constexpr (parthenon::IsCoord<parthenon::UniformSpherical>()) {
            const Real rsph =
                std::sqrt(camera_pts(n, sample::x) * camera_pts(n, sample::x) +
                          camera_pts(n, sample::y) * camera_pts(n, sample::y) +
                          camera_pts(n, sample::z) * camera_pts(n, sample::z));
            const Real th = std::acos(camera_pts(n, sample::z) / rsph);
            Real phi_sph = std::atan2(camera_pts(n, sample::y), camera_pts(n, sample::x));
            phi_sph += (phi_sph < 0.0) * 2.0 * M_PI;
            x(pidx) = rsph;
            y(pidx) = th;
            z(pidx) = phi_sph;
          } else {
            PARTHENON_FAIL("Unsupported coordinate system");
          }
          vx(pidx) = camera_pts(n, sample::nx);
          vy(pidx) = camera_pts(n, sample::ny);
          vz(pidx) = camera_pts(n, sample::nz);
          r(pidx) = 0.0;
          g(pidx) = 0.0;
          b(pidx) = 0.0;
          a(pidx) = 0.0;
          cam_id(pidx) = camera_idx(n, 0);
          pix_id(pidx) = camera_idx(n, 1);
        });

  } // loop over blocks in MeshData

  return TaskStatus::complete;
}

TaskStatus Tracer(MeshData<Real> *md) {
  namespace rt = particles::ray_tracer;
  namespace vr = particles::riot_viz;

  auto pm = md->GetParentPointer();
  auto vrend = pm->packages.Get("riot_viz");
  auto render_params = vrend->Param<RenderingParams>("rendering_params");
  auto &scene_info = vrend->Param<SceneInfo>("scene_info");
  auto desc = parthenon::MakePackDescriptor(md, scene_info.field);
  ;
  auto v = desc.GetPack(md);

  auto desc_ps =
      parthenon::MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y,
                                         swarm_position::z, rt::vx, rt::vy, rt::vz, vr::r,
                                         vr::g, vr::b, vr::a>(vr::particles::name());
  auto ps = desc_ps.GetPack(md);

  auto desc_ps_int =
      parthenon::MakeSwarmPackDescriptor<vr::camera_id>(vr::particles::name());
  auto ps_int = desc_ps_int.GetPack(md);

  return RayTrace::Trace<Composer, false, false>(ps, v, md->GetParentPointer(), ps_int,
                                                 render_params);
}

TaskStatus InitializeNodalValues(MeshData<Real> *md) {
  using TE = parthenon::TopologicalElement;

  auto pm = md->GetMeshPointer();
  auto orig_fields =
      pm->packages.Get("riot_viz")->Param<std::vector<std::string>>("orig_fields");
  auto node_fields = pm->packages.Get("riot_viz")->Param<SceneInfo>("scene_info").field;
  auto desc_orig = parthenon::MakePackDescriptor(md, orig_fields);
  auto vorig = desc_orig.GetPack(md);
  auto desc_node = parthenon::MakePackDescriptor(md, node_fields);
  auto vnode = desc_node.GetPack(md);

  PARTHENON_REQUIRE_THROWS(vorig.GetNBlocks() == vnode.GetNBlocks(), "oops");
  PARTHENON_REQUIRE_THROWS(vorig.GetNBlocks() == md->NumBlocks(), "oops");

  int dj = pm->ndim > 1 ? 1 : 0;
  int dk = pm->ndim > 2 ? 1 : 0;

  using lt = RiotUtils::LoopType<>;
  auto idx_space =
      lt::GetIndexSpace(IndexDomain::interior, 0, vorig.GetNBlocks(), md, TE::NN);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const auto dkl = dk;
        const auto djl = dj;

        for (int iv = vorig.GetLowerBound(b); iv <= vorig.GetUpperBound(b); iv++) {
          auto cell = RiotLoop::make_var_view(idx_range, vorig, iv);
          RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
            Real nodal_val = 0.0;
            int cnt = 0;
            for (int kk = -1 * dkl; kk <= 0; kk++) {
              for (int jj = -1 * djl; jj <= 0; jj++) {
                for (int ii = -1; ii <= 0; ii++) {
                  nodal_val += cell(k + kk, j + jj, i + ii);
                  cnt++;
                }
              }
            }
            vnode(b, iv, k, j, i) = nodal_val / cnt;
          });
        }
      });

  return TaskStatus::complete;
}

TaskCollection Render(Mesh *pm, Real time) {
  using TQ = TaskQualifier;
  TaskCollection tc;
  TaskID none;
  // this must be 1 for now until pack_size != -1 is supported for swarms in parthenon
  constexpr int num_partitions = 1; // pm->DefaultNumPartitions();
  auto &reg = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = reg[i];
    auto md_sp = pm->mesh_data.Get();
    auto md = md_sp.get();

    auto set_nodal_vals = tl.AddTask(none, InitializeNodalValues, md);
    auto init_particles = tl.AddTask(none, InitializeCameras, md);
    auto [itl, push] = tl.AddSublist(set_nodal_vals | init_particles, {0, 100000});
    auto reset_comms = itl.AddTask(none, parthenon::ResetSwarmsCommunicationMesh, md_sp);
    auto send = itl.AddTask(reset_comms, parthenon::SendSwarmsMesh, md_sp);
    auto recv = itl.AddTask(send | reset_comms, parthenon::ReceiveSwarmsMesh, md_sp);
    auto transport = itl.AddTask(TQ::completion | TQ::global_sync, recv, Tracer, md);
  }
  auto &dump_reg = tc.AddRegion(1);
  auto &tl = dump_reg[0];
  tl.AddTask(none, DumpImages, pm, time);
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

TaskStatus DumpImages(Mesh *pm, Real time) {
  namespace vr = particles::riot_viz;

  auto vrend = pm->packages.Get("riot_viz");
  auto &dump_id = *(vrend->MutableParam<int>("dump_id"));
  bool timebar = vrend->Param<bool>("timebar");
  auto &time_rgb = vrend->Param<std::vector<Real>>("timebar_rgb");
  auto time_start = vrend->Param<Real>("first_dump_time");
  auto time_stop = vrend->Param<Real>("last_dump_time");
  auto scene_info = vrend->MutableParam<SceneInfo>("scene_info");

  auto &images = scene_info->images;
  auto md = pm->mesh_data.Get("base").get();

  auto desc_ps = parthenon::MakeSwarmPackDescriptor<vr::r, vr::g, vr::b, vr::a>(
      vr::particles::name());
  auto ps = desc_ps.GetPack(md);

  auto desc_ps_int = parthenon::MakeSwarmPackDescriptor<vr::camera_id, vr::pixel_id>(
      vr::particles::name());
  auto ps_int = desc_ps_int.GetPack(md);

  auto cast = [=](Real val) {
    return static_cast<std::uint8_t>(std::clamp(val, 0.0, 1.0) * 255.0 + 0.5);
  };

  int pix_set = 0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, ps.GetMaxFlatIndex(),
      KOKKOS_LAMBDA(const int idx, int &num_set) {
        auto [blk, n] = ps.GetBlockParticleIndices(idx);
        auto &swarm_d = ps.GetContext(blk);
        if (swarm_d.IsActive(n)) {
          const int cam_id = ps_int(blk, vr::camera_id(), n);
          const int pix_id = ps_int(blk, vr::pixel_id(), n);

          // remember that the sign of alpha may have been flipped to indicate this
          // particle was done being integrated
          const Real a = std::abs(ps(blk, vr::a(), n));
          const Real r = (a > 0) ? ps(blk, vr::r(), n) / a : 0.0;
          const Real g = (a > 0) ? ps(blk, vr::g(), n) / a : 0.0;
          const Real b = (a > 0) ? ps(blk, vr::b(), n) / a : 0.0;

          images(cam_id, 4 * pix_id + 0) = cast(r);
          images(cam_id, 4 * pix_id + 1) = cast(g);
          images(cam_id, 4 * pix_id + 2) = cast(b);
          images(cam_id, 4 * pix_id + 3) = cast(a);

          num_set++;

          swarm_d.MarkParticleForRemoval(n);
        }
      },
      Kokkos::Sum<int>(pix_set));

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
        max_label_width = std::max(max_label_width, layer.label.size());
        auto range_str = FormatMinMax(layer.min_range, layer.max_range);
        max_range_width = std::max(max_range_width, range_str.size());
      }
      max_label_width = 8 * max_label_width + 2;
      max_range_width = 8 * max_range_width + 2;
      int w = camera[i].nwidth;
      int h = camera[i].nheight;
      // set the colormap pixels
      int t = camera[i].colorbar_thickness;
      int total_h = h + (camera[i].ncolorbar + timebar) * t;
      int ilabel = 0;
      if (timebar) {
        int jstart = 4 * w * h;
        for (int j = jstart; j < jstart + 4 * w * t; j += 4) {
          int ix = ((j - jstart) % (4 * w)) / 4;
          Real x = (time - time_start) / (time_stop - time_start);
          int ixt = x * w;
          if (ix <= ixt) {
            host_images(i, j + 0) = cast(time_rgb[0]);
            host_images(i, j + 1) = cast(time_rgb[1]);
            host_images(i, j + 2) = cast(time_rgb[2]);
            host_images(i, j + 3) = 255;
          } else {
            host_images(i, j + 0) = cast(0.7 * time_rgb[0]);
            host_images(i, j + 1) = cast(0.7 * time_rgb[1]);
            host_images(i, j + 2) = cast(0.7 * time_rgb[2]);
            host_images(i, j + 3) = 255;
          }
        }
        draw_text(&host_images(i, 0), w, total_h, w / 2, h + 1,
                  "Time = " + FormatReal(time), cast(1.0 - time_rgb[0]),
                  cast(1.0 - time_rgb[1]), cast(1.0 - time_rgb[2]));
      }
      for (auto &layer : camera[i].layer) {
        if (!layer.colorbar) continue;
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
          Real a = layer.acmap.interpToReal(x);
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
              host_images(i, j + 0) = cast(darken * layer.rcmap.interpToReal(x));
              host_images(i, j + 1) = cast(darken * layer.gcmap.interpToReal(x));
              host_images(i, j + 2) = cast(darken * layer.bcmap.interpToReal(x));
              host_images(i, j + 3) = 255;
            }
          }
        }
        draw_text(&host_images(i, 0), w, total_h, max_label_width / 2,
                  h + (ilabel + 1 + timebar) * t - 9, layer.label, 255, 255, 255);
        draw_text(&host_images(i, 0), w, total_h, w - max_range_width / 2,
                  h + (ilabel + 1 + timebar) * t - 9,
                  FormatMinMax(layer.min_range, layer.max_range), 255, 255, 255);
        ilabel++;
      }

      std::string name = camera[i].filename + std::format("{:04d}", dump_id) + ".png";
      if (!stbi_write_png(name.c_str(), w, total_h,
                          4, // RGBA
                          &host_images(i, 0), w * 4)) {
        throw std::runtime_error("Failed to write PNG");
      }
    }
  }
  dump_id++;

  for (int b = 0; b < md->NumBlocks(); b++) {
    md->GetSwarmData(b)->Get(vr::particles::name())->RemoveMarkedParticles();
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, images.GetDim(2) - 1,
      0, images.GetDim(1) - 1,
      KOKKOS_LAMBDA(const int j, const int i) { images(j, i) = 0; });

  return TaskStatus::complete;
}

} // namespace riot_viz

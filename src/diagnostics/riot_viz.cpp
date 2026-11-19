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
#include <iomanip>
#include <sstream>

#include "riot_viz.hpp"

#include "riot_utils/riot_loops.hpp"

// the stb implementations are compiled as C, in stb_impl.c
#include <stb_image_write.h>
#include <stb_image_resize2.h>

#include <font8x8.h>

namespace riot_viz {

void AddCamera(ParameterInput *pin, const int img_id, const int ndim, SceneInfo &scene) {
  const std::string cam_block = "riot_viz/camera" + std::to_string(img_id);
  const int cam_id = scene.camera.size();
  scene.camera.push_back(CameraInfo{});
  auto &camera = scene.camera.back();
  auto &rp = scene.rendering_params;

  // the lighting, and where the rays stop, for all the layers of the camera
  auto light_x =
      pin->GetOrAddReal(cam_block, "light_x", 0.0,
                        "Position of the point light that shades slices and contours");
  auto light_y =
      pin->GetOrAddReal(cam_block, "light_y", 0.0,
                        "Position of the point light that shades slices and contours");
  auto light_z =
      pin->GetOrAddReal(cam_block, "light_z", 0.0,
                        "Position of the point light that shades slices and contours");
  auto ambient = pin->GetOrAddReal(cam_block, "light_ambient", 1.0,
                                   "Ambient (unshaded) lighting term");
  auto diffuse = pin->GetOrAddReal(cam_block, "light_diffuse", 0.0,
                                   "Diffuse lighting coefficient, falling off as 1/r^2");
  auto opacity_threshold =
      pin->GetOrAddReal(cam_block, "opacity_threshold", 1.e-2,
                        "Remaining transparency below which a ray stops");

  rp.xl.push_back(light_x);
  rp.yl.push_back(light_y);
  rp.zl.push_back(light_z);
  rp.ambient.push_back(ambient);
  rp.diffuse.push_back(diffuse);
  rp.alpha_tol.push_back(opacity_threshold);

  // the index of a field among those of the scene, which it is added to if new
  auto field_index = [&](const std::string &f) -> int {
    auto it = std::find(scene.field.begin(), scene.field.end(), f);
    if (it != scene.field.end()) return it - scene.field.begin();
    scene.field.push_back(f);
    return scene.field.size() - 1;
  };

  auto layers = pin->GetVector<std::string>(cam_block, "layers");
  rp.NewCamera();
  for (auto &layer : layers) {
    const auto type = VizTypeMap.at(pin->GetString(layer, "type", MapKeys(VizTypeMap)));
    auto field = pin->GetString(layer, "field");
    const auto use_grad = GradTypeMap.at(
        pin->GetOrAddString(layer, "field_use_grad", "none", MapKeys(GradTypeMap),
                            "Gradient operator to apply to the field"));
    const auto scale_type = ScaleTypeMap.at(
        pin->GetOrAddString(layer, "field_scale", "linear", MapKeys(ScaleTypeMap),
                            "Scaling of the color axis"));
    auto field_alpha =
        pin->GetOrAddString(layer, "field_alpha", field, "Field that drives opacity");
    const auto alpha_use_grad = GradTypeMap.at(
        pin->GetOrAddString(layer, "field_alpha_use_grad", "none", MapKeys(GradTypeMap),
                            "Gradient operator to apply to the opacity field"));
    auto label =
        pin->GetOrAddString(layer, "label", field, "Text label of the color bar");

    // volume render, slice, and contour_slice params
    auto colorbar = pin->GetOrAddBoolean(
        layer, "colorbar", false,
        "Whether to append a color bar for the layer below the image");
    camera.ncolorbar += colorbar;
    auto red = pin->GetOrAddVector<Real>(layer, "red", {0.0, 0.0},
                                         "Color-map control points in [0,1]");
    auto green = pin->GetOrAddVector<Real>(layer, "green", {0.0, 0.0},
                                           "Color-map control points in [0,1]");
    auto blue = pin->GetOrAddVector<Real>(layer, "blue", {0.0, 0.0},
                                          "Color-map control points in [0,1]");
    auto alpha = pin->GetOrAddVector<Real>(
        layer, "alpha", {0.0, 0.0},
        "Opacity control points, as extinction per unit length for volume layers");
    auto cmin = pin->GetOrAddReal(layer, "min_value", 0.0,
                                  "Lower end of the data range of the color map");
    auto cmax = pin->GetOrAddReal(layer, "max_value", 1.0,
                                  "Upper end of the data range of the color map");
    auto amin = pin->GetOrAddReal(layer, "min_alpha_value", cmin,
                                  "Lower end of the data range of the opacity map");
    auto amax = pin->GetOrAddReal(layer, "max_alpha_value", cmax,
                                  "Upper end of the data range of the opacity map");

    // contour and contour_slice params
    auto contours =
        pin->GetOrAddVector<Real>(layer, "contours", {}, "Iso-values to draw");
    auto contour_alpha = pin->GetOrAddVector<Real>(layer, "contour_alpha", {},
                                                   "Per-contour opacities in [0,1]");

    // contour params
    auto contour_red =
        pin->GetOrAddVector<Real>(layer, "contour_red", {}, "Per-contour colors");
    auto contour_green =
        pin->GetOrAddVector<Real>(layer, "contour_green", {}, "Per-contour colors");
    auto contour_blue =
        pin->GetOrAddVector<Real>(layer, "contour_blue", {}, "Per-contour colors");

    // slice params
    auto slice_pos = pin->GetOrAddVector<Real>(layer, "slice_location", {},
                                               "A point on the slice plane");
    auto slice_normal =
        pin->GetOrAddVector<Real>(layer, "slice_normal", {}, "Normal of the slice plane");
    auto slice_alpha = pin->GetOrAddReal(layer, "slice_alpha", 1.0,
                                         "Opacity of the slice, clamped to [0,1]");
    auto slice_transparent_outside = pin->GetOrAddVector<bool>(
        layer, "transparent_outside_range", std::vector<bool>(2, false),
        "Whether the slice is transparent below and above its data range");

    if (contours.size() > 0) {
      for (const auto *c :
           {&contour_red, &contour_green, &contour_blue, &contour_alpha}) {
        PARTHENON_REQUIRE_THROWS(
            c->size() == contours.size(),
            "size of contour list must match across values and colors");
      }
    }
    for (auto &ca : contour_alpha)
      ca = std::clamp(ca, 0.0, 1.0);

    std::vector<Real> slice_x{std::numeric_limits<Real>::max(),
                              std::numeric_limits<Real>::max(),
                              std::numeric_limits<Real>::max()};
    std::vector<Real> slice_n = {0.0, 0.0, 0.0};
    if ((slice_pos.size() == slice_normal.size()) && slice_pos.size() == 3) {
      slice_x = {slice_pos[0], slice_pos[1], slice_pos[2]};
      slice_n = {slice_normal[0], slice_normal[1], slice_normal[2]};
    }

    const int var_index = field_index(field);
    const int alpha_var_index = field_index(field_alpha);

    auto masks = MakeMasks(pin, layer);

    auto scale = [&](const Real val) {
      if (scale_type == ScaleType::log) {
        PARTHENON_REQUIRE_THROWS(
            val > 0.0, "Trying to take the log of a negative number is a no-no.");
        return std::log10(val);
      }
      return val;
    };
    const Real range_min = scale(cmin);
    const Real range_max = scale(cmax);
    const auto red_box = MakeNormalizedDataBox(red);
    const auto green_box = MakeNormalizedDataBox(green);
    const auto blue_box = MakeNormalizedDataBox(blue);
    const auto alpha_box = MakeNormalizedDataBox(alpha);

    rp.type.back().push_back(type);
    rp.pack_idx.back().push_back(var_index);
    rp.alpha_pack_idx.back().push_back(alpha_var_index);
    rp.field_scale.back().push_back(scale_type);
    rp.range_min.back().push_back(range_min);
    rp.range_max.back().push_back(range_max);
    rp.alpha_range_min.back().push_back(amin);
    rp.alpha_range_max.back().push_back(amax);
    rp.use_grad.back().push_back(use_grad);
    rp.alpha_use_grad.back().push_back(alpha_use_grad);
    rp.r.back().push_back(red_box.getOnDevice());
    rp.g.back().push_back(green_box.getOnDevice());
    rp.b.back().push_back(blue_box.getOnDevice());
    rp.a.back().push_back(alpha_box.getOnDevice());
    rp.contours.back().push_back(contours);
    rp.rc.back().push_back(contour_red);
    rp.gc.back().push_back(contour_green);
    rp.bc.back().push_back(contour_blue);
    rp.ac.back().push_back(contour_alpha);
    rp.slice_x.back().push_back(slice_x);
    rp.slice_n.back().push_back(slice_n);
    rp.slice_alpha.back().push_back(std::clamp(slice_alpha, 0.0, 1.0));
    rp.slice_transparent_outside.back().push_back(slice_transparent_outside);
    rp.masks.back().push_back(masks);

    // add this layer to the camera
    camera.layer.emplace_back(red_box, green_box, blue_box, alpha_box, label, colorbar,
                              range_min, range_max, type, scale_type);
  }

  // now set some info at the camera/image level
  camera.filename = pin->GetString(cam_block, "name");
  auto cloc = pin->GetVector<Real>(cam_block, "location");
  auto floc = pin->GetVector<Real>(cam_block, "focus");
  auto up =
      pin->GetOrAddVector<Real>(cam_block, "up", {0, 0, 1}, "Up vector of the camera");
  auto width = pin->GetReal(cam_block, "target_width");
  auto height = pin->GetReal(cam_block, "target_height");
  camera.nwidth = pin->GetInteger(cam_block, "nwidth");
  camera.nheight = pin->GetInteger(cam_block, "nheight");
  camera.scale = pin->GetOrAddReal(cam_block, "scale_output", 1.0,
                                   "Factor to resize the written image by");
  camera.colorbar_thickness = pin->GetOrAddInteger(cam_block, "colorbar_thickness", 10,
                                                   "Height in pixels of each color bar");

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
        }
      } else {
        PARTHENON_THROW("Unsupported coordinate system in volume renderer.");
      }
      if (lambda_use > 0.5 * std::numeric_limits<Real>::max()) {
        // the ray misses the mesh
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
  Params &params = pkg->AllParams();

  // figure out when viz should happen
  auto dt_viz =
      pin->GetOrAddReal("riot_viz", "dt", -1.0, "Render every dt in simulation time");
  auto t_viz =
      pin->GetOrAddVector<Real>("riot_viz", "t", {}, "Explicit list of render times");
  auto tlim = pin->GetReal("parthenon/time", "tlim");

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
  bool timebar = pin->GetOrAddBoolean("riot_viz", "timebar", false,
                                      "Whether to draw a time bar below each image");
  params.Add("timebar", timebar);
  auto timebar_rgb = pin->GetOrAddVector<Real>("riot_viz", "timebar_rgb", {1.0, 1.0, 1.0},
                                               "Color of the time bar");
  params.Add("timebar_rgb", timebar_rgb);

  int ndim = 1 + (pin->GetInteger("parthenon/mesh", "nx2") > 1) +
             (pin->GetInteger("parthenon/mesh", "nx3") > 1);

  const std::string block_base = "riot_viz/camera";
  auto blocks = pin->GetBlockNamesWithPrefix(block_base);
  std::set<int> cam_block_ids;
  int max_size = 0;
  SceneInfo scene;
  for (const auto &block_name : blocks) {
    if (block_name.length() > block_base.size()) {
      auto suffix = block_name[block_base.size()];
      if (std::isdigit(suffix)) {
        auto id = atoi(block_name.substr(block_base.size()).c_str());
        PARTHENON_REQUIRE_THROWS(cam_block_ids.count(id) == 0,
                                 "Two camera input blocks are numbered identically.");
        cam_block_ids.insert(id);
        AddCamera(pin, id, ndim, scene);
        auto &camera = scene.camera.back();
        max_size = std::max(max_size, camera.nwidth * camera.nheight +
                                          (camera.ncolorbar + timebar) *
                                              camera.colorbar_thickness * camera.nwidth);
      }
    }
  }

  // don't forget the 4 because it's 4 channels (RGBA)
  scene.images = parthenon::ParArray2D<uint8_t>("Volume Render images",
                                                scene.camera.size(), 4 * max_size);
  scene.host_images = Kokkos::create_mirror_view(scene.images);

  // Every camera point, face by face
  int npts = 0;
  for (int i = 0; i < 6; i++) {
    npts += scene.face_pts[i][sample::x].size();
  }
  scene.camera_pts = parthenon::ParArray2D<Real>("camera points", npts,
                                                 static_cast<int>(sample::nvalues));
  scene.camera_idx = parthenon::ParArray2D<int>("camera indexing", npts, 3);
  auto chost = Kokkos::create_mirror_view(Kokkos::HostSpace(), scene.camera_pts);
  auto cid_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), scene.camera_idx);
  for (int i = 0, p = 0; i < 6; i++) {
    for (int j = 0; j < scene.face_pts[i][sample::x].size(); j++, p++) {
      for (int n = 0; n < sample::nvalues; n++) {
        chost(p, n) = scene.face_pts[i][n][j];
      }
      cid_host(p, 0) = scene.camera_id[i][j];
      cid_host(p, 1) = scene.pixel_id[i][j];
      cid_host(p, 2) = i;
    }
  }
  Kokkos::deep_copy(scene.camera_pts, chost);
  Kokkos::deep_copy(scene.camera_idx, cid_host);

  const int initial_pool_size =
    pin->GetOrAddInteger("riot_viz", "initial_swarm_pool_size", 256);

  PARTHENON_REQUIRE(initial_pool_size > 0,
                  "riot_viz/initial_swarm_pool_size must be positive");

  Metadata mswarm({Metadata::Provides, Metadata::None});
  mswarm.SetInitialSwarmPoolReservation(initial_pool_size);
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

  Metadata m({Metadata::OneCopy, Metadata::Node});
  std::vector<std::string> orig_fields = scene.field;
  std::string suffix = "_nodal_";
  for (auto &f : scene.field) {
    f += suffix;
    pkg->AddField(f, m);
  }
  params.Add("scene_info", scene, true);
  params.Add("orig_fields", orig_fields);
  params.Add("tracer_info", impl::TracerInfo(scene.field, RenderingParams(scene.rendering_params)), true);

  params.Add("nodal_bounds", impl::NodalBounds(), true);
  params.Add("show_volume", ShowVolume(), true);

  return pkg;
}

bool TimeToRender(Mesh *pm, Real time) {
  if (pm->packages.AllPackages().count("riot_viz") == 0) return false;
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

  auto pm = md->GetMeshPointer();
  auto &viz = pm->packages.Get("riot_viz");
  const auto &si = viz->Param<SceneInfo>("scene_info");

  const int ndim = pm->ndim;

  // Every camera point, with the camera, pixel, and mesh face it is on
  const auto &camera_pts = si.camera_pts;
  const auto &camera_idx = si.camera_idx;
  const int npts = camera_pts.extent_int(0);
  if (npts == 0) return TaskStatus::complete;

  // A block face is on the mesh boundary unless it borders another block, and faces along
  // inactive directions always are
  const int nblocks = md->NumBlocks();
  parthenon::ParArray2D<bool> on_boundary("camera boundary faces", nblocks, 6);
  parthenon::ParArray2D<Real> block_box("camera block boxes", nblocks, 6);
  auto bhost = Kokkos::create_mirror_view(Kokkos::HostSpace(), on_boundary);
  auto box_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), block_box);
  for (int b = 0; b < nblocks; b++) {
    auto *pmb = md->GetBlockData(b)->GetBlockPointer();
    for (int f = 0; f < 6; f++) {
      bhost(b, f) =
          f >= 2 * ndim || pmb->boundary_flag[f] != parthenon::BoundaryFlag::block;
    }
    for (int d = 0; d < 3; d++) {
      const auto dir = static_cast<parthenon::CoordinateDirection>(d + 1);
      box_host(b, 2 * d) = pmb->block_size.xmin(dir);
      box_host(b, 2 * d + 1) = pmb->block_size.xmax(dir);
    }
  }
  Kokkos::deep_copy(on_boundary, bhost);
  Kokkos::deep_copy(block_box, box_host);

  // Rays start where they meet the volume anything could show in, or at their camera
  // point if it is in it already. Those that miss it are background.
  using OnBlock = impl::CameraPointOnBlock;
  const auto &show_volume = viz->Param<ShowVolume>("show_volume");
  Kokkos::Array<Real, 3> mesh_lo, mesh_hi;
  for (int d = 0; d < 3; d++) {
    const auto dir = static_cast<parthenon::CoordinateDirection>(d + 1);
    mesh_lo[d] = pm->mesh_size.xmin(dir);
    mesh_hi[d] = pm->mesh_size.xmax(dir);
  }
  auto &images = si.images;
  parthenon::ParArray1D<int> start_kind("camera start kinds", npts);
  parthenon::ParArray2D<Real> start("camera starts", npts, 3);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, npts - 1,
      KOKKOS_LAMBDA(const int p) {
        start_kind(p) = OnBlock::start_at_point;
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
          for (int d = 0; d < nd; d++)
            miss = miss || xm[d] < mesh_lo[d] || xm[d] > mesh_hi[d];
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

  const OnBlock on_block{block_box, camera_pts,  camera_idx,
                         on_boundary, start_kind, start};

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
    for (int d = 0; d < 3; d++) {
      md_lo[d] = std::min(md_lo[d], box_host(b, 2 * d));
      md_hi[d] = std::max(md_hi[d], box_host(b, 2 * d + 1));
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
  parthenon::ParArray1D<int> counts("camera counts", nblocks);
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, 0, 0, 0, nblocks - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
        int count = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(member, ncand),
            [&](const int k, int &c) { c += on_block(b, cand(k)); }, count);
        Kokkos::single(Kokkos::PerTeam(member), [&]() { counts(b) = count; });
      });
  auto counts_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), counts);

  // make space for the particles
  parthenon::ParArray1D<parthenon::NewParticlesContext> new_context("New context",
                                                                    nblocks);
  auto new_context_h = new_context.GetHostMirror();
  for (int b = 0; b < nblocks; b++) {
    auto *swarm = md->GetSwarmData(b)->Get(pvr::particles::name()).get();
    new_context_h(b) = swarm->AddEmptyParticles(counts_h(b));
  }
  new_context.DeepCopy(new_context_h);

  // the packs are snapshots, so get them after making space for the rays
  const auto &tinfo = viz->Param<impl::TracerInfo>("tracer_info");
  auto pack = tinfo.desc_ps.GetPack(md);
  auto pack_int = tinfo.desc_ps_int.GetPack(md);

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
                const int pidx = new_context(b).GetNewParticleIndex(offset);
                Real &x = pack(b, swarm_position::x(), pidx);
                Real &y = pack(b, swarm_position::y(), pidx);
                Real &z = pack(b, swarm_position::z(), pidx);
                // nvcc does not let a device lambda first capture a variable in an if
                // constexpr branch, so what the branches use is read before them
                const Real px = camera_pts(n, sample::x);
                const Real py = camera_pts(n, sample::y);
                const Real pz = camera_pts(n, sample::z);
                const Real block_xmax1 = block_box(b, 1);
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
                pack(b, prt::vx(), pidx) = camera_pts(n, sample::nx);
                pack(b, prt::vy(), pidx) = camera_pts(n, sample::ny);
                pack(b, prt::vz(), pidx) = camera_pts(n, sample::nz);
                pack(b, pvr::r(), pidx) = 0.0;
                pack(b, pvr::g(), pidx) = 0.0;
                pack(b, pvr::b(), pidx) = 0.0;
                pack(b, pvr::a(), pidx) = 0.0;
                pack_int(b, pvr::camera_id(), pidx) = camera_idx(n, 0);
                pack_int(b, pvr::pixel_id(), pidx) = camera_idx(n, 1);
              }
              offset += on;
            });
      });

  return TaskStatus::complete;
}

TaskStatus Tracer(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto viz = pm->packages.Get("riot_viz");
  auto &scene_info = viz->Param<SceneInfo>("scene_info");
  auto tinfo = viz->MutableParam<impl::TracerInfo>("tracer_info");

  if (!tinfo->desc) tinfo->initialize(md);

  auto [v, ps, ps_int] = tinfo->GetPacks(md);

  auto &bounds = viz->Param<impl::NodalBounds>("nodal_bounds");

  return RayTrace::Trace<Composer>(ps, v, md->GetParentPointer(), ps_int,
                                   tinfo->render_params, tinfo->pack_idx_view,
                                   bounds.may_show, viz->Param<ShowVolume>("show_volume"),
                                   scene_info.images);
}

TaskStatus InitializeNodalValues(MeshData<Real> *md) {
  using TE = parthenon::TopologicalElement;

  auto pm = md->GetMeshPointer();
  auto viz = pm->packages.Get("riot_viz");
  auto orig_fields =
      viz->Param<std::vector<std::string>>("orig_fields");
  auto &scene_info = viz->Param<SceneInfo>("scene_info");
  auto node_fields = scene_info.field;
  auto desc_orig = parthenon::MakePackDescriptor(md, orig_fields);
  auto vorig = desc_orig.GetPack(md);

  auto pack_idx_view = impl::PackIdxView(desc_orig, orig_fields, "pack_idx_view");
  const int nfields = orig_fields.size();

  auto desc_node = parthenon::MakePackDescriptor(md, node_fields);
  auto vnode = desc_node.GetPack(md);
  auto node_pack_idx_view =
      impl::PackIdxView(desc_node, node_fields, "node_pack_idx_view");

  PARTHENON_REQUIRE_THROWS(vorig.GetNBlocks() == md->NumBlocks() &&
                               vnode.GetNBlocks() == md->NumBlocks(),
                           "The field packs must cover every block of the mesh data.");

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
  const int ngc = parthenon::Globals::nghost;
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
  auto bounds = viz->MutableParam<impl::NodalBounds>("nodal_bounds");
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

  bounds->may_show = std::move(may_show);

  return TaskStatus::complete;
}

// The volume anything could show in, from the boxes of the blocks on every rank
TaskStatus ReduceShowVolume(Mesh *pm) {
  auto viz = pm->packages.Get("riot_viz");
  const auto &bounds = viz->Param<impl::NodalBounds>("nodal_bounds");
  auto show_volume = viz->MutableParam<ShowVolume>("show_volume");
  const int ncam = viz->Param<SceneInfo>("scene_info").camera.size();

  // lower bounds are negated, so that one MPI_MAX reduces them along with the upper ones
  std::vector<Real> box(6 * ncam);
  for (int i = 0; i < 6 * ncam; i++) {
    box[i] = i % 2 == 0 ? -bounds.show_box[i] : bounds.show_box[i];
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, box.data(), box.size(),
                                    MPI_PARTHENON_REAL, MPI_MAX, MPI_COMM_WORLD));
#endif

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
  using TQ = TaskQualifier;
  TaskCollection tc;
  TaskID none;
  // this must be 1 for now until pack_size != -1 is supported for swarms in parthenon
  constexpr int num_partitions = 1; // pm->DefaultNumPartitions();
  auto &reg = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = reg[i];
    auto md = pm->mesh_data.Get().get();
    tl.AddTask(none, InitializeNodalValues, md);
  }
  // The cameras start rays where they meet the volume anything could show in, which
  // takes the nodal values of every block
  auto &show_reg = tc.AddRegion(1);
  show_reg[0].AddTask(none, ReduceShowVolume, pm);
  auto &trace_reg = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = trace_reg[i];
    auto md_sp = pm->mesh_data.Get();
    auto md = md_sp.get();

    auto init_particles = tl.AddTask(none, InitializeCameras, md);
    auto [itl, push] = tl.AddSublist(init_particles, {0, 100000});

    auto remove = itl.AddTask(none, parthenon::RemoveMarkedParticlesMesh, md_sp, pvr::particles::name());
    auto reset_comms = itl.AddTask(none, parthenon::ResetSwarmsCommunicationMesh, md_sp);
    auto send = itl.AddTask(reset_comms | remove, parthenon::SendSwarmsMesh, md_sp);
    auto recv = itl.AddTask(send | reset_comms, parthenon::ReceiveSwarmsMesh, md_sp);
    auto defrag = itl.AddTask(recv, parthenon::DefragSwarmsMesh, md_sp, 0.25);
    auto transport = itl.AddTask(TQ::completion | TQ::global_sync, defrag, Tracer, md);
    // The last pass only marks its rays for removal, and a remesh before the next render
    // would bring them back as active rays
    tl.AddTask(push, parthenon::RemoveMarkedParticlesMesh, md_sp, pvr::particles::name());
  }
  auto &dump_reg = tc.AddRegion(1);
  auto &tl = dump_reg[0];
  tl.AddTask(none, DumpImages, pm, time);
  return tc;
}

namespace {
// Draw the 8x8 glyph of c with its top left corner at (x0, y0) of an RGBA image
void DrawChar(uint8_t *rgba, int width, int height, int x0, int y0, char c,
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

// Draw text starting, or centered, at (x, y) of an RGBA image
void DrawText(uint8_t *rgba, int width, int height, int x, int y, const std::string &text,
              std::uint8_t r = 255, std::uint8_t g = 255, std::uint8_t b = 255,
              std::uint8_t a = 255, bool centered = true) {
  int num_chars = text.size();
  int pixel_width = 8 * num_chars;
  int offset = centered ? pixel_width / 2 : 0;
  for (char c : text) {
    DrawChar(rgba, width, height, x - offset, y, c, r, g, b, a);

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

// A tightly packed RGBA image resampled to dst_w by dst_h
std::vector<std::uint8_t> ResizeRGBA(const void *src, int src_w, int src_h, int dst_w,
                                     int dst_h) {
  std::vector<std::uint8_t> dst(static_cast<std::size_t>(dst_w) * dst_h * 4);
  stbir_resize(src, src_w, src_h, 0, dst.data(), dst_w, dst_h, 0, STBIR_RGBA,
               STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM);
  return dst;
}

// Write an image as a PNG, resized by scale
int WritePNG(char const *filename, int w, int h, int comp, const void *data,
             int stride_in_bytes, Real scale) {
  if (scale == 1.0) return stbi_write_png(filename, w, h, comp, data, stride_in_bytes);

  int nw = scale * w;
  int nh = scale * h;
  auto rescaled = ResizeRGBA(data, w, h, nw, nh);
  return stbi_write_png(filename, nw, nh, comp, rescaled.data(), nw * 4);
}
} // namespace

TaskStatus DumpImages(Mesh *pm, Real time) {
  auto viz = pm->packages.Get("riot_viz");
  auto &dump_id = *(viz->MutableParam<int>("dump_id"));
  bool timebar = viz->Param<bool>("timebar");
  auto &time_rgb = viz->Param<std::vector<Real>>("timebar_rgb");
  auto time_start = viz->Param<Real>("first_dump_time");
  auto time_stop = viz->Param<Real>("last_dump_time");
  auto scene_info = viz->MutableParam<SceneInfo>("scene_info");

  auto &images = scene_info->images;

  Kokkos::fence();
  auto &host_images = scene_info->host_images;
  Kokkos::deep_copy(host_images, images);
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Reduce(parthenon::Globals::my_rank == 0 ? MPI_IN_PLACE : host_images.data(),
                 host_images.data(), static_cast<int>(host_images.size()), MPI_UINT8_T,
                 MPI_MAX, 0, MPI_COMM_WORLD));
#endif // MPI_PARALLEL
  if (parthenon::Globals::my_rank == 0) {
    auto label_of = [](const LayerInfo &layer) {
      return layer.scale_type == ScaleType::log ? "log10(" + layer.label + ")"
                                                : layer.label;
    };
    auto &camera = scene_info->camera;
    for (int i = 0; i < camera.size(); i++) {
      // an opaque pixel, at offset j of the image
      auto set_pixel = [&](const int j, const uint8_t r, const uint8_t g,
                           const uint8_t b) {
        host_images(i, j + 0) = r;
        host_images(i, j + 1) = g;
        host_images(i, j + 2) = b;
        host_images(i, j + 3) = 255;
      };
      size_t max_label_width = 0;
      size_t max_range_width = 0;
      for (auto &layer : camera[i].layer) {
        if (!layer.colorbar) continue;
        max_label_width = std::max(max_label_width, label_of(layer).size());
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
      int ilabel = 0;
      if (timebar) {
        int jstart = 4 * w * h;
        Real x = (time - time_start) / (time_stop - time_start);
        int ixt = x * w;
        for (int j = jstart; j < jstart + 4 * w * t; j += 4) {
          int ix = ((j - jstart) % (4 * w)) / 4;
          const Real dim = ix <= ixt ? 1.0 : 0.7;
          set_pixel(j, real_to_uint8_t(dim * time_rgb[0]),
                    real_to_uint8_t(dim * time_rgb[1]),
                    real_to_uint8_t(dim * time_rgb[2]));
        }
        DrawText(&host_images(i, 0), w, total_h, w / 2, h + 1,
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
            set_pixel(j, 0, 0, 0);
            continue;
          }
          Real x =
              (1.0 * (ix - max_label_width)) / (w - max_label_width - max_range_width);
          int iy = t - (j - jstart) / (4 * w) - 1;
          Real a = layer.acmap.interpToReal(x) / amax;
          int iyl = a * (t - 2) + 1;
          if (iy == 0) {
            set_pixel(j, 0, 0, 0);
          } else if (iy == iyl && layer.type == VizType::volume) {
            const uint8_t dash = 255 * (ix % 2);
            set_pixel(j, dash, dash, dash);
          } else {
            Real darken = (iy > iyl && layer.type == VizType::volume) ? 0.5 : 1.0;
            set_pixel(j, real_to_uint8_t(darken * layer.rcmap.interpToReal(x)),
                      real_to_uint8_t(darken * layer.gcmap.interpToReal(x)),
                      real_to_uint8_t(darken * layer.bcmap.interpToReal(x)));
          }
        }
        DrawText(&host_images(i, 0), w, total_h, max_label_width / 2,
                 h + (ilabel + 1 + timebar) * t - 9, label_of(layer), 255, 255, 255);
        DrawText(&host_images(i, 0), w, total_h, w - max_range_width / 2,
                 h + (ilabel + 1 + timebar) * t - 9,
                 FormatMinMax(layer.min_range, layer.max_range), 255, 255, 255);
        ilabel++;
      }

      std::ostringstream ss;
      ss << camera[i].filename << std::setfill('0') << std::setw(4) << dump_id << ".png";
      std::string name = ss.str();
      if (!WritePNG(name.c_str(), w, total_h, 4, &host_images(i, 0), w * 4, scale)) {
        PARTHENON_THROW("Failed to write " + name);
      }
    }
  }
  dump_id++;

  // clear the images for the next dump
  Kokkos::deep_copy(images, 0);

  return TaskStatus::complete;
}

} // namespace riot_viz

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
#ifndef RIOT_PGEN_CAD_HPP_
#define RIOT_PGEN_CAD_HPP_
// This file was made in part with generative AI.

#include <array>
#include <unordered_map>
#include <unordered_set>

#include "region_primitives.hpp"

// a cell/tree node in a cell-based amr mesh representing a CAD part
struct PartCell {
  int64_t index = -1;
  int level;
  bool value;
  std::vector<PartCell> child;
  PartCell() = default;
  PartCell(const int64_t id, const int lev) : index(id), level(lev) {}

  template <typename F, typename G>
  auto fill_corners(F &unflatten, G &flatten, int max_level, int ndim) const {
    std::vector<int64_t> corners(1 << ndim);
    auto [k, j, i] = unflatten(index);
    int stride = (1 << (max_level - level));
    corners[0] = flatten(k, j, i);
    corners[1] = flatten(k, j, i + stride);
    if (ndim > 1) {
      corners[2] = flatten(k, j + stride, i);
      corners[3] = flatten(k, j + stride, i + stride);
      if (ndim > 2) {
        corners[4] = flatten(k + stride, j, i);
        corners[5] = flatten(k + stride, j, i + stride);
        corners[6] = flatten(k + stride, j + stride, i);
        corners[7] = flatten(k + stride, j + stride, i + stride);
      }
    }
    return std::make_tuple(k, j, i, corners);
  }

  // Linear interpolation using FMA: a + t*(b - a)
  double lerp(double a, double b, double t) const { return std::fma(t, b - a, a); }

  // val[0..7] is ordered as:
  // 0:(0,0,0)  1:(1,0,0)  2:(0,1,0)  3:(1,1,0)
  // 4:(0,0,1)  5:(1,0,1)  6:(0,1,1)  7:(1,1,1)
  template <typename accessor_t>
  double trilinear_fma(const accessor_t &val, double wx, double wy, double wz) const {
    // Interpolate along x
    double c00 = lerp(val(0), val(1), wx); // z=0, y=0
    double c10 = lerp(val(2), val(3), wx); // z=0, y=1
    double c01 = lerp(val(4), val(5), wx); // z=1, y=0
    double c11 = lerp(val(6), val(7), wx); // z=1, y=1

    // Interpolate along y
    double c0 = lerp(c00, c10, wy); // z=0
    double c1 = lerp(c01, c11, wy); // z=1

    // Interpolate along z
    return lerp(c0, c1, wz);
  }

  template <typename accessor_t>
  double bilinear_fma(const accessor_t &val, double wx, double wy) const {
    // Interpolate in x-direction for bottom and top rows
    double c0 = lerp(val(0), val(1), wx); // y = 0
    double c1 = lerp(val(2), val(3), wx); // y = 1

    // Interpolate in y-direction
    return lerp(c0, c1, wy);
  }

  template <typename F, typename G>
  Real interp(const Real x, const Real y, const Real z, std::array<Real, 3> &xmin,
              std::array<Real, 3> &xmax, const std::unordered_map<int64_t, bool> &v,
              F &unflatten, G &flatten, const int max_level, const int ndim) const {
    if (child.size() == 0) {
      auto [k, j, i, corners] = fill_corners(unflatten, flatten, max_level, ndim);
      auto &c = corners;
      auto val = [&](const int n) { return v.at(c[n]); };
      Real wx = (x - xmin[0]) / (xmax[0] - xmin[0]);
      if (ndim > 1) {
        Real wy = (y - xmin[1]) / (xmax[1] - xmin[1]);
        if (ndim == 2) {
          return bilinear_fma(val, wx, wy);
        } else {
          Real wz = (z - xmin[2]) / (xmax[2] - xmin[2]);
          return trilinear_fma(val, wx, wy, wz);
        }
      }
      return lerp(val(0), val(1), wx);
    }

    Real xmid = 0.5 * (xmin[0] + xmax[0]);
    Real ymid = 0.5 * (xmin[1] + xmax[1]);
    Real zmid = 0.5 * (xmin[2] + xmax[2]);
    int i = x > xmid;
    int j = y > ymid;
    int k = z > zmid;
    int cid = i + 2 * (ndim > 1) * (j + 2 * (ndim > 2) * k);
    if (i)
      xmin[0] = xmid;
    else
      xmax[0] = xmid;
    if (j)
      xmin[1] = ymid;
    else
      xmax[1] = ymid;
    if (k)
      xmin[2] = zmid;
    else
      xmax[2] = zmid;
    return child[cid].interp(x, y, z, xmin, xmax, v, unflatten, flatten, max_level, ndim);
  }

  template <typename F, typename G, typename H>
  void refine(const std::unordered_map<int64_t, bool> &values, int max_level,
              std::unordered_set<int64_t> &new_points, F &unflatten, G &flatten_cell,
              H &flatten_node, int ndim) {
    if (level == max_level) return;
    if (child.size() > 0) {
      for (auto &c : child) {
        c.refine(values, max_level, new_points, unflatten, flatten_cell, flatten_node,
                 ndim);
      }
      return;
    }
    auto [k0, j0, i0, corner] = fill_corners(unflatten, flatten_node, max_level, ndim);
    bool val0 = values.at(corner[0]);
    bool same = true;
    for (int i = 1; i < corner.size(); i++) {
      if (val0 != values.at(corner[i])) {
        same = false;
        break;
      }
    }
    if (!same) { // then refine
      int new_stride = (1 << (max_level - level - 1));
      child.reserve(1 << ndim);
      // index of first child is same as parent
      child.emplace_back(index, level + 1);
      // index of i + stride/2
      child.emplace_back(flatten_cell(k0, j0, i0 + new_stride), level + 1);
      // etc
      if (ndim > 1) {
        child.emplace_back(flatten_cell(k0, j0 + new_stride, i0), level + 1);
        child.emplace_back(flatten_cell(k0, j0 + new_stride, i0 + new_stride), level + 1);
        if (ndim > 2) {
          child.emplace_back(flatten_cell(k0 + new_stride, j0, i0), level + 1);
          child.emplace_back(flatten_cell(k0 + new_stride, j0, i0 + new_stride),
                             level + 1);
          child.emplace_back(flatten_cell(k0 + new_stride, j0 + new_stride, i0),
                             level + 1);
          child.emplace_back(
              flatten_cell(k0 + new_stride, j0 + new_stride, i0 + new_stride), level + 1);
        }
      }
      for (auto &c : child) {
        auto [kc, jc, ic, cc] = c.fill_corners(unflatten, flatten_node, max_level, ndim);
        for (auto cid : cc) {
          new_points.insert(cid);
        }
      }
    }
  }
};

class PartMesh {
 public:
  PartMesh(const int maximum_level, const std::array<int, 3> &ncells,
           const std::array<Real, 3> &bbox_min, const std::array<Real, 3> &bbox_max)
      : max_level(maximum_level), nx(ncells), xmin(bbox_min), xmax(bbox_max),
        ndim(1 + static_cast<int>(ncells[1] > 1) + static_cast<int>(ncells[2] > 1)) {
    for (int i = 0; i < 3; i++)
      dx_base[i] = (xmax[i] - xmin[i]) / nx[i];
    cells.resize(nx[0] * nx[1] * nx[2]);
    int max_ref = (1 << max_level);
    std::array<int, 3> nfine{nx[0] * max_ref, nx[1] * max_ref, nx[2] * max_ref};
    for (int64_t k = 0; k < nx[2]; k++) {
      for (int64_t j = 0; j < nx[1]; j++) {
        for (int64_t i = 0; i < nx[0]; i++) {
          int cell_index = i + nx[0] * (j + nx[1] * k);
          int64_t fine_cell_index =
              i * max_ref + nfine[0] * (j * max_ref + nfine[1] * k * max_ref);
          cells[cell_index].index = fine_cell_index;
          cells[cell_index].level = 0;
        }
      }
    }
  }

  // this is the main function to build a cell-based AMR mesh representation of a CAD
  // part.  It works by iteratively refining cells where node-based evaluations of
  // the mask function differ, i.e. cells that definitely intersect the boundary of a
  // part.  It is possible for this method to miss things, like a sharp feature that
  // pierces a cell face, but does not overlap a node.
  template <typename F>
  void build(F &mask_func) {
    // fill in the nodes of the base mesh
    int stride = (1 << max_level);
    std::array<int, 3> ncell_max{nx[0] * stride, nx[1] * stride, nx[2] * stride};
    std::array<int, 3> nface_max{ncell_max[0] + 1, ncell_max[1] + 1, ncell_max[2] + 1};
    std::array<Real, 3> dx_fine{dx_base[0] / stride, dx_base[1] / stride,
                                dx_base[2] / stride};
    std::unordered_set<int64_t> new_points;

    // maps back and forth between flattened and unflattened index spaces for nodes/cells
    flatten_cell = [=](const int64_t k, const int64_t j, const int64_t i) {
      int64_t cell_id = i + ncell_max[0] * (j + ncell_max[1] * k);
      return cell_id;
    };
    flatten_node = [=](const int64_t k, const int64_t j, const int64_t i) {
      int64_t node_id = i + nface_max[0] * (j + nface_max[1] * k);
      return node_id;
    };
    unflatten = [=](const int64_t cell_id) {
      const int64_t k = cell_id / (ncell_max[0] * ncell_max[1]);
      const int64_t j = (cell_id - k * (ncell_max[0] * ncell_max[1])) / ncell_max[0];
      const int64_t i = cell_id % ncell_max[0];
      return std::make_tuple(k, j, i);
    };
    auto pid_to_xyz = [=](const int64_t pid) {
      const int64_t k = pid / (nface_max[0] * nface_max[1]);
      const int64_t j = (pid - k * (nface_max[0] * nface_max[1])) / nface_max[0];
      const int64_t i = pid % nface_max[0];
      return std::make_tuple(xmin[0] + i * dx_fine[0], xmin[1] + j * dx_fine[1],
                             xmin[2] + k * dx_fine[2]);
    };

    // positions of the nodes of the base (coarsest) mesh
    int n = 0;
    sample_positions_t xs((nx[0] + 1) * (nx[1] + 1) * (nx[2] + 1));
    for (int64_t k = 0; k < nface_max[2]; k += stride) {
      for (int64_t j = 0; j < nface_max[1]; j += stride) {
        for (int64_t i = 0; i < nface_max[0]; i += stride) {
          int64_t node_index = flatten_node(k, j, i);
          auto [xp, yp, zp] = pid_to_xyz(node_index);
          xs(n, 0) = xp;
          xs(n, 1) = yp;
          xs(n, 2) = zp;
          n++;
        }
      }
    }
    // evaluate the mask to see if each node is inside or outside the CAD part
    auto mask_vals = mask_func(xs);
    n = 0;
    for (int64_t k = 0; k < nface_max[2]; k += stride) {
      for (int64_t j = 0; j < nface_max[1]; j += stride) {
        for (int64_t i = 0; i < nface_max[0]; i += stride) {
          int64_t node_index = flatten_node(k, j, i);
          values[node_index] = mask_vals[n++];
        }
      }
    }

    // build level by level until max level or there is no more refinement necessary
    for (int level = 1; level <= max_level; level++) {
      // loop over all cells and refine where node values of the mask differ
      // keep a list of all the new node ids produced by refining
      // if no new nodes are produced, no refinement was done, and the mesh is complete
      new_points.clear();
      for (auto &c : cells) {
        c.refine(values, max_level, new_points, unflatten, flatten_cell, flatten_node,
                 ndim);
      }
      if (new_points.size() == 0) break;

      // get the positions of all the new nodes
      xs.resize(new_points.size());
      int n = 0;
      for (auto pid : new_points) {
        auto [x, y, z] = pid_to_xyz(pid);
        xs(n, 0) = x;
        xs(n, 1) = y;
        xs(n, 2) = z;
        n++;
      }
      // evaluate the mask at the new points
      auto mask_vals = mask_func(xs);
      n = 0;
      for (auto pid : new_points) {
        values[pid] = mask_vals[n++];
      }
    }
  }

  // interpolate on the mesh to decide whether a point is inside or outside the CAD part
  bool mask(const Real x, const Real y, const Real z) const {
    if (x < xmin[0] || x >= xmax[0]) return false;
    if (y < xmin[1] || y >= xmax[1]) return false;
    if (z < xmin[2] || z >= xmax[2]) return false;
    int i = (x - xmin[0]) / dx_base[0];
    int j = (y - xmin[1]) / dx_base[1];
    int k = (z - xmin[2]) / dx_base[2];
    int cindex = i + nx[0] * (j + nx[1] * k);
    std::array<Real, 3> cmin{xmin[0] + i * dx_base[0], xmin[1] + j * dx_base[1],
                             xmin[2] + k * dx_base[2]};
    std::array<Real, 3> cmax{xmin[0] + (i + 1) * dx_base[0],
                             xmin[1] + (j + 1) * dx_base[1],
                             xmin[2] + (k + 1) * dx_base[2]};
    auto val = cells[cindex].interp(x, y, z, cmin, cmax, values, unflatten, flatten_node,
                                    max_level, ndim);
    if (val > 0.0 && val < 1.e-9) PARTHENON_WARN("CAD mask val > 0 missed.\n");
    return (val >= 1.e-9);
  }

 private:
  int max_level;
  std::array<int, 3> nx;
  std::array<double, 3> xmin, xmax, dx_base;
  int ndim;

  std::vector<PartCell> cells;
  std::unordered_map<int64_t, bool> values;
  std::function<std::tuple<int64_t, int64_t, int64_t>(const int64_t)> unflatten;
  std::function<int64_t(const int64_t, const int64_t, const int64_t)> flatten_node,
      flatten_cell;
};

#endif

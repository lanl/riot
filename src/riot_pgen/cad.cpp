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

#include "cad.hpp"
#include "region_primitives.hpp"

// ---- headers (OCCT) ----
// github.com/Open-Cascade-SAS/OCCT
#include <BRepBndLib.hxx>
#include <BRepClass3d_SolidClassifier.hxx>
#include <STEPCAFControl_Reader.hxx>
#include <ShapeFix_Shape.hxx>
#include <TDataStd_Name.hxx>
#include <TDocStd_Document.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Solid.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <gp_Pnt.hxx>

// ---- headers (STL) ----
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// =================== Utilities ===================
//----------------------------------------------------------------------------------------
//! \fn  bool PointInAABB
//! \brief test whether a point is inside a bounding box
bool PointInAABB(const gp_Pnt &p, const Bnd_Box &box, Real padTol) {
  if (box.IsVoid()) return false;
  Standard_Real xmin, ymin, zmin, xmax, ymax, zmax;
  box.Get(xmin, ymin, zmin, xmax, ymax, zmax);
  return (p.X() >= xmin - padTol && p.X() <= xmax + padTol) &&
         (p.Y() >= ymin - padTol && p.Y() <= ymax + padTol) &&
         (p.Z() >= zmin - padTol && p.Z() <= zmax + padTol);
}

//----------------------------------------------------------------------------------------
//! \fn  static TopoDS_Shape HealShape
//! \brief try to deal with CAD parts that are not water tight
static TopoDS_Shape HealShape(const TopoDS_Shape &in, const double tol = 1.0e-7) {
  Handle(ShapeFix_Shape) fixer = new ShapeFix_Shape(in);
  fixer->SetPrecision(tol);
  fixer->SetMaxTolerance(tol * 10.0);
  fixer->Perform();
  return fixer->Shape();
}

//----------------------------------------------------------------------------------------
//! \fn  std::string GetShapeLabelName
//! \brief
std::string GetShapeLabelName(const TDF_Label &label) {
  Handle(TDataStd_Name) nameAttr;
  if (label.FindAttribute(TDataStd_Name::GetID(), nameAttr)) {
    TCollection_AsciiString ascii(nameAttr->Get());
    return ascii.ToCString();
  }
  return {};
}

//----------------------------------------------------------------------------------------
//! \fn  std::optional<TopoDS_Solid> LoadSolidFromSTEP
//! \brief extract a CAD part from a STEP file
std::optional<TopoDS_Solid> LoadSolidFromSTEP(const std::string &filename,
                                              const std::string &regionName) {
  // 1. Create a document and XDE shape tool
  Handle(TDocStd_Document) doc;
  Handle(XCAFApp_Application) app = XCAFApp_Application::GetApplication();
  app->NewDocument("MDTV-XCAF", doc);

  STEPCAFControl_Reader reader;
  reader.SetColorMode(Standard_False);
  reader.SetNameMode(Standard_True);
  reader.SetLayerMode(Standard_False);

  IFSelect_ReturnStatus status = reader.ReadFile(filename.c_str());
  if (status != IFSelect_RetDone) {
    std::cerr << "Failed to read STEP file: " << filename << std::endl;
    return std::nullopt;
  }

  // 2. Transfer to XDE document
  if (!reader.Transfer(doc)) {
    std::cerr << "Failed to transfer STEP file into XDE document.\n";
    return std::nullopt;
  }

  // 3. Access the shape tool
  Handle(XCAFDoc_ShapeTool) shapeTool = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
  if (shapeTool.IsNull()) {
    std::cerr << "No ShapeTool found in document.\n";
    return std::nullopt;
  }

  // 4. Iterate over shapes and match by name
  TDF_LabelSequence labels;
  shapeTool->GetFreeShapes(labels);

  for (Standard_Integer i = 1; i <= labels.Length(); ++i) {
    TDF_Label label = labels.Value(i);
    auto labelName = GetShapeLabelName(label);
    TCollection_ExtendedString xname;
    if (labelName == regionName) {
      TopoDS_Shape shape_orig = shapeTool->GetShape(label);
      TopoDS_Shape shape = HealShape(shape_orig);
      // If the shape itself isn’t a solid, search inside for one
      if (shape.ShapeType() == TopAbs_SOLID) {
        return TopoDS::Solid(shape);
      } else {
        for (TopExp_Explorer ex(shape, TopAbs_SOLID); ex.More(); ex.Next()) {
          return TopoDS::Solid(ex.Current());
        }
      }
    }
  }

  std::cerr << "Region \"" << regionName << "\" not found in file " << filename << "\n";
  return std::nullopt;
}

//----------------------------------------------------------------------------------------
//! \fn  auto SetBaseDx
//! \brief
auto SetBaseDx(ParameterInput *pin, parthenon::RegionSize &ds) {
  int nlevels = 1;
  if (pin->DoesParameterExist("parthenon/mesh", "refinement")) {
    std::string refine = pin->GetString("parthenon/mesh", "refinement");
    if (refine == "adaptive") {
      nlevels = pin->GetInteger("parthenon/mesh", "numlevel");
    }
  }
  std::array<Real, 3> dx;
  for (int d = X1DIR; d <= X3DIR; d++) {
    auto dir = static_cast<parthenon::CoordinateDirection>(d);
    dx[d - X1DIR] = (ds.xmax(dir) - ds.xmin(dir)) / ds.nx(dir);
  }
  return std::make_tuple(nlevels, dx);
}

//----------------------------------------------------------------------------------------
//! \fn  mask_func_t cad
//! \brief
mask_func_t cad(ParameterInput *pin, const std::string &block_name) {
  std::string cad_name = pin->GetString(block_name, "cadfile");
  std::string reg_name = pin->GetString(block_name, "name");
  auto solid_opt = LoadSolidFromSTEP(cad_name, reg_name);
  PARTHENON_REQUIRE(static_cast<bool>(solid_opt), "Failed to find region in cad file");
  auto solid = solid_opt.value();

  // BRep
  auto healedShape = HealShape(TopoDS_Shape(solid), 1.e-6);
  auto classifier = std::make_shared<BRepClass3d_SolidClassifier>();
  classifier->Load(healedShape);
  auto aabb = std::make_shared<Bnd_Box>();
  BRepBndLib::Add(healedShape, *aabb, false);
  // End BRep

  const Real tol = 1.e-9;

  std::array<Real, 3> xmin, xmax;
  aabb->Get(xmin[0], xmin[1], xmin[2], xmax[0], xmax[1], xmax[2]);
  auto [ds, bs] = parthenon::Mesh::GetRegionSizes(pin);
  for (int i = 0; i < 3; i++) {
    auto dir = static_cast<parthenon::CoordinateDirection>(i + X1DIR);
    xmin[i] = std::max(xmin[i], ds.xmin(dir));
    xmax[i] = std::min(xmax[i], ds.xmax(dir));
  }
  auto [n_amr_levels, dx_base] = SetBaseDx(pin, ds);
  Real dx_max =
      pin->GetOrAddReal(block_name, "sample_dx_max", -1.0,
                        "The maximum dx of the base Part mesh representing a CAD part.");
  if (dx_max > 0.0) {
    for (int i = 0; i < 3; i++)
      dx_base[i] = std::min(dx_base[i], dx_max);
  }
  int nlev_min = pin->GetInteger("regions", "nlev_min");
  int nlev_max = pin->GetInteger("regions", "nlev_max");
  std::array<int, 3> npart{static_cast<int>((xmax[0] - xmin[0]) / dx_base[0]) + 1,
                           static_cast<int>((xmax[1] - xmin[1]) / dx_base[1]) + 1,
                           static_cast<int>((xmax[2] - xmin[2]) / dx_base[2]) + 1};
  // make the base mesh at least 16 on a side, somewhat arbitrarily
  for (int i = 0; i < 3; i++)
    npart[i] = std::max(npart[i], 16);

  // build a lambda that tests whether a set of points is inside or outside
  // of the part we just loaded using OCCT's classifier
  auto cad_test = [=](const sample_positions_t &x) {
    std::vector<bool> mask(x.size());
    for (int i = 0; i < x.size(); ++i) {
      gp_Pnt pos(x(i, 0), x(i, 1), x(i, 2));
      if (!PointInAABB(pos, *aabb, tol)) {
        mask[i] = false;
        continue;
      }
      classifier->Perform(pos, tol);
      TopAbs_State st = classifier->State();
      mask[i] = (st == TopAbs_IN || st == TopAbs_ON);
    }
    return mask;
  };

  // now use that lambda to build a cell-based AMR mesh representation of the part
  // i.e. a structured set of points where we know the classification as a basis
  // fast queries when we are actually initializing the domain
  auto cad_mask = std::make_shared<PartMesh>(
      (dx_max > 0.0 ? 0 : nlev_max + n_amr_levels - 1), npart, xmin, xmax);
  cad_mask->build(cad_test);

  // this is actually the mask function we'll use for this cad region
  // it uses the cell-based AMR representation to make fast queries to decide the mask
  // value at a given {x, y, z}
  return base_region_loop(
      [=](const Real x, const Real y, const Real z) { return cad_mask->mask(x, y, z); });
}

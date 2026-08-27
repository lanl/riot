
#include "ray_trace.hpp"

namespace RayTrace {

TaskStatus Trace(MeshData<Real> *md, StateDescriptor *pkg) {


  auto pm = md->GetMeshPointer();
  auto &mesh_xmin = pm->mesh_size.xmin_;
  auto &mesh_xmax = pm->mesh_size.xmax_;
  const int ndim = pm->ndim;
  int dj = ndim > 1 ? 1 : 0;
  int dk = ndim > 2 ? 1 : 0;

  auto &vars = pkg->Param<std::vector<std::string>>("ray_trace_vars");

  // don't use the riot::MakePack/GetPack because we want *all* blocks
  auto resolved_pkgs = pm->resolved_packages.get();
  auto desc = parthenon::MakePackDescriptor<parthenon::variable_names::any>(resolved_pkgs, vars);



}



} // namespace RayTrace
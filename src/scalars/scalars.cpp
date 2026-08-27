//========================================================================================
// (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
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

#include <map>
#include <set>
#include <string>
#include <vector>

#include "scalars.hpp"

#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "variables.hpp"

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

using namespace parthenon::package::prelude;

namespace Scalars {

//----------------------------------------------------------------------------------------
//! \fn  std::shared_ptr<StateDescriptor> Scalars::Initialize
//! \brief
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  namespace ccmat = cell_variables::cell_averaged::mat;

  auto physics = std::make_shared<StateDescriptor>("scalars");
  Params &params = physics->AllParams();

  std::string base_name("scalars");
  int numscalar = 0;
  std::vector<std::string> svec, bulkvec;
  std::set<std::string> bulkscalars;
  std::map<std::string, std::set<int>> sparse_scalars;

  for (;;) {
    std::string name = base_name + std::to_string(numscalar);
    if (!pin->DoesBlockExist(name)) {
      break;
    }
    std::string label = pin->GetOrAddString(name, "label", name);
    if (!pin->DoesParameterExist(name, "matid")) {
      // Bulk-tied scalars are dense, not sparse. Also since they're
      // not tied to any mass, they're simply advected with the bulk
      // velocity. No prim vs conserved distinction required.
      // TODO(JMM): Is that what we want?
      if (sparse_scalars.count(label) > 0) {
        PARTHENON_THROW("Scalar var " + label +
                        " registered as both bulk- and mass-tied");
      }
      Metadata m =
          Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
                    Metadata::FillGhost, Metadata::Advected, Metadata::WithFluxes});
      physics->AddField(label, m);
      bulkscalars.insert(label);
    } else {
      int matid = pin->GetInteger(name, "matid");
      if (bulkscalars.count(label) > 0) {
        PARTHENON_THROW("Scalar var " + label +
                        " registered as both bulk- and mass-tied");
      }
      sparse_scalars[label].insert(matid);
    }
    numscalar++;
  }

  // Sparse scalars are material-tied and are allocated/de-allocated
  // with the material they are tied to.
  std::string control_field = ccmat::rho::name();
  Metadata ms_cons = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
                               Metadata::Sparse, Metadata::FillGhost, Metadata::Advected,
                               Metadata::WithFluxes});
  Metadata ms_prim =
      Metadata({Metadata::Cell, Metadata::Sparse, Metadata::Derived, Metadata::OneCopy});
  for (auto &s : sparse_scalars) {
    std::string prim_label = "prim." + s.first;
    ms_cons.Associate(prim_label);
    std::vector<int> sparse_ids;
    sparse_ids.insert(sparse_ids.begin(), s.second.begin(), s.second.end());
    physics->AddSparsePool(s.first, ms_cons, control_field, sparse_ids);
    physics->AddSparsePool(prim_label, ms_prim, control_field, sparse_ids);
    svec.push_back(prim_label);
    svec.push_back(s.first);
  }

  bulkvec.insert(bulkvec.begin(), bulkscalars.begin(), bulkscalars.end());
  for (auto &var : bulkscalars) {
    sparse_scalars[var] = {};
  }

  std::vector<std::string> all_scalar_vars;
  all_scalar_vars.insert(all_scalar_vars.end(), bulkvec.begin(), bulkvec.end());
  all_scalar_vars.insert(all_scalar_vars.end(), svec.begin(), svec.end());

  params.Add("scalar_vars", svec);
  params.Add("bulk_tied_scalar_vars", bulkvec);
  params.Add("all_scalar_vars", all_scalar_vars);
  params.Add("scalar_tie_map", sparse_scalars);

  physics->FillDerivedMesh = SetPrims;

  return physics;
}

//----------------------------------------------------------------------------------------
//! \fn  void Scalars::SetPrims
//! \brief
void SetPrims(MeshData<Real> *md) {
  using parthenon::PackIdx;

  auto var_list =
      md->GetParentPointer()->packages.Get("scalars")->Param<std::vector<std::string>>(
          "scalar_vars");
  const int cmrho = var_list.size();
  var_list.push_back("c.c.mat.rho");

  static auto desc = riot::MakePackDescriptor(var_list, md);
  auto vmesh = desc.GetPack(md);

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::entire, 0, vmesh.GetNBlocks(), md,
                                     parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const int irlo = vmesh.GetLowerBound(b, PackIdx(cmrho));
        const int irhi = vmesh.GetUpperBound(b, PackIdx(cmrho));

        for (int n = 0; n < cmrho; n += 2) {
          const int pstart = vmesh.GetLowerBound(b, PackIdx(n));
          const int pstop = vmesh.GetUpperBound(b, PackIdx(n));
          const int cstart = vmesh.GetLowerBound(b, PackIdx(n + 1));
          if (pstart >= 0) { // allocated on this block
            for (int p = pstart; p <= pstop; ++p) {
              const int sid = vmesh(b, p).sparse_id;
              int rho_id;
              for (rho_id = irlo; rho_id <= irhi; rho_id++) {
                if (sid == vmesh(b, rho_id).sparse_id) break;
              }
              const int c = cstart + p - pstart;
              auto prim = RiotLoop::make_var_view(idx_range, vmesh, p);
              auto cons = RiotLoop::make_var_view(idx_range, vmesh, c);
              auto ccmat_rho = RiotLoop::make_var_view(idx_range, vmesh, rho_id);
              RiotLoop::inner(idx_range, [&](const auto kji) {
                const Real cons0 = cons(kji);
                const Real rho = ccmat_rho(kji);
                const Real mask = (rho > 0);
                prim(kji) = mask * cons0 / (std::abs(rho) + 1e-16);
                cons(kji) = rho * prim(kji);
              });
            }
          }
        }
      });
}

} // namespace Scalars

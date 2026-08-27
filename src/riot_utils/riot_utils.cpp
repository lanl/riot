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
#include <cctype>
#include <string>
#include <utility>

#include <spiner/databox.hpp>

#include <interface/metadata.hpp>
#include <parthenon/package.hpp>

#include "riot_utils.hpp"
#include "variables.hpp"

using namespace parthenon::package::prelude;
using parthenon::MakeVarLabel;
using parthenon::MetadataFlag;

namespace RiotUtils {

//----------------------------------------------------------------------------------------
//! \fn  void RiotUtils::ToUpper
//! \brief
void ToUpper(std::string &s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::toupper(c); });
}

//----------------------------------------------------------------------------------------
//! \fn  void RiotUtils::ToLower
//! \brief
void ToLower(std::string &s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
}

// For a flag collection, generate the list of vars associated with each other.
// Returns a std::pair<std::vector<std::string>, std::vector<std::string>>
// The vars in the second vector are associated with the vars in the first.
// This is used in a few key places in the code:
//
// 1. The associated machinery is used to tie primitive vars to
//    conserved for anonymous advection
//----------------------------------------------------------------------------------------
//! \fn  VarNamePairList RiotUtils::GetAssociatedVars
//! \brief
VarNamePairList GetAssociatedVars(MeshData<Real> *md,
                                  const Metadata::FlagCollection &flags) {
  // make the assumption that all vars exist on all blocks, even if they aren't alloc'd
  MeshBlockData<Real> *pbd = (md->GetBlockData(0)).get();
  return GetAssociatedVars(pbd, flags);
}

//----------------------------------------------------------------------------------------
//! \fn  VarNamePairList RiotUtils::GetAssociatedVars
//! \brief
VarNamePairList GetAssociatedVars(MeshBlockData<Real> *bd,
                                  const Metadata::FlagCollection &flags) {
  std::vector<std::string> flagged_vars, assoc_vars;
  auto vars = bd->GetVariablesByFlag(flags).vars();
  for (auto &var : vars) {
    auto &var_name = var->label();
    auto assoc_name = var->getAssociated();
    if (assoc_name != var_name && var->IsSparse()) {
      assoc_name = MakeVarLabel(assoc_name, var->GetSparseID());
    }
    auto &assoc_var = bd->Get(assoc_name);
    flagged_vars.push_back(var->label());
    assoc_vars.push_back(assoc_var.label());
  }
  return std::make_pair(flagged_vars, assoc_vars);
}

//----------------------------------------------------------------------------------------
//! \fn  std::vector<std::string> RiotUtils::GetUnsplitVarNames
//! \brief
std::vector<std::string> GetUnsplitVarNames(Mesh *pmesh) {
  static const parthenon::MetadataFlag OperatorSplit =
      Metadata::GetOrAddFlag(riot::metadata::OperatorSplit);
  parthenon::Metadata::FlagCollection flags;
  flags.Exclude(OperatorSplit);
  auto varnames = pmesh->GetVariableNames(flags);
  if (std::find(varnames.begin(), varnames.end(), block_active_flag) == varnames.end()) {
    varnames.push_back(block_active_flag);
  }
  return varnames;
}

auto UniformlyResampleTimeSeries(const parthenon::HostArray2D<Real> &table,
                                 const std::string &new_name) {
  const int nrows = table.extent(0);
  PARTHENON_REQUIRE(nrows >= 2, "Time series must have at least 2 rows");
  PARTHENON_REQUIRE(table.extent(1) == 2, "Table must have exactly 2 columns");

  Real min_dt = std::numeric_limits<Real>::max();
  for (int i = 0; i < nrows - 1; ++i) {
    Real dt = table(i + 1, 0) - table(i, 0);
    PARTHENON_REQUIRE(dt > 0, "Time series must be monotonically increasing");
    min_dt = std::min(min_dt, dt);
  }

  const Real t_start = table(0, 0);
  const Real t_end = table(nrows - 1, 0);
  const int new_nrows = static_cast<int>((t_end - t_start) / min_dt) + 1;

  parthenon::HostArray2D<Real> result(new_name, new_nrows, 2);

  int j = 0;
  for (int i = 0; i < new_nrows; ++i) {
    const Real t = t_start + i * min_dt;
    result(i, 0) = t;

    while (j < nrows - 1 && table(j + 1, 0) < t) {
      ++j;
    }

    if (j >= nrows - 1) {
      result(i, 1) = table(nrows - 1, 1);
    } else if (table(j, 0) == t) {
      result(i, 1) = table(j, 1);
    } else {
      const Real t0 = table(j, 0);
      const Real t1 = table(j + 1, 0);
      const Real v0 = table(j, 1);
      const Real v1 = table(j + 1, 1);
      const Real alpha = (t - t0) / (t1 - t0);
      result(i, 1) = v0 + alpha * (v1 - v0);
    }
  }

  return result;
}

} // namespace RiotUtils

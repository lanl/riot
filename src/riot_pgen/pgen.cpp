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

// C++ includes
#include <functional>
#include <string>
#include <unordered_map>

#include "riot_pgen/pgen.hpp"

#include <singularity-eos/eos/eos.hpp>

namespace riot {

namespace internal {
// A problem holds exactly one of ProblemGenerator (per block) or
// MeshProblemGenerator. The modifier and package are optional.
struct RiotProblem {
  pgen_t ProblemGenerator = nullptr;
  mesh_pgen_t MeshProblemGenerator = nullptr;
  pmod_t ProblemModifier = nullptr;
  ppkg_t ProblemPackage = nullptr;
};
std::unordered_map<std::string, RiotProblem> pgen_dict;

//----------------------------------------------------------------------------------------
//! \fn  auto& riot::internal::GetProb
//! \brief
static auto &GetProb(ParameterInput *pin) {
  std::string name = pin->GetOrAddString("riot", "problem", "missing_problem_name");

  if (pgen_dict.count(name) == 0) {
    std::stringstream s;
    s << "Invalid problem name in input file.  Valid options include:" << std::endl;
    for (const auto &p : pgen_dict) {
      s << "   " << p.first << std::endl;
    }
    PARTHENON_THROW(s);
  }
  return pgen_dict.at(name);
}
} // namespace internal

//----------------------------------------------------------------------------------------
//! \fn  void riot::RegisterProblem
//! \brief Register a per-block problem generator.
void RegisterProblem(const std::string &name, pgen_t pgen, pmod_t pmod, ppkg_t ppkg) {
  internal::RiotProblem prob;
  prob.ProblemGenerator = std::move(pgen);
  prob.ProblemModifier = std::move(pmod);
  prob.ProblemPackage = std::move(ppkg);
  internal::pgen_dict[name] = std::move(prob);
}

//----------------------------------------------------------------------------------------
//! \fn  void riot::RegisterProblem
//! \brief Register a per-block problem generator with a package but no modifier.
void RegisterProblem(const std::string &name, pgen_t pgen, ppkg_t ppkg) {
  RegisterProblem(name, std::move(pgen), pmod_t(nullptr), std::move(ppkg));
}

//----------------------------------------------------------------------------------------
//! \fn  void riot::RegisterProblem
//! \brief Register a mesh-level problem generator.
void RegisterProblem(const std::string &name, mesh_pgen_t pgen, pmod_t pmod,
                     ppkg_t ppkg) {
  internal::RiotProblem prob;
  prob.MeshProblemGenerator = std::move(pgen);
  prob.ProblemModifier = std::move(pmod);
  prob.ProblemPackage = std::move(ppkg);
  internal::pgen_dict[name] = std::move(prob);
}

//----------------------------------------------------------------------------------------
//! \fn  void riot::RegisterProblem
//! \brief Register a mesh-level problem generator with a package but no modifier.
void RegisterProblem(const std::string &name, mesh_pgen_t pgen, ppkg_t ppkg) {
  RegisterProblem(name, std::move(pgen), pmod_t(nullptr), std::move(ppkg));
}

//----------------------------------------------------------------------------------------
//! \fn  void riot::ProblemGenerator
//! \brief Mesh-level entry point given to parthenon. Per-block generators are called
//! once for each block in the partition.
void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md) {
  if (md->NumBlocks() == 0) return;
  auto &prob = internal::GetProb(pin);
  if (prob.MeshProblemGenerator != nullptr) {
    prob.MeshProblemGenerator(pmesh, pin, md);
  } else if (prob.ProblemGenerator != nullptr) {
    for (int b = 0; b < md->NumBlocks(); ++b) {
      prob.ProblemGenerator(md->GetBlockData(b)->GetBlockPointer(), pin);
    }
  } else {
    PARTHENON_THROW("Problem generator is a null pointer!");
  }

  ClipMaterials(md, pin);
  ValidateTemperature(md);
  ValidateVfracs(md);
}

//----------------------------------------------------------------------------------------
//! \fn  void riot::ProblemModifier
//! \brief
void ProblemModifier(parthenon::ParthenonManager *pman) {
  auto &prob = internal::GetProb(pman->pinput.get());
  if (prob.ProblemModifier != nullptr) {
    prob.ProblemModifier(pman);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  std::shared_ptr<StateDescriptor> riot::ProblemPackage
//! \brief
std::shared_ptr<StateDescriptor> ProblemPackage(ParameterInput *pin) {
  auto &prob = internal::GetProb(pin);
  if (prob.ProblemPackage != nullptr) {
    return prob.ProblemPackage(pin);
  } else {
    // always add empty package with the problem name
    std::string name = pin->GetString("riot", "problem");
    auto pkg = std::make_shared<StateDescriptor>(name);
    return pkg;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void riot::AllocateAllSparse
//! \brief
void AllocateAllSparse(MeshData<Real> *md) {
  for (int b = 0; b < md->NumBlocks(); ++b) {
    auto &rc = md->GetBlockData(b);
    auto pmb = rc->GetBlockPointer();
    for (auto &var : rc->GetVariableVector()) {
      if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void riot::ClipMaterials
//! \brief
void ClipMaterials(MeshData<Real> *md, ParameterInput *pin) {
  namespace ccmat = cell_variables::cell_averaged::mat;

  auto hydro_pkg = md->GetMeshPointer()->packages.Get("hydro");
  const auto mass_frac_thresh = hydro_pkg->Param<Real>("mass_frac_thresh");
  const auto vol_frac_thresh = hydro_pkg->Param<Real>("vol_frac_thresh");

  auto v = riot::MakePack<ccmat::rho, ccmat::volume_fraction>(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ProblemGenerator::ClipMaterials", DevExecSpace(), 0,
      v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const int nmat = v.GetSize(b, ccmat::volume_fraction());
        Real rhotot = 0;
        for (int m = 0; m < nmat; ++m) {
          rhotot += v(b, ccmat::rho(m), k, j, i);
        }
        Real vfrac_new_sum = 0;
        if (rhotot > 0) {
          for (int m = 0; m < nmat; ++m) {
            Real fracmass = v(b, ccmat::rho(m), k, j, i) / rhotot;
            Real fracvol = v(b, ccmat::volume_fraction(m), k, j, i);
            if ((std::abs(fracmass) < mass_frac_thresh) ||
                (std::abs(fracvol) < vol_frac_thresh)) {
              v(b, ccmat::rho(m), k, j, i) = 0;
              v(b, ccmat::volume_fraction(m), k, j, i) = 0;
            }
            vfrac_new_sum += v(b, ccmat::volume_fraction(m), k, j, i);
          }
        }
        if (vfrac_new_sum > 0) {
          for (int m = 0; m < nmat; ++m) {
            v(b, ccmat::volume_fraction(m), k, j, i) /= vfrac_new_sum;
          }
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn  void riot::ValidateTemperature
//! \brief don't let bulk temperature go unset for PTE purposes
void ValidateTemperature(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  auto v = riot::MakePack<ccbulk::temperature>(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ProblemGenerator::SetTemp", DevExecSpace(), 0,
      v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        if (v(b, ccbulk::temperature(), k, j, i) < TINY_NUMBER) {
          v(b, ccbulk::temperature(), k, j, i) = 293.15;
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn  void riot::ValidateVfracs
//! \brief Check that material volume fractions sum approximately to 1. The check is
//! applied to each block separately, as it was when this ran per block.
void ValidateVfracs(MeshData<Real> *md) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  auto v = riot::MakePack<ccmat::volume_fraction>(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const auto total_cells_block =
      md->GetBlockData(0)->GetBlockPointer()->cellbounds.GetTotal(IndexDomain::interior);

  // TODO(JMM): sqrt(sum)/N is not an RMS (that would be sqrt(sum/N)), so this
  // threshold is very loose. Tightening it requires that hse and
  // ionized_shocktube_analytic set volume fractions.
  for (int b = 0; b < v.GetNBlocks(); ++b) {
    Real vfrac_delta = 0;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "ProblemGenerator::ValidateVfracs",
        DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &delta) {
          Real vfsum = 0;
          const int nmat = v.GetSize(b, ccmat::volume_fraction());
          for (int m = 0; m < nmat; ++m) {
            vfsum += v(b, ccmat::volume_fraction(m), k, j, i);
          }
          delta += (1. - vfsum) * (1. - vfsum);
        },
        Kokkos::Sum<Real>(vfrac_delta));

    vfrac_delta = std::sqrt(vfrac_delta) / total_cells_block;
    PARTHENON_REQUIRE_THROWS(vfrac_delta < 1e-1,
                             "Volume fraction must approximately sum to 1 in all cells");
  }
}

} // namespace riot

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
struct RiotProblem {
 public:
  RiotProblem() = default;
  RiotProblem(pgen_t &p)
      : ProblemGenerator(p), ProblemModifier(nullptr), ProblemPackage(nullptr) {}
  RiotProblem(pgen_t &pgen, pmod_t &pmod)
      : ProblemGenerator(pgen), ProblemModifier(pmod), ProblemPackage(nullptr) {}
  RiotProblem(pgen_t &pgen, ppkg_t &ppkg)
      : ProblemGenerator(pgen), ProblemModifier(nullptr), ProblemPackage(ppkg) {}
  RiotProblem(pgen_t &pgen, pmod_t &pmod, ppkg_t &ppkg)
      : ProblemGenerator(pgen), ProblemModifier(pmod), ProblemPackage(ppkg) {}
  pgen_t ProblemGenerator;
  pmod_t ProblemModifier;
  ppkg_t ProblemPackage;
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
//! \brief
//! JMM: I *think* I need to do these as overloads, not templates,
//! because I need these implemented here so the pgen_dict is pgen.cpp
//! scope.
void RegisterProblem(const std::string &name, pgen_t pgen) {
  internal::pgen_dict[name] = internal::RiotProblem(pgen);
}

//----------------------------------------------------------------------------------------
//! \fn  void riot::RegisterProblem
//! \brief
void RegisterProblem(const std::string &name, pgen_t pgen, pmod_t pmod) {
  internal::RiotProblem prob(pgen, pmod);
  internal::pgen_dict[name] = prob;
}

//----------------------------------------------------------------------------------------
//! \fn  void riot::RegisterProblem
//! \brief
void RegisterProblem(const std::string &name, pgen_t pgen, ppkg_t ppkg) {
  internal::RiotProblem prob(pgen, ppkg);
  internal::pgen_dict[name] = prob;
}

//----------------------------------------------------------------------------------------
//! \fn  void riot::RegisterProblem
//! \brief
void RegisterProblem(const std::string &name, pgen_t pgen, pmod_t pmod, ppkg_t ppkg) {
  internal::RiotProblem prob(pgen, pmod, ppkg);
  internal::pgen_dict[name] = prob;
}

//----------------------------------------------------------------------------------------
//! \fn  void riot::ProblemGenerator
//! \brief
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto &prob = internal::GetProb(pin);
  if (prob.ProblemGenerator == nullptr) {
    PARTHENON_THROW("Problem generator is a null pointer!");
  }
  prob.ProblemGenerator(pmb, pin);

  ClipMaterials(pmb, pin);
  ValidateTemperature(pmb);
  ValidateVfracs(pmb);
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
//! \fn  void riot::ClipMaterials
//! \brief
void ClipMaterials(MeshBlock *pmb, ParameterInput *pin) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;

  auto hydro_pkg = pmb->packages.Get("hydro");
  const auto mass_frac_thresh = hydro_pkg->Param<Real>("mass_frac_thresh");
  const auto vol_frac_thresh = hydro_pkg->Param<Real>("vol_frac_thresh");

  auto &rc = pmb->meshblock_data.Get();
  auto v = riot::MakePack<ccmat::rho, ccmat::volume_fraction>(rc.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  const int b = 0;
  pmb->par_for(
      "ProblemGenerator::ClipMaterials", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
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

// don't let bulk temperature go unset for PTE purposes
void ValidateTemperature(MeshBlock *pmb) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  // don't let bulk temperature go unset for PTE purposes
  auto &rc = pmb->meshblock_data.Get();
  auto v = riot::MakePack<ccbulk::temperature>(rc.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  const int b = 0;
  pmb->par_for(
      "ProblemGenerator::SetTemp", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        if (v(b, ccbulk::temperature(), k, j, i) < TINY_NUMBER) {
          v(b, ccbulk::temperature(), k, j, i) = 293.15;
        }
      });
}

// Check that material volume fractions sum approximately to 1.
void ValidateVfracs(MeshBlock *pmb) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  // don't let bulk temperature go unset for PTE purposes
  auto &rc = pmb->meshblock_data.Get();
  auto v = riot::MakePack<ccmat::volume_fraction>(rc.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real vfrac_delta = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "ProblemGenerator::ValidateVfracs",
      DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &delta) {
        Real vfsum = 0;
        const int nmat = v.GetSize(0, ccmat::volume_fraction());
        for (int m = 0; m < nmat; ++m) {
          vfsum += v(0, ccmat::volume_fraction(m), k, j, i);
        }
        delta += (1. - vfsum) * (1. - vfsum);
      },
      Kokkos::Sum<Real>(vfrac_delta));

  const auto total_cells_block = pmb->cellbounds.GetTotal(IndexDomain::interior);
  vfrac_delta = std::sqrt(vfrac_delta) / total_cells_block;
  PARTHENON_REQUIRE_THROWS(vfrac_delta < 1e-1,
                           "Volume fraction must approximately sum to 1 in all cells");
}

} // namespace riot

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
#ifndef RIOT_PGEN_PGEN_HPP_
#define RIOT_PGEN_PGEN_HPP_
// This file was made in part with generative AI.

// C++ includes
#include <functional>
#include <memory>
#include <string>

// Parthenon includes
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

// singularity includes
#include <singularity-eos/eos/eos.hpp>

// internal includes
#include "microphysics/eos_riot.hpp"
#include "riot_pgen/conduction_analytic.hpp"
#include "riot_pgen/ei_relax.hpp"
#include "riot_pgen/gaussian_conduction.hpp"
#include "riot_pgen/linear_modes.hpp"
#include "riot_pgen/pipe.hpp"
#include "riot_pgen/rad_shock.hpp"
#include "variables.hpp"

// add the name of a namespace that contains your new ProblemGenerator
#define FOREACH_PROBLEM                                                                  \
  PROBLEM(ei_relax)                                                                      \
  PROBLEM(gacc)                                                                          \
  PROBLEM(gaussian_conduction)                                                           \
  PROBLEM(gresho)                                                                        \
  PROBLEM(hse)                                                                           \
  PROBLEM(ionized_shocktube_analytic)                                                    \
  PROBLEM(quirk)                                                                         \
  PROBLEM(region_pgen)                                                                   \
  PROBLEM(rm)                                                                            \
  PROBLEM(rt)                                                                            \
  PROBLEM(shock_tube)                                                                    \
  PROBLEM(taylor_green)                                                                  \
  PROBLEM(tn_test)

namespace riot {

using pgen_t = std::function<void(MeshBlock *, ParameterInput *)>;
using pmod_t = std::function<void(parthenon::ParthenonManager *pman)>;
using ppkg_t = std::function<std::shared_ptr<StateDescriptor>(ParameterInput *)>;

// JMM: I'm not sure why, but the function pointers here must be
// passed by value, not reference.
void RegisterProblem(const std::string &name, pgen_t pgen);
void RegisterProblem(const std::string &name, pgen_t pgen, pmod_t pmod);
void RegisterProblem(const std::string &name, pgen_t pgen, ppkg_t ppkg);
void RegisterProblem(const std::string &name, pgen_t pgen, pmod_t pmod, ppkg_t ppkg);
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
void ProblemModifier(parthenon::ParthenonManager *pman);
void RegisterAllRiotProblems();
std::shared_ptr<StateDescriptor> ProblemPackage(ParameterInput *pin);
// This is a pgen utility you can use to clip materials if you belive
// it is appropriate to do so. Stick at the end of a specific problem
// generator, such as a custom regions pgen
void ClipMaterials(MeshBlock *pmb, ParameterInput *pin);
// don't let bulk temperature go unset for PTE purposes
void ValidateTemperature(MeshBlock *pmb);
// This checks that material volume fractions sum approximately to 1.
void ValidateVfracs(MeshBlock *pmb);

} // namespace riot

#define RIOT_PROBLEM(name) RegisterProblem(#name, name::ProblemGenerator)

// Declare all the problem generators
#define PROBLEM(name)                                                                    \
  namespace name {                                                                       \
  void ProblemGenerator(MeshBlock *, ParameterInput *);                                  \
  } // namespace name
FOREACH_PROBLEM
#undef PROBLEM

#define PROBLEM(name) RIOT_PROBLEM(name);
namespace riot {
inline void RegisterAllRiotProblems() {
  // Handle the originals
  FOREACH_PROBLEM;

  // Any that need to be handled with special care
  // TODO(JMM): This could be cleaned up in 1 of 2 ways:
  //
  // 1. Modify parthenon so that problems always contain a pgen, a
  //    modifier, and a package
  //
  // 2. Could make default modifiers and packages, so that these could
  //    be slurped in with the generator in the macro nonsense above
  RegisterProblem("conduction_analytic", conduction_analytic::ProblemGenerator,
                  conduction_analytic::ProblemModifier);
  RegisterProblem("linear_modes", linear_modes::ProblemGenerator,
                  linear_modes::ProblemModifier, linear_modes::ProblemPackage);

  // Radiation transport problems with pgen-defined custom boundary conditions
  RegisterProblem("shock", rad_shock::ProblemGenerator, rad_shock::ProblemModifier);
  RegisterProblem("pipe", crooked::ProblemGenerator, crooked::ProblemModifier);
}
#undef PROBLEM

} // namespace riot

#undef FOREACH_PROBLEM

#endif // RIOT_PGEN_PGEN_HPP_

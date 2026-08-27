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
#ifndef RIOT_DRIVER_HPP_
#define RIOT_DRIVER_HPP_
// This file was made in part with generative AI.

#include <memory>

#include <parthenon/driver.hpp>
using namespace parthenon::driver::prelude;
using parthenon::Packages_t;
using parthenon::StateDescriptor;

#include "plugins.hpp"
#include "variables.hpp"

namespace riot {

using TaskCollectionFnPtr = TaskCollection (*)(Mesh *pm, parthenon::SimTime &tm,
                                               const Real dt);

class RiotDriver : public EvolutionDriver {
 public:
  using Integrator_t = parthenon::LowStorageIntegrator;
  using IntegratorPtr_t = std::unique_ptr<Integrator_t>;
  RiotDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm);
  void RiotStepInit();
  TaskCollection RiotStepTasks();
  TaskCollection RiotPostStepTasks();
  virtual TaskListStatus Step();
  void ReportBlockHistogram();
  void ReportMemUsage();
  static Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);
  static std::vector<TaskCollectionFnPtr> OperatorSplitTasks;
  static void RegisterPgens();

 private:
  IntegratorPtr_t integrator;
  Real dt_init, dt_init_fact;
  TaskStatus SetBlockCosts(MeshData<Real> *md);
  std::vector<int> nmat_hist;
  std::vector<uint64_t> mem_usage;
  std::vector<uint64_t> max_rss;
  // raw pointers so that Kokkos::Views in Params get freed before Kokkos::Finalize()
  StateDescriptor *riot_pkg, *hydro_pkg, *mix_pkg, *tn_pkg, *mat_pkg, *lset_pkg,
      *gravity_pkg, *strength_pkg, *ion_pkg, *laser_pkg;
  bool do_hydro, do_strength, do_mix, do_tn, do_levelsets, do_gravity,
      do_multigroup_diffusion, do_ionization, do_lasers, curvilinear;
  bool fixed_fluid, use_general_pte, sparse_dealloc;
  riot_plugins::Plugins plugins;

  uint64_t GetMaxRss();
  void SetDiagnostics();
};

//----------------------------------------------------------------------------------------
//! \fn  Real BlockCost
//! \brief
inline Real BlockCost(int n) {
  n--;
  return std::exp(-0.137137 * n) + 0.494136 * n;
}

} // namespace riot

#endif // RIOT_DRIVER_HPP_

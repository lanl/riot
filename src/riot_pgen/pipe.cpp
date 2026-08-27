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

// C/C++ headers
#include <algorithm> // min, max
#include <cmath>     // sqrt()
#include <cstdio>    // fopen(), fprintf(), freopen()
#include <iostream>  // endl
#include <limits>    // numeric_limits
#include <sstream>   // stringstream
#include <string>    // c_str()

// Parthenon includes
#include <globals.hpp>
#include <mesh/mesh_refinement.hpp>
#include <parthenon_manager.hpp>

// Singularity includes
#include <singularity-eos/eos/eos.hpp>

// Riot includes
#include "radiation_transport/transport_utils/transport_utils.hpp"
#include "riot_pgen/pgen.hpp"
#include "riot_pgen/pipe.hpp"
#include "riot_utils/riot_utils.hpp"
#include "variables.hpp"

namespace crooked {

struct PipeParams {
  Real temp_s, temp_c;
  Real rho_o, rho_i;
  Real pres_co, pres_ci, pres_s;
  Real gm1, cv;
};

PipeParams pipe_params;

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator
//! \brief Sets initial conditions for crooked pipe test problem
//!        XY setup follows Southworth et al. 2024, LA-UR-24-20140
//!        RZ setup follows Till et al. 2018, LA-UR-17-28830
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace ccrad = cell_variables::cell_averaged::rad;

  PARTHENON_REQUIRE(pmb->packages.Get("riot")->Param<bool>("fixed_fluid"),
                    "Crooked pipe pgen requires a fixed fluid (physics/fixed_fluid).");
  PARTHENON_REQUIRE(pmb->packages.Get("riot")->Param<bool>("do_radiation_transport"),
                    "Crooked pipe pgen requires radiation transport "
                    "(radiation_transport/do_radiation_transport).");

  region_pgen::ProblemGenerator(pmb, pin);

  auto &rc = pmb->meshblock_data.Get();
  for (auto &var : rc->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }

  // Read global parameters
  pipe_params.temp_s = 5.803e6; // 0.5 keV
  pipe_params.temp_c = 5.803e5; // 0.05 keV
  pipe_params.rho_o = 10.0;
  pipe_params.rho_i = 0.01;
  pipe_params.gm1 =
      pin->GetReal("material0", "Gamma", "Adiabatic Index of material0") - 1.0;
  pipe_params.cv =
      pin->GetReal("material0", "Cv", "Specific heat at constant volume for material0");
  PARTHENON_REQUIRE(pipe_params.rho_o == 10.0 && pipe_params.rho_i == 0.01,
                    "Issue detected in crooked pipe input file!");

  // Derived
  const Real u_co = pipe_params.rho_o * pipe_params.cv * pipe_params.temp_c;
  const Real u_ci = pipe_params.rho_i * pipe_params.cv * pipe_params.temp_c;
  const Real u_s = pipe_params.rho_i * pipe_params.cv * pipe_params.temp_s;
  pipe_params.pres_co = pipe_params.gm1 * u_co;
  pipe_params.pres_ci = pipe_params.gm1 * u_ci;
  pipe_params.pres_s = pipe_params.gm1 * u_s;

  // Capture variables for kernel
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  // Initialize intensity field (hydro already handled by regions)
  const auto rad_pkg = GetRadPackage(pmb->packages);
  PARTHENON_REQUIRE(rad_pkg->Param<bool>("fixed_pgen_opac"),
                    "Crooked pipe pgen requires fixed pgen opacities "
                    "(radiation_transport/fixed_pgen_opac).");
  const int ngroups = rad_pkg->Param<int>("ngroups");
  const int nangles = rad_pkg->Param<int>("nangles");
  const auto unit_utils = rad_pkg->Param<UnitUtils>("unit_utils");
  const auto fbnd = *(rad_pkg->MutableParam<ParArray1D<Real>>("fbnd_d"));
  static auto descr =
      MakePackDescriptor<ccrad::intensity>((pmb->resolved_packages).get());
  auto vr = descr.GetPack(rc.get());
  auto ppr = pipe_params;
  pmb->par_for(
      "ProblemGenerator::pipe1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real trad = ppr.temp_c;
        for (int gg = 0; gg < ngroups; ++gg) {
          const Real ee = Emissivity(gg, trad, fbnd, ngroups, unit_utils);
          for (int aa = 0; aa < nangles; ++aa) {
            vr(0, ccrad::intensity(GAI(nangles, gg, aa)), k, j, i) = ee;
          }
        }
      });

  // Manually set opacities
  static auto desco =
      MakePackDescriptor<ccmat::rho, ccrad::aa>((pmb->resolved_packages).get());
  auto vo = desco.GetPack(rc.get());
  pmb->par_for(
      "ProblemGenerator::pipe2", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real &rho = vo(0, ccmat::rho(0), k, j, i);
        const bool outside = (rho == 10.0);
        for (int gg = 0; gg < ngroups; ++gg) {
          vo(0, ccrad::aa(gg), k, j, i) = rho * (outside * 200.0 + (!outside) * 20.0);
        }
      });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemModifier
//! \brief Enroll the custom crooked-pipe boundary conditions and moment output
void ProblemModifier(parthenon::ParthenonManager *pman) {
  using BF = parthenon::BoundaryFace;
  pman->app_input->RegisterBoundaryCondition(BF::inner_x1, "ic", crooked::PipeInnerX1);
  pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "ic", crooked::PipeOuterX1);
}

//----------------------------------------------------------------------------------------
//! \fn void PipeInnerX1
//! \brief Custom InnerX1 BC for crooked pipe problem
void PipeInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace ccrad = cell_variables::cell_averaged::rad;
  auto pmb = mbd->GetBlockPointer();
  const auto nb = IndexRange{0, 0};

  static auto desc =
      GetBoundaryPackDescriptorMap<ccmat::rho, ccmat::volume_fraction, ccbulk::velocity,
                                   ccbulk::temperature, ccbulk::pressure>(mbd);
  auto v = desc[coarse].GetPack(mbd.get());
  const auto &coords = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  if (v.GetMaxNumberOfVars() > 0) {
    auto pp = pipe_params;
    pmb->par_for_bndry(
        "PipeInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::CC,
        coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          const Real x2v = coords.Xc<X2DIR>(j);
          const bool inside = (std::abs(x2v) <= 0.5);
          v(0, ccmat::rho(0), k, j, i) = (inside) ? pp.rho_i : pp.rho_o;
          v(0, ccmat::volume_fraction(0), k, j, i) = 1.0;
          v(0, ccbulk::velocity(0), k, j, i) = 0.0;
          v(0, ccbulk::velocity(1), k, j, i) = 0.0;
          v(0, ccbulk::velocity(2), k, j, i) = 0.0;
          v(0, ccbulk::temperature(), k, j, i) = (inside) ? pp.temp_s : pp.temp_c;
          v(0, ccbulk::pressure(), k, j, i) = (inside) ? pp.pres_s : pp.pres_co;
        });
  }

  const auto rad_pkg = GetRadPackage(pmb->packages);
  const int ngroups = rad_pkg->Param<int>("ngroups");
  const int nangles = rad_pkg->Param<int>("nangles");
  const auto unit_utils = rad_pkg->Param<UnitUtils>("unit_utils");
  const auto fbnd = *(rad_pkg->MutableParam<ParArray1D<Real>>("fbnd_d"));

  // Intensity
  static auto descr = GetBoundaryPackDescriptorMap<ccrad::intensity>(mbd);
  auto vr = descr[coarse].GetPack(mbd.get());
  if (vr.GetMaxNumberOfVars() > 0) {
    auto ppr = pipe_params;
    pmb->par_for_bndry(
        "PipeInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::CC,
        coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          const Real x2v = coords.Xc<X2DIR>(j);
          const bool outside = (std::abs(x2v) > 0.5);
          const Real trad = (outside) ? ppr.temp_c : ppr.temp_s;
          for (int gg = 0; gg < ngroups; ++gg) {
            const Real ee = Emissivity(gg, trad, fbnd, ngroups, unit_utils);
            for (int aa = 0; aa < nangles; ++aa) {
              vr(0, ccrad::intensity(GAI(nangles, gg, aa)), k, j, i) = ee;
            }
          }
        });
  }

  // Opacity
  static auto desco = GetBoundaryPackDescriptorMap<ccrad::aa, ccrad::ss>(mbd);
  auto vo = desco[coarse].GetPack(mbd.get());
  if (vo.GetMaxNumberOfVars() > 0) {
    const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const auto &range = bounds.GetBoundsI(IndexDomain::interior);
    const int ref = range.s;
    pmb->par_for_bndry(
        "PipeInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::CC,
        coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          const Real x2v = coords.Xc<X2DIR>(j);
          const bool outside = (std::abs(x2v) > 0.5);
          for (int gg = 0; gg < ngroups; ++gg) {
            vo(0, ccrad::aa(gg), k, j, i) = outside * vo(0, ccrad::aa(gg), k, j, ref);
            vo(0, ccrad::ss(gg), k, j, i) = outside * vo(0, ccrad::ss(gg), k, j, ref);
          }
        });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PipeOuterX1
//! \brief Custom OuterX1 BC for crooked pipe problem
void PipeOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace ccrad = cell_variables::cell_averaged::rad;
  auto pmb = mbd->GetBlockPointer();
  const auto nb = IndexRange{0, 0};

  static auto desc =
      GetBoundaryPackDescriptorMap<ccmat::rho, ccmat::volume_fraction, ccbulk::velocity,
                                   ccbulk::temperature, ccbulk::pressure>(mbd);
  auto v = desc[coarse].GetPack(mbd.get());
  const auto &coords = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  if (v.GetMaxNumberOfVars() > 0) {
    auto pp = pipe_params;
    pmb->par_for_bndry(
        "PipeOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
        coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          const Real x2v = coords.Xc<X2DIR>(j);
          const bool inside = (std::abs(x2v) <= 0.5);
          v(0, ccmat::rho(0), k, j, i) = (inside) ? pp.rho_i : pp.rho_o;
          v(0, ccmat::volume_fraction(0), k, j, i) = 1.0;
          v(0, ccbulk::velocity(0), k, j, i) = 0.0;
          v(0, ccbulk::velocity(1), k, j, i) = 0.0;
          v(0, ccbulk::velocity(2), k, j, i) = 0.0;
          v(0, ccbulk::temperature(), k, j, i) = pp.temp_c;
          v(0, ccbulk::pressure(), k, j, i) = (inside) ? pp.pres_co : pp.pres_ci;
        });
  }

  const auto rad_pkg = GetRadPackage(pmb->packages);
  const int ngroups = rad_pkg->Param<int>("ngroups");
  const int nangles = rad_pkg->Param<int>("nangles");
  const auto unit_utils = rad_pkg->Param<UnitUtils>("unit_utils");
  const auto fbnd = *(rad_pkg->MutableParam<ParArray1D<Real>>("fbnd_d"));

  // Intensity
  static auto descr = GetBoundaryPackDescriptorMap<ccrad::intensity>(mbd);
  auto vr = descr[coarse].GetPack(mbd.get());
  if (vr.GetMaxNumberOfVars() > 0) {
    auto ppr = pipe_params;
    pmb->par_for_bndry(
        "PipeOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
        coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          const Real trad = ppr.temp_c;
          for (int gg = 0; gg < ngroups; ++gg) {
            const Real ee = Emissivity(gg, trad, fbnd, ngroups, unit_utils);
            for (int aa = 0; aa < nangles; ++aa) {
              vr(0, ccrad::intensity(GAI(nangles, gg, aa)), k, j, i) = ee;
            }
          }
        });
  }

  // Opacity
  static auto desco = GetBoundaryPackDescriptorMap<ccrad::aa, ccrad::ss>(mbd);
  auto vo = desco[coarse].GetPack(mbd.get());
  if (vo.GetMaxNumberOfVars() > 0) {
    const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const auto &range = bounds.GetBoundsI(IndexDomain::interior);
    const int ref = range.e;
    pmb->par_for_bndry(
        "PipeOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
        coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          for (int gg = 0; gg < ngroups; ++gg) {
            vo(0, ccrad::aa(gg), k, j, i) = vo(0, ccrad::aa(gg), k, j, ref);
            vo(0, ccrad::ss(gg), k, j, i) = vo(0, ccrad::ss(gg), k, j, ref);
          }
        });
  }
}

} // namespace crooked

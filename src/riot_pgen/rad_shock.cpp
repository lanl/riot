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
#include <parthenon_manager.hpp>

// Singularity includes
#include <singularity-eos/eos/eos.hpp>

// Riot includes
#include "microphysics/opacity_models.hpp"
#include "radiation_transport/transport_utils/transport_utils.hpp"
#include "riot_pgen/pgen.hpp"
#include "riot_pgen/rad_shock.hpp"
#include "riot_utils/riot_utils.hpp"
#include "variables.hpp"

namespace rad_shock {

struct ShockParams {
  Real rho_l, vx_l, temp_l, u_l, pres_l;
  Real rho_r, vx_r, temp_r, u_r, pres_r;
  Real gm1, cv, xd;
};

ShockParams shock_params;

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator
//! \brief Sets initial conditions for a radiating L/R interface
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace ccrad = cell_variables::cell_averaged::rad;

  // Read global parameters
  shock_params.rho_l = pin->GetReal("problem", "rho_l", "L-state density");
  shock_params.rho_r = pin->GetReal("problem", "rho_r", "R-state density");
  shock_params.vx_l = pin->GetReal("problem", "vx_l", "L-state vel-x");
  shock_params.vx_r = pin->GetReal("problem", "vx_r", "R-state vel-x");
  shock_params.temp_l = pin->GetReal("problem", "temp_l", "L-state temperature");
  shock_params.temp_r = pin->GetReal("problem", "temp_r", "R-state temperature");
  shock_params.gm1 =
      pin->GetReal("material0", "Gamma", "Adiabatic index of material0") - 1.0;
  shock_params.cv =
      pin->GetReal("material0", "Cv", "Specific heat at constant volume for material0");
  shock_params.xd = pin->GetReal("problem", "xd", "x-position of discontinuity");

  // Derived
  shock_params.u_l = shock_params.rho_l * shock_params.cv * shock_params.temp_l;
  shock_params.u_r = shock_params.rho_r * shock_params.cv * shock_params.temp_r;
  shock_params.pres_l = shock_params.gm1 * shock_params.u_l;
  shock_params.pres_r = shock_params.gm1 * shock_params.u_r;

  // Capture variables for kernel
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  // Prep setting MeshBlockData
  auto &rc = pmb->meshblock_data.Get();
  for (auto &var : rc->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }

  // Packing
  static auto desc = MakePackDescriptor<ccmat::rho, ccmat::volume_fraction,
                                        ccbulk::total_material_energy, ccbulk::momentum>(
      (pmb->resolved_packages).get());
  auto v = desc.GetPack(rc.get());

  // Set initial condition
  auto sp = shock_params;
  auto &coords = pmb->coords;
  pmb->par_for(
      "ProblemGenerator::shock1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x1v = coords.Xc<X1DIR>(i);
        const Real rho = (x1v >= sp.xd) ? sp.rho_r : sp.rho_l;
        const Real vx = (x1v >= sp.xd) ? sp.vx_r : sp.vx_l;
        const Real u_bulk = (x1v >= sp.xd) ? sp.u_r : sp.u_l;
        v(0, ccmat::rho(0), k, j, i) = rho;
        v(0, ccmat::volume_fraction(0), k, j, i) = 1.0;
        v(0, ccbulk::momentum(0), k, j, i) = rho * vx;
        v(0, ccbulk::momentum(1), k, j, i) = 0.0;
        v(0, ccbulk::momentum(2), k, j, i) = 0.0;
        v(0, ccbulk::total_material_energy(), k, j, i) = u_bulk + 0.5 * rho * SQR(vx);
      });

  // Now initialize intensity field
  if (pmb->packages.Get("riot")->Param<bool>("do_radiation_transport")) {
    const auto rad_pkg = GetRadPackage(pmb->packages);
    const int ngroups = rad_pkg->Param<int>("ngroups");
    const int nangles = rad_pkg->Param<int>("nangles");
    const auto unit_utils = rad_pkg->Param<UnitUtils>("unit_utils");
    const auto fbnd = *(rad_pkg->MutableParam<ParArray1D<Real>>("fbnd_d"));
    static auto descr =
        MakePackDescriptor<ccrad::intensity>((pmb->resolved_packages).get());
    auto vr = descr.GetPack(rc.get());
    auto spr = shock_params;
    pmb->par_for(
        "ProblemGenerator::shock2", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real x1v = coords.Xc<X1DIR>(i);
          const Real trad = (x1v >= spr.xd) ? spr.temp_r : spr.temp_l;
          for (int gg = 0; gg < ngroups; ++gg) {
            const Real ee = Emissivity(gg, trad, fbnd, ngroups, unit_utils);
            for (int aa = 0; aa < nangles; ++aa) {
              vr(0, ccrad::intensity(GAI(nangles, gg, aa)), k, j, i) = ee;
            }
          }
        });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemModifier
//! \brief Enroll the custom radiating-shock boundary conditions and moment output
void ProblemModifier(parthenon::ParthenonManager *pman) {
  using BF = parthenon::BoundaryFace;
  pman->app_input->RegisterBoundaryCondition(BF::inner_x1, "ic", rad_shock::ShockInnerX1);
  pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "ic", rad_shock::ShockOuterX1);
}

//----------------------------------------------------------------------------------------
//! \fn void ShockInnerX1
//! \brief Custom InnerX1 BC for radiating shock
void ShockInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
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
  if (v.GetMaxNumberOfVars() > 0) {
    auto sp = shock_params;
    pmb->par_for_bndry(
        "ShockInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::CC,
        coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          v(0, ccmat::rho(0), k, j, i) = sp.rho_l;
          v(0, ccmat::volume_fraction(0), k, j, i) = 1.0;
          v(0, ccbulk::velocity(0), k, j, i) = sp.vx_l;
          v(0, ccbulk::velocity(1), k, j, i) = 0.0;
          v(0, ccbulk::velocity(2), k, j, i) = 0.0;
          v(0, ccbulk::temperature(), k, j, i) = sp.temp_l;
          v(0, ccbulk::pressure(), k, j, i) = sp.pres_l;
        });
  }

  if (pmb->packages.Get("riot")->Param<bool>("do_radiation_transport")) {
    const auto rad_pkg = GetRadPackage(pmb->packages);
    const int ngroups = rad_pkg->Param<int>("ngroups");
    const int nangles = rad_pkg->Param<int>("nangles");
    const auto unit_utils = rad_pkg->Param<UnitUtils>("unit_utils");
    const auto fbnd = *(rad_pkg->MutableParam<ParArray1D<Real>>("fbnd_d"));

    // Intensity
    static auto descr = GetBoundaryPackDescriptorMap<ccrad::intensity>(mbd);
    auto vr = descr[coarse].GetPack(mbd.get());
    if (vr.GetMaxNumberOfVars() > 0) {
      auto spr = shock_params;
      pmb->par_for_bndry(
          "ShockInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::CC,
          coarse, false,
          KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
            const Real trad = spr.temp_l;
            for (int gg = 0; gg < ngroups; ++gg) {
              const Real ee = Emissivity(gg, trad, fbnd, ngroups, unit_utils);
              for (int aa = 0; aa < nangles; ++aa) {
                vr(0, ccrad::intensity(GAI(nangles, gg, aa)), k, j, i) = ee;
              }
            }
          });
    }

    // Opacity
    auto &mat_pkg = pmb->packages.Get("materials");
    const auto opac_a = mat_pkg->Param<ParArray1D<RiotOpacity::MeanOpacA>>("d.d.opac_a");
    const auto opac_s = mat_pkg->Param<ParArray1D<RiotOpacity::MeanOpacS>>("d.d.opac_s");
    const auto opac_from_matid = mat_pkg->Param<ParArray1D<int>>("d.opac_from_matid");
    static auto desco = GetBoundaryPackDescriptorMap<ccrad::aa, ccrad::ss>(mbd);
    auto vo = desco[coarse].GetPack(mbd.get());
    if (vo.GetMaxNumberOfVars() > 0) {
      auto spr = shock_params;
      pmb->par_for_bndry(
          "ShockInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::CC,
          coarse, false,
          KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
            const int opac_id = opac_from_matid(0);
            for (int gg = 0; gg < ngroups; ++gg) {
              vo(0, ccrad::aa(gg), k, j, i) =
                  opac_a(opac_id).AbsorptionCoefficient(spr.rho_l, spr.temp_l, gg);
              vo(0, ccrad::ss(gg), k, j, i) =
                  opac_s(opac_id).ScatteringCoefficient(spr.rho_l, spr.temp_l, gg);
            }
          });
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ShockOuterX1
//! \brief Custom OuterX1 BC for radiating shock
void ShockOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
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
  if (v.GetMaxNumberOfVars() > 0) {
    auto sp = shock_params;
    pmb->par_for_bndry(
        "ShockOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
        coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          v(0, ccmat::rho(0), k, j, i) = sp.rho_r;
          v(0, ccmat::volume_fraction(0), k, j, i) = 1.0;
          v(0, ccbulk::velocity(0), k, j, i) = sp.vx_r;
          v(0, ccbulk::velocity(1), k, j, i) = 0.0;
          v(0, ccbulk::velocity(2), k, j, i) = 0.0;
          v(0, ccbulk::temperature(), k, j, i) = sp.temp_r;
          v(0, ccbulk::pressure(), k, j, i) = sp.pres_r;
        });
  }

  if (pmb->packages.Get("riot")->Param<bool>("do_radiation_transport")) {
    const auto rad_pkg = GetRadPackage(pmb->packages);
    const int ngroups = rad_pkg->Param<int>("ngroups");
    const int nangles = rad_pkg->Param<int>("nangles");
    const auto unit_utils = rad_pkg->Param<UnitUtils>("unit_utils");
    const auto fbnd = *(rad_pkg->MutableParam<ParArray1D<Real>>("fbnd_d"));

    // Intensity
    static auto descr = GetBoundaryPackDescriptorMap<ccrad::intensity>(mbd);
    auto vr = descr[coarse].GetPack(mbd.get());
    if (vr.GetMaxNumberOfVars() > 0) {
      auto spr = shock_params;
      pmb->par_for_bndry(
          "ShockOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
          coarse, false,
          KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
            const Real trad = spr.temp_r;
            for (int gg = 0; gg < ngroups; ++gg) {
              const Real ee = Emissivity(gg, trad, fbnd, ngroups, unit_utils);
              for (int aa = 0; aa < nangles; ++aa) {
                vr(0, ccrad::intensity(GAI(nangles, gg, aa)), k, j, i) = ee;
              }
            }
          });
    }

    // Opacity
    auto &mat_pkg = pmb->packages.Get("materials");
    const auto opac_a = mat_pkg->Param<ParArray1D<RiotOpacity::MeanOpacA>>("d.d.opac_a");
    const auto opac_s = mat_pkg->Param<ParArray1D<RiotOpacity::MeanOpacS>>("d.d.opac_s");
    const auto opac_from_matid = mat_pkg->Param<ParArray1D<int>>("d.opac_from_matid");
    static auto desco = GetBoundaryPackDescriptorMap<ccrad::aa, ccrad::ss>(mbd);
    auto vo = desco[coarse].GetPack(mbd.get());
    if (vo.GetMaxNumberOfVars() > 0) {
      auto spr = shock_params;
      pmb->par_for_bndry(
          "ShockOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
          coarse, false,
          KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
            const int opac_id = opac_from_matid(0);
            for (int gg = 0; gg < ngroups; ++gg) {
              vo(0, ccrad::aa(gg), k, j, i) =
                  opac_a(opac_id).AbsorptionCoefficient(spr.rho_r, spr.temp_r, gg);
              vo(0, ccrad::ss(gg), k, j, i) =
                  opac_s(opac_id).ScatteringCoefficient(spr.rho_r, spr.temp_r, gg);
            }
          });
    }
  }
}

} // namespace rad_shock

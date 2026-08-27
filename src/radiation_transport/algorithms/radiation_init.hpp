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
#ifndef RADIATION_ALGORITHMS_RADIATION_INIT_HPP_
#define RADIATION_ALGORITHMS_RADIATION_INIT_HPP_
// This file was made in part with generative AI.

// C++ headers
#include <limits>
#include <memory>
#include <string>
#include <vector>

// Parthenon headers
#include <parthenon/package.hpp>

// Riot headers
#include "radiation_transport/transport_utils/transport_utils.hpp"
#include "riot_utils/riot_loops.hpp"

using namespace parthenon::package::prelude;

//----------------------------------------------------------------------------------------
// Shared initialization for the radiation transport initialization (none, zero, thermal).
//----------------------------------------------------------------------------------------
namespace RadiationInit {

// The input blocks from which the shared radiation initialization type is read.
constexpr char init_block[] = "radiation_transport/init";

//----------------------------------------------------------------------------------------
//! \fn void RadiationPostInitialization
//! \brief
inline void RadiationPostInitialization(Mesh *pm, ParameterInput *pin,
                                        MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace ccrad = cell_variables::cell_averaged::rad;
  if (md->NumBlocks() == 0) return;

  // Radiation package
  const auto rad_pkg = GetRadPackage(pm->packages);
  const int ngroups = rad_pkg->Param<int>("ngroups");
  const int nangles = rad_pkg->Param<int>("nangles");

  // Set opacities
  // NOTE(@pdmullen): We are not required to set opacities in the ProblemGenerator,
  // however, this makes them visible in the initial condition which is useful for viz
  static auto desc =
      MakePackDescriptor<ccmat::rho, ccmat::volume_fraction, ccbulk::temperature,
                         ccrad::aa, ccrad::ss>((pm->resolved_packages).get());
  auto v = desc.GetPack(md);

  auto &mat_pkg = pm->packages.Get("materials");
  const auto &opac_a = mat_pkg->Param<ParArray1D<RiotOpacity::MeanOpacA>>("d.d.opac_a");
  const auto &opac_s = mat_pkg->Param<ParArray1D<RiotOpacity::MeanOpacS>>("d.d.opac_s");
  const auto &opac_from_matid = mat_pkg->Param<ParArray1D<int>>("d.opac_from_matid");

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::entire, 0, v.GetNBlocks(), md,
                                     parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
          for (int gg = 0; gg < ngroups; ++gg) {
            // Set opacities
            Real &aa = v(b, ccrad::aa(gg), k, j, i) = 0.0;
            Real &ss = v(b, ccrad::ss(gg), k, j, i) = 0.0;
            const Real &temp = v(b, ccbulk::temperature(), k, j, i);
            for (int m = 0; m < v.GetSize(b, ccmat::rho()); ++m) {
              const Real &vfracm = v(b, ccmat::volume_fraction(m), k, j, i);
              const Real rhom = v(b, ccmat::rho(m), k, j, i) / (vfracm + 1.0e-100);
              const int &mat_id = v(b, ccmat::rho(m)).sparse_id;
              const int &phase_id = v(b, ccmat::rho(m)).v;
              const int opac_id = opac_from_matid(mat_id) + phase_id;
              const Real aam = (rhom > 0)
                                   ? opac_a(opac_id).AbsorptionCoefficient(rhom, temp, gg)
                                   : 0.0;
              const Real ssm = (rhom > 0)
                                   ? opac_s(opac_id).ScatteringCoefficient(rhom, temp, gg)
                                   : 0.0;
              aa += vfracm * aam;
              ss += vfracm * ssm;
            }
          }
        });
      });

  // Extract initialization type
  const std::string init_type = rad_pkg->Param<std::string>("initialization_type");
  if (init_type == "none") return; // maintain pgen-set specific intensity

  // Set specific intensity
  static auto desc_i =
      MakePackDescriptor<ccrad::intensity>((pm->resolved_packages).get());
  auto vi = desc_i.GetPack(md);

  // Capture variables for kernel
  const auto unit_utils = rad_pkg->Param<UnitUtils>("unit_utils");
  const auto fbnd = *(rad_pkg->MutableParam<ParArray1D<Real>>("fbnd_d"));
  const bool thermal = (init_type == "thermal");

  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
          const Real &temp = v(b, ccbulk::temperature(), k, j, i);
          for (int gg = 0; gg < ngroups; ++gg) {
            // Initialize specific intensity to thermal (or zero if not thermal)
            const Real ee = (thermal && v(b, ccrad::aa(gg), k, j, i) > 0.0)
                                ? Emissivity(gg, temp, fbnd, ngroups, unit_utils)
                                : 0.0;
            for (int aa = 0; aa < nangles; ++aa) {
              vi(b, ccrad::intensity(GAI(nangles, gg, aa)), k, j, i) = ee;
            }
          }
        });
      });
}

// Individual shared init options.
namespace InitOption {
//----------------------------------------------------------------------------------------
//! \fn void AddInitType
//! \brief
inline void AddInitType(ParameterInput *pin, Params &params) {
  // Extract init type
  const std::string init_type =
      pin->GetOrAddString(init_block, "initialization", "thermal",
                          {"none", "zero", "thermal"}, "Radiation initializations");
  params.Add("initialization_type", init_type);
}

} // namespace InitOption

//----------------------------------------------------------------------------------------
//! \fn EnrollRadiationInit
inline void EnrollRadiationInit(StateDescriptor *rad_pkg, ParameterInput *pin,
                                Params &params) {
  InitOption::AddInitType(pin, params);
  rad_pkg->PostInitializationMesh = RadiationPostInitialization;
}

} // namespace RadiationInit

#endif // RADIATION_ALGORITHMS_RADIATION_INIT_HPP_

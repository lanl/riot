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
#ifndef IONIZATION_IONIZATION_HPP_
#define IONIZATION_IONIZATION_HPP_
// This file was made in part with generative AI.

#include <memory>

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

#include <singularity-eos/closure/mixed_cell_models.hpp>
#include <singularity-eos/eos/eos.hpp>

#include "ionization/conduction_equation.hpp"
#include "ionization/ionization_base.hpp"
#include "microphysics/eos_riot.hpp"
#include "riot_utils/riot_loops.hpp"
#include "riot_utils/riot_utils.hpp"
#include "variables.hpp"

namespace Ionization {
using namespace parthenon::package::prelude;

ElectronThermalConductivityModel
ElectronConductivityModelEnumFromString(const std::string &model);
IonThermalConductivityModel IonConductivityModelEnumFromString(const std::string &model);
CoulombLogarithmKind ParseCoulombLogarithmKind(std::string coulomb_logarithm);
PlasmaViscosityModel PlasmaViscosityEnumFromString(const std::string &model);

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskStatus CalculateElectronPDVWork(MeshData<Real> *u0, MeshData<Real> *dudt);
void ComputeFreeElectronNumberDensity(MeshData<Real> *md);
parthenon::TaskCollection ElectronIonCouplingStep(Mesh *pm, parthenon::SimTime &tm,
                                                  const Real dt);
parthenon::TaskStatus ElectronIonEquilibration(MeshData<Real> *md, Real dt);

void ConvertElectronEnergyEntropyWork(MeshData<Real> *u0, const EntropyDirection dir,
                                      const IndexDomain domain);
void ConvertElectronEnergyToEntropyMesh(Mesh *pm, ParameterInput *pin,
                                        parthenon::SimTime &tm);
TaskStatus ConvertElectronEnergyToEntropy(MeshData<Real> *u0, IndexDomain domain);
TaskStatus ConvertElectronEntropyToEnergy(MeshData<Real> *u0, IndexDomain domain);
Real EstimateTimestepMesh(MeshData<Real> *md);

template <TransportSpecies Species>
parthenon::TaskCollection ConductionStep(Mesh *pmesh, parthenon::SimTime &tm,
                                         const Real dt);

extern template parthenon::TaskCollection
ConductionStep<TransportSpecies::Electron>(Mesh *pmesh, parthenon::SimTime &tm,
                                           const Real dt);

extern template parthenon::TaskCollection
ConductionStep<TransportSpecies::Ion>(Mesh *pmesh, parthenon::SimTime &tm, const Real dt);

TaskStatus CalculateElectronThermalDiffusionCoefficient(MeshData<Real> *md_base,
                                                        MeshData<Real> *md, Real dt);

TaskStatus CalculateIonThermalDiffusionCoefficient(MeshData<Real> *md_base,
                                                   MeshData<Real> *md, Real dt);

template <TransportSpecies Species>
TaskStatus UpdateStateFromConduction(MeshData<Real> *md_base, MeshData<Real> *md_u,
                                     const Real dt);

template <TransportSpecies Species>
TaskStatus SetLocal(MeshData<Real> *md, MeshData<Real> *md_out);

template <class var_t, class D_t, class rhs_t>
parthenon::TaskStatus
CalculateRHSFluxes(std::shared_ptr<parthenon::MeshData<Real>> &md,
                   std::shared_ptr<parthenon::MeshData<Real>> &md_rhs) {
  using namespace parthenon;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using TE = parthenon::TopologicalElement;

  const int ndim = md->GetMeshPointer()->ndim;

  static auto desc = parthenon::MakePackDescriptor<var_t, D_t>(md.get());
  static auto desc_rhs =
      parthenon::MakePackDescriptor<rhs_t>(md_rhs.get(), {}, {PDOpt::WithFluxes});
  auto pack = desc.GetPack(md.get());
  auto pack_rhs = desc_rhs.GetPack(md_rhs.get());

  // The face-centered flux/D live on face elements while the state var_t is
  // cell-centered, and the gradient reaches the low-side neighbor, so this uses (k, j,
  // i).
  for (int dim = 0; dim < ndim; ++dim) {
    const auto te = dim == 0 ? TE::F1 : (dim == 1 ? TE::F2 : TE::F3);
    const int ioff = dim == 0;
    const int joff = dim == 1;
    const int koff = dim == 2;
    const int dir = dim + 1;
    using lt = RiotUtils::LoopType<LoopConstraint::NoGhost>;
    auto idx_space =
        lt::GetIndexSpace(IndexDomain::interior, 0, pack.GetNBlocks(), md.get(), te);
    RiotLoop::outer(
        idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
          const auto &coords = pack.GetCoordinates(b);
          RiotLoop::inner(idx_range, [&](const int k, const int j, const int i) {
            const Real one_over_dx = 1.0 / coords.Dxc(dim + 1, k, j, i);
            pack_rhs.flux(b, dir, rhs_t(), k, j, i) =
                -pack(b, te, D_t(0), k, j, i) *
                (pack(b, var_t(), k, j, i) -
                 pack(b, var_t(), k - koff, j - joff, i - ioff)) *
                one_over_dx;
          });
        });
  }

  return TaskStatus::complete;
}

using namespace RiotEOS;
KOKKOS_FORCEINLINE_FUNCTION
void ComputeElectronIonCouplingTimescale(const EOS &eos_i, const EOS &eos_e,
                                         const Real &rho, const Real &zbar,
                                         const Real &tele, const Real &tion, Real &tau_ei,
                                         CoulombLogarithmKind logKind) {
  using namespace CouplingModelConstants;

  // ion mass
  Real mi = amu * eos_i.MeanAtomicMass();
  // ion number density
  Real ni = rho / mi;
  // electron number density
  Real ne = zbar * ni;

  tau_ei = TauEILandauSpitzer(tele, tion, ne, ni, mi, zbar, logKind);

} // ComputeElectronIonCouplingTimescale

// for a single state (zero d), evolve the electron and ion internal energies
// according to some electron-ion coupling coefficient/time scale.
// there are multiple approximations to this coupling coefficient including
// just using a constant, so we should support all these
template <typename Lambda_t, class pack_t, typename EosMap_t>
KOKKOS_FORCEINLINE_FUNCTION void ElectronIonEquilibrationOne(
    const RiotEOS::EOS_Array_t &ion_eos, const RiotEOS::EOS_Array_t &electron_eos,
    const EosMap_t &eos_map, Lambda_t &lambda, const Real &dt, const bool constant_tau_ei,
    const Real tau_ei_const, CoulombLogarithmKind logKind, const pack_t &v, const int b,
    const int k, const int j, const int i) {

  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;
  using namespace CouplingModelConstants;
  const int nmat = v.GetSize(b, ccmat::rho());

  const Real tele0 = v(b, ccbulk::electron_temperature(), k, j, i);
  const Real tion0 = v(b, ccbulk::temperature(), k, j, i);
  Real tion1 = tion0;
  Real tele1 = tele0;
  Real tion_last, tele_last, tau_ei_mean, rhocvi_sum, rhocve_sum, rhosum, ne;
  Real tau_eim;

  // fixed point iterations on tion1 and tele1 - hardcoded not to iterate
  // currently
  const int maxiter = 1;
  for (int iteration = 0; iteration < maxiter; iteration++) {

    ne = rhosum = tau_ei_mean = rhocvi_sum = rhocve_sum = 0.;
    // compute specific heats, coupling coefficients and effective averages
    for (int m = 0; m < nmat; m++) {
      auto &eosi = ion_eos(eos_map[m]);
      auto &eose = electron_eos(eos_map[m]);
      const Real rhom = v(b, cm::rho(m), k, j, i);
      const Real fvm = v(b, ccmat::volume_fraction(m), k, j, i);
      const Real zbarm = v(b, cm::ionization_zbar(m), k, j, i);
      ComputeElectronIonCouplingTimescale(ion_eos(eos_map[m]), electron_eos(eos_map[m]),
                                          rhom, zbarm, tele1, tion1, tau_eim, logKind);

      const Real cvem = eose.SpecificHeatFromDensityTemperature(rhom, tele1, lambda[m]);
      const Real cvim = eosi.SpecificHeatFromDensityTemperature(rhom, tion1, lambda[m]);

      // compute ne
      Real mi = amu * electron_eos(eos_map[m]).MeanAtomicMass();
      Real ni = rhom / mi;
      ne += fvm * ni * zbarm;

      rhosum += fvm * rhom;
      tau_eim = (constant_tau_ei) ? tau_ei_const : tau_eim;

      tau_ei_mean += fvm * rhom * cvem / tau_eim;
      rhocvi_sum += fvm * rhom * cvim;
      rhocve_sum += fvm * rhom * cvem;
    }

    tau_ei_mean = 1.0 / tau_ei_mean;

    // first order exponential integration
    Real alpha = (rhocve_sum + rhocvi_sum) / (tau_ei_mean * rhocvi_sum * rhocve_sum);
    // linear term here is the equilibrium solution for t->infinity
    Real linterm = (tele0 * rhocve_sum + tion0 * rhocvi_sum) / (rhocvi_sum + rhocve_sum);
    // exponential term is the correction to the equilibrium solution
    Real expterm = (tion0 - tele0) * std::exp(-alpha * dt) / (rhocvi_sum + rhocve_sum);
    tion_last = tion1;
    tele_last = tele1;
    tion1 = linterm + rhocve_sum * expterm;
    tele1 = linterm - rhocvi_sum * expterm;
    Real err = std::pow(tion_last - tion1, 2) + std::pow(tele_last - tele1, 2);
    // if (std::sqrt(err) < 1e-9) {
    // printf("Fixed point iteration converged after %d iterations: err = %10.3e\n",
    // iteration, err); break;
    //}
  }

  // store new temperatures in state and recover electron internal energy density.
  // bulk internal energy already includes electrons, so does not need to
  // change, since internal energy is conserved. ions will be updated in FillDerived.
  v(b, ccbulk::temperature(), k, j, i) = tion1;
  v(b, ccbulk::electron_temperature(), k, j, i) = tele1;
  v(b, ccbulk::electron_internal_energy(), k, j, i) += rhocve_sum * (tele1 - tele0);

} // ElectronIonEquilibrationOne

// Subtract off electron internal energy if needed so that
// etot = e_ion_int + e_electron_int + 1/2 rho v^2
// performed in-place
enum ConversionDirection { ToTotal, ToIon };
template <ConversionDirection DIR, typename InnerIndexRangeType, typename Pack_t>
KOKKOS_FORCEINLINE_FUNCTION void
ConvertEnergyPressureBulkIon(const InnerIndexRangeType &idx_range, const Pack_t &v,
                             const int b) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  constexpr Real sgn = (DIR == ConversionDirection::ToIon) ? -1.0 : 1.0;

  auto pv = RiotLoop::make_pack_view(idx_range, v);
  RiotLoop::inner(idx_range, [&](const auto kji) {
    pv(ccbulk::internal_energy(), kji) =
        pv(ccbulk::internal_energy(), kji) +
        sgn * pv(ccbulk::electron_internal_energy(), kji);
    pv(ccbulk::pressure(), kji) =
        pv(ccbulk::pressure(), kji) + sgn * pv(ccbulk::electron_pressure(), kji);
  });
}

template <typename Pack_t, typename IndexRangeType>
KOKKOS_FORCEINLINE_FUNCTION void
ComputeElectronTemperature(const IndexRangeType &idx_range, const Pack_t &v,
                           const RiotEOS::EOS_Array_t &electron_eos,
                           const parthenon::ParArray1D<int> &eos_from_matid,
                           const Real TOL, const int b, const int nmat) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;
  using namespace RootFinding1D;

  constexpr Real Tmin = 1e-50; // TODO(JMM): use table bounds?
  constexpr Real Tmax = 1e20;

  // TODO(JMM): This probably doesn't vectorize
  auto pv = RiotLoop::make_pack_view(idx_range, v);
  RiotLoop::inner(idx_range, [&](const auto kji) {
    Real T = pv(ccbulk::electron_temperature(), kji);
    const Real uu = pv(ccbulk::electron_internal_energy(), kji);
    // TODO(JMM): Should we set it to ion temperature?
    Real Tguess = ((T < Tmin) || (T > Tmax)) ? 0.5 * (Tmin + Tmax) : T;

    const auto [k, j, i] = idx_range.GetKJI(kji);
    auto f = [&](const Real T_trial) {
      Real usum = 0;
      for (int m = 0; m < nmat; ++m) {
        const Real cm_rho = v(b, cm::rho(m), k, j, i);
        const Real ccmat_rho = v(b, ccmat::rho(m), k, j, i); // = vfrac * cm_rho
        if (ccmat_rho > 0) { // don't include contribution from masked materials
          RiotEOS::LambdaIndexerSingleCoord lambda(v, b, m, k, j, i);
          const int mat_id = v(b, ccmat::rho(m)).sparse_id;
          const int phase_id = v(b, ccmat::rho(m)).v;
          auto &eosm = electron_eos(eos_from_matid(mat_id) + phase_id);
          usum += ccmat_rho *
                  eosm.InternalEnergyFromDensityTemperature(cm_rho, T_trial, lambda);
        }
      }
      return usum;
    };
    auto status = regula_falsi(f, uu, Tguess, Tmin, Tmax, TOL, TOL, T);
    if (status == Status::FAIL) {
      // set to ion temperature
      T = v(b, ccbulk::temperature(), k, j, i);
    }
    pv(ccbulk::electron_temperature(), kji) = T;
  });
  // TODO(JMM): Should we update electron internal energy to the above
  // sum for consistency if the solve fails?
}

// Updates per-material electron energy, electron pressure, total
// pressure, and total bulk modulus
template <typename Pack_t, typename IndexRangeType>
KOKKOS_FORCEINLINE_FUNCTION void PerMaterialEnergyPressureBmod(
    const IndexRangeType &idx_range, const Pack_t &v, const RiotEOS::EOS_Array_t &ion_eos,
    const RiotEOS::EOS_Array_t &electron_eos,
    const parthenon::ParArray1D<int> &eos_from_matid, const int b, const int nmat) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;

  auto pv = RiotLoop::make_pack_view(idx_range, v);

  for (int n = 0; n < nmat; ++n) {
    const int mat_id = v(b, ccmat::rho(n)).sparse_id;
    const int phase_id = v(b, ccmat::rho(n)).v;
    auto &eos_e = electron_eos(eos_from_matid(mat_id) + phase_id);

    auto pv_n = RiotLoop::make_sparse_pack_view(idx_range, v, n);
    auto eos_e_loop = [&](const auto &eos) {
      RiotLoop::inner(idx_range, [&](const auto kji) {
        RiotEOS::LambdaIndexerSingle pl(pv_n, kji);

        const Real T_e = pv(ccbulk::electron_temperature(), kji);
        const Real rho_i = pv_n(cm::rho(), kji);
        const Real vfrac = pv_n(ccmat::volume_fraction(), kji);

        // compute per-material electron pressure, electron bulk modulus,
        // and electron gruneisen coefficient.
        const bool mask = vfrac > 0;
        const Real P_e_m =
            (mask ? eos.PressureFromDensityTemperature(rho_i, T_e, pl) : 0.0);
        const Real bmod_e_m =
            (mask ? eos.BulkModulusFromDensityTemperature(rho_i, T_e, pl) : 0.0);
        const Real Gamma_e_m =
            (mask ? eos.GruneisenParamFromDensityTemperature(rho_i, T_e, pl) : 0.0);
        const Real sie_e_m =
            (mask ? eos.InternalEnergyFromDensityTemperature(rho_i, T_e, pl) : 0.0);

        // Accumulate per-material electron pressure into total electron pressure
        pv(ccbulk::electron_pressure(), kji) =
            (n == 0) ? vfrac * P_e_m
                     : (pv(ccbulk::electron_pressure(), kji) + vfrac * P_e_m);
        // per-material bulk modulus is given by: K^m = K^m_i + K^m_e
        pv_n(cm::bulk_modulus(), kji) += bmod_e_m;
        // For consistency, update per material total pressure to include electron
        // pressure
        pv_n(cm::pressure(), kji) += mask * P_e_m;

        // compute bulk electron bulk modulus and electron gruneisen parameter
        pv(ccbulk::electron_bulk_modulus(), kji) =
            (n > 0) * pv(ccbulk::electron_bulk_modulus(), kji) + bmod_e_m * vfrac;
        pv(ccbulk::electron_gruneisen_parameter(), kji) =
            (n > 0) * pv(ccbulk::electron_gruneisen_parameter(), kji) + Gamma_e_m * vfrac;

        // Store per-material electron sie and internal energy
        pv_n(cm::electron_sie(), kji) = sie_e_m;
        pv_n(ccmat::electron_internal_energy(), kji) = vfrac * rho_i * sie_e_m;
      });
    };
    eos_e.EvaluateDevice(eos_e_loop);
    idx_range.TeamBarrier();
  }
}

TaskStatus ComputePlasmaViscousFluxes(MeshData<Real> *md);
TaskStatus ComputePlasmaDiffusionFluxes(MeshData<Real> *md);
void FillDerivedIonization(MeshData<Real> *md);
void CalculatePlasmaViscosity(MeshData<Real> *md);

} // namespace Ionization

#endif // IONIZATION_IONIZATION_HPP_

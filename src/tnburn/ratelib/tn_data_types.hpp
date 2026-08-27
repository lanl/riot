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
#ifndef TNBURN_RATELIB_TN_DATA_TYPES_
#define TNBURN_RATELIB_TN_DATA_TYPES_

// C++ includes
#include <algorithm>
#include <string>
#include <vector>

// Parthenon includes
#include <parthenon/package.hpp>

// Ports of call includes
#include <ports-of-call/portability.hpp>

// Spiner includes
#include <spiner/databox.hpp>

using namespace parthenon::package::prelude;

template <typename T>
constexpr bool inline isin(std::vector<T> v, const T &value) {
  return std::find(v.begin(), v.end(), value) != v.end();
}

// Note that this isn't fault tolerant.
template <typename T>
inline int findindex(std::vector<T> v, T value) {
  return std::distance(v.begin(), std::find(v.begin(), v.end(), value));
}

using DataBox = Spiner::DataBox<Real>;

using unit_system = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;
constexpr Real amu = unit_system::amu;
constexpr Real ergs_per_MeV{1.0e6 * unit_system::eV};
constexpr Real MeV_to_kelvin = ergs_per_MeV / unit_system::boltzmann;

struct ReactionData {
  static constexpr int max_size = 3;
  int num_reactants;
  int num_products;
  int num_temp;
  std::array<int, max_size> reactants;
  std::array<int, max_size> products;
  std::array<int, max_size> product_multiplicities;
  Real Qval;
  DataBox Temperatures; // Could possibly forego this one.
  DataBox InputEnergy;
  DataBox SigmaVBar;
  std::array<DataBox, max_size> EnergyOut; // Does this seem like a viable option here?
  ReactionData getOnDevice() {
    ReactionData retval;
    retval.num_reactants = this->num_reactants;
    retval.num_products = this->num_products;
    retval.num_temp = this->num_temp;
    retval.reactants = this->reactants;
    retval.products = this->products;
    retval.product_multiplicities = this->product_multiplicities;
    retval.Qval = Qval;
    retval.Temperatures = this->Temperatures.getOnDevice();
    retval.InputEnergy = this->InputEnergy.getOnDevice();
    retval.SigmaVBar = this->SigmaVBar.getOnDevice();
    for (int i = 0; i < this->num_products; i++)
      retval.EnergyOut[i] = this->EnergyOut[i].getOnDevice();
    return retval;
  }
  // Eventually, this could be generalized to different unit sets.
  // When this is called, the
  void setup_arrays_for_code_units( // const temperature_units tu, const length_units lu,
                                    // const mass_units mu, const time_units time_u,
      const std::vector<int> &all_isotopes, const std::vector<Real> &masses) {
    Real tmult = MeV_to_kelvin;
    Real emult = ergs_per_MeV;
    Real massmult = amu; // Masses start in AMU
    // There are always two reactants, so if there is only one recorded, then we need to
    // take that into account
    if (num_reactants == 1) reactants[1] = reactants[0];
    num_reactants = 2;
    Real sigmavmult =
        1.0; // Starts out in cm^3/s; we need to divide by product(mass(reactant_isotope))
    for (int i = 0; i < 2; i++) {
      const int inx = findindex(all_isotopes, reactants[i]);
      const Real mass = masses[inx];
      sigmavmult /= (massmult * mass);
    }
    for (int i = 0; i < Temperatures.size(); i++) {
      Temperatures(i) *= tmult;
      InputEnergy(i) *= emult;
      SigmaVBar(i) *= sigmavmult;
    }
    for (int j = 0; j < num_products; j++) {
      for (int i = 0; i < EnergyOut[j].dim(1); i++) {
        EnergyOut[j](i) *= emult;
      }
    }
    Qval *= emult;
    Real ltemp0 = std::log(Temperatures(0));
    Real ltemp1 = std::log(Temperatures(Temperatures.size() - 1));
    InputEnergy.setRange(0, ltemp0, ltemp1, Temperatures.size());
    SigmaVBar.setRange(0, ltemp0, ltemp1, Temperatures.size());
    for (int i = 0; i < num_products; i++)
      EnergyOut[i].setRange(0, ltemp0, ltemp1, Temperatures.size());
  }
};

#endif

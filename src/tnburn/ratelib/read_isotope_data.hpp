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
#ifndef TNBURN_RATELIB_READ_ISOTOPE_DATA_
#define TNBURN_RATELIB_READ_ISOTOPE_DATA_

// C++ includes
#include <string>
#include <vector>

// TNBurn includes
#include "tn_data_types.hpp"

// HDF5 includes
#ifdef SPINER_USE_HDF
#include <hdf5.h>
#include <hdf5_hl.h>
#endif

namespace ratelib {
#ifdef SPINER_USE_HDF
inline void read_tn_reactions(const std::string filename,
                              const std::vector<std::string> reactions,
                              std::vector<int> &all_isotopes, std::vector<Real> &masses,
                              std::vector<int> &charges,
                              std::vector<ReactionData> &data) {
  const int num_reactions = reactions.size();
  herr_t status;
  hid_t file, reaction, reactions_group, mass_group;
  file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  reactions_group = H5Gopen(file, "reactions", H5P_DEFAULT);
  data.resize(reactions.size());
  for (int i = 0; i < reactions.size(); i++) {
    const auto &r = reactions[i];
    ReactionData &d = data[i];
    reaction = H5Gopen(reactions_group, r.c_str(), H5P_DEFAULT);
    status += H5LTget_attribute_int(reactions_group, r.c_str(), "num_reactants",
                                    &d.num_reactants);
    status += H5LTget_attribute_int(reactions_group, r.c_str(), "num_products",
                                    &d.num_products);
    status += H5LTget_attribute_int(reactions_group, r.c_str(), "num_temp", &d.num_temp);
    status += H5LTget_attribute_double(reactions_group, r.c_str(), "Qval", &d.Qval);
    status += H5LTget_attribute_int(reactions_group, r.c_str(), "reactants",
                                    d.reactants.data());
    status +=
        H5LTget_attribute_int(reactions_group, r.c_str(), "products", d.products.data());
    status += H5LTget_attribute_int(reactions_group, r.c_str(), "product_multiplicities",
                                    d.product_multiplicities.data());
    status += d.Temperatures.loadHDF(reaction, "Temperatures");
    status += d.InputEnergy.loadHDF(reaction, "InputEnergy");
    status += d.SigmaVBar.loadHDF(reaction, "SigmaVBar");
    for (int j = 0; j < d.num_products; j++) {
      std::string name = "EnergyOut" + std::to_string(j);
      status += d.EnergyOut[j].loadHDF(reaction, name.c_str());
    }
    H5Gclose(reaction);
  }
  H5Gclose(reactions_group);
  mass_group = H5Gopen(file, "Masses", H5P_DEFAULT);
  int num_isotopes;
  status += H5LTget_attribute_int(file, "Masses", "num_isotopes", &num_isotopes);
  all_isotopes.resize(num_isotopes);
  masses.resize(num_isotopes);
  charges.resize(num_isotopes);
  status += H5LTget_attribute_int(file, "Masses", "zaids", all_isotopes.data());
  status += H5LTget_attribute_double(file, "Masses", "masses", masses.data());
  status += H5LTget_attribute_int(file, "Masses", "charges", charges.data());
  H5Gclose(mass_group);
  H5Fclose(file);
}
#endif
} // namespace ratelib
#endif

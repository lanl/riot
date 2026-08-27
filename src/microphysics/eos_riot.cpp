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

// system includes
#include <map>
#include <memory>
#include <sstream>

// riot includes
#include "eos_riot.hpp"

#define GET(param) pin->GetReal(block_name, #param)

namespace RiotEOS {

// NOTE(JMM): Anything output by HDF5 as an attribute needs to
// be identical on every rank. This includes the ParameterInput
// and parmas like eos_from_matid. ParameterInput's GetOrAdd has
// side effects so it's important it follows the same code path
// on all ranks. The easiest way to make this happen is to
// modify `InitializeEOS` so it can optionally just return an
// empty EOS object. We use MPI to clean that up later so this
// is safe to do.
#define RETURN(actually_load, ...)                                                       \
  if (actually_load) {                                                                   \
    return __VA_ARGS__;                                                                  \
  } else {                                                                               \
    return EOS();                                                                        \
  }

//----------------------------------------------------------------------------------------
//! \fn  EOS RiotEOS::InitializeEOS
//! \brief
EOS InitializeEOS(ParameterInput *pin, const std::string &block_name, bool is_electron,
                  bool load) {
  using namespace singularity;
  const bool do_ionization = pin->GetBoolean("physics", "ionization");

  auto valid_eos_types = RiotEOS::GetEOSNames(RiotEOS::EOS());
  std::string eos_type = pin->GetString(block_name, std::string("eos_type"),
                                        valid_eos_types, "Equation of state to use");
  auto shift = pin->GetOrAddBoolean("hydro", "eos_shift", false, "Shift the EOS");
  auto reference_sie =
      pin->GetOrAddReal("hydro", "eos_reference_sie", 0.0, "Reference energy for shift");

  ZSplitting zsplit = ZSplitting::None;
  if (do_ionization) {
    const bool is_zsplit = pin->GetOrAddBoolean(block_name, "zsplit", false);
    if (is_zsplit) {
      zsplit = is_electron ? ZSplitting::Electrons : ZSplitting::Ions;
    }
  }

  const Real Abar = pin->GetOrAddReal(block_name, "mean_atomic_mass", 2.0);
  const Real Zbar = pin->GetOrAddReal(block_name, "mean_atomic_number", 1.0);
  singularity::MeanAtomicProperties AZbar(Abar, Zbar);

  // Parse inputs
  if (eos_type.compare(IdealGas::EosType()) == 0) {
    RETURN(load,
           ZSplit(Shift(IdealGas(GET(Gamma) - 1.0, GET(Cv), AZbar), shift, reference_sie),
                  zsplit));
  } else if (eos_type.compare(Gruneisen::EosType()) == 0) {
    RETURN(load,
           ZSplit(Shift(Gruneisen(GET(C0), GET(s1), GET(s2), GET(s3), GET(G0), GET(b),
                                  GET(rho0), GET(T0), GET(P0), GET(Cv), AZbar),
                        shift, reference_sie),
                  zsplit));
  } else if (eos_type.compare(JWL::EosType()) == 0) {
    PARTHENON_REQUIRE_THROWS(zsplit == ZSplitting::None, "Z split not supported for JWL");
    RETURN(load,
           Shift(JWL(GET(A), GET(B), GET(R1), GET(R2), GET(w), GET(rho0), GET(Cv), AZbar),
                 shift, reference_sie));
  } else if (eos_type.compare(DavisProducts::EosType()) == 0) {
    PARTHENON_REQUIRE_THROWS(zsplit == ZSplitting::None,
                             "Z split not supported for DavisProducts");
    RETURN(load, Shift(DavisProducts(GET(a), GET(b), GET(k), GET(n), GET(vc), GET(pc),
                                     GET(Cv), AZbar),
                       shift, reference_sie));
  } else if (eos_type.compare(DavisReactants::EosType()) == 0) {
    PARTHENON_REQUIRE_THROWS(zsplit == ZSplitting::None,
                             "Z split not supported for DavisReactants");
    RETURN(load,
           Shift(DavisReactants(GET(rho0), GET(e0), GET(P0), GET(T0), GET(A), GET(B),
                                GET(C), GET(G0), GET(Z), GET(alpha), GET(Cv0), AZbar),
                 shift, reference_sie));
#ifdef SPINER_USE_HDF
  } else if (eos_type.compare(SpinerEOSDependsRhoT::EosType()) == 0) {
    auto filename =
        pin->GetString(block_name, "filename", "File name to read EOS data from");
    auto reprod = pin->GetOrAddBoolean(block_name, "reproducibility_mode", false,
                                       "If true, makes the initial guess for temperature "
                                       "inversions identical every time. false (default) "
                                       "allows caching of previous guesses.");

    const auto subtable = pin->GetOrAddBoolean(block_name, "use_subtable", false);
    singularity::TableSplit tsplit =
        (subtable && do_ionization) ? (is_electron ? singularity::TableSplit::ElectronOnly
                                                   : singularity::TableSplit::IonCold)
                                    : singularity::TableSplit::Total;
    PARTHENON_REQUIRE_THROWS(
        !((zsplit != ZSplitting::None) && (tsplit != singularity::TableSplit::Total)),
        "You should not use a subtable with z splitting!");

    if (pin->DoesParameterExist(block_name, "sesame_id")) {
      auto matid = pin->GetInteger(block_name, "sesame_id",
                                   "Sesame material ID. Takes precedence over a name.");
      RETURN(load, Shift(SpinerEOSDependsRhoT(filename, matid, tsplit, reprod), shift,
                         reference_sie));
    } else if (pin->DoesParameterExist(block_name, "sesame_name")) {
      auto name =
          pin->GetString(block_name, "sesame_name",
                         "Material name in SP5 file. Matid takes precedence if set.");
      RETURN(load, Shift(SpinerEOSDependsRhoT(filename, name, tsplit, reprod), shift,
                         reference_sie));
    } else {
      std::stringstream msg;
      msg << "Neither sesame_id nor sesame_name exists for material " << block_name
          << std::endl;
      PARTHENON_THROW(msg);
    }
  } else if (eos_type.compare(SpinerEOSDependsRhoSie::EosType()) == 0) {
    auto filename =
        pin->GetString(block_name, "filename", "File name to read EOS data from");
    auto reprod = pin->GetOrAddBoolean(block_name, "reproducibility_mode", false,
                                       "If true, makes the initial guess for temperature "
                                       "inversions identical every time. false (default) "
                                       "allows caching of previous guesses.");

    const auto subtable = pin->GetOrAddBoolean(block_name, "use_subtable", false);
    singularity::TableSplit tsplit =
        (subtable && do_ionization) ? (is_electron ? singularity::TableSplit::ElectronOnly
                                                   : singularity::TableSplit::IonCold)
                                    : singularity::TableSplit::Total;
    PARTHENON_REQUIRE_THROWS(
        !((zsplit != ZSplitting::None) && (tsplit != singularity::TableSplit::Total)),
        "You should not use a subtable with z splitting!");

    if (pin->DoesParameterExist(block_name, "sesame_id")) {
      auto matid = pin->GetInteger(block_name, "sesame_id",
                                   "Sesame material ID. Takes precedence over a name.");
      RETURN(load, Shift(SpinerEOSDependsRhoSie(filename, matid, tsplit, reprod), shift,
                         reference_sie));
    } else if (pin->DoesParameterExist(block_name, "sesame_name")) {
      auto name =
          pin->GetString(block_name, "sesame_name",
                         "Material name in SP5 file. Matid takes precedence if set.");
      RETURN(load, Shift(SpinerEOSDependsRhoSie(filename, name, tsplit, reprod), shift,
                         reference_sie));
    } else {
      std::stringstream msg;
      msg << "Neither sesame_id nor sesame_name exists for material " << block_name
          << std::endl;
      PARTHENON_THROW(msg);
    }
#endif
  } else if (eos_type.compare(singularity::IdealElectrons::EosType()) == 0) {
    RETURN(load, Shift(IdealElectrons(AZbar), shift, reference_sie));
  } else {
    std::stringstream error_mesg;
    error_mesg << __func__ << ": " << eos_type << " is an invalid EOS selection"
               << std::endl;
    PARTHENON_THROW(error_mesg);
  }
  // should never get here, but need a return
  return EOS();
}

} // namespace RiotEOS

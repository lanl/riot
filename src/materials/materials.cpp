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

#include "materials.hpp"

// C++ includes
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

// Parthenon includes
#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

// Singularity includes
#include <singularity-eos/base/serialization_utils.hpp>
#include <singularity-eos/base/spiner_table_utils.hpp>
#include <singularity-eos/closure/mixed_cell_models.hpp>
#include <singularity-eos/eos/eos.hpp>
#include <singularity-utils/indexable_types.hpp>

// HDF5 includes (for isotope data reading)
#ifdef SPINER_USE_HDF
#include <hdf5.h>
#include <hdf5_hl.h>
#endif

// other Riot includes
#include "microphysics/eos_riot.hpp"
#include "microphysics/opacity_models.hpp"
#include "microphysics/strength_models.hpp"
#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"

namespace Materials {

template <typename T>
using ParArrayHost1D = typename parthenon::ParArray1D<T>::HostMirror;

//----------------------------------------------------------------------------------------
//! \fn  std::shared_ptr<StateDescriptor> Materials::Initialize
//! \brief Adds material related quantities (e.g., EOS, opacity, etc.)
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;
  using parthenon::Metadata;
  using parthenon::MetadataFlag;
  using parthenon::ParArray1D;
  using parthenon::SparsePool;

  auto materials = std::make_shared<StateDescriptor>("materials");
  Params &params = materials->AllParams();

  // Material IDs, AMR controls
  std::vector<int> matids;
  std::vector<int> max_lev_mat_v;
  std::vector<int> max_lev_bnd_v;
  std::vector<std::string> all_mats;

  // Base EOS
  // TODO(JMM): In case of ionization, these are for ions.
  std::vector<RiotEOS::EOS> eos_dvec;
  std::vector<RiotEOS::EOS> eos_host;
  std::vector<int> eos_from_matid;

  // Electron EOS
  // TODO(JMM): Add bulk electron number options
  std::vector<RiotEOS::EOS> electron_eos_dvec, electron_eos_host;

  // Opacities
  int global_ngroups;
  std::vector<Real> global_group_bounds;
  bool group_bounds_set = false;
  std::vector<RiotOpacity::MeanOpacA> opac_a_dvec;
  std::vector<RiotOpacity::MeanOpacA> opac_a_host;
  std::vector<RiotOpacity::MeanOpacS> opac_s_dvec;
  std::vector<RiotOpacity::MeanOpacS> opac_s_host;
  std::vector<int> opac_from_matid;

  // Strength
  std::vector<bool> strong;
  std::vector<int> strength_mats;
  std::vector<int> strength_map;
  std::vector<Strength::StressModel> strength_models;
  std::vector<Strength::stress_model> strength_model_ids;

  // Isotopes
  std::vector<std::vector<std::string>> iso_names;
  std::vector<std::vector<int>> iso_zaids;
  std::vector<int> num_iso_per_mat;
  std::vector<Real> iso_fractions;

  // Phases
  std::vector<int> nphase;

  // -------------------------------------------------------------------------------------
  // Initialize materials vector
  std::string base_name("material");
  const int max_level = pin->GetOrAddInteger("parthenon/mesh", "numlevel", 1) - 1;
  const bool do_strength = pin->GetBoolean("physics", "strength");
  int nummat = 0, numstr = 0;
  while (true) {
    // TODO(JMM): This loop requires matids be contiguous starting from 0. We might want
    // to add some validation or error message, in case someone tries to add a set of
    // matids that are NOT contiguous. Would require adding features to ParameterInput.
    std::string name = base_name + std::to_string(nummat);
    if (!pin->DoesBlockExist(name)) { // block does not exist
      break;
    }

    // Read and store AMR level
    auto max_lev = pin->GetOrAddInteger(name, "max_mat_level", 0,
                                        "Maximum level to refine around a material");
    auto bnd_lev =
        pin->GetOrAddInteger(name, "max_bnd_level", 0,
                             "Maximum level to refine around interfaces of a material");
    if (max_lev == -1) max_lev = max_level;
    if (bnd_lev == -1) bnd_lev = max_level;
    max_lev_mat_v.push_back(max_lev);
    max_lev_bnd_v.push_back(bnd_lev);

    // Handle strength-supporting materials
    auto str = pin->GetOrAddBoolean(name, "strong", false,
                                    "Does the material have material strength");
    str = str && do_strength;
    strong.push_back(str);
    if (str) {
      strength_mats.push_back(nummat);
      strength_map.push_back(numstr);
      numstr++;
    } else {
      strength_map.push_back(-1);
    }

    // Pushback on materials vectors
    matids.push_back(nummat);
    all_mats.push_back(parthenon::MakeVarLabel(ccmat::rho::name(), nummat));
    nummat++;
  }
  PARTHENON_REQUIRE(nummat > 0, "Found no Material blocks in input")

  // Custom Metadata flags
  auto BurnFlag = Metadata::GetOrAddFlag(riot::metadata::TNBurn);

  // Material Density
  // TODO(JCD): how should these thresholds really be set?
  Metadata m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
                         Metadata::Conserved, Metadata::Sparse, Metadata::FillGhost,
                         Metadata::WithFluxes});
  std::string control_field = ccmat::rho::name();
  m.SetSparseThresholds(1.e-30, 1.e-32, 0.0);
  auto ccmat_rho = SparsePool::Make<ccmat::rho>(m, control_field);

  // Material Volume Fraction
  m = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Sparse, Metadata::Derived,
                Metadata::OneCopy, Metadata::FillGhost, Metadata::ForceRemeshComm,
                Metadata::Restart});
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  auto ccmat_volume_fraction = SparsePool::Make<ccmat::volume_fraction>(m, control_field);

  // Material Volumetric Internal Energy
  m = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Sparse, Metadata::Derived,
                Metadata::OneCopy});
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  auto ccmat_internal_energy = SparsePool::Make<ccmat::internal_energy>(m, control_field);

  // Material-volume-averaged Density and Specific Internal Energy
  auto cm_rho = SparsePool::Make<cm::rho>(m, control_field);
  auto cm_sie = SparsePool::Make<cm::sie>(m, control_field);

  // Material Temperature, Pressure, Bulk Modulus, and Specific Heat
  m = Metadata({Metadata::Cell, Metadata::Sparse, Metadata::Derived, Metadata::OneCopy});
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  auto cm_temperature = SparsePool::Make<cm::temperature>(m, control_field);
  auto cm_pressure = SparsePool::Make<cm::pressure>(m, control_field);
  auto cm_bulk_modulus = SparsePool::Make<cm::bulk_modulus>(m, control_field);
  auto cm_cv = SparsePool::Make<cm::specific_heat>(m, control_field);

  // JMM: Caching for root finds used by some equations of state
  auto cm_lT_cache = SparsePool::Make<cm::lT_cache>(m, control_field);
  auto cm_lr_cache = SparsePool::Make<cm::lr_cache>(m, control_field);

  // Deviatoric stress
  m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
                Metadata::Conserved, Metadata::Sparse, Metadata::FillGhost,
                Metadata::WithFluxes});
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  auto mat_stress = SparsePool::Make<ccmat::deviatoric_stress>(m, control_field);

  // Equivalent plastic strain
  m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
                Metadata::Conserved, Metadata::Sparse, Metadata::FillGhost,
                Metadata::WithFluxes, Metadata::Advected});
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  m.Associate(cm::equivalent_plastic_strain::name());
  auto mat_eps = SparsePool::Make<ccmat::equivalent_plastic_strain>(m, control_field);

  // Material-volume-averaged Deviatoric Stress, Equivalent Plastic Strain, and Shear Mod
  m = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Sparse, Metadata::Derived,
                Metadata::OneCopy});
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  auto stress = SparsePool::Make<cm::deviatoric_stress>(m, control_field);
  auto eps = SparsePool::Make<cm::equivalent_plastic_strain>(m, control_field);
  auto gmod = SparsePool::Make<cm::shear_modulus>(m, control_field);

  // Material J2
  m = Metadata({Metadata::Cell, Metadata::Sparse, Metadata::Derived, Metadata::OneCopy});
  auto strength_j2 = SparsePool::Make<cm::strength_j2>(m, control_field);

  // Phase fractions and phase density sums (when invoking phases)
  m = Metadata({Metadata::Cell, Metadata::Sparse, Metadata::Derived, Metadata::OneCopy});
  auto phase_fraction = SparsePool::Make<cm::phase_fraction>(m, control_field);
  auto phase_rho_sum = SparsePool::Make<ccmat::phase_rho_sum>(m, control_field);

  // Isotopes
  // TODO(JMM): Should these be defined in the tnburn package?
  m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
                Metadata::Conserved, Metadata::Sparse, Metadata::FillGhost,
                Metadata::WithFluxes, Metadata::Advected, BurnFlag});
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  m.Associate(cm::iso::name());
  auto mat_iso = SparsePool::Make<ccmat::iso>(m, control_field);

  // Material-volume-averaged Isotopes
  m = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Sparse, Metadata::Derived,
                Metadata::OneCopy, BurnFlag});
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  auto iso = SparsePool::Make<cm::iso>(m, control_field);

  // Ionization State
  m = Metadata({Metadata::Cell, Metadata::Sparse, Metadata::WithFluxes,
                Metadata::Advected, Metadata::Independent});
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  m.Associate(cm::ionization_zbar::name());
  auto ccmat_ionization_zbar = SparsePool::Make<ccmat::ionization_zbar>(m, control_field);

  // Material-volume-averaged Ionization State
  m = Metadata({Metadata::Cell, Metadata::Sparse, Metadata::OneCopy, Metadata::Derived,
                Metadata::FillGhost});
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  auto cm_ionization_zbar = SparsePool::Make<cm::ionization_zbar>(m, control_field);

  // Material-volume-averaged Volumetric Internal Energy
  m = Metadata({Metadata::Cell, Metadata::Sparse, Metadata::OneCopy, Metadata::Derived});
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  auto cm_ue = SparsePool::Make<ccmat::electron_internal_energy>(m, control_field);
  // auto cm_sie_e = SparsePool::Make<cm::electron_sie>(m, control_field);

  // Electron specific internal energy
  m = Metadata({Metadata::Cell, Metadata::Sparse, Metadata::Derived, Metadata::OneCopy});
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  auto cm_sie_e = SparsePool::Make<cm::electron_sie>(m, control_field);

  // -------------------------------------------------------------------------------------
  // Shared Memory
  bool use_mpi_shared_memory = pin->GetOrAddBoolean(
      "riot", "use_mpi_shared_memory", true,
      "Store material (mainly EOS) data in MPI shared memory. Can be a big memory "
      "savings on many-core CPU systems. Does not work on GPUs.");
#ifndef MPI_PARALLEL
  if (use_mpi_shared_memory) {
    PARTHENON_DEBUG_WARN("MPI shared memory requested but MPI is not enabled. Not using "
                         "MPI shared memory.");
    use_mpi_shared_memory = false;
  }
#endif
  const bool global_rank0 = (parthenon::Globals::my_rank == 0); // 0 with MPI disabled

  // -------------------------------------------------------------------------------------
  // Initialize material opacity group structure
  // NOTE(@pdmullen): We currently mandate that all materials provide opacity models
  // whose underlying group structures are identical.  There are a number of ways of
  // specifying the group structure.  First, it can be specified via the <materials>
  // block.  If that is omitted, it can be obtained via reading from the singularity-opac
  // table (if a tabular opacity).  In the following, we attempt to discover the group
  // structure and ensure that all material models agree said structure.
  auto get_opac_blocks = [&](const std::string &block_name, int nph) {
    std::vector<std::string> blocks;
    if (nph == 1) {
      blocks.push_back(pin->DoesParameterExist(block_name, "opac")
                           ? pin->GetString(block_name, "opac")
                           : block_name);
    } else {
      for (int n = 0; n < nph; ++n) {
        const std::string opac_param = std::string("opac") + std::to_string(n);
        blocks.push_back(pin->DoesParameterExist(block_name, opac_param)
                             ? pin->GetString(block_name, opac_param)
                             : block_name);
      }
    }
    return blocks;
  };
  auto bounds_match = [](const std::vector<Real> &a, const std::vector<Real> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      if (std::isinf(a[i]) || std::isinf(b[i])) {
        if (!(std::isinf(a[i]) && std::isinf(b[i]) && (a[i] > 0) == (b[i] > 0)))
          return false;
      } else if (std::abs(a[i] - b[i]) >= 1.0e-10) {
        return false;
      }
    }
    return true;
  };
  const bool group_bounds_from_input =
      pin->DoesParameterExist("materials", "group_bounds");
  if (group_bounds_from_input) {
    global_group_bounds = pin->GetVector<Real>("materials", "group_bounds");
    PARTHENON_REQUIRE(global_group_bounds.size() >= 2,
                      "materials/group_bounds needs at least 2 edges (ngroups+1).");
    global_ngroups = static_cast<int>(global_group_bounds.size()) - 1;
    const int ngroups_in = pin->GetOrAddInteger("materials", "ngroups", global_ngroups,
                                                "Number of radiation frequency groups.");
    PARTHENON_REQUIRE(ngroups_in == global_ngroups,
                      "materials/ngroups is inconsistent with materials/group_bounds "
                      "(expected ngroups == group_bounds.size() - 1).");
    group_bounds_set = true;
  } else {
    // Scan every block for a provided singularity-opac table
    auto seed_or_validate = [&](int t_ngroups, const std::vector<Real> &t_bounds,
                                const std::string &table) {
      if (!group_bounds_set) {
        global_ngroups = t_ngroups;
        global_group_bounds = t_bounds;
        group_bounds_set = true;
      } else {
        PARTHENON_REQUIRE(t_ngroups == global_ngroups &&
                              bounds_match(t_bounds, global_group_bounds),
                          "Inconsistent group structure in " + table);
      }
    };
    for (const auto matid : matids) {
      const std::string block_name = std::string("material") + std::to_string(matid);
      const int nph = pin->GetOrAddInteger(block_name, "nphase", 1);
      for (const auto &opac_block : get_opac_blocks(block_name, nph)) {
        if (pin->GetOrAddString(opac_block, "opac_a", "none") == "table") {
          const std::string table = pin->GetString(opac_block, "opac_a_filename");
          RiotOpacity::MeanOpacA peek(table);
          seed_or_validate(peek.ngroups(), peek.GetGroupBounds(), table);
        }
        if (pin->GetOrAddString(opac_block, "opac_s", "none") == "table") {
          const std::string table = pin->GetString(opac_block, "opac_s_filename");
          RiotOpacity::MeanOpacS peek(table);
          seed_or_validate(peek.ngroups(), peek.GetGroupBounds(), table);
        }
      }
    }
    // No explicit group structure: default to gray. This covers all non-radiating runs.
    if (!group_bounds_set) {
      global_ngroups = 1;
      global_group_bounds = {0.0, std::numeric_limits<Real>::infinity()};
      group_bounds_set = true;
    }
  }
  PARTHENON_REQUIRE(global_ngroups >= 1, "Resolved a non-positive number of groups.");

  // -------------------------------------------------------------------------------------
  // Construct materials
  bool use_general_pte = pin->GetOrAddBoolean(
      "materials", "use_general_pte", false,
      "Force the full PTE solver, even if an analytic (ideal gas) solver will do.");
  const bool do_ionization = pin->GetBoolean("physics", "ionization");
  const bool do_tn = pin->GetOrAddBoolean("physics", "tn", false);

  for (const auto matid : matids) {
    std::string block_name = std::string("material") + std::to_string(matid);
    // NOTE(JMM): This phase index introduces an additional index in variables whether
    // they are single- or multi-phase, and this shows up in the output. If you're
    // confused by a funny variable shape, that might be why.
    auto nph = pin->GetOrAddInteger(block_name, "nphase", 1);
    std::vector<int> phase_size(1, nph);
    nphase.push_back(nph);

    // Add in this matid to sparse pools
    AddMatId(matid, phase_size, ccmat_rho, ccmat_volume_fraction, ccmat_internal_energy,
             cm_rho, cm_sie, cm_temperature, cm_pressure, cm_bulk_modulus, cm_cv);

    // Phases
    // TODO(JMM): Should we remove phase_rho_sum? It could be scratch var, but it's kind
    // of useful for visualization when phases are actually active.
    if (do_tn) {
      AddMatId(matid, phase_size, phase_fraction);
      AddMatId(matid, {1, 1}, phase_rho_sum);
    }

    // Electrons
    if (do_ionization) {
      AddMatId(matid, phase_size, ccmat_ionization_zbar);
      AddMatId(matid, phase_size, cm_ionization_zbar);
      AddMatId(matid, phase_size, cm_ue);
      AddMatId(matid, phase_size, cm_sie_e);
    }

    // Strength
    if (do_strength && strong[matid]) {
      PARTHENON_REQUIRE(nph == 1, "Strength-supporting multiphase mats not supported.");
      auto model_block =
          pin->GetString(block_name, "strength_model", "Strength model to use");
      strength_models.push_back(Strength::InitStressModel(pin, model_block));
      strength_model_ids.push_back(strength_models.back().type());
      std::vector<int> shape(1, 5);
      AddMatId(matid, shape, mat_stress, stress);
      shape[0] = 1;
      AddMatId(matid, shape, mat_eps, eps, gmod, strength_j2);
    }

    // Isotopes
    int niso = 0;
    std::string iname = std::string("isotope") + std::to_string(niso);
    std::vector<std::string> mat_iso_names;
    std::vector<int> mat_iso_zaids;
    while (pin->DoesParameterExist(block_name, iname)) {
      mat_iso_names.push_back(pin->GetString(block_name, iname));
      mat_iso_zaids.push_back(std::stoi(mat_iso_names[niso]));
      iso_fractions.push_back(pin->DoesParameterExist(block_name, iname + "_mfrac")
                                  ? pin->GetReal(block_name, iname + "_mfrac")
                                  : 0.0);
      niso++;
      iname = std::string("isotope") + std::to_string(niso);
    }
    iso_names.push_back(mat_iso_names);
    iso_zaids.push_back(mat_iso_zaids);
    if (niso > 0) {
      std::vector<int> shape(1, niso);
      AddMatId(matid, shape, mat_iso, iso);
    }

    num_iso_per_mat.push_back(niso);

    // For adding EOS...
    auto valid_eos_types = RiotEOS::GetEOSNames(RiotEOS::EOS());
    auto add_eos = [&](const std::string &eos_block) {
      auto eos_type = pin->GetString(eos_block, "eos_type");
      bool is_ideal = (eos_type.compare(singularity::IdealGas::EosType()) == 0) ||
                      (eos_type.compare(singularity::IdealElectrons::EosType()) == 0);
      if (!is_ideal) use_general_pte = true;

      // NOTE(JMM): 1 rank stash pointers into params.  Anything output by HDF5 as an
      // attribute needs to be identical on every rank. This includes the ParameterInput
      // and parmas like eos_from_matid. ParameterInput's GetOrAdd has side effects so
      // it's important it follows the same code path on all ranks. The easiest way to
      // make this happen is to modify `InitializeEOS` so it can optionally just return an
      // empty EOS object. We use MPI to clean that up later so this
      // is safe to do.
      const bool actually_load = (global_rank0) || !use_mpi_shared_memory;
      eos_host.push_back(RiotEOS::InitializeEOS(pin, eos_block, false, actually_load));
    };

    // For adding electron EOS...
    auto add_electron_eos = [&](const std::string &eos_block) {
      const bool actually_load = (global_rank0) || !use_mpi_shared_memory;
      electron_eos_host.push_back(
          RiotEOS::InitializeEOS(pin, eos_block, true, actually_load));
    };

    // For adding opacities...
    auto add_opac = [&](const std::string &opac_block) {
      using namespace singularity::photons;

      // First get underlying absorption opacity model
      RiotOpacity::OpacA opac_a;
      std::string opac_a_type = pin->GetOrAddString(opac_block, "opac_a", "none");
      std::string opac_a_table = "autogen-opac-a-";
      if (opac_a_type == "none") {
        opac_a = singularity::photons::Gray(0.0);
      } else if (opac_a_type == "constant") {
        const Real kappa_a = pin->GetOrAddReal(opac_block, "kappa_a", 0.0);
        opac_a = Gray(kappa_a);
      } else if (opac_a_type == "powerlaw") {
        const Real coef_kappa_a = pin->GetOrAddReal(opac_block, "kappa0_a", 0.0);
        const Real rho_exp = pin->GetOrAddReal(opac_block, "kappa_Rhopower_a", 0.0);
        const Real temp_exp = pin->GetOrAddReal(opac_block, "kappa_Tpower_a", 0.0);
        const Real nu_exp = pin->GetOrAddReal(opac_block, "kappa_Nupower_a", 0.0);
        const Real nu_ref = pin->GetOrAddReal(opac_block, "kappa_Nuref_a", 1.0);
        opac_a = PowerLaw(coef_kappa_a, rho_exp, temp_exp, nu_exp, nu_ref);
      } else if (opac_a_type == "table") {
        opac_a_table = pin->GetString(opac_block, "opac_a_filename");
      }

      // Now get underlying scattering opacity model
      RiotOpacity::OpacS opac_s;
      std::string opac_s_type = pin->GetOrAddString(opac_block, "opac_s", "none");
      std::string opac_s_table = "autogen-opac-s-";
      if (opac_s_type == "none") {
        opac_s = GrayS(0.0, 1.0);
      } else if (opac_s_type == "constant") {
        const Real kappa_s = pin->GetOrAddReal(opac_block, "kappa_s", 0.0);
        opac_s = GrayS(kappa_s, 1.0);
      } else if (opac_s_type == "table") {
        opac_s_table = pin->GetString(opac_block, "opac_s_filename");
      }

      // Construct Absorption MeanOpacity
      RiotOpacity::MeanOpacA mg_opac_a;
      if (opac_a_type == "table") {
        mg_opac_a = MeanOpacityBase(opac_a_table);
      } else {
        const Real lRhoMin = pin->GetOrAddReal(opac_block, "lRhoMin_a", -1.0);
        const Real lRhoMax = pin->GetOrAddReal(opac_block, "lRhoMax_a", 1.0);
        const Real lTMin = pin->GetOrAddReal(opac_block, "lTMin_a", -1.0);
        const Real lTMax = pin->GetOrAddReal(opac_block, "lTMax_a", 1.0);
        const int NRho = pin->GetOrAddInteger(opac_block, "NRho_a", 2);
        const int NT = pin->GetOrAddInteger(opac_block, "NT_a", 2);
        const int NNuPerGroup = pin->GetOrAddInteger(opac_block, "NNuPerGroup_a", 64);
        mg_opac_a = MeanOpacityBase(opac_a, // underlying opacity model
                                    lRhoMin, lRhoMax, NRho, lTMin, lTMax,
                                    NT,                  // table bounds and # entries
                                    global_group_bounds, // frequency bounds (Hz)
                                    global_ngroups,      // number of groups
                                    NNuPerGroup);
      }

      // Construct Scattering MeanOpacity
      RiotOpacity::MeanOpacS mg_opac_s;
      if (opac_s_type == "table") {
        mg_opac_s = MeanSOpacityBase(opac_s_table);
      } else {
        const Real lRhoMin = pin->GetOrAddReal(opac_block, "lRhoMin_s", -1.0);
        const Real lRhoMax = pin->GetOrAddReal(opac_block, "lRhoMax_s", 1.0);
        const Real lTMin = pin->GetOrAddReal(opac_block, "lTMin_s", -1.0);
        const Real lTMax = pin->GetOrAddReal(opac_block, "lTMax_s", 1.0);
        const int NRho = pin->GetOrAddInteger(opac_block, "NRho_s", 2);
        const int NT = pin->GetOrAddInteger(opac_block, "NT_s", 2);
        const int NNuPerGroup = pin->GetOrAddInteger(opac_block, "NNuPerGroup_s", 64);
        mg_opac_s = MeanSOpacityBase(opac_s, // underlying opacity model
                                     lRhoMin, lRhoMax, NRho, lTMin, lTMax,
                                     NT,                  // table bounds and # entries
                                     global_group_bounds, // frequency bounds (Hz)
                                     global_ngroups,      // number of groups
                                     NNuPerGroup);
      }

      opac_a_host.push_back(mg_opac_a);
      opac_s_host.push_back(mg_opac_s);
      const bool save_opac_a_table =
          pin->GetOrAddBoolean(opac_block, "save_opac_a_file", false);
      const bool save_opac_s_table =
          pin->GetOrAddBoolean(opac_block, "save_opac_s_file", false);
      if (parthenon::Globals::my_rank == 0) {
        if (save_opac_a_table && opac_a_type != "table") {
          mg_opac_a.Save(opac_a_table + opac_block + ".sp5");
        }
        if (save_opac_s_table && opac_s_type != "table") {
          mg_opac_s.Save(opac_s_table + opac_block + ".sp5");
        }
      }
    };

    // EOS, electron EOS, and opacity enrollment
    eos_from_matid.push_back(eos_host.size());
    opac_from_matid.push_back(opac_a_host.size());
    if (nph == 1) {
      // EOS
      std::string eosblock = (pin->DoesParameterExist(block_name, "eos"))
                                 ? pin->GetString(block_name, "eos")
                                 : block_name;
      add_eos(eosblock);

      // electron EOS
      if (do_ionization) {
        std::string electron_eos_param = "electron_eos";
        PARTHENON_REQUIRE_THROWS(
            pin->DoesParameterExist(block_name, electron_eos_param),
            "Electron EOS must be provided for every material if ionization is enabled.");
        std::string electron_eos_block = pin->GetString(block_name, electron_eos_param);
        add_electron_eos(electron_eos_block);
      }

      // Opacities
      std::string opacblock = (pin->DoesParameterExist(block_name, "opac"))
                                  ? pin->GetString(block_name, "opac")
                                  : block_name;
      add_opac(opacblock);
    } else if (nph > 1) { // multi-phase material
      for (int n = 0; n < nph; n++) {
        // EOS
        std::string eos_param = std::string("eos") + std::to_string(n);
        PARTHENON_REQUIRE_THROWS(pin->DoesParameterExist(block_name, eos_param),
                                 "For multi-phase EOS's must be separate blocks.");
        std::string eosblock = pin->GetString(block_name, eos_param);
        add_eos(eosblock);

        // Electron EOS
        if (do_ionization) {
          std::string electron_eos_param =
              std::string("electron_eos") + std::to_string(n);
          PARTHENON_REQUIRE_THROWS(
              pin->DoesParameterExist(block_name, electron_eos_param),
              "Electron EOS must be provided for every material and phase if ionization "
              "is enabled.");
          std::string electron_eos_block = pin->GetString(block_name, electron_eos_param);
          add_electron_eos(electron_eos_block);
        }

        // Opacities
        std::string opac_param = std::string("opac") + std::to_string(n);
        std::string opacblock = (pin->DoesParameterExist(block_name, opac_param))
                                    ? pin->GetString(block_name, opac_param)
                                    : block_name;
        add_opac(opacblock);
      }
    }
  }

  // Handle share memory EOS instantiation (if enabled)
#ifdef MPI_PARALLEL
  if (use_mpi_shared_memory) {
    auto shared_memory_eos = [&](std::vector<RiotEOS::EOS> &eos_h) {
      // shared memory communicator, ranks
      MPI_Comm shm_comm;
      MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                          &shm_comm);

      int shm_rank, shm_size;
      MPI_Comm_rank(shm_comm, &shm_rank);
      MPI_Comm_size(shm_comm, &shm_size);

      // serialize
      std::size_t packed_size, shared_size;
      singularity::VectorSerializer<RiotEOS::EOS> serializer;
      if (global_rank0) {
        serializer = singularity::VectorSerializer<RiotEOS::EOS>(eos_h);
        packed_size = serializer.SerializedSizeInBytes();
        shared_size = serializer.SharedMemorySizeInBytes();
      }

      // Send sizes
      MPI_Bcast(&packed_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
      MPI_Bcast(&shared_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

      // Allocate shared memory
      char *packed_data = (char *)malloc(packed_size);
      if (global_rank0) {
        serializer.Serialize(packed_data);
        serializer.Finalize(); // cleans up owned EOSs
      }
      RiotUtils::BcastBytes(packed_data, packed_size, 0, MPI_COMM_WORLD);

      singularity::SharedMemSettings settings = singularity::DEFAULT_SHMEM_STNGS;

      char *shared_data;
      char *mpi_base_pointer;
      int mpi_unit;
      MPI_Aint query_size;
      MPI_Win window;
      MPI_Win_allocate_shared((shm_rank == 0) ? shared_size : 0, 1, MPI_INFO_NULL,
                              shm_comm, &mpi_base_pointer, &window);
      // pointer to shared memory
      MPI_Win_shared_query(window, MPI_PROC_NULL, &query_size, &mpi_unit, &shared_data);

      // shared memory mutex
      MPI_Win_lock_all(MPI_MODE_NOCHECK, window);

      // Set singularity shared memory settings
      settings.data = shared_data;
      settings.is_domain_root = (shm_rank == 0);

      singularity::VectorSerializer<RiotEOS::EOS> deserializer;
      deserializer.DeSerialize(packed_data, settings);

      MPI_Win_unlock_all(window);
      MPI_Barrier(shm_comm);
      free(packed_data);
      MPI_Comm_free(&shm_comm);

      // extract EOS
      eos_h = deserializer.eos_objects;
    };

    shared_memory_eos(eos_host);
    if (do_ionization) shared_memory_eos(electron_eos_host);
  } // use_mpi_shared_memory
#endif // MPI_PARALLEL

  // Obtain vector of EOS...
  for (auto eos : eos_host) {
    eos_dvec.push_back(eos.GetOnDevice());
  }

  // ... electron EOS...
  if (do_ionization) {
    for (auto electron_eos : electron_eos_host) {
      electron_eos_dvec.push_back(electron_eos.GetOnDevice());
    }
  }

  bool add_lR = false;
  bool add_lT = false;
  auto check_lambdas = [&](auto &eos_vec) {
    for (auto &eos : eos_vec) {
      if (eos.template NeedsLambda<singularity::IndexableTypes::LogDensity>())
        add_lR = true;
      if (eos.template NeedsLambda<singularity::IndexableTypes::LogTemperature>())
        add_lT = true;
      if (!do_ionization &&
          eos.template NeedsLambda<singularity::IndexableTypes::MeanIonizationState>()) {
        PARTHENON_THROW(
            "EOS requires mean ionizations tate, but ionization is not enabled!");
      }
    }
  };
  check_lambdas(eos_host);
  check_lambdas(electron_eos_host);
  for (const auto matid : matids) {
    std::vector<int> phase_size = {nphase[matid]};
    if (add_lR) AddMatId(matid, phase_size, cm_lr_cache);
    if (add_lT) AddMatId(matid, phase_size, cm_lT_cache);
  }

  // ...and opacity objects on device
  for (auto opac : opac_a_host) {
    opac_a_dvec.push_back(opac.GetOnDevice());
  }
  for (auto opac : opac_s_host) {
    opac_s_dvec.push_back(opac.GetOnDevice());
  }

  // Add the sparse pools
  AddPools(materials.get(), ccmat_rho, ccmat_volume_fraction, ccmat_internal_energy,
           cm_rho, cm_sie, cm_temperature, cm_pressure, cm_bulk_modulus, cm_cv,
           mat_stress, stress, mat_eps, eps, gmod, strength_j2, mat_iso, iso,
           phase_rho_sum, phase_fraction, ccmat_ionization_zbar, cm_ionization_zbar,
           cm_ue, cm_sie_e, cm_lr_cache, cm_lT_cache);

  // Instantiate device ParArray1D of EOS and Opacity objects, setup PTE list vector
  const int num_eos = eos_host.size();
  const int num_opac = opac_a_host.size();
  PARTHENON_REQUIRE(num_opac == opac_s_host.size(), "Issue encountered in opacity init!");

  auto [eos_device, eos_hcopy] = RiotUtils::VectorToViewPair(eos_dvec, "EOS Device");
  auto [opac_a_device, opac_a_hcopy] =
      RiotUtils::VectorToViewPair(opac_a_dvec, "opac_a Device");
  auto [opac_s_device, opac_s_hcopy] =
      RiotUtils::VectorToViewPair(opac_s_dvec, "opac_s Device");

  // electron EOS special because it's optional
  auto electron_eos_device = ParArray1D<RiotEOS::EOS>("Electron EOS Device", num_eos);
  auto electron_eos_hcopy =
      Kokkos::create_mirror_view(Kokkos::HostSpace(), electron_eos_device);
  if (do_ionization) {
    for (int m = 0; m < num_eos; ++m) {
      electron_eos_hcopy(m) = electron_eos_dvec[m];
    }
  }
  if (do_ionization) Kokkos::deep_copy(electron_eos_device, electron_eos_hcopy);

  auto max_lev_mat_d = RiotUtils::VectorToDevice(max_lev_mat_v, "max_lev_mat");
  auto max_lev_bnd_d = RiotUtils::VectorToDevice(max_lev_bnd_v, "max_lev_bnd");
  auto eos_from_matid_d = RiotUtils::VectorToDevice(eos_from_matid, "eos starting index");
  auto opac_from_matid_d =
      RiotUtils::VectorToDevice(opac_from_matid, "opac starting index");
  auto strength_map_d =
      RiotUtils::VectorToDevice(strength_map, "point from matid to strid");
  auto [strength_models_d, strength_models_h] =
      RiotUtils::VectorToViewPair(strength_models, "Flow stress models");
  auto [strength_model_ids_d, strength_model_ids_h] =
      RiotUtils::VectorToViewPair(strength_model_ids, "Flow stress model ids");
  auto nphase_d = RiotUtils::VectorToDevice(nphase, "number of phases for matid");
  auto pte_matlist_d = RiotUtils::VectorToDevice(matids, "PTE matlist");
  auto num_iso_per_mat_d = RiotUtils::VectorToDevice(num_iso_per_mat, "num isos");
  auto iso_fractions_d = RiotUtils::VectorToDevice(iso_fractions, "iso fractions");

  // JMM: strong is special because std::vector<bool>
  auto [strong_d, strong_h] = RiotUtils::VectorToViewPair(strong, "Strength flag");
  auto [strength_mats_d, strength_mats_h] =
      RiotUtils::VectorToViewPair(strength_mats, "IDs of strength mats");

  // Store relevant mat quantities in params
  params.Add("max_lev_mat", max_lev_mat_d);
  params.Add("max_lev_bnd", max_lev_bnd_d);
  // EOS
  params.Add("d.d.EOS", eos_device);
  params.Add("d.h.EOS", eos_hcopy);
  params.Add("h.h.EOS", eos_host);
  params.Add("num_eos", eos_host.size());
  params.Add("d.EOS_from_matid", eos_from_matid_d);
  params.Add("h.EOS_from_matid", eos_from_matid);
  // Electron EOS
  if (do_ionization) {
    params.Add("d.d.electron_EOS", electron_eos_device);
    params.Add("d.h.electron_EOS", electron_eos_hcopy);
    params.Add("h.h.electron_EOS", electron_eos_host);
  }
  // Opacities
  params.Add("d.d.opac_a", opac_a_device);
  params.Add("d.h.opac_a", opac_a_hcopy);
  params.Add("h.h.opac_a", opac_a_host);
  params.Add("d.d.opac_s", opac_s_device);
  params.Add("d.h.opac_s", opac_s_hcopy);
  params.Add("h.h.opac_s", opac_s_host);
  params.Add("num_opac", opac_a_host.size());
  params.Add("d.opac_from_matid", opac_from_matid_d);
  params.Add("h.opac_from_matid", opac_from_matid);
  params.Add("ngroups", global_ngroups);
  params.Add("group_bounds", global_group_bounds);
  // Strength
  params.Add("d.strong", strong_d);
  params.Add("strength_mats", strength_mats);
  params.Add("h.strength_mats", strength_mats_h);
  params.Add("d.strength_map", strength_map_d);
  params.Add("numstr", numstr);
  params.Add("d.strength_models", strength_models_d);
  params.Add("d.strength_model_ids", strength_model_ids_d);
  // Phases
  params.Add("d.nphase", nphase_d);
  params.Add("h.nphase", nphase);
  // PTE
  params.Add("h.pte_matlist", matids);
  params.Add("d.pte_matlist", pte_matlist_d);
  params.Add("max_array_size", num_eos);
  params.Add("nummat", nummat);
  params.Add("all_mats", all_mats);
  params.Add("matids", matids);
  params.Add("use_general_pte", use_general_pte);

  // Isotope data configuration
  // Check for isotope_data block for reading masses/charges from HDF5
  bool read_isotope_data = pin->DoesBlockExist("isotope_data");
  std::string isotope_filename = "isotope_data.hdf5";
  if (read_isotope_data) {
    isotope_filename =
        pin->GetOrAddString("isotope_data", "filename", "isotope_data.hdf5");
  }
  params.Add("read_isotope_data", read_isotope_data);
  params.Add("isotope_filename", isotope_filename);

  // Isotopes
  params.Add("Isotope Names", iso_names);
  params.Add("Isotope Zaids", iso_zaids);
  params.Add("Isotope Fractions", iso_fractions);
  params.Add("d.Isotope Fractions", iso_fractions_d);
  params.Add("num_iso_per_mat", num_iso_per_mat_d);

  // Read isotope masses and charges from HDF5 if requested
  // This happens during initialization since masses/charges are fixed material properties
  if (read_isotope_data) {
#ifdef SPINER_USE_HDF
    std::vector<std::vector<Real>> iso_masses_per_mat;
    std::vector<std::vector<int>> iso_charges_per_mat;

    // Read data for each material
    for (const auto &mat_zaids : iso_zaids) {
      if (mat_zaids.empty()) {
        iso_masses_per_mat.push_back(std::vector<Real>());
        iso_charges_per_mat.push_back(std::vector<int>());
        continue;
      }

      std::vector<Real> mat_masses;
      std::vector<int> mat_charges;
      read_isotope_data_from_hdf5(isotope_filename, mat_zaids, mat_masses, mat_charges);
      iso_masses_per_mat.push_back(mat_masses);
      iso_charges_per_mat.push_back(mat_charges);
    }

    // Store in params for access by other packages
    params.Add("Isotope Masses", iso_masses_per_mat);
    params.Add("Isotope Charges", iso_charges_per_mat);
#else
    PARTHENON_THROW(
        "read_isotope_data=true requires HDF5 support (SPINER_USE_HDF). "
        "Rebuild with HDF5 enabled or set read=false in <isotope_data> block.");
#endif
  }

  // Alert user if invoking ideal gas PTE solver
  if (!use_general_pte && global_rank0) {
    fprintf(stderr, "All EoSs are ideal gases, using ideal gas PTE solver.\n");
  }

  //--------------------------------------------------------------------------------------
  // PTE Statistics
  bool track_pte_statistics = false;
  bool pte_stats_avg_fields = false;
  bool pte_stats_reset_fields = false;
  std::string pte_stats_mode = "averaged_light";
  if (use_general_pte) {
    track_pte_statistics =
        pin->GetOrAddBoolean("materials", "track_pte_statistics", false,
                             "Output diagnostic statistics on PTE solver successes, "
                             "failures, iteration count, etc.");
    if (track_pte_statistics) {
      std::string pte_stats_mode = pin->GetOrAddString(
          "materials", "pte_stats_mode", "averaged_light",
          {"instantaneous", "averaged", "averaged_light"},
          "Only active if track_pte_statistics=true. instantaneous is whatever the last "
          "iteration was at output. averaged computes average values. averaged_light "
          "records the numerator and denominator of these running averages, i.e., the "
          "total number of iterations and the total number of calls, but not their "
          "raito.");
      std::vector<parthenon::MetadataFlag> flags = {Metadata::Cell, Metadata::Derived,
                                                    Metadata::OneCopy};
      pte_stats_reset_fields = (pte_stats_mode == "instantaneous");
      if ((pte_stats_mode == "averaged") || (pte_stats_mode == "averaged_light")) {
        flags.push_back(Metadata::ForceRemeshComm);
        flags.push_back(Metadata::Restart);
      }
      m = parthenon::Metadata(flags);
      materials->AddField<diag::pte_niter>(m);
      materials->AddField<diag::pte_nfails>(m);
      materials->AddField<diag::pte_nbackups>(m);
      materials->AddField<diag::pte_ncalls>(m);

      if (pte_stats_mode == "averaged") {
        pte_stats_avg_fields = true;
        m = parthenon::Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
        materials->AddField<diag::pte_avg_niter>(m);
        materials->AddField<diag::pte_avg_nbackups>(m);
        materials->AddField<diag::pte_fail_fraction>(m);
      }
    }

    // Singularity PTE controls/tolerances
    singularity::MixParams pte_params;
    pte_params.pte_max_iter_per_mat = pin->GetOrAddInteger(
        "materials", "pte_max_iter_per_mat", pte_params.pte_max_iter_per_mat,
        "Maximum number of iterations the PTE solver will take is this parameter times "
        "number of materials in the cell");

    // JMM: The default bounds here are rather loose
    pte_params.pte_rel_tolerance_p = pin->GetOrAddReal(
        "materials", "pte_rel_tolerance_p", pte_params.pte_rel_tolerance_p,
        "How small the PTE solver pressure residual (compared to the pressure) must be "
        "for PTE success");
    pte_params.pte_rel_tolerance_e = pin->GetOrAddReal(
        "materials", "pte_rel_tolerance_e", pte_params.pte_rel_tolerance_e,
        "How small the PTE solver energy residual (compared to all internal energy in a "
        "cell) must be before PTE success");
    pte_params.pte_abs_tolerance_p = pin->GetOrAddReal(
        "materials", "pte_abs_tolerance_p", pte_params.pte_abs_tolerance_p,
        "How small the PTE solver pressure residual (in absolute terms) must be "
        "for PTE success");

    // JMM: These could be even looser if you are encountering PTE
    // failures frequently as large as 1e-2 for the relative
    // tolerances
    pte_params.pte_abs_tolerance_p_sufficient =
        pin->GetOrAddReal("materials", "pte_abs_tolerance_p_sufficient",
                          pte_params.pte_abs_tolerance_p_sufficient,
                          "If the PTE solver fails, this residual is considered good "
                          "enough to avoid fallback strategies");
    pte_params.pte_rel_tolerance_p_sufficient =
        pin->GetOrAddReal("materials", "pte_rel_tolerance_p_sufficient",
                          pte_params.pte_rel_tolerance_p_sufficient,
                          "If the PTE solver fails, this residual is considered good "
                          "enough to avoid fallback strategies");
    pte_params.pte_rel_tolerance_e_sufficient =
        pin->GetOrAddReal("materials", "pte_rel_tolerance_e_sufficient",
                          pte_params.pte_rel_tolerance_e_sufficient,
                          "If the PTE solver fails, this residual is considered good "
                          "enough to avoid fallback strategies");

    params.Add("pte_params", pte_params);
  }
  params.Add("track_pte_statistics", track_pte_statistics);
  params.Add("pte_stats_mode", pte_stats_mode);
  params.Add("pte_stats_avg_fields", pte_stats_avg_fields);
  params.Add("pte_stats_reset_fields", pte_stats_reset_fields);

  return materials;
}

int CountMaterials(ParameterInput *pin) {
  const std::string mat_prefix = "material";
  // "blocks" also contains the <materials> block, which we don't want
  // to count.
  auto blocks = pin->GetBlockNamesWithPrefix(mat_prefix);
  return std::max(0, static_cast<int>(blocks.size()) - 1);
}

#ifdef SPINER_USE_HDF
// Helper function to read isotope data from HDF5 file
void read_isotope_data_from_hdf5(const std::string &filename,
                                 const std::vector<int> &requested_zaids,
                                 std::vector<Real> &masses, std::vector<int> &charges) {
  herr_t status = 0;
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    PARTHENON_THROW("Failed to open isotope data file: " + filename);
  }

  hid_t mass_group = H5Gopen(file, "Masses", H5P_DEFAULT);
  if (mass_group < 0) {
    H5Fclose(file);
    PARTHENON_THROW("Failed to open /Masses group in file: " + filename);
  }

  int num_isotopes;
  status = H5LTget_attribute_int(file, "Masses", "num_isotopes", &num_isotopes);
  if (status < 0) {
    H5Gclose(mass_group);
    H5Fclose(file);
    PARTHENON_THROW("Failed to read num_isotopes from file: " + filename);
  }

  std::vector<int> all_zaids(num_isotopes);
  std::vector<Real> all_masses(num_isotopes);
  std::vector<int> all_charges(num_isotopes);

  status = H5LTget_attribute_int(file, "Masses", "zaids", all_zaids.data());
  status += H5LTget_attribute_double(file, "Masses", "masses", all_masses.data());
  status += H5LTget_attribute_int(file, "Masses", "charges", all_charges.data());

  if (status < 0) {
    H5Gclose(mass_group);
    H5Fclose(file);
    PARTHENON_THROW("Failed to read isotope data attributes from file: " + filename);
  }

  // Extract only requested isotopes in the order they were requested
  masses.clear();
  charges.clear();
  for (const auto &zaid : requested_zaids) {
    auto it = std::find(all_zaids.begin(), all_zaids.end(), zaid);
    if (it != all_zaids.end()) {
      int idx = std::distance(all_zaids.begin(), it);
      masses.push_back(all_masses[idx]);
      charges.push_back(all_charges[idx]);
    } else {
      H5Gclose(mass_group);
      H5Fclose(file);
      PARTHENON_THROW("Isotope " + std::to_string(zaid) +
                      " not found in file: " + filename);
    }
  }

  H5Gclose(mass_group);
  H5Fclose(file);
}
#endif
} // namespace Materials

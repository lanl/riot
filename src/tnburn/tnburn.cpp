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

// C++ includes
#include <array>
#include <vector>

// Parthenon includes
#include <kokkos_abstraction.hpp>

// Riot incldues
#include "materials/materials.hpp"
#include "ratelib/read_isotope_data.hpp"
#include "riot_utils/riot_loops.hpp"
#include "tnburn.hpp"
#include "variables.hpp"

namespace TNBurn {

using RiotLimits::MAX_MATERIALS;

//----------------------------------------------------------------------------------------
//! \fn  std::shared_ptr<StateDescriptor> TNBurn::Initialize
//! \brief
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            std::vector<std::vector<int>> isotope_names) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;
  using parthenon::ParArray1D;
  using parthenon::ParArray2D;
  using parthenon::ParArray3D;
  using parthenon::SparsePool;
  auto tnburn = std::make_shared<StateDescriptor>("TNBurn");
  Params &params = tnburn->AllParams();

  // NOTE(@pdmullen): TN burn is currently incompatible with sparse physics
  // TODO(@chadmeyer): Potentially address this and lift the PARTHENON_REQUIRE?
  PARTHENON_REQUIRE(!(pin->GetOrAddBoolean("physics", "sparse_physics", true)),
                    "TN does not currently support sparse physics!");

  // Extract custom Metadata flags
  auto BurnFlag = Metadata::GetOrAddFlag(riot::metadata::TNBurn);

  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
              Metadata::Conserved, Metadata::Sparse, Metadata::FillGhost,
              Metadata::Advected, Metadata::WithFluxes, BurnFlag});
  m.Associate(cm::tn_specific_reactions::name());
  auto tn_reaction_density =
      SparsePool::Make<ccmat::tn_reaction_density>(m, ccmat::rho::name());
  m = Metadata(
      {Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::Sparse, BurnFlag});
  auto tn_specific_reactions =
      SparsePool::Make<cm::tn_specific_reactions>(m, ccmat::rho::name());

  auto const tn_block_name = std::string("tnburn");

  // All isotope and TN data configuration now in <isotope_data> block
  const std::string tn_filename =
      pin->GetOrAddString("isotope_data", "filename", default_tn_fname);
  // populate the list of desired reactions
  std::vector<std::string> reactions_list;
  reactions_list.reserve(10);
  int reaction_count = 0;
  auto reaction_name = std::string("reaction") + std::to_string(reaction_count);
  while (pin->DoesParameterExist(tn_block_name, reaction_name)) {
    reactions_list.emplace_back(pin->GetString(tn_block_name, reaction_name));
    reaction_count += 1;
    reaction_name = std::string("reaction") + std::to_string(reaction_count);
  }

  std::vector<int> isotopes;
  std::vector<Real> masses;
  std::vector<int> charges;
  std::vector<ReactionData> reactions;
  // Read in from file
#ifdef SPINER_USE_HDF
  if (reaction_count > 0) {
    ratelib::read_tn_reactions(tn_filename, reactions_list, isotopes, masses, charges,
                               reactions);
  }
#else
  if (reaction_count > 0) {
    PARTHENON_FAIL("TN reactions require HDF5 support (SPINER_USE_HDF). "
                   "Use ndi2spiner tool to generate HDF5 file from NDI data.");
  }
#endif
  auto reactions_d = ParArray1D<ReactionData>("Reaction Data", reactions.size());
  auto reactions_h = Kokkos::create_mirror_view(reactions_d);
  for (int i = 0; i < reactions.size(); i++) {
    reactions_h[i] = reactions[i].getOnDevice();
  }
  Kokkos::deep_copy(reactions_d, reactions_h);
  params.Add("reaction_data", reactions_d);
  // That takes care of the nuclear data we need.  Now, we need some objects
  // to make it easier to index material offsets.  We need to map isotope index
  // to the location in the isotopes array, identify relevant isotopes per
  // reaction, and get a list of possible reactions per material.

  const int nummat = isotope_names.size();
  const int num_reactions = reactions.size();
  std::vector<int> reaction_shape{num_reactions};

  // Find out if isotopes will be deposited locally or just disappear.  If we
  // don't deposit them, their mass and energy will just be gone (unless we had a
  // CPT package or something to account for that)
  std::vector<bool> deposit_isotope(isotopes.size());
  for (int i = 0; i < isotopes.size(); i++) {
    std::string deposit = "deposit_locally_" + std::to_string(isotopes[i]);
    deposit_isotope[i] = pin->GetOrAddBoolean(tn_block_name, deposit,
                                              true); // by default we deposit it all
  }

  // Indexing needs to be something like [mat][reaction][thing]
  // This might seem a little excessive, but it is all pre-computable to make indexing
  // trivial in the kernels.
  std::vector<int> num_reactions_per_mat(nummat);
  std::vector<std::vector<int>> reaction_list_per_mat(nummat);
  std::vector<std::vector<std::array<int, 2>>> reactant_indices_per_mat_reaction(nummat);
  std::vector<std::vector<int>> num_products_per_mat_reaction(nummat);
  std::vector<std::vector<std::array<int, 3>>> product_indices_per_mat_reaction(nummat);
  std::vector<std::vector<std::array<int, 3>>> energy_indices_per_mat_reaction(nummat);
  std::vector<std::vector<std::array<int, 3>>> product_multiplicities_per_mat_reaction(
      nummat);

  for (int m = 0; m < nummat; m++) {
    auto &reactant_indices_per_reaction = reactant_indices_per_mat_reaction[m];
    auto &num_products_per_reaction = num_products_per_mat_reaction[m];
    auto &product_indices_per_reaction = product_indices_per_mat_reaction[m];
    auto &product_multiplicities_per_reaction =
        product_multiplicities_per_mat_reaction[m];
    auto &energy_indices_per_reaction = energy_indices_per_mat_reaction[m];
    auto &reaction_list = reaction_list_per_mat[m];
    auto &num_reactions_mat = num_reactions_per_mat[m];
    num_reactions_mat = 0;
    const auto &mat_iso = isotope_names[m];
    if (mat_iso.size() == 0) {
      continue; // continues the mat loop.  These "inner" vectors will be empty.  We
                // already set reactions to zero
    }
    for (int r = 0; r < num_reactions; r++) {
      const auto &reaction = reactions[r];
      bool active = true; // until proven otherwise;
      // First, check if the reactants are present.
      for (int i = 0; i < 2; i++) { // Is it acceptable to do this? Do I really know that
                                    // there are exactly 2 reactants?
        if (!isin(mat_iso, reaction.reactants[i])) {
          active = false;
          break;
        }
      }
      if (!active) continue; // continues the reactions loop. This reaction is not active
      // I assume that if all the reactants are present, we will do the reaction.
      // If any products are missing or are labeled as not deposited, they will
      // be skipped below.  In this way we could, for instance, never deposit
      // neutrons by not allocating space for them.
      reaction_list.push_back(r);
      auto &rs = reaction.reactants;
      reactant_indices_per_reaction.push_back(
          std::array<int, 2>{findindex(mat_iso, rs[0]), findindex(mat_iso, rs[1])});
      num_products_per_reaction.push_back(0);
      product_indices_per_reaction.push_back({0, 0, 0});
      product_multiplicities_per_reaction.push_back({0, 0, 0});
      energy_indices_per_reaction.push_back({0, 0, 0});
      int &numprod = num_products_per_reaction[num_reactions_mat];
      auto &product_indices = product_indices_per_reaction[num_reactions_mat];
      auto &product_multiplicities =
          product_multiplicities_per_reaction[num_reactions_mat];
      auto &energy_indices = energy_indices_per_reaction[num_reactions_mat];
      num_reactions_mat++;
      for (int i = 0; i < reaction.num_products; i++) {
        const auto &p = reaction.products[i];
        if (!isin(mat_iso, p)) continue;
        if (!deposit_isotope[findindex(isotopes, p)]) continue;
        product_indices[numprod] = findindex(mat_iso, p);
        product_multiplicities[numprod] = reaction.product_multiplicities[i];
        energy_indices[numprod] = i;
        numprod++;
      }
    }
    if (num_reactions_mat > 0) {
      // Note that if there are *any* reactions active, we'll allocate space for all the
      // potential reactions Even if these are not active in *this* material
      tn_specific_reactions.Add(m, reaction_shape);
      tn_reaction_density.Add(m, reaction_shape);
    }
  }
  // Now these need to be copied to device
  auto [num_reactions_per_mat_d, num_reactions_per_mat_h] = RiotUtils::VectorToViewPair(
      num_reactions_per_mat, "Number of Reactions active in this material");

  ParArray2D<int> reaction_list_per_mat_d("Which reactions are active in this material",
                                          nummat, num_reactions);
  auto reaction_list_per_mat_h = Kokkos::create_mirror_view(reaction_list_per_mat_d);
  for (int i = 0; i < nummat; i++)
    for (int j = 0; j < num_reactions_per_mat[i]; j++)
      reaction_list_per_mat_h(i, j) = reaction_list_per_mat[i][j];
  Kokkos::deep_copy(reaction_list_per_mat_d, reaction_list_per_mat_h);

  ParArray3D<int> reactant_indices_per_mat_reaction_d(
      "material isotope indices of reactants", nummat, num_reactions, 2);
  auto reactant_indices_per_mat_reaction_h =
      Kokkos::create_mirror_view(reactant_indices_per_mat_reaction_d);
  for (int i = 0; i < nummat; i++)
    for (int j = 0; j < num_reactions_per_mat[i]; j++)
      for (int k = 0; k < 2; k++)
        reactant_indices_per_mat_reaction_h(i, j, k) =
            reactant_indices_per_mat_reaction[i][j][k];
  Kokkos::deep_copy(reactant_indices_per_mat_reaction_d,
                    reactant_indices_per_mat_reaction_h);

  ParArray2D<int> num_products_per_mat_reaction_d("Number of deposited products", nummat,
                                                  num_reactions);
  auto num_products_per_mat_reaction_h =
      Kokkos::create_mirror_view(num_products_per_mat_reaction_d);
  for (int i = 0; i < nummat; i++)
    for (int j = 0; j < num_reactions_per_mat[i]; j++)
      num_products_per_mat_reaction_h(i, j) = num_products_per_mat_reaction[i][j];
  Kokkos::deep_copy(num_products_per_mat_reaction_d, num_products_per_mat_reaction_h);

  ParArray3D<int> product_indices_per_mat_reaction_d(
      "material isotope indices of deposited products", nummat, num_reactions, 3);
  auto product_indices_per_mat_reaction_h =
      Kokkos::create_mirror_view(product_indices_per_mat_reaction_d);
  for (int i = 0; i < nummat; i++)
    for (int j = 0; j < num_reactions_per_mat[i]; j++)
      for (int k = 0; k < num_products_per_mat_reaction[i][j]; k++)
        product_indices_per_mat_reaction_h(i, j, k) =
            product_indices_per_mat_reaction[i][j][k];
  Kokkos::deep_copy(product_indices_per_mat_reaction_d,
                    product_indices_per_mat_reaction_h);

  ParArray3D<int> energy_indices_per_mat_reaction_d(
      "energy return indices for deposited products", nummat, num_reactions, 3);
  auto energy_indices_per_mat_reaction_h =
      Kokkos::create_mirror_view(energy_indices_per_mat_reaction_d);
  for (int i = 0; i < nummat; i++)
    for (int j = 0; j < num_reactions_per_mat[i]; j++)
      for (int k = 0; k < num_products_per_mat_reaction[i][j]; k++)
        energy_indices_per_mat_reaction_h(i, j, k) =
            energy_indices_per_mat_reaction[i][j][k];
  Kokkos::deep_copy(energy_indices_per_mat_reaction_d, energy_indices_per_mat_reaction_h);

  ParArray3D<int> product_multiplicities_per_mat_reaction_d("product multiplicities",
                                                            nummat, num_reactions, 3);
  auto product_multiplicities_per_mat_reaction_h =
      Kokkos::create_mirror_view(product_multiplicities_per_mat_reaction_d);
  for (int i = 0; i < nummat; i++)
    for (int j = 0; j < num_reactions_per_mat[i]; j++)
      for (int k = 0; k < num_products_per_mat_reaction[i][j]; k++)
        product_multiplicities_per_mat_reaction_h(i, j, k) =
            product_multiplicities_per_mat_reaction[i][j][k];
  Kokkos::deep_copy(product_multiplicities_per_mat_reaction_d,
                    product_multiplicities_per_mat_reaction_h);

  ParArray3D<Real> reactant_masses_per_mat_reaction_d("reactant masses", nummat,
                                                      num_reactions, 2);
  auto reactant_masses_per_mat_reaction_h =
      Kokkos::create_mirror_view(reactant_masses_per_mat_reaction_d);
  for (int i = 0; i < nummat; i++)
    for (int j = 0; j < num_reactions_per_mat[i]; j++)
      for (int k = 0; k < 2; k++) {
        const int inx = findindex(
            isotopes, isotope_names[i][reactant_indices_per_mat_reaction_h(i, j, k)]);
        const Real mass = masses[inx];
        reactant_masses_per_mat_reaction_h(i, j, k) = mass * amu;
      }
  Kokkos::deep_copy(reactant_masses_per_mat_reaction_d,
                    reactant_masses_per_mat_reaction_h);

  ParArray3D<Real> product_masses_per_mat_reaction_d("product masses", nummat,
                                                     num_reactions, 3);
  auto product_masses_per_mat_reaction_h =
      Kokkos::create_mirror_view(product_masses_per_mat_reaction_d);
  for (int i = 0; i < nummat; i++)
    for (int j = 0; j < num_reactions_per_mat[i]; j++)
      for (int k = 0; k < num_products_per_mat_reaction[i][j]; k++) {
        const int inx = findindex(
            isotopes, isotope_names[i][product_indices_per_mat_reaction_h(i, j, k)]);
        const Real mass = masses[inx];
        product_masses_per_mat_reaction_h(i, j, k) = mass * amu;
      }
  Kokkos::deep_copy(product_masses_per_mat_reaction_d, product_masses_per_mat_reaction_h);

  // Finally, we must make these parameters

  params.Add("all_isotopes", isotopes);
  params.Add("isotope_masses", masses);
  params.Add("isotope_charges", charges);
  params.Add("num_reactions_per_mat", num_reactions_per_mat_d);
  params.Add("reaction_list_per_mat", reaction_list_per_mat_d);
  params.Add("reactant_indices_per_mat_reaction", reactant_indices_per_mat_reaction_d);
  params.Add("num_products_per_mat_reaction", num_products_per_mat_reaction_d);
  params.Add("product_indices_per_mat_reaction", product_indices_per_mat_reaction_d);
  params.Add("energy_indices_per_mat_reaction", energy_indices_per_mat_reaction_d);
  params.Add("product_multiplicities_per_mat_reaction",
             product_multiplicities_per_mat_reaction_d);
  params.Add("num_reactions", num_reactions);
  params.Add("reactant_masses_per_mat_reaction", reactant_masses_per_mat_reaction_d);
  params.Add("product_masses_per_mat_reaction", product_masses_per_mat_reaction_d);

  tnburn->AddSparsePool(tn_specific_reactions);
  tnburn->AddSparsePool(tn_reaction_density);
  tnburn->FillDerivedMesh = FillDerived;

  // Add history variables
  auto HstSum = parthenon::UserHistoryOperation::sum;
  using parthenon::HistoryOutputVar;
  parthenon::HstVar_list hst_vars = {};
  for (int mat = 0; mat < nummat; mat++) {
    for (int rxn = 0; rxn < num_reactions_per_mat[mat]; rxn++) {
      hst_vars.emplace_back(
          HstSum,
          [=](MeshData<Real> *mymd) {
            return TNBurn::IntegratedReactionCount(mymd, mat, rxn);
          },
          "TotalReactions_" + std::to_string(mat) + "_" +
              reactions_list[reaction_list_per_mat[mat][rxn]]);
    }
  }
  params.Add(parthenon::hist_param_key, hst_vars);

  // Source term dU/dt variables: all independent, non-operator-split fields.
  // Use GetOrAddFlag: this package may initialize before the OperatorSplit user
  // flag is registered elsewhere. The subset is resolved later (at step time), so
  // the flag only needs to exist here, not yet be set on any variables.
  using FC_t = Metadata::FlagCollection;
  auto op_split = Metadata::GetOrAddFlag(riot::metadata::OperatorSplit);
  tnburn->RegisterMeshDataSubset(
      "dudt", RiotUtils::MakePackageDudtRequirements({}, FC_t({Metadata::Independent}) -
                                                             FC_t({op_split})));

  return tnburn;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus TNBurn::CalculateTNBurnSource
//! \brief Compute the TN reaction rates.  There are sources on mass, energy, isotopics,
//! and reaction count. There are also resultant sources on momentum, and any per-material
//! (or per mass) quantities which are separately handled in shared_sources.
TaskStatus CalculateTNBurnSource(MeshData<Real> *state, MeshData<Real> *src,
                                 const Real dt) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using parthenon::ParArray1D;
  using parthenon::ParArray2D;
  using parthenon::ParArray3D;

  if (state->NumBlocks() == 0) return TaskStatus::complete;
  auto pmb = state->GetBlockData(0)->GetBlockPointer();
  auto pm = state->GetParentPointer();
  const int nblocks = state->NumBlocks();
  auto const &hydro = pmb->packages.Get("hydro");
  auto const &tnburn = pmb->packages.Get("TNBurn");
  auto const &materials = pmb->packages.Get("materials");
  const Real vol_frac_thresh = hydro->Param<Real>("vol_frac_thresh");
  auto num_reactions = tnburn->Param<int>("num_reactions");
  auto num_reactions_per_mat = tnburn->Param<ParArray1D<int>>("num_reactions_per_mat");
  auto reaction_list_per_mat = tnburn->Param<ParArray2D<int>>("reaction_list_per_mat");
  auto reactant_indicies_per_mat_reaction =
      tnburn->Param<ParArray3D<int>>("reactant_indices_per_mat_reaction");
  auto num_products_per_mat_reaction =
      tnburn->Param<ParArray2D<int>>("num_products_per_mat_reaction");
  auto product_indices_per_mat_reaction =
      tnburn->Param<ParArray3D<int>>("product_indices_per_mat_reaction");
  auto energy_indices_per_mat_reaction =
      tnburn->Param<ParArray3D<int>>("energy_indices_per_mat_reaction");
  auto product_multiplicities_per_mat_reaction =
      tnburn->Param<ParArray3D<int>>("product_multiplicities_per_mat_reaction");
  auto reaction_data = tnburn->Param<ParArray1D<ReactionData>>("reaction_data");
  auto reactant_masses_per_mat_reaction =
      tnburn->Param<ParArray3D<Real>>("reactant_masses_per_mat_reaction");
  auto product_masses_per_mat_reaction =
      tnburn->Param<ParArray3D<Real>>("product_masses_per_mat_reaction");
  auto num_iso_per_mat = materials->Param<ParArray1D<int>>("num_iso_per_mat");

  // RHS variables; the inputs.  The reaction rates do not depend on much!
  auto v = riot::MakePack<ccbulk::temperature, ccmat::iso, ccmat::volume_fraction,
                          ccmat::rho, cm::phase_fraction>(state);
  // Variables for which this package produces source terms
  auto dv = riot::MakePack<ccbulk::total_material_energy, ccmat::iso,
                           ccmat::tn_reaction_density, ccmat::rho>(src);

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, state,
                                     parthenon::TopologicalElement::CC);
  idx_space.template AddPerPointScratch<Real>(3);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto rate = RiotLoop::GetPerPointScratch<Real>(idx_range);
        auto sigvbar = RiotLoop::GetPerPointScratch<Real>(idx_range);
        auto logt = RiotLoop::GetPerPointScratch<Real>(idx_range);
        const int nmat = v.GetSize(b, ccmat::rho());

        // First, zero out everything
        for (int r = dv.GetLowerBound(b); r <= dv.GetUpperBound(b); r++) {
          auto var = RiotLoop::make_var_view(idx_range, dv, r);
          RiotLoop::inner(idx_range, [&](const auto kji) { var(kji) = 0.0; });
        }

        // Loop over all entries in ccmat::rho
        int offset_tnr = 0;
        int offset_iso = 0;
        for (int m = 0; m < nmat; m++) {
          // Globalmatid will be the same for all rhos of same mat.  This is as expected.
          auto globalmatid = v(b, ccmat::rho(m)).sparse_id;

          // Loop over all reactions that are active for *this* material.  Could be zero.
          for (int r = 0; r < num_reactions_per_mat(globalmatid); r++) {
            // Calculate the rate for this reaction
            // These are local offsets (isotope id of this material)
            const auto &r_id_1 = reactant_indicies_per_mat_reaction(globalmatid, r, 0);
            const auto &r_id_2 = reactant_indicies_per_mat_reaction(globalmatid, r, 1);
            auto reactant_1 =
                RiotLoop::make_var_view(idx_range, v, ccmat::iso(offset_iso + r_id_1));
            auto reactant_2 =
                RiotLoop::make_var_view(idx_range, v, ccmat::iso(offset_iso + r_id_2));
            auto phase_frac =
                RiotLoop::make_var_view(idx_range, v, cm::phase_fraction(m));
            auto vfrac = RiotLoop::make_var_view(idx_range, v, ccmat::volume_fraction(m));
            auto temp = RiotLoop::make_var_view(idx_range, v, ccbulk::temperature());
            const auto &rd = reaction_data(reaction_list_per_mat(globalmatid, r));

            // Now we're ready to get the rate
            // First take the logarithm of the temperature
            // Maybe this should be a fastlog?
            RiotLoop::inner(idx_range,
                            [&](const auto kji) { logt(kji) = std::log(temp(kji)); });
            idx_range.TeamBarrier();

            // Second, get sigmavbar from the table (function of temperature)
            RiotLoop::inner(idx_range, [&](const auto kji) {
              sigvbar(kji) = rd.SigmaVBar.interpToReal(logt(kji));
            });
            idx_range.TeamBarrier();

            // Finally, calculate the rate
            RiotLoop::inner(idx_range, [&](const auto kji) {
              rate(kji) = (vfrac(kji) > vol_frac_thresh) * reactant_1(kji) *
                          reactant_2(kji) * sigvbar(kji) * phase_frac(kji) *
                          phase_frac(kji) / (vfrac(kji) + 1.0e-100);
            });
            idx_range.TeamBarrier();

            // With the rate, we can now calculate the source terms
            auto int_eng =
                RiotLoop::make_var_view(idx_range, dv, ccbulk::total_material_energy());
            auto dreactant1 =
                RiotLoop::make_var_view(idx_range, dv, ccmat::iso(offset_iso + r_id_1));
            auto dreactant2 =
                RiotLoop::make_var_view(idx_range, dv, ccmat::iso(offset_iso + r_id_2));
            auto drho = RiotLoop::make_var_view(idx_range, dv, ccmat::rho(m));
            auto dtnr = RiotLoop::make_var_view(
                idx_range, dv,
                ccmat::tn_reaction_density(offset_tnr +
                                           reaction_list_per_mat(globalmatid, r)));
            const Real &atomic_mass_reactant_1 =
                reactant_masses_per_mat_reaction(globalmatid, r, 0);
            const Real &atomic_mass_reactant_2 =
                reactant_masses_per_mat_reaction(globalmatid, r, 1);
            // Source terms related to reactants
            // There is an interpolation here, which perhaps could be split out
            RiotLoop::inner(idx_range, [&](const auto kji) {
              int_eng(kji) -= rate(kji) * rd.InputEnergy.interpToReal(logt(kji));
              const Real drho1 = rate(kji) * atomic_mass_reactant_1;
              const Real drho2 = rate(kji) * atomic_mass_reactant_2;
              dreactant1(kji) -= drho1;
              dreactant2(kji) -= drho2;
              drho(kji) -= (drho1 + drho2);
              dtnr(kji) += rate(kji);
            });
            idx_range.TeamBarrier();

            // Source terms related to products
            for (int p = 0; p < num_products_per_mat_reaction(globalmatid, r); p++) {
              const auto &pid = product_indices_per_mat_reaction(globalmatid, r, p);
              const auto &eid = energy_indices_per_mat_reaction(globalmatid, r, p);
              // For each *deposited* product, add in the product energies
              auto dproduct =
                  RiotLoop::make_var_view(idx_range, dv, ccmat::iso(offset_iso + pid));
              const auto &mult =
                  product_multiplicities_per_mat_reaction(globalmatid, r, p);
              const Real &atomic_mass_product_p =
                  product_masses_per_mat_reaction(globalmatid, r, p);
              RiotLoop::inner(idx_range, [&](const auto kji) {
                int_eng(kji) +=
                    rate(kji) * rd.EnergyOut[eid].interpToReal(logt(kji)) * mult;
                const Real drhop = rate(kji) * atomic_mass_product_p * mult;
                drho(kji) += drhop;
                dproduct(kji) += drhop;
              });
              idx_range.TeamBarrier();
            }
          } // Loop over reaction in this mat

          // Only jump here if this is the last phase for this material
          if (m < nmat - 1) {
            if (globalmatid != v(b, ccmat::rho(m + 1)).sparse_id) {
              if (num_reactions_per_mat(globalmatid) > 0) offset_tnr += num_reactions;
              offset_iso += num_iso_per_mat(globalmatid);
            }
          }
          idx_range.TeamBarrier();
        } // Loop over material
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void TNBurn::FillDerived
//! \brief
void FillDerived(MeshData<Real> *md) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;
  using parthenon::ParArray1D;

  Mesh *pm = md->GetMeshPointer();
  auto const &tnburn = pm->packages.Get("TNBurn");
  auto num_reactions = tnburn->Param<int>("num_reactions");
  auto num_reactions_per_mat = tnburn->Param<ParArray1D<int>>("num_reactions_per_mat");

  auto &materials = pm->packages.Get("materials");
  const int max_array_size = materials->Param<int>("max_array_size");
  const auto &nphase = materials->Param<parthenon::ParArray1D<int>>("d.nphase");
  PARTHENON_REQUIRE(max_array_size <= MAX_MATERIALS,
                    "Number of materials exceeds MAX_MATERIALS compile-time limit");

  auto v = riot::MakePack<ccmat::rho, ccmat::phase_rho_sum, ccmat::iso, cm::iso,
                          cm::phase_fraction, ccmat::tn_reaction_density,
                          cm::tn_specific_reactions>(md);

  const int nblocks = md->NumBlocks();
  if (nblocks == 0) return;

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::entire, 0, nblocks, md,
                                     parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        // sparse_id -> dense material index; identical for every cell of the block, so
        // it lives on the stack (bounded by MAX_MATERIALS) rather than in team scratch.
        std::array<int, MAX_MATERIALS> mat_map;
        const int nmat = v.GetSize(b, ccmat::phase_rho_sum());
        const int niso = v.GetSize(b, ccmat::iso());

        // TODO(JMM): Should the phases stuff be its own fill
        // derived? I think for now it belongs in TN. But if we were
        // able to add ordering to the per-package fill derived
        // calls... maybe phases should be their own thing.

        // Fill in phase rho sum
        int iphase = 0;
        for (int m = 0; m < nmat; m++) {
          const int sparse_id = v(b, ccmat::rho(iphase)).sparse_id;
          mat_map[sparse_id] = m;
          auto phase_rho_sum =
              RiotLoop::make_var_view(idx_range, v, ccmat::phase_rho_sum(m));
          auto phase_rho = RiotLoop::make_var_view(idx_range, v, ccmat::rho(iphase));
          RiotLoop::inner(idx_range,
                          [&](const auto kji) { phase_rho_sum(kji) = phase_rho(kji); });
          iphase++;
          idx_range.TeamBarrier();
          // if the material is multiphase, add in those as well
          for (int p = 1; p < nphase(m); p++) {
            idx_range.TeamBarrier(); // Should only need the barrier in multiphase
            auto phase_rho_p = RiotLoop::make_var_view(idx_range, v, ccmat::rho(iphase));
            RiotLoop::inner(idx_range, [&](const auto kji) {
              phase_rho_sum(kji) += phase_rho_p(kji);
            });
            iphase++;
            idx_range.TeamBarrier();
          }
        }

        // Now fill in phase fractions
        for (int p = 0; p < nmat; p++) {
          const int m = mat_map[v(b, ccmat::rho(p)).sparse_id];
          auto cons = RiotLoop::make_var_view(idx_range, v, ccmat::rho(p));
          auto prim = RiotLoop::make_var_view(idx_range, v, cm::phase_fraction(p));
          auto rho = RiotLoop::make_var_view(idx_range, v, ccmat::phase_rho_sum(m));
          RiotLoop::inner(idx_range, [&](const auto kji) {
            prim(kji) = cons(kji) * (rho(kji) > 0.0 ? 1.0 / rho(kji) : 0.0);
          });
        }

        // Fill in isotopic information
        for (int iso = 0; iso < niso; iso++) {
          const int m = mat_map[v(b, ccmat::iso(iso)).sparse_id];
          auto cons = RiotLoop::make_var_view(idx_range, v, ccmat::iso(iso));
          auto prim = RiotLoop::make_var_view(idx_range, v, cm::iso(iso));
          auto rho = RiotLoop::make_var_view(idx_range, v, ccmat::phase_rho_sum(m));
          RiotLoop::inner(idx_range, [&](const auto kji) {
            prim(kji) = cons(kji) * (rho(kji) > 0.0 ? 1.0 / rho(kji) : 0.0);
          });
        }

        // Fill in TN specific data
        // TODO: would it be better to just loop over the tn_specific_reactions space
        // and point back to a matid?
        int offset = 0;
        for (int m = 0; m < nmat; m++) { // For every material
          int globalmatid = v(b, v.GetIndex(b, ccmat::phase_rho_sum(m))).sparse_id; // ???
          if (num_reactions_per_mat(globalmatid) > 0) { // With TN active
            auto rho = RiotLoop::make_var_view(idx_range, v, ccmat::phase_rho_sum(m));
            for (int r = 0; r < num_reactions; r++) { // For all reactions in the problem
              auto rhoq = RiotLoop::make_var_view(idx_range, v,
                                                  ccmat::tn_reaction_density(offset + r));
              auto q = RiotLoop::make_var_view(idx_range, v,
                                               cm::tn_specific_reactions(offset + r));
              RiotLoop::inner(idx_range, [&](const auto kji) {
                q(kji) = (rho(kji) > 0.0 ? rhoq(kji) / rho(kji) : 0.0);
              });
            }
            offset += num_reactions;
          }
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn  Real TNBurn::IntegratedReactionCount
//! \brief User History
Real IntegratedReactionCount(MeshData<Real> *md, const int mat, const int reaction) {
  namespace ccmat = cell_variables::cell_averaged::mat;
  using parthenon::MakePackDescriptor;
  using parthenon::ParArray1D;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto const &tnburn = pm->packages.Get("TNBurn");
  auto num_reactions_per_mat = tnburn->Param<ParArray1D<int>>("num_reactions_per_mat");
  static auto desc =
      MakePackDescriptor<ccmat::tn_reaction_density, ccmat::rho>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  using TE = parthenon::TopologicalElement;
  using rt = RiotUtils::ReductionType<Kokkos::Sum<Real>>;
  auto idx_space =
      rt::GetIndexSpace(IndexDomain::interior, 0, vmesh.GetNBlocks(), md, TE::CC);
  return RiotLoop::outer_reduce(
      idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
        auto coords = vmesh.GetCoordinates(b);
        // Hoist the material search and reaction-offset accumulation out of the inner
        // reduction: find the matching material's reaction-component offset i_tn once.
        // sparse_id is unique per material, so at most one material matches.
        int i_tn = 0;
        bool found = false;
        for (int n = 0; n < vmesh.GetSize(b, ccmat::rho()); n++) {
          if (vmesh(b, ccmat::rho(n)).sparse_id == mat) {
            found = true;
            break;
          }
          i_tn += num_reactions_per_mat(vmesh(b, ccmat::rho(n)).sparse_id);
        }
        if (!found) return;
        // tn_reaction_density is indexed by reaction component (i_tn + reaction), not by
        // material, so use a single-variable view at that resolved component index.
        auto rxn = RiotLoop::make_var_view(idx_range, vmesh,
                                           ccmat::tn_reaction_density(i_tn + reaction));
        RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &lsum) {
          const auto [k, j, i] = idx_range.GetKJI(idx);
          lsum += rxn(idx) * coords.CellVolume(k, j, i);
        });
      });
}

} // namespace TNBurn

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

#include <memory>
#include <string>

#include <ports-of-call/robust_utils.hpp>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

#include "bhr3/bhr3.1.hpp"
#include "mix.hpp"
#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"

using parthenon::ParArray1D;

namespace Mix {

//----------------------------------------------------------------------------------------
//! \fn  void Mix::Initialize
//! \brief Adds mix related quantities to the Physics and reads in parameters
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace fm = face_variables::mat;

  // BHR mix is currently implemented for Cartesian coordinates only: the source terms and
  // fluxes assume Cartesian gradients/divergences (no metric or curvature terms).
  // Extending to curvilinear grids is possible but has not been done, so fail loudly at
  // set-up rather than silently run with the wrong geometry.
  if (!parthenon::IsCoord<parthenon::UniformCartesian>()) {
    PARTHENON_FAIL("mix (BHR) is only supported with Cartesian coordinates; "
                   "UniformCylindrical and UniformSpherical are not yet implemented");
  }

  auto mix = std::make_shared<StateDescriptor>("mix");
  Params &params = mix->AllParams();
  std::string name;
  Metadata m;

  // NOTE(@chadmeyer): There are a ton of parameters that could be read in, but most of
  // them have defaults
  auto c_1 = pin->GetOrAddReal("mix", "c_1", 1.6);
  auto c_2 = pin->GetOrAddReal("mix", "c_2", 1.77);
  auto c_3 = pin->GetOrAddReal("mix", "c_3", 0.0);
  auto c_4 = pin->GetOrAddReal("mix", "c_4", 1.1);
  auto c_a1 = pin->GetOrAddReal("mix", "c_a1", 2.8);
  auto c_a2 = pin->GetOrAddReal("mix", "c_a2", 1.0);
  auto c_a3 = pin->GetOrAddReal("mix", "c_a3", 1.0);
  auto c_b2 = pin->GetOrAddReal("mix", "c_b2", 1.8);
  auto c_r1 = pin->GetOrAddReal("mix", "c_r1", 0.3);
  auto c_r2 = pin->GetOrAddReal("mix", "c_r2", 0.6);
  auto c_r4 = pin->GetOrAddReal("mix", "c_r4", 1.8);
  auto c_ap = pin->GetOrAddReal("mix", "c_ap", 0.1);
  auto c_ar = pin->GetOrAddReal("mix", "c_ar", 0.0);
  auto c_au = pin->GetOrAddReal("mix", "c_au", 0.4);
  auto sigma_c = pin->GetOrAddReal("mix", "sigma_c", 1.0);
  auto sigma_a = pin->GetOrAddReal("mix", "sigma_a", 1.0);
  auto sigma_b = pin->GetOrAddReal("mix", "sigma_b", 1.0);
  auto sigma_k = pin->GetOrAddReal("mix", "sigma_k", 1.0);
  auto sigma_visc = pin->GetOrAddReal("mix", "sigma_visc", 0.6);
  auto sigma_epsilon = pin->GetOrAddReal("mix", "sigma_epsilon", 0.1);
  auto c_1v = pin->GetOrAddReal("mix", "c_1v", 1.3);
  auto c_2v = pin->GetOrAddReal("mix", "c_2v", 1.77);
  auto c_3v = pin->GetOrAddReal("mix", "c_3v", 0.0);
  auto c_4v = pin->GetOrAddReal("mix", "c_4v", 1.24);
  auto c_mu = pin->GetOrAddReal("mix", "c_mu", 0.28);
  // NOTE(@chadmeyer): The following probably shouldn't have default values
  auto tke_0 = pin->GetOrAddReal("mix", "K0", 0.01);
  auto S0 = pin->GetOrAddReal("mix", "S0", 0.0001);

  // The object should be trivially copyable but outside of a par-array; I'm not sure
  // how to keep it in persistent memory on device otherwise (and I don't want to copy it
  // every cycle)
  ParArray1D<MixModel::BHR3_1> bhr_d("BHR object", 1);
  auto bhr_h = Kokkos::create_mirror_view(bhr_d);
  bhr_h[0] =
      MixModel::BHR3_1(c_1, c_2, c_3, c_4, c_a1, c_a2, c_a3, c_b2, c_r1, c_r2, c_r4, c_ap,
                       c_ar, c_au, sigma_c, sigma_a, sigma_b, sigma_k, sigma_visc,
                       sigma_epsilon, c_1v, c_2v, c_3v, c_4v, c_mu, tke_0, S0);
  Kokkos::deep_copy(bhr_d, bhr_h);
  params.Add("bhr_d", bhr_d);
  params.Add("bhr_h", bhr_h);
  params.Add("K0", tke_0);
  params.Add("S0", S0);

  // Independent components of the tensor.  max(3, 2*numvel), but numvel==3
  const int tensorsize = 6;
  std::vector<int> tensor_arr_size(1, tensorsize); // Indep. comp. of symmetric tensor
  std::vector<int> vector_arr_size(1, 3);          // Size numvel (not numdim)
  m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
                Metadata::Conserved, Metadata::FillGhost, Metadata::Vector,
                Metadata::Advected, Metadata::WithFluxes},
               tensor_arr_size);
  m.Associate(ccbulk::reynolds_stress::name());
  mix->AddField(ccbulk::rho_reynolds_stress::name(), m);
  m = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Vector, Metadata::Derived,
                Metadata::OneCopy},
               tensor_arr_size);
  mix->AddField(ccbulk::reynolds_stress::name(), m);
  m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
                Metadata::Conserved, Metadata::FillGhost, Metadata::Vector,
                Metadata::Advected, Metadata::WithFluxes},
               vector_arr_size);
  m.Associate(ccbulk::bhr_a::name());
  mix->AddField(ccbulk::rho_bhr_a::name(), m);
  m = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Vector, Metadata::Derived,
                Metadata::OneCopy},
               vector_arr_size);
  mix->AddField(ccbulk::bhr_a::name(), m);
  m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
                Metadata::Conserved, Metadata::FillGhost, Metadata::Advected,
                Metadata::WithFluxes});
  m.Associate(ccbulk::bhr_b::name());
  mix->AddField(ccbulk::rho_bhr_b::name(), m);
  m.Associate(ccbulk::bhr_SD::name());
  mix->AddField(ccbulk::rho_bhr_SD::name(), m);
  m.Associate(ccbulk::bhr_ST::name());
  mix->AddField(ccbulk::rho_bhr_ST::name(), m);
  m = Metadata(
      {Metadata::Cell, Metadata::Intensive, Metadata::Derived, Metadata::OneCopy});
  mix->AddField(ccbulk::bhr_b::name(), m);
  mix->AddField(ccbulk::bhr_ST::name(), m);
  mix->AddField(ccbulk::bhr_SD::name(), m);
  mix->FillDerivedMesh = FillDerived;
  mix->EstimateTimestepMesh = EstimateTimestepMesh;

  // Diffusive mass-flux register, one sparse component per material.
  //
  // The BHR viscous-flux kernel deposits the per-material diffusive mass flux here as a
  // face quantity; ComputeAnonFluxes then reads it back to build the anonymously advected
  // fluxes on all other per-material conserved variables. This lives in its own face
  // field (rather than c.c.mat.rho's real flux register) because the raw diffusive mass
  // flux is needed as a distinct quantity to advect the other fields with.
  //
  // CellMemAligned is required for correctness, not just performance. It allocates this
  // face field with the cell-aligned layout that Parthenon gives real flux registers on
  // Cell fields; the kernels index it by cell index kji (through flux pack views) and
  // deposit its values onto ccmat::rho's real flux register, so the two must share that
  // layout. Without the flag the field would get the extra-face-per-direction layout and
  // the indices would not line up. If, in the fullness of time, other packages need
  // diffusive flux registers, this should be promoted to the materials package.
  m = Metadata(
      {Metadata::Face, Metadata::OneCopy, Metadata::Sparse, Metadata::CellMemAligned});
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  auto diffusive_flux = SparsePool::Make<fm::diffusive_fluxes>(m, ccmat::rho::name());
  int nummat = 0;
  while (pin->DoesBlockExist("material" + std::to_string(nummat))) {
    diffusive_flux.Add(nummat, {1, 1});
    nummat++;
  }
  mix->AddSparsePool(diffusive_flux);
  // Source term dU/dt variables: all independent, non-operator-split fields.
  using FC_t = Metadata::FlagCollection;
  auto op_split = Metadata::GetOrAddFlag(riot::metadata::OperatorSplit);
  mix->RegisterMeshDataSubset("dudt",
                              RiotUtils::MakePackageDudtRequirements(
                                  {}, FC_t({Metadata::Independent}) - FC_t({op_split})));
  return mix;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Mix::Mut
//! \brief Turbulent viscosity mu_t at a single cell from the packed state. Bundles the
//! tke/sqrt(tke)/mut chain that the diffusive-source stencil needs at each of the three
//! stencil points, so the coefficient is evaluated once per cell and shared across every
//! diffused BHR quantity (only the per-variable sigma/rho scaling differs).
template <typename PackView, typename Index>
KOKKOS_FORCEINLINE_FUNCTION Real Mut(const PackView &pv, MixModel::BHR3_1 &bhr,
                                     const Index &kji) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  const Real k =
      bhr.tke(pv(ccbulk::reynolds_stress(0), kji), pv(ccbulk::reynolds_stress(1), kji),
              pv(ccbulk::reynolds_stress(2), kji));
  return bhr.mut(pv(ccbulk::rho(), kji), pv(ccbulk::bhr_ST(), kji), std::sqrt(k));
}

//----------------------------------------------------------------------------------------
//! \fn  void Mix::CalculateMixDiffSourceAlongDir
//! \brief Adds the DIR-dependent piece of the diffusive BHR source terms. The diffusive
//! sources, d/dx_DIR( mu_t/sigma * d/dx_DIR(q) ), separate cleanly by direction: the full
//! Laplacian-like operator is the sum over active dimensions of the single-direction term
//! this function computes. It is therefore called once per active dimension by
//! CalculateMixSource (X1, then X2, then X3), and each call adds only its own d/dx_DIR
//! contribution -- accumulating onto the registers the algebraic (direction-independent)
//! pass already initialized, hence the += into dvv.
//!
//! Within a sweep, mu_t is computed once per cell into a symmetric (+/-1) halo scratch
//! buffer, then the three-point stencil (three_points_to_source) is applied to each
//! diffused quantity: SD, ST, b, the three a components, and the Rsize Reynolds-stress
//! components.
template <parthenon::CoordinateDirection DIR>
void CalculateMixDiffSourceAlongDir(MeshData<Real> *state, MeshData<Real> *src,
                                    ParArray1D<MixModel::BHR3_1> mix_model) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using TE = parthenon::TopologicalElement;
  using PortsOfCall::Robust::ratio;

  auto v = riot::MakePack<ccbulk::rho, ccbulk::reynolds_stress, ccbulk::bhr_a,
                          ccbulk::bhr_b, ccbulk::bhr_SD, ccbulk::bhr_ST>(state);
  auto dv =
      riot::MakePack<ccbulk::rho_reynolds_stress, ccbulk::rho_bhr_a, ccbulk::rho_bhr_b,
                     ccbulk::rho_bhr_SD, ccbulk::rho_bhr_ST>(src);

  const int numvel = 3;
  const int Rsize = std::max(3, 2 * numvel);
  const int nblocks = v.GetNBlocks();
  if (nblocks == 0) return;

  // Symmetric (+/-1) halo for the direction being swept, so mu_t is produced on the two
  // face-neighbors the three-point stencil reads as well as the center cell.
  using halo =
      std::tuple_element_t<DIR - 1,
                           std::tuple<RiotUtils::halo::pm_i_t, RiotUtils::halo::pm_j_t,
                                      RiotUtils::halo::pm_k_t>>;

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, state, TE::CC);
  RiotLoop::AddPerPointScratch<Real, halo>(idx_space, 1);
  const auto dd = idx_space.GetDelta(DIR);

  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto &coords = v.GetCoordinates(b);
        const Real dh =
            (DIR == X1DIR ? coords.Dxc<X1DIR>(0)
                          : (DIR == X2DIR ? coords.Dxc<X2DIR>(0) : coords.Dxc<X3DIR>(0)));
        auto &bhr = mix_model(0);

        auto halo_range = idx_range.template AddHalo<halo>();
        auto mut = RiotLoop::GetPerPointScratch<Real>(halo_range);
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto dvv = RiotLoop::make_pack_view(idx_range, dv);

        // Produce mu_t on the center cell and both face-neighbors.
        RiotLoop::inner(halo_range,
                        [&](const auto kji) { mut(kji) = Mut(pv, bhr, kji); });
        idx_range.TeamBarrier();

        // Apply d/dx( coef * d/dx(q) ) for each diffused quantity. coef carries the
        // per-variable 1/sigma (and 1/rho scaling); an outer factor restores the
        // conserved (rho-weighted) form where needed. All reuse the same mu_t triplet.
        auto diffuse = [&](auto q_field, auto out_field, const Real inv_sigma,
                           const Real coef_rho_pow, const Real factor_rho_pow) {
          RiotLoop::inner(idx_range, [&](const auto kji) {
            const auto kL = kji - dd;
            const auto kR = kji + dd;
            auto coef = [&](const auto idx) {
              Real c = mut(idx) * inv_sigma;
              if (coef_rho_pow == 1.0)
                c = ratio(c, pv(ccbulk::rho(), idx));
              else if (coef_rho_pow == 2.0) {
                const Real r = pv(ccbulk::rho(), idx);
                c = ratio(c, r * r);
              }
              return c;
            };
            Real s =
                three_points_to_source(pv(q_field, kL), pv(q_field, kji), pv(q_field, kR),
                                       coef(kL), coef(kji), coef(kR), dh);
            if (factor_rho_pow == 1.0)
              s *= pv(ccbulk::rho(), kji);
            else if (factor_rho_pow == 2.0) {
              const Real r = pv(ccbulk::rho(), kji);
              s *= r * r;
            }
            dvv(out_field, kji) += s;
          });
        };

        // SD, ST: coef = mut/sigma, no rho scaling.
        diffuse(ccbulk::bhr_SD(), ccbulk::rho_bhr_SD(), 1.0 / bhr.sigma_d(), 0.0, 0.0);
        diffuse(ccbulk::bhr_ST(), ccbulk::rho_bhr_ST(), 1.0 / bhr.sigma_t(), 0.0, 0.0);
        // b: coef = mut/(rho^2 sigma_b), factor rho^2.
        diffuse(ccbulk::bhr_b(), ccbulk::rho_bhr_b(), 1.0 / bhr.sigma_b(), 2.0, 2.0);
        // a(d): coef = mut/(rho sigma_a), factor rho.
        for (int d = 0; d < numvel; d++)
          diffuse(ccbulk::bhr_a(d), ccbulk::rho_bhr_a(d), 1.0 / bhr.sigma_a(), 1.0, 1.0);
        // Rij: coef = mut/sigma_k, no rho scaling.
        for (int d = 0; d < Rsize; d++)
          diffuse(ccbulk::reynolds_stress(d), ccbulk::rho_reynolds_stress(d),
                  1.0 / bhr.sigma_k(), 0.0, 0.0);
      });
}

//----------------------------------------------------------------------------------------
//! \fn  void Mix::CalculateMixSource
//! \brief The source terms on the BHR variables, assembled in two parts. First, the
//! algebraic (0-D and advective) sources are computed in a single fused per-cell pass
//! below. These are the direction-independent terms: they use full gradient/divergence
//! tensors built from all active dimensions at once, so a single pass writes the complete
//! contribution to each source register. Second, the diffusive sources -- the only
//! directionally-separable part of BHR -- are added by CalculateMixDiffSourceAlongDir,
//! called once per active dimension so each sweep accumulates its own d/dx_DIR term onto
//! the registers this pass initialized.
TaskStatus CalculateMixSource(MeshData<Real> *state, MeshData<Real> *src) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using TE = parthenon::TopologicalElement;

  auto pm = state->GetParentPointer();
  auto &mix = pm->packages.Get("mix");
  auto mix_model = mix->Param<ParArray1D<MixModel::BHR3_1>>("bhr_d");

  auto v =
      riot::MakePack<ccbulk::rho, ccbulk::reynolds_stress, ccbulk::bhr_a, ccbulk::bhr_b,
                     ccbulk::bhr_SD, ccbulk::bhr_ST, ccbulk::pressure, ccbulk::velocity>(
          state);
  auto dv =
      riot::MakePack<ccbulk::rho_reynolds_stress, ccbulk::rho_bhr_a, ccbulk::rho_bhr_b,
                     ccbulk::rho_bhr_SD, ccbulk::rho_bhr_ST>(src);

  const int ndim = pm->ndim;
  const int numvel = 3;
  const int nblocks = v.GetNBlocks();
  if (nblocks == 0) return TaskStatus::complete;

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, state, TE::CC);
  // Neighbor-cell offsets; zero in collapsed dimensions, so transverse gradients vanish
  // in reduced dimensionality without an explicit ndim branch on the offset.
  const auto di = idx_space.GetDelta(X1DIR);
  const auto dj = idx_space.GetDelta(X2DIR);
  const auto dk = idx_space.GetDelta(X3DIR);

  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto &coords = v.GetCoordinates(b);
        const Real dx = coords.Dxc<X1DIR>(0);
        const Real dy = coords.Dxc<X2DIR>(0);
        const Real dz = coords.Dxc<X3DIR>(0);
        auto &bhr = mix_model(0);

        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto dvv = RiotLoop::make_pack_view(idx_range, dv);

        RiotLoop::inner(idx_range, [&](const auto kji) {
          // Centered first derivatives d(field)/dx_dir. Transverse offsets are zero in
          // collapsed dims, so those derivatives evaluate to 0 without an ndim branch.
          auto ddir = [&](auto field, const auto off, const Real d2) {
            return (pv(field, kji + off) - pv(field, kji - off)) / d2;
          };
          // grad_u[c][dir] = d(u_c)/dx_dir
          Real gu[3][3];
          for (int c = 0; c < numvel; c++) {
            gu[c][0] = ddir(ccbulk::velocity(c), di, 2.0 * dx);
            gu[c][1] = (ndim > 1) ? ddir(ccbulk::velocity(c), dj, 2.0 * dy) : 0.0;
            gu[c][2] = (ndim > 2) ? ddir(ccbulk::velocity(c), dk, 2.0 * dz) : 0.0;
          }
          // grad_P, grad_rho, grad_b along each dir
          Real gP[3], grho[3], gb[3];
          gP[0] = ddir(ccbulk::pressure(), di, 2.0 * dx);
          grho[0] = ddir(ccbulk::rho(), di, 2.0 * dx);
          gb[0] = ddir(ccbulk::bhr_b(), di, 2.0 * dx);
          gP[1] = (ndim > 1) ? ddir(ccbulk::pressure(), dj, 2.0 * dy) : 0.0;
          grho[1] = (ndim > 1) ? ddir(ccbulk::rho(), dj, 2.0 * dy) : 0.0;
          gb[1] = (ndim > 1) ? ddir(ccbulk::bhr_b(), dj, 2.0 * dy) : 0.0;
          gP[2] = (ndim > 2) ? ddir(ccbulk::pressure(), dk, 2.0 * dz) : 0.0;
          grho[2] = (ndim > 2) ? ddir(ccbulk::rho(), dk, 2.0 * dz) : 0.0;
          gb[2] = (ndim > 2) ? ddir(ccbulk::bhr_b(), dk, 2.0 * dz) : 0.0;

          // div_aa[c] = sum_d d/dx_d(a_c a_d); div_a = sum_d d/dx_d(a_d);
          // div_u = sum_d (du_d/dx_d - da_d/dx_d) = div(ubar)
          auto ddprod = [&](int c, int d, const auto off, const Real d2) {
            return (pv(ccbulk::bhr_a(c), kji + off) * pv(ccbulk::bhr_a(d), kji + off) -
                    pv(ccbulk::bhr_a(c), kji - off) * pv(ccbulk::bhr_a(d), kji - off)) /
                   d2;
          };
          Real div_aa[3];
          for (int c = 0; c < numvel; c++) {
            div_aa[c] = ddprod(c, 0, di, 2.0 * dx);
            if (ndim > 1) div_aa[c] += ddprod(c, 1, dj, 2.0 * dy);
            if (ndim > 2) div_aa[c] += ddprod(c, 2, dk, 2.0 * dz);
          }
          const Real da_x = ddir(ccbulk::bhr_a(0), di, 2.0 * dx);
          const Real da_y = (ndim > 1) ? ddir(ccbulk::bhr_a(1), dj, 2.0 * dy) : 0.0;
          const Real da_z = (ndim > 2) ? ddir(ccbulk::bhr_a(2), dk, 2.0 * dz) : 0.0;
          const Real div_a = da_x + da_y + da_z;
          const Real div_u = (gu[0][0] - da_x) + (gu[1][1] - da_y) + (gu[2][2] - da_z);

          // Local state at this cell.
          const Real rho = pv(ccbulk::rho(), kji);
          const Real ax = pv(ccbulk::bhr_a(0), kji);
          const Real ay = pv(ccbulk::bhr_a(1), kji);
          const Real az = pv(ccbulk::bhr_a(2), kji);
          const Real SD = pv(ccbulk::bhr_SD(), kji);
          const Real ST = pv(ccbulk::bhr_ST(), kji);
          const Real bb = pv(ccbulk::bhr_b(), kji);
          const Real Rxx = pv(ccbulk::reynolds_stress(0), kji);
          const Real Ryy = pv(ccbulk::reynolds_stress(1), kji);
          const Real Rzz = pv(ccbulk::reynolds_stress(2), kji);
          const Real Rxy = (numvel > 1) ? pv(ccbulk::reynolds_stress(3), kji) : 0.0;
          const Real Rxz = (numvel == 3) ? pv(ccbulk::reynolds_stress(4), kji) : 0.0;
          const Real Ryz = (numvel == 3) ? pv(ccbulk::reynolds_stress(5), kji) : 0.0;

          const Real tke = bhr.tke(Rxx, Ryy, Rzz);
          const Real sqrtk = std::sqrt(tke);
          const Real a_dot_grad_p = ax * gP[0] + ay * gP[1] + az * gP[2];
          const Real rRgu = bhr.rho_R_grad_u(rho, Rxx, Ryy, Rzz, Rxy, Rxz, Ryz, gu[0][0],
                                             gu[0][1], gu[0][2], gu[1][0], gu[1][1],
                                             gu[1][2], gu[2][0], gu[2][1], gu[2][2]);

          // Symmetric Reynolds-stress table indexed [component i][component j]; gu is the
          // velocity-gradient table indexed [component i][derivative direction].
          const Real Rij[3][3] = {{Rxx, Rxy, Rxz}, {Rxy, Ryy, Ryz}, {Rxz, Ryz, Rzz}};
          const Real *dpdi = gP;

          // Algebraic sources.
          dvv(ccbulk::rho_bhr_b(), kji) =
              bhr.b_source(bb, gb[0], gb[1], gb[2], ax, ay, az, rho, grho[0], grho[1],
                           grho[2], SD, sqrtk);
          dvv(ccbulk::rho_bhr_ST(), kji) =
              bhr.ST_source(ST, tke, sqrtk, rho, rRgu, a_dot_grad_p, div_u);
          dvv(ccbulk::rho_bhr_SD(), kji) =
              bhr.SD_source(SD, tke, sqrtk, rho, rRgu, a_dot_grad_p, div_u);
          const Real a_comp[3] = {ax, ay, az};
          for (int d = 0; d < numvel; d++) {
            dvv(ccbulk::rho_bhr_a(d), kji) =
                bhr.ai_source(a_comp[d], dpdi[d], rho, sqrtk, SD, bb, div_aa[d], ax, ay,
                              az, Rij[d][0], Rij[d][1], Rij[d][2], grho[0], grho[1],
                              grho[2], gu[d][0], gu[d][1], gu[d][2], div_a);
          }
          for (int d = 0; d < 3; d++) { // Diagonal Reynolds-stress components
            dvv(ccbulk::rho_reynolds_stress(d), kji) = bhr.Rii_source(
                Rij[d][d], a_comp[d], dpdi[d], rho, tke, sqrtk, SD, a_dot_grad_p, rRgu,
                Rij[d][0], Rij[d][1], Rij[d][2], gu[d][0], gu[d][1], gu[d][2]);
          }
          if (numvel > 1) { // Rxy (i=0, j=1)
            dvv(ccbulk::rho_reynolds_stress(3), kji) = bhr.Rij_source(
                Rij[0][1], a_comp[0], a_comp[1], dpdi[0], dpdi[1], rho, sqrtk, SD,
                Rij[0][0], Rij[0][1], Rij[0][2], Rij[1][0], Rij[1][1], Rij[1][2],
                gu[0][0], gu[0][1], gu[0][2], gu[1][0], gu[1][1], gu[1][2]);
          }
          if (numvel == 3) { // Rxz (i=0,j=2) and Ryz (i=1,j=2)
            for (int d = 0; d < 2; d++) {
              dvv(ccbulk::rho_reynolds_stress(4 + d), kji) = bhr.Rij_source(
                  Rij[d][2], a_comp[d], a_comp[2], dpdi[d], dpdi[2], rho, sqrtk, SD,
                  Rij[d][0], Rij[d][1], Rij[d][2], Rij[2][0], Rij[2][1], Rij[2][2],
                  gu[d][0], gu[d][1], gu[d][2], gu[2][0], gu[2][1], gu[2][2]);
            }
          }
        });
      });

  // Diffusive sources: the direction-separable part of BHR. Each active dimension adds
  // its own d/dx_DIR term onto the registers the algebraic pass above initialized.
  CalculateMixDiffSourceAlongDir<X1DIR>(state, src, mix_model);
  if (ndim > 1) CalculateMixDiffSourceAlongDir<X2DIR>(state, src, mix_model);
  if (ndim > 2) CalculateMixDiffSourceAlongDir<X3DIR>(state, src, mix_model);

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Mix::ComputeViscousFluxesAlongDir
//! \brief Single-direction diffusive mass and energy fluxes for sweep direction DIR. The
//! face flux at kji, F = c * 0.5(a(kji) + a(kL)) * (q(kL) - q(kji))/dh, reads cell kji
//! and its lower neighbor kL = kji - delta. The turbulent viscosity mu_t and turbulent
//! kinetic energy are produced once per cell into halo scratch (reused by the faces on
//! both sides), then the energy flux accumulates the TKE contribution first, followed by
//! the per-material enthalpy contribution in material order. The diffusive material mass
//! flux is deposited on the fm::diffusive_fluxes face state field (consumed by
//! ComputeAnonFluxes) and the enthalpy uses the intrinsic (per-material-volume) density.
//! Adds to the total energy flux register hydro already wrote (hence +=).
template <parthenon::CoordinateDirection DIR, typename VPack>
void ComputeViscousFluxesAlongDir(MeshData<Real> *md, const VPack &v,
                                  ParArray1D<MixModel::BHR3_1> mix_model) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat; // partial (conserved) density
  namespace cm = cell_variables::material_averaged;     // intrinsic density, p, sie
  namespace fm = face_variables::mat;
  using TE = parthenon::TopologicalElement;

  const int nblocks = v.GetNBlocks();
  if (nblocks == 0) return;

  using lt = RiotUtils::LoopType<>;
  constexpr TE face = (DIR == X1DIR) ? TE::F1 : (DIR == X2DIR ? TE::F2 : TE::F3);
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, md, face);

  // Single-offset {-1, 0} halo: the flux at kji reads only cell kji and cell kji - delta.
  using halo = std::tuple_element_t<
      DIR - 1, std::tuple<RiotLoop::halo::minus_i_t, RiotLoop::halo::minus_j_t,
                          RiotLoop::halo::minus_k_t>>;
  RiotLoop::AddPerPointScratch<Real, halo>(idx_space, 4); // tke, mu_t, mass_fraction, Q
  const auto delta = idx_space.GetDelta(DIR);

  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const int nmat = v.GetSize(b, ccmat::rho());
        auto &coords = v.GetCoordinates(b);
        const Real dh = coords.template Dxc<DIR>(0);
        auto &bhr = mix_model(0);

        auto halo_range = idx_range.template AddHalo<halo>();
        auto tke = RiotLoop::GetPerPointScratch<Real>(halo_range);
        auto mu_t = RiotLoop::GetPerPointScratch<Real>(halo_range);
        auto mass_fraction = RiotLoop::GetPerPointScratch<Real>(halo_range);
        auto Q = RiotLoop::GetPerPointScratch<Real>(halo_range);

        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto fv = RiotLoop::make_flux_pack_view(idx_range, v, DIR);

        // F(kji) += (q(kL) - q(kji)) / dh * 0.5 (a(kji) + a(kL)) * c, where kL = kji -
        // delta is the lower neighbor across the face at kji.
        auto diff_flux = [&](auto &qbuf, auto &abuf, const Real c, auto &&emit) {
          RiotLoop::inner(idx_range, [&](const auto kji) {
            const auto kL = kji - delta;
            emit(kji) += (qbuf(kL) - qbuf(kji)) / dh * 0.5 * (abuf(kji) + abuf(kL)) * c;
          });
        };

        // Produce tke and mu_t once per cell. mu_t goes through Mix::Mut, which
        // recomputes tke internally so its std::sqrt intermediate is rounded
        // independently of the tke buffer stored here for the energy flux.
        RiotLoop::inner(halo_range, [&](const auto kji) {
          tke(kji) = bhr.tke(pv(ccbulk::reynolds_stress(0), kji),
                             pv(ccbulk::reynolds_stress(1), kji),
                             pv(ccbulk::reynolds_stress(2), kji));
          mu_t(kji) = Mut(pv, bhr, kji);
        });
        idx_range.TeamBarrier();

        // TKE component of the energy flux (added first).
        diff_flux(tke, mu_t, 1.0 / bhr.sigma_k(), [&](const auto kji) -> Real & {
          return fv(ccbulk::total_material_energy(), kji);
        });

        for (int m = 0; m < nmat; m++) {
          // Per-material (sparse) fields are read through a sparse pack view bound to
          // material slot m; the bulk (non-sparse) fields stay on pv.
          auto pvm = RiotLoop::make_sparse_pack_view(idx_range, v, m);

          // Mass fraction = partial density / bulk density (assumes bulk density > 0).
          RiotLoop::inner(halo_range, [&](const auto kji) {
            mass_fraction(kji) = pvm(ccmat::rho(), kji) / pv(ccbulk::rho(), kji);
          });
          idx_range.TeamBarrier();

          // Diffusive material mass flux -> fm::diffusive_fluxes face state field.
          diff_flux(mass_fraction, mu_t, 1.0 / bhr.sigma_c(),
                    [&](const auto kji) -> Real & {
                      return pvm(face, fm::diffusive_fluxes(), kji);
                    });
          idx_range.TeamBarrier();

          // Specific enthalpy h = sie + p/rho uses the intrinsic density.
          RiotLoop::inner(halo_range, [&](const auto kji) {
            const Real rho_int = pvm(cm::rho(), kji);
            Q(kji) = rho_int > 0.0 ? mu_t(kji) * (pvm(cm::sie(), kji) +
                                                  pvm(cm::pressure(), kji) / rho_int)
                                   : 0.0;
          });
          idx_range.TeamBarrier();

          // Enthalpy component of the energy flux: the diffused quantity is the mass
          // fraction and the face coefficient is Q (which already carries mu_t).
          diff_flux(mass_fraction, Q, 1.0 / bhr.sigma_c(), [&](const auto kji) -> Real & {
            return fv(ccbulk::total_material_energy(), kji);
          });
          idx_range.TeamBarrier();
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn  void Mix::ComputeViscousFluxes
//  \brief Calculates the fluxes of the diffusive source terms of BHR mix (mass and energy
//  diffusion)

TaskStatus ComputeViscousFluxes(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat; // partial (conserved) density
  namespace cm = cell_variables::material_averaged;     // intrinsic density, p, sie
  namespace fm = face_variables::mat;

  auto pm = md->GetParentPointer();
  auto &mix = pm->packages.Get("mix");
  auto mix_model = mix->Param<ParArray1D<MixModel::BHR3_1>>("bhr_d");

  std::vector<int> matids;
  std::set<parthenon::PDOpt> opts = {parthenon::PDOpt::WithFluxes};
  // The diffusive mass flux is deposited on fm::diffusive_fluxes (a face field);
  // ComputeAnonFluxes consumes it. cm::pressure/cm::sie/cm::rho are the intrinsic
  // (per-material-volume) thermodynamic quantities; ccmat::rho is the partial
  // (per-cell-volume) conserved density used to form mass fractions.
  auto v = riot::MakePack<ccbulk::rho, ccmat::rho, cm::rho, cm::pressure, cm::sie,
                          ccbulk::reynolds_stress, ccbulk::bhr_ST,
                          ccbulk::total_material_energy, fm::diffusive_fluxes>(md, matids,
                                                                               opts);

  if (v.GetNBlocks() == 0) return TaskStatus::complete;
  const int ndim = pm->ndim;

  ComputeViscousFluxesAlongDir<X1DIR>(md, v, mix_model);
  if (ndim > 1) ComputeViscousFluxesAlongDir<X2DIR>(md, v, mix_model);
  if (ndim > 2) ComputeViscousFluxesAlongDir<X3DIR>(md, v, mix_model);

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Mix::ComputeStressFluxesAlongDir
//! \brief Single-direction non-diffusive flux (Reynolds-stress and a-term momentum and
//! energy fluxes) for sweep direction DIR. The face at cell index kji sits between its
//! lower neighbor kji - delta and cell kji; the flux is the simple face average
//! 0.5 (F(kji - delta) + F(kji)). The per-cell integrands rho*(v.R_row) - P*a (energy)
//! and rho*R_row[c] (momentum c) are produced once per cell into halo scratch (so each is
//! reused by the faces on both sides) and then averaged. R_row is the DIR-normal row of
//! the symmetric Reynolds stress; the a and momentum components follow the same normal
//! direction. This adds to the flux registers hydro already wrote (hence +=).
template <parthenon::CoordinateDirection DIR, typename VPack, typename FPack>
void ComputeStressFluxesAlongDir(MeshData<Real> *md, const VPack &v, const FPack &f) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  using TE = parthenon::TopologicalElement;

  const int nblocks = v.GetNBlocks();
  if (nblocks == 0) return;

  using lt = RiotUtils::LoopType<>;
  constexpr TE face = (DIR == X1DIR) ? TE::F1 : (DIR == X2DIR ? TE::F2 : TE::F3);
  auto idx_space = lt::GetIndexSpace(IndexDomain::interior, 0, nblocks, md, face);

  // Single-offset {-1, 0} halo: the flux at kji reads only cell kji and cell kji - delta.
  using halo = std::tuple_element_t<
      DIR - 1, std::tuple<RiotLoop::halo::minus_i_t, RiotLoop::halo::minus_j_t,
                          RiotLoop::halo::minus_k_t>>;
  // One integrand buffer for the energy flux plus one per momentum component.
  RiotLoop::AddPerPointScratch<Real, halo>(idx_space, 4);
  const auto delta = idx_space.GetDelta(DIR);

  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto halo_range = idx_range.template AddHalo<halo>();
        auto Qe = RiotLoop::GetPerPointScratch<Real>(halo_range);  // energy integrand
        auto Qm0 = RiotLoop::GetPerPointScratch<Real>(halo_range); // momentum(0)
        auto Qm1 = RiotLoop::GetPerPointScratch<Real>(halo_range); // momentum(1)
        auto Qm2 = RiotLoop::GetPerPointScratch<Real>(halo_range); // momentum(2)

        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto fv = RiotLoop::make_flux_pack_view(idx_range, f, DIR);

        // Produce the per-cell integrands over the halo range (covers kji and kji-delta).
        RiotLoop::inner(halo_range, [&](const auto kji) {
          // Component indices into reynolds_stress {Rxx,Ryy,Rzz,Rxy,Rxz,Ryz} for the
          // DIR-normal row nc (= X1->0, X2->1, X3->2): diagonal + two off-diagonals.
          constexpr int nc = static_cast<int>(DIR) - 1;
          constexpr int c0 = (nc == 0) ? 0 : (nc == 1 ? 3 : 4);
          constexpr int c1 = (nc == 0) ? 3 : (nc == 1 ? 1 : 5);
          constexpr int c2 = (nc == 0) ? 4 : (nc == 1 ? 5 : 2);

          const Real rho = pv(ccbulk::rho(), kji);
          const Real Rn0 = pv(ccbulk::reynolds_stress(c0), kji);
          const Real Rn1 = pv(ccbulk::reynolds_stress(c1), kji);
          const Real Rn2 = pv(ccbulk::reynolds_stress(c2), kji);
          const Real vx = pv(ccbulk::velocity(0), kji);
          const Real vy = pv(ccbulk::velocity(1), kji);
          const Real vz = pv(ccbulk::velocity(2), kji);
          const Real P = pv(ccbulk::pressure(), kji);
          const Real an = pv(ccbulk::bhr_a(nc), kji);

          Qe(kji) = rho * (vx * Rn0 + vy * Rn1 + vz * Rn2) - P * an;
          Qm0(kji) = rho * Rn0;
          Qm1(kji) = rho * Rn1;
          Qm2(kji) = rho * Rn2;
        });
        idx_range.TeamBarrier();

        // Face-average the integrands onto the flux registers (energy first, then
        // momentum in component order).
        RiotLoop::inner(idx_range, [&](const auto kji) {
          const auto kL = kji - delta;
          fv(ccbulk::total_material_energy(), kji) += (Qe(kji) + Qe(kL)) * 0.5;
          fv(ccbulk::momentum(0), kji) += (Qm0(kji) + Qm0(kL)) * 0.5;
          fv(ccbulk::momentum(1), kji) += (Qm1(kji) + Qm1(kL)) * 0.5;
          fv(ccbulk::momentum(2), kji) += (Qm2(kji) + Qm2(kL)) * 0.5;
        });
      });
}

//----------------------------------------------------------------------------------------
//! \fn  void Mix::ComputeStressFluxes
//! \brief Computes the non-diffusive fluxes when BHR is present due to the Reynolds
//! stress and a terms (momentum and energy terms)
TaskStatus ComputeStressFluxes(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;

  auto v = riot::MakePack<ccbulk::rho, ccbulk::pressure, ccbulk::reynolds_stress,
                          ccbulk::bhr_a, ccbulk::velocity>(md);
  auto f = riot::MakePack<ccbulk::momentum, ccbulk::total_material_energy>(
      md, std::vector<int>{}, std::set<parthenon::PDOpt>{parthenon::PDOpt::WithFluxes});

  if (v.GetNBlocks() == 0) return TaskStatus::complete;
  const int ndim = md->GetParentPointer()->ndim;

  ComputeStressFluxesAlongDir<X1DIR>(md, v, f);
  if (ndim > 1) ComputeStressFluxesAlongDir<X2DIR>(md, v, f);
  if (ndim > 2) ComputeStressFluxesAlongDir<X3DIR>(md, v, f);

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Mix::EstimateTimestepMesh
//! \brief Computes the timestep restrictions due to mix.  The diffusive timestep is
//! basically the diffusion coefficient (mu_t/sigma).  There are also
//! advective-like timesteps (speed of a) and 0-d timesteps (exponential decay).
Real EstimateTimestepMesh(MeshData<Real> *rc) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;

  auto pm = rc->GetParentPointer();
  auto &mix = pm->packages.Get("mix");
  auto bhr_d = mix->Param<ParArray1D<MixModel::BHR3_1>>("bhr_d");

  auto v = riot::MakePack<ccbulk::rho, ccbulk::bhr_a, ccbulk::bhr_ST, ccbulk::bhr_SD,
                          ccbulk::reynolds_stress>(rc);

  const int ndim = pm->ndim;
  const int numvel = 3;

  using TE = parthenon::TopologicalElement;
  using rt = RiotUtils::ReductionType<Kokkos::Min<Real>>;
  auto idx_space =
      rt::GetIndexSpace(IndexDomain::interior, 0, v.GetNBlocks(), rc, TE::CC);
  return RiotLoop::outer_reduce(
      idx_space, KOKKOS_LAMBDA(const rt::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        auto &coords = v.GetCoordinates(b);
        auto &bhr = bhr_d(0);
        RiotLoop::inner_reduce(idx_range, [&](const auto idx, Real &ldt) {
          const auto [k, j, i] = idx_range.GetKJI(idx);

          // hyperbolic (advective-like) dt; build inverse first
          Real dth = std::abs(pv(ccbulk::bhr_a(0), idx)) / coords.Dxc<X1DIR>(i);
          // TODO(JCD): is the Dx below correct?
          for (int d = 1; d < numvel; d++) {
            dth = std::max(dth, std::abs(pv(ccbulk::bhr_a(d), idx)) / coords.Dx(d));
          }
          dth = (dth > 0.0 ? 1.0 / (dth * ndim) : 1e30);

          // parabolic dt
          Real dxmin = std::min(coords.Dxc<X1DIR>(0),
                                std::min(coords.Dxc<X2DIR>(0), coords.Dxc<X3DIR>(0)));
          const Real tke = bhr.tke(pv(ccbulk::reynolds_stress(0), idx),
                                   pv(ccbulk::reynolds_stress(1), idx),
                                   pv(ccbulk::reynolds_stress(2), idx));
          const Real sqrt_tke = std::sqrt(tke);
          Real dtp =
              bhr.mut(pv(ccbulk::rho(0), idx), pv(ccbulk::bhr_ST(0), idx), sqrt_tke);
          dtp = (dtp > 0.0 ? bhr.sigma_min() * dxmin * dxmin / (2.0 * ndim * dtp) : 1e30);

          // Now 0D terms
          // This is the exponential term (driving to isotropy)
          const Real &SD = pv(ccbulk::bhr_SD(0), idx);
          Real dt0 = bhr.c_max() * sqrt_tke * (SD > 0.0 ? 1.0 / SD : 0.0);
          dt0 = (dt0 > 0.0 ? 0.95 / dt0 : 1e30);

          ldt = std::min(ldt, std::min(dtp, std::min(dt0, dth)));
        });
      });
}

//----------------------------------------------------------------------------------------
//! \fn  void Mix::FillDerived
//! \brief Fills derived variables for BHR
void FillDerived(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;

  if (md->NumBlocks() == 0) return;
  auto pm = md->GetParentPointer();
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto &mix = pmb->packages.Get("mix");
  auto bhr_d = mix->Param<ParArray1D<MixModel::BHR3_1>>("bhr_d");

  auto v = riot::MakePack<ccbulk::rho, ccbulk::reynolds_stress, ccbulk::bhr_a,
                          ccbulk::bhr_b, ccbulk::bhr_SD, ccbulk::bhr_ST,
                          ccbulk::rho_reynolds_stress, ccbulk::rho_bhr_a,
                          ccbulk::rho_bhr_b, ccbulk::rho_bhr_SD, ccbulk::rho_bhr_ST>(md);

  using lt = RiotUtils::LoopType<>;
  auto idx_space = lt::GetIndexSpace(IndexDomain::entire, 0, v.GetNBlocks(), md,
                                     parthenon::TopologicalElement::CC);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        for (int f = 0; f < v.GetSize(b, ccbulk::reynolds_stress()); f++) {
          // NOTE(@chadmeyer): only the first three components should be potentially
          // floored;
          if (f < 3) {
            const Real tke = bhr_d[0].tke_0();
            RiotLoop::inner(idx_range, [&](const auto kji) {
              const Real ccbulk_rho = pv(ccbulk::rho(), kji);
              pv(ccbulk::rho_reynolds_stress(f), kji) =
                  (pv(ccbulk::rho_reynolds_stress(f), kji) != 0.0
                       ? std::max(tke * 1.0e-6 * 2.0 / 3.0 * ccbulk_rho,
                                  pv(ccbulk::rho_reynolds_stress(f), kji))
                       : 0.0);
            });
          }
          RiotLoop::inner(idx_range, [&](const auto kji) {
            const Real ccbulk_rho = pv(ccbulk::rho(), kji);
            const Real u2p = ((ccbulk_rho > 0.0) / (ccbulk_rho + (ccbulk_rho <= 0.0)));
            pv(ccbulk::reynolds_stress(f), kji) =
                pv(ccbulk::rho_reynolds_stress(f), kji) * u2p;
          });
        }
        for (int f = 0; f < v.GetSize(b, ccbulk::bhr_a()); f++) {
          RiotLoop::inner(idx_range, [&](const auto kji) {
            const Real ccbulk_rho = pv(ccbulk::rho(), kji);
            const Real u2p = ((ccbulk_rho > 0.0) / (ccbulk_rho + (ccbulk_rho <= 0.0)));
            pv(ccbulk::bhr_a(f), kji) = pv(ccbulk::rho_bhr_a(f), kji) * u2p;
          });
        }
        RiotLoop::inner(idx_range, [&](const auto kji) {
          const Real ccbulk_rho = pv(ccbulk::rho(), kji);
          const Real u2p = ((ccbulk_rho > 0.0) / (ccbulk_rho + (ccbulk_rho <= 0.0)));
          pv(ccbulk::rho_bhr_SD(), kji) = std::max(0.0, pv(ccbulk::rho_bhr_SD(), kji));
          pv(ccbulk::rho_bhr_ST(), kji) = std::max(0.0, pv(ccbulk::rho_bhr_ST(), kji));
          pv(ccbulk::rho_bhr_b(), kji) =
              std::max(0.0, std::min(ccbulk_rho, pv(ccbulk::rho_bhr_b(), kji)));
          // we should consider disabling
          pv(ccbulk::rho_bhr_b(), kji) *=
              (pv(ccbulk::rho_bhr_b(), kji) >= 1.0e-6 * ccbulk_rho);
          pv(ccbulk::bhr_b(), kji) = pv(ccbulk::rho_bhr_b(), kji) * u2p;
          pv(ccbulk::bhr_SD(), kji) = pv(ccbulk::rho_bhr_SD(), kji) * u2p;
          pv(ccbulk::bhr_ST(), kji) = pv(ccbulk::rho_bhr_ST(), kji) * u2p;
        });
      });
}
} // namespace Mix

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
#ifndef VARIABLES_HPP_
#define VARIABLES_HPP_
// This file was made in part with generative AI.

#include <map>
#include <set>
#include <string>
#include <vector>

#include <pack/sparse_pack/sparse_pack.hpp>
#include <parthenon/package.hpp>
#include <utils/type_list.hpp>

using namespace parthenon::package::prelude;

#define CELL_DELTA_SPARSEID 999999
#define XSTR(s) #s
#define STR(s) XSTR(s)
#define CELL_DELTA_NAME "c.c.cell_delta_" STR(CELL_DELTA_SPARSEID)
constexpr int cell_delta_id = CELL_DELTA_SPARSEID;
constexpr char block_active_flag[] = CELL_DELTA_NAME;
#undef CELL_DELTA_SPARSEID
#undef XSTR
#undef STR
#undef CELL_DELTA_NAME

#define VARIABLE_SCALAR(ns, varname, sparse)                                             \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
    static constexpr bool is_sparse() { return sparse; }                                 \
  }

#define VARIABLE_VECTOR(ns, varname, sparse, ncomp)                                      \
  struct varname : public parthenon::variable_names::base_t<false, ncomp> {              \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false, ncomp>(std::forward<Ts>(args)...) {}  \
    static std::string name() { return #ns "." #varname; }                               \
    static constexpr bool is_sparse() { return sparse; }                                 \
  }

#define VARIABLE_TENSOR(ns, varname, sparse, ncomp1, ncomp2)                             \
  struct varname : public parthenon::variable_names::base_t<false, ncomp1, ncomp2> {     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false, ncomp1, ncomp2>(                      \
              std::forward<Ts>(args)...) {}                                              \
    static std::string name() { return #ns "." #varname; }                               \
    static constexpr bool is_sparse() { return sparse; }                                 \
  }

#define VARIABLE_FACE(ns, varname, sparse)                                               \
  struct varname : public parthenon::variable_names::base_w_tt_t<                        \
                       false, parthenon::TopologicalType::Face> {                        \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_w_tt_t<false,                                  \
                                                 parthenon::TopologicalType::Face>(      \
              std::forward<Ts>(args)...) {}                                              \
    static std::string name() { return #ns "." #varname; }                               \
    static constexpr bool is_sparse() { return sparse; }                                 \
  }

#define PARTICLE_VARIABLE(type, ns, varname)                                             \
  struct varname : public parthenon::swarm_variable_names::base_t<type> {                \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::swarm_variable_names::base_t<type>(std::forward<Ts>(args)...) {}    \
    static std::string name() { return #ns "." #varname; }                               \
  }

namespace cell_variables {
namespace cell_averaged {
namespace bulk {
VARIABLE_SCALAR(c.c.bulk, rho, false);
VARIABLE_SCALAR(c.c.bulk, total_material_energy, false);
VARIABLE_SCALAR(c.c.bulk, internal_energy, false);
VARIABLE_VECTOR(c.c.bulk, momentum, false, 3);
VARIABLE_VECTOR(c.c.bulk, velocity, false, 3);
VARIABLE_SCALAR(c.c.bulk, pressure, false);
VARIABLE_SCALAR(c.c.bulk, bulk_modulus, false);
VARIABLE_SCALAR(c.c.bulk, electron_bulk_modulus, false);
VARIABLE_SCALAR(c.c.bulk, electron_gruneisen_parameter, false);
VARIABLE_SCALAR(c.c.bulk, temperature, false);
VARIABLE_SCALAR(c.c.bulk, electron_pressure, false);
VARIABLE_SCALAR(c.c.bulk, electron_temperature, false);
VARIABLE_SCALAR(c.c.bulk, electron_internal_energy, false);
VARIABLE_SCALAR(c.c.bulk, electron_entropy, false);
VARIABLE_SCALAR(c.c.bulk, electron_number_density, false);
VARIABLE_SCALAR(c.c.bulk, ionization_zbar, false);
VARIABLE_FACE(c.c.bulk, face_signal, false);
VARIABLE_VECTOR(c.c.bulk, max_signal, false, 3);
VARIABLE_VECTOR(c.c.bulk, face_velocity, false, 9);
VARIABLE_VECTOR(c.c.bulk, strain_rate, false, 6);
VARIABLE_SCALAR(c.c.bulk, shear_modulus, false);
// NOTE(@chadmeyer): Distinguish b/w bulk prim/cons?
VARIABLE_VECTOR(c.c.bulk, reynolds_stress, false, 6);
VARIABLE_VECTOR(c.c.bulk, bhr_a, false, 3);
VARIABLE_SCALAR(c.c.bulk, bhr_b, false);
VARIABLE_SCALAR(c.c.bulk, bhr_ST, false);
VARIABLE_SCALAR(c.c.bulk, bhr_SD, false);
VARIABLE_VECTOR(c.c.bulk, rho_reynolds_stress, false, 6);
VARIABLE_VECTOR(c.c.bulk, rho_bhr_a, false, 3);
VARIABLE_SCALAR(c.c.bulk, rho_bhr_b, false);
VARIABLE_SCALAR(c.c.bulk, rho_bhr_ST, false);
VARIABLE_SCALAR(c.c.bulk, rho_bhr_SD, false);
VARIABLE_SCALAR(c.c.bulk, electron_thermal_conductivity_face, false);
VARIABLE_SCALAR(c.c.bulk, ion_shear_viscosity, false);
VARIABLE_SCALAR(c.c, cell_delta, false);
VARIABLE_SCALAR(c.c.bulk, laser_deposition, false);
VARIABLE_SCALAR(c.c.bulk, laser_energy_density, false);
VARIABLE_SCALAR(c.c.bulk, laser_tau_max, false);
} // namespace bulk

namespace mat {
VARIABLE_SCALAR(c.c.mat, rho, true);
VARIABLE_SCALAR(c.c.mat, phase_rho_sum, true);
VARIABLE_SCALAR(c.c.mat, internal_energy, true);
VARIABLE_SCALAR(c.c.mat, volume_fraction, true);
VARIABLE_SCALAR(c.c.mat, iso, true);
VARIABLE_SCALAR(c.c.mat, rho_sh, true);
VARIABLE_SCALAR(c.c.mat, tn_reaction_density, true);
VARIABLE_VECTOR(c.c.mat, deviatoric_stress, true, 5);
VARIABLE_SCALAR(c.c.mat, equivalent_plastic_strain, true);
VARIABLE_SCALAR(c.c.mat, ionization_zbar, true);
VARIABLE_SCALAR(c.c.mat, electron_internal_energy, true);
} // namespace mat

namespace rad {
// NOTE(): These are the multigroup radiation transport fields. Their true per-cell
// component count is the number of energy groups (or ngroups * the number of angles for
// the specific intensity).
//
// The `ncomp` template argument on VARIABLE_VECTOR below is therefore a PLACEHOLDER of 1
// and does NOT reflect the real component count. This is only safe because the transport
// loops do not yet invoke the parthenon::loop_abstraction framework (i.e., they invoke
// pure flat par_for loops)
VARIABLE_VECTOR(c.c.rad, intensity, false, 1); // ngroups * nangles
VARIABLE_VECTOR(c.c.rad, aa, false, 1);        // ngroups
VARIABLE_VECTOR(c.c.rad, ss, false, 1);        // ngroups
VARIABLE_VECTOR(c.c.rad, moments, false, 1);   // ngroups
VARIABLE_VECTOR(c.c.rad, s1, false, 1);        // ngroups
VARIABLE_VECTOR(c.c.rad, s2, false, 1);        // ngroups
VARIABLE_VECTOR(c.c.rad, s3, false, 1);        // ngroups
VARIABLE_VECTOR(c.c.rad, divfa, false, 1);     // ngroups
VARIABLE_VECTOR(c.c.rad, tauw, false, 1);      // ngroups
VARIABLE_SCALAR(c.c.rad, temperature, false);
} // namespace rad

} // namespace cell_averaged

namespace material_averaged {
VARIABLE_SCALAR(c.m, rho, true);
VARIABLE_SCALAR(c.m, phase_fraction, true);
VARIABLE_SCALAR(c.m, sie, true);
VARIABLE_SCALAR(c.m., specific_heat, true); // TODO(JMM): Electrons?
VARIABLE_SCALAR(c.m, electron_sie, true);
VARIABLE_SCALAR(c.m, temperature, true);
VARIABLE_SCALAR(c.m, pressure, true);
VARIABLE_SCALAR(c.m, bulk_modulus, true);
VARIABLE_SCALAR(c.m, iso, true);
VARIABLE_SCALAR(c.m, rho_sh, true);
VARIABLE_SCALAR(c.m, he_psi, true);
VARIABLE_SCALAR(c.m, awsd_zeta, true);
VARIABLE_SCALAR(c.m, awsd_tsh, true);
VARIABLE_VECTOR(c.m, deviatoric_stress, true, 5);
VARIABLE_SCALAR(c.m, equivalent_plastic_strain, true);
VARIABLE_SCALAR(c.m, shear_modulus, true);
VARIABLE_SCALAR(c.m, strength_j2, true);
VARIABLE_SCALAR(c.m, tn_specific_reactions, true);
VARIABLE_SCALAR(c.m, ionization_zbar, true);
// JMM: Used for initial guesses for EOS root finding.
VARIABLE_SCALAR(c.m, lT_cache, true);
VARIABLE_SCALAR(c.m, lr_cache, true);
} // namespace material_averaged
} // namespace cell_variables

// Face-centered, material-averaged quantities. Presently only the diffusive
// mass flux register used by the BHR mix model lives here.
namespace face_variables {
namespace mat {
VARIABLE_FACE(f.m, diffusive_fluxes, true);
} // namespace mat
} // namespace face_variables
namespace node_variables {
VARIABLE_SCALAR(n, electron_number_density, false);
} // namespace node_variables

namespace particles {
namespace laser {
// this is not a scalar variable, it's the swarm variable
VARIABLE_SCALAR(p.l, particles, false);
PARTICLE_VARIABLE(Real, p.l, t);
PARTICLE_VARIABLE(Real, p.l, energy);
PARTICLE_VARIABLE(Real, p.l, vx);
PARTICLE_VARIABLE(Real, p.l, vy);
PARTICLE_VARIABLE(Real, p.l, vz);
PARTICLE_VARIABLE(Real, p.l, wavelength);
} // namespace laser
} // namespace particles

// PTE diagnostics
namespace diag {
VARIABLE_SCALAR(diag, pte_niter, false);
VARIABLE_SCALAR(diag, pte_nfails, false);
VARIABLE_SCALAR(diag, pte_nbackups, false);
VARIABLE_SCALAR(diag, pte_ncalls, false);
VARIABLE_SCALAR(diag, pte_avg_niter, false);
VARIABLE_SCALAR(diag, pte_avg_nbackups, false);
VARIABLE_SCALAR(diag, pte_fail_fraction, false);
} // namespace diag

namespace RadiationDiffusion {
namespace MultiGroupVars {
// NOTE(): These are the multigroup radiation-diffusion fields. Their true
// per-cell component count is the number of energy groups, which is a RUNTIME parameter
// ("ngroup"), not a compile-time constant. The actual storage is sized at field
// registration time via the Metadata shape {ngroup} (see multigroup_diffusion-package
// .cpp); the group index is walked at runtime as `Egroup(g)` for g in [0, ngroup).
//
// The `ncomp` template argument on VARIABLE_VECTOR below is therefore a PLACEHOLDER of 1
// and does NOT reflect the real component count. This is only safe because the P1 loops
// access these fields by DIRECT pack indexing -- `pack(b, Egroup(g), k, j, i)` -- inside
// RiotLoop::inner, and never build a pack_view / sparse_pack_view over them. The loop
// abstraction's pack views bake `var_t::size()` in at compile time (see
// loop_abstraction/pack_view.hpp), so a runtime component count is fundamentally
// incompatible with them; the direct-indexing form used by conduction_equation.hpp is
// the escape hatch we follow here.
VARIABLE_VECTOR(rmg, Egroup, false, 1);     // ngroups (see NOTE above; runtime-sized)
VARIABLE_FACE(rmg, Fgroup, false);          // ngroups
VARIABLE_FACE(rmg, D, false);               // ngroups
VARIABLE_VECTOR(rmg, kappa_cell, false, 1); // ngroups
VARIABLE_FACE(rmg, kappa_face, false);      // ngroups
VARIABLE_VECTOR(rmg, diag_loc, false, 1);   // ngroups
VARIABLE_VECTOR(rmg, sigma, false, 1);      // ngroups
VARIABLE_VECTOR(rmg, dSdT, false, 1);       // ngroups
VARIABLE_SCALAR(rmg, dTc, false);
VARIABLE_SCALAR(rmg, temperature0, false);
VARIABLE_FACE(rmg, face_area, false);
VARIABLE_SCALAR(rmg, volume, false);
VARIABLE_FACE(rmg, DeltaX, false);
} // namespace MultiGroupVars
} // namespace RadiationDiffusion

namespace Ionization {
VARIABLE_FACE(ion, D, false);
VARIABLE_SCALAR(ion, Dcell, false);
VARIABLE_SCALAR(ion, diag_loc, false);
VARIABLE_SCALAR(ion, delta, false);

VARIABLE_SCALAR(ion, temperature_old, false);
VARIABLE_SCALAR(ion, temp_tstep_criterion, false);
} // namespace Ionization

namespace Levelsets {
VARIABLE_SCALAR(levelset, levelset, false);
VARIABLE_SCALAR(levelset, levelset0, false);
VARIABLE_SCALAR(levelset, dudt_reinitialize, false);
} // namespace Levelsets

// JMM: We want to expose these so plugins can use them. Don't undef.
// TODO: If not undefing these causes problems, think of another
// solution
/*
#undef VARIABLE_SCALAR
#undef VARIABLE_VECTOR
#undef VARIABLE_TENSOR
#undef VARIABLE_FACE
*/

namespace riot {

#define DNAME(name) static constexpr char name[] = #name
namespace container_names {
DNAME(u0);
DNAME(u1);
} // namespace container_names
#undef DNAME

namespace metadata {
constexpr char OperatorSplit[] = "OperatorSplit";

// Some variables need to be tagged with the package they are relevant
// for for the purpose of anonymous advection.
constexpr char TNBurn[] = "TNBurn";
} // namespace metadata

//----------------------------------------------------------------------------------------
//! \fn  auto riot::MakePackDescriptor
//! \brief matid overload
template <class... Ts, typename Data_t>
inline auto MakePackDescriptor(const Data_t &data, const std::vector<int> &matids = {},
                               const std::set<parthenon::PDOpt> &options = {}) {
  parthenon::StateDescriptor *psd;
  if constexpr (std::is_same<Data_t, parthenon::StateDescriptor *>::value) {
    psd = data;
  } else {
    Mesh *pm = data->GetMeshPointer();
    psd = (pm->resolved_packages).get();
  }
  static_assert(sizeof...(Ts) > 0, "Must have at least one variable type for type pack");
  if (matids.size() == 0) {
    return parthenon::MakePackDescriptor<Ts...>(
        psd, std::vector<parthenon::MetadataFlag>{}, options);
  }
  std::vector<std::string> vars{Ts::name()...};
  std::set<int> matid_set(matids.begin(), matids.end());
  auto selector = [&](int vidx, const parthenon::VarID &id, const Metadata &md) {
    return ((vars[vidx] == id.label()) ||
            ((vars[vidx] == id.base_name) && matid_set.count(id.sparse_id)));
  };
  parthenon::impl::PackDescriptor base_desc(psd, vars, selector, options);
  return typename parthenon::SparsePack<Ts...>::Descriptor(base_desc);
}

//----------------------------------------------------------------------------------------
//! \fn  auto riot::MakePackDescriptor
//! \brief Used in regions pgen
template <class... Ts>
inline auto MakePackDescriptor(parthenon::SparsePack<Ts...> pack,
                               parthenon::StateDescriptor *psd) {
  return parthenon::MakePackDescriptor<Ts...>(psd);
}

//----------------------------------------------------------------------------------------
//! \fn  auto riot::MakePackDescriptor
//! \brief vverload that takes metadata flags.  note: required <any>
template <typename T>
inline auto MakePackDescriptor(const std::vector<parthenon::MetadataFlag> flags, T *md) {
  using parthenon::variable_names::any;
  auto pm = md->GetParentPointer();
  StateDescriptor *resolved_pkgs = (pm->resolved_packages).get();
  return parthenon::MakePackDescriptor<any>(resolved_pkgs, flags);
}

//----------------------------------------------------------------------------------------
//! \fn  auto riot::MakePackDescriptor
//! \brief overload for string and uid.  note: no template <any>
template <typename F, typename T>
inline auto MakePackDescriptor(const std::vector<F> &flags, T *md) {
  using parthenon::variable_names::any;
  auto pm = md->GetParentPointer();
  StateDescriptor *resolved_pkgs = (pm->resolved_packages).get();
  return parthenon::MakePackDescriptor(resolved_pkgs, flags);
}

//----------------------------------------------------------------------------------------
//! \fn  auto riot::GetPack
//! \brief
template <typename Desc_t, typename Data_t>
inline auto GetPack(const Desc_t &desc, const Data_t &data) {
  bool has_cell_active_field = false;
  bool set = false;
  if constexpr (std::is_same<Data_t, MeshData<Real> *>::value) {
    has_cell_active_field = data->GetBlockData(0)->HasVariable(block_active_flag);
    set = true;
  }
  if constexpr (std::is_same<Data_t, MeshBlockData<Real> *>::value) {
    has_cell_active_field = data->HasVariable(block_active_flag);
    set = true;
  }
  PARTHENON_REQUIRE(set, "Passed an invalid type to riot::GetPack");
  if (has_cell_active_field) {
    auto include_blocks = data->AllocationStatus(block_active_flag);
    return desc.GetPack(data, include_blocks);
  } else {
    return desc.GetPack(data);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  auto riot::MakePack
//! \brief Make a static pack descriptor internally.
template <typename... Vars, typename Data_t, typename... Args>
inline auto MakePack(const Data_t &data, const Args &...args) {
  using desc_t = typename parthenon::SparsePack<Vars...>::Descriptor;
  static std::map<std::tuple<Args...>, desc_t> m;

  Mesh *pm = data->GetMeshPointer();
  StateDescriptor *resolved_pkgs = (pm->resolved_packages).get();

  // JMM: For whatever reason the compiler doesn't like it->second
  // For it = std::find(...) in the case where args... is empty
  auto key = std::make_tuple(args...);
  if (!m.count(key)) {
    m.emplace(key, MakePackDescriptor<Vars...>(resolved_pkgs, args...));
  }
  return riot::GetPack(m.at(key), data);
}

} // namespace riot

#endif // VARIABLES_HPP_

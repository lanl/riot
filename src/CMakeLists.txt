# ========================================================================================
#  (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001 for Los
#  Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
#  for the U.S. Department of Energy/National Nuclear Security Administration. All rights
#  in the program are reserved by Triad National Security, LLC, and the U.S. Department
#  of Energy/National Nuclear Security Administration. The Government is granted for
#  itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
#  license in this material to reproduce, prepare derivative works, distribute copies to
#  the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

# This file was modified with the assistance of generative AI

set (SRC_LIST
  # alphabetical

  diagnostics/diagnostics.cpp
  diagnostics/diagnostics.hpp
  diagnostics/doubleshell.cpp
  diagnostics/dsplanar.cpp
  diagnostics/energies.cpp
  diagnostics/masses.cpp
  diagnostics/masses.hpp

  gravity/gravity.cpp
  gravity/gravity.hpp

  hydro/advection.hpp
  hydro/calculate_fluxes.cpp
  hydro/hydro.cpp
  hydro/hydro.hpp
  hydro/riemann.hpp

  laser/laser.cpp
  laser/laser.hpp
  laser/laser_props.cpp
  laser/laser_props.hpp

  levelsets/levelsets.hpp
  levelsets/levelsets.cpp

  ionization/ionization_base.hpp
  ionization/ionization.cpp
  ionization/ionization.hpp

  materials/materials.cpp
  materials/materials.hpp

  microphysics/eos_riot.cpp
  microphysics/eos_riot.hpp
  microphysics/opacity_models.hpp
  microphysics/pte_closure.hpp
  microphysics/pte_closure_general.cpp
  microphysics/pte_closure_ideal.cpp
  microphysics/strength_models.hpp

  mix/anon_fluxes.cpp
  mix/mix.cpp
  mix/mix.hpp
  mix/bhr3/bhr3.1.hpp

  multiphysics/fill_shared_derived.cpp
  multiphysics/fill_shared_derived.hpp

  prescribed_sources/prescribed_sources.cpp
  prescribed_sources/prescribed_sources.hpp
  plugins.hpp

  radiation_diffusion/multigroup_diffusion.hpp
  radiation_diffusion/multigroup_diffusion-package.cpp 
  radiation_diffusion/multigroup_diffusion-solver.cpp
  radiation_diffusion/multigroup_diffusion-tasks.cpp
  radiation_diffusion/multigroup_diffusion-tasks-init.cpp
  radiation_diffusion/material_helpers.hpp
  radiation_diffusion/polylog.hpp

  radiation_transport/algorithms/explicit.cpp
  radiation_transport/algorithms/explicit.hpp
  radiation_transport/algorithms/jacobi.cpp
  radiation_transport/algorithms/jacobi.hpp
  radiation_transport/algorithms/radiation_shared.hpp
  radiation_transport/algorithms/radiation_init.hpp
  radiation_transport/algorithms/radiation_bcs.hpp
  radiation_transport/angular_grids/angular_grid_utils.hpp
  radiation_transport/angular_grids/geodesic_grid.cpp
  radiation_transport/angular_grids/geodesic_grid.hpp
  radiation_transport/angular_grids/latlon_grid.cpp
  radiation_transport/angular_grids/latlon_grid.cpp
  radiation_transport/transport_utils/transport_utils.hpp

  riot_pgen/conduction_analytic.cpp
  riot_pgen/conduction_analytic.hpp
  riot_pgen/ei_relax.cpp
  riot_pgen/ei_relax.hpp
  riot_pgen/gacc.cpp
  riot_pgen/gaussian_conduction.cpp
  riot_pgen/gaussian_conduction.hpp
  riot_pgen/gresho.cpp
  riot_pgen/hse.cpp
  riot_pgen/ionized_shocktube_analytic.cpp
  riot_pgen/linear_modes.hpp
  riot_pgen/linear_modes.cpp
  riot_pgen/pgen.hpp
  riot_pgen/pgen.cpp
  riot_pgen/pipe.cpp
  riot_pgen/pipe.hpp
  riot_pgen/rad_shock.cpp
  riot_pgen/rad_shock.hpp
  riot_pgen/quirk.cpp
  riot_pgen/regions.cpp
  riot_pgen/regions.hpp
  riot_pgen/region_pgen.cpp
  riot_pgen/region_primitives.cpp
  riot_pgen/region_primitives.hpp
  riot_pgen/region_python.cpp
  riot_pgen/region_python.hpp
  riot_pgen/rm.cpp
  riot_pgen/rm.hpp
  riot_pgen/rt.cpp
  riot_pgen/shock_tube.cpp
  riot_pgen/taylor_green.cpp
  riot_pgen/tn_test.cpp

  reconstruction/reconstruction.hpp

  riot.cpp
  riot_driver.cpp
  riot_driver.hpp
  riot_limits.hpp.in

  riot_utils/sparse_update.hpp
  riot_utils/riot_loops.hpp
  riot_utils/riot_utils.hpp
  riot_utils/riot_utils.cpp
  riot_utils/table_utils.cpp
  riot_utils/table_utils.hpp
  riot_utils/type_maps.hpp

  scalars/scalars.cpp
  scalars/scalars.hpp

  strength/strength.cpp
  strength/strength.hpp

  tnburn/shared_sources.cpp
  tnburn/tnburn.cpp
  tnburn/tnburn.hpp
  tnburn/ratelib/tn_data_types.hpp
  tnburn/ratelib/read_isotope_data.hpp

  tracers/tracers.cpp
  tracers/tracers.hpp

  template_main.hpp

  variables.hpp
)

if (RIOT_BUILD_CAD)
  set (SRC_LIST ${SRC_LIST} riot_pgen/cad.cpp riot_pgen/cad.hpp)
  add_compile_definitions(RIOT_ENABLE_CAD)
endif()

if (RIOT_BUILD_PYTHON)
  set (SRC_LIST ${SRC_LIST} riot_utils/py_caller.hpp riot_utils/py_init.cpp riot_utils/py_init.hpp)
  add_compile_definitions(RIOT_ENABLE_PYTHON)
endif()

# Generate the compile-time limits header from the configure-time cache variables
# (see RIOT_MAX_* in the top-level CMakeLists.txt). The output lands in the
# generated/ include directory added below.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/riot_limits.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/riot_limits.hpp
  @ONLY)

add_library(riotlib ${SRC_LIST})

# JMM: This is how to add kernel register usage diagnostics in amdclang
# Keep so we have it but lets not expose in a build option right now.
# set_source_files_properties(
#   hydro/calculate_fluxes.cpp
#   PROPERTIES
#   COMPILE_OPTIONS "-Rpass-analysis=kernel-resource-usage"
# )
# set_source_files_properties(
#   microphysics/pte_closure_general.cpp
#   PROPERTIES
#   COMPILE_OPTIONS "-Rpass-analysis=kernel-resource-usage"
# )

target_include_directories(riotlib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  $<$<BOOL:${RIOT_BUILD_SINGULARITY}>:${PROJECT_SOURCE_DIR}/external/singularity-eos>
  $<$<BOOL:${RIOT_BUILD_SINGULARITY}>:${PROJECT_SOURCE_DIR}/external/singularity-eos/util/spiner>
  $<$<BOOL:${RIOT_BUILD_SINGULARITY}>:${PROJECT_SOURCE_DIR}/external/utils/herumi-fmath>
  $<$<BOOL:${RIOT_BUILD_SINGULARITY}>:${PROJECT_SOURCE_DIR}/external/utils/ports-of-call>
  $<$<BOOL:${RIOT_BUILD_SINGULARITY}>:${PROJECT_SOURCE_DIR}/external/singularity-opac>
  $<$<BOOL:${RIOT_BUILD_SINGULARITY}>:${PROJECT_SOURCE_DIR}/external/singularity-opac/utils>
  $<$<BOOL:${RIOT_BUILD_SINGULARITY}>:${PROJECT_SOURCE_DIR}/external/singularity-eos/utils>
  $<$<BOOL:${RIOT_BUILD_SINGULARITY}>:${PROJECT_SOURCE_DIR}/external/singularity-eos/utils/ports-of-call>
  $<$<BOOL:${RIOT_BUILD_SINGULARITY}>:${PROJECT_SOURCE_DIR}/external/singularity-eos/utils/spiner>
  $<$<BOOL:${RIOT_BUILD_SINGULARITY}>:${PROJECT_SOURCE_DIR}/external/singularity-eos/utils/variant/include>
  $<$<BOOL:${RIOT_BUILD_PYTHON}>:${Python_INCLUDE_DIRS}>
  $<$<BOOL:${RIOT_BUILD_PYTHON}>:${Python_NumPy_INCLUDE_DIRS}>
  $<$<BOOL:${RIOT_BUILD_CAD}>:${OpenCASCADE_INCLUDE_DIR}>
)

target_compile_features(riotlib PUBLIC cxx_std_17)

if (Kokkos_ENABLE_CUDA)
  target_compile_options(riotlib PUBLIC --expt-relaxed-constexpr)
endif()

if (RIOT_ENABLE_SANITIZE)
  target_compile_options(riotlib PUBLIC -fsanitize=address -fno-omit-frame-pointer)
  target_link_libraries(riotlib PUBLIC -fsanitize=address)
endif()

target_link_libraries(riotlib PUBLIC
  $<IF:$<BOOL:${RIOT_BUILD_PARTHENON}>,parthenon,Parthenon::parthenon>
  $<IF:$<BOOL:${RIOT_BUILD_SINGULARITY}>,singularity-eos,singularity-eos::singularity-eos>
)

target_compile_options(riotlib
    PUBLIC
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-Wno-psabi>
)


if (RIOT_BUILD_PYTHON)
  target_link_libraries(riotlib PUBLIC Python::Python Python::NumPy)
endif()

if (RIOT_BUILD_CAD)
  target_link_libraries(riotlib PUBLIC ${OpenCASCADE_LIBRARIES})
endif()

if (RIOT_BUILD_PYTHON)
  install(DIRECTORY ${CMAKE_BINARY_DIR}/singularity-eos/python
    DESTINATION .
    COMPONENT runtime
    FILES_MATCHING
    PATTERN "singularity_eos.cpython*"
    PATTERN "CMakeFiles/*" EXCLUDE
  )
endif()

# Disable executable automatically for builds with downstream codes
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
   message(STATUS "Standalone mode. Adding riot executable")
   add_executable(riot main.cpp)
   
   target_link_libraries(riot PRIVATE riotlib)

   install(TARGETS riot
     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
   )

   set_target_properties(riot PROPERTIES
     INSTALL_RPATH_USE_LINK_PATH TRUE
   )
endif()

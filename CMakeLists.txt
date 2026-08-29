# ========================================================================================
#  (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
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
# This file was made in part with generative AI.

include(GNUInstallDirs)

# Turn off python and fortran C++ bindings
set(Fortran_BINDINGS OFF CACHE BOOL "")
set(Python_BINDINGS OFF CACHE BOOL "")

# Load machine specific defaults such as architecture and mpi launch command.
# Command line argument takes precedence over environment variable.
# Loading this before project definition to allow setting the compiler.
# In fact, for some reason, this needs to be above the `cmake_minimum_required`.
if (MACHINE_CFG)
  if(EXISTS "${MACHINE_CFG}")
    include(${MACHINE_CFG})
  else()
    message(FATAL_ERROR "Given machine configuration at "
      "${MACHINE_CFG} not found.")
  endif()
elseif (DEFINED ENV{MACHINE_CFG})
  if(EXISTS "$ENV{MACHINE_CFG}")
    include($ENV{MACHINE_CFG})
  else()
    message(FATAL_ERROR "Given machine configuration from environment variable "
      "MACHINE_CFG at $ENV{MACHINE_CFG} not found.")
  endif()
else()
  message(WARNING "Not using any machine configuration. Consider creating a configuration "
    "file following the examples in ${PROJECT_SOURCE_DIR}/external/parthenon/cmake/machine_cfgs/ and then "
    "point the MACHINE_CFG variable to your custom file."
    "Note, that the machine file can be placed in any directory (also outside the repo).")
endif()

# Boiler plate
cmake_minimum_required(VERSION 3.26)
project(riot
  LANGUAGES C CXX
  VERSION 2026.08.29
)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)

# TODO(JMM): Tests? Copy what's in riot? Hack something in?
# TODO(JMM): Pgen?
include(CTest)
include(CMakeDependentOption)
option(RIOT_ENABLE_UNIT_TESTS "Enable unit tests" OFF)
option(RIOT_ENABLE_INTEGRATION_TESTS "Enable integration tests" OFF)
option(RIOT_ENABLE_REGRESSION_TESTS "Enable regression tests" OFF)
option(RIOT_ENABLE_CUDA "Enable cuda for riot and all dependencies" OFF)
option(RIOT_ENABLE_HDF5 "Enable HDF5 for riot and all dependencies" ON)
option(RIOT_ENABLE_MPI "Enable MPI for riot and all dependencies" ON)
option(RIOT_ENABLE_OPENMP "Enable OpenMP for riot and parthenon" OFF)
option(RIOT_ENABLE_WARNINGS "Enable compiler warnings" OFF)
option(RIOT_BUILD_CATCH2 "Build included version of Catch2" ON)
option(RIOT_BUILD_PARTHENON "Build included version of Parthenon" ON)
option(RIOT_BUILD_SINGULARITY "Build included version of Singularity" ON)
option(RIOT_BUILD_PYTHON "Build Python interface" ON)
option(RIOT_BUILD_CAD "Build support for using CAD files" OFF)
option(RIOT_ENABLE_SANITIZE "Build with address sanitizer" OFF)

option(RIOT_ENABLE_NDI "Enable NDI for ndi2spiner tool only" OFF)
cmake_dependent_option(RIOT_BUILD_NDI "Build included version of NDI"
  ON RIOT_ENABLE_NDI OFF)

# Compile-time limits. These control fixed-size storage (mostly in
# hydro/calculate_fluxes and the radiation_diffusion package). Note the
# material limits are per-block, so an individual problem can have more
# than RIOT_MAX_MATERIALS so long as there aren't more than that many
# materials co-located on a meshblock. RIOT_MAX_GROUPS bounds the
# per-point scratch capacity of the multigroup radiation solver (the true
# group count is a runtime parameter <= this cap). These get baked into a
# generated header at configure time (see src/riot_limits.hpp.in).
set(RIOT_MAX_MATERIALS 16 CACHE STRING "Maximum number of materials")
set(RIOT_MAX_STRONG    16 CACHE STRING "Maximum number of strong materials")
set(RIOT_MAX_ADV       32 CACHE STRING "Maximum number of advected quantities")
set(RIOT_MAX_GROUPS    64 CACHE STRING "Maximum number of radiation energy groups")

# Don't allow in-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
  message(FATAL_ERROR
    "You cannot build in a source directory (or any directory with a CMakeLists.txt file). "
    "Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

# If the user doesn't specify a build type, prefer RelWithDebInfo
set(default_build_type "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# If this is a debug build, set kokkos debug on
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message(STATUS "Enabling Kokkos debug mode")
  set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Most general debug settings")
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL
     "Bounds checking on Kokkos views")
  set(Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK ON CACHE BOOL
      "Sanity checks on Kokkos DualView")
endif()

# Python requires position-independent code. Not sure this is good for us...
if (RIOT_BUILD_PYTHON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# append to CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# Include compiler flag checking utilities
include(AddCompilerFlagIfSupported)

# TODO(JMM): do we want parent flags for both singularity and
# parthenon? I dunno.
set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL
    "Kokkos aggressive vectorization" FORCE)
# TODO(JMM): This option will eventually be removed.
# set(Kokkos_ENABLE_IMPL_VIEW_LEGACY OFF CACHE BOOL
#     "Use int-based, not span-based views." FORCE)

# CUDA
if(RIOT_ENABLE_CUDA)
  set(SINGULARITY_USE_CUDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "" FORCE)
endif()

# HDF5
if(RIOT_ENABLE_HDF5)
  set(HDF5_PREFER_PARALLEL ${RIOT_ENABLE_MPI})
  find_package(HDF5 COMPONENTS C HL)
  if (NOT HDF5_FOUND)
    message(FATAL_ERROR "HDF5 is required but couldn't be found. "
      "If you want to build RIOT without HDF5, please rerun "
      "CMake with -DRIOT_ENABLE_HDF5=OFF")
  endif()
  if (RIOT_ENABLE_MPI AND (NOT HDF5_IS_PARALLEL))
    message(FATAL_ERROR "Both MPI and HDF5 are enabled "
      "but only a serial version of HDF5 was found. Please install "
      "a parallel version of HDF5 (or point CMake to it by adding its path "
      "to the CMAKE_PREFIX_PATH environment variable), or disable either MPI "
      "or HDF5 by rerunning CMake with -DRIOT_ENABLE_MPI=OFF or "
      "-DRIOT_ENABLE_HDF5=OFF.")
  endif()
  set(SINGULARITY_USE_HDF5 ON CACHE BOOL "" FORCE)
  set(SINGULARITY_USE_SPINER ON CACHE BOOL "" FORCE)
  set(SINGULARITY_USE_SPINER_WITH_HDF5 ON CACHE BOOL "" FORCE)
  set(PARTHENON_DISABLE_HDF5 OFF CACHE BOOL "" FORCE)
else()
  set(SINGULARITY_USE_SPINER OFF CACHE BOOL "" FORCE)
  set(SINGULARITY_USE_SPINER_WITH_HDF5 OFF CACHE BOOL "" FORCE)
  set(SINGULARITY_USE_HDF5 OFF CACHE BOOL "" FORCE)
  set(SINGULARITY_USE_SPINER_WITH_HDF5 OFF CACHE BOOL "" FORCE)
  set(PARTHENON_DISABLE_HDF5 ON CACHE BOOL "" FORCE)
endif()

if(RIOT_ENABLE_MPI)
  find_package(MPI COMPONENTS CXX)
  set(PARTHENON_DISABLE_MPI OFF CACHE BOOL "" FORCE)
else()
  set(PARTHENON_DISABLE_MPI ON CACHE BOOL "" FORCE)
endif()

if(RIOT_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  set(PAR_LOOP_LAYOUT MDRANGE_LOOP CACHE STRING "" FORCE)
else()
  set(PARTHENON_DISABLE_OPENMP ON CACHE BOOL "" FORCE)
endif()

add_compiler_flag_if_supported(-Wno-unused-variable)
add_compiler_flag_if_supported(-Wno-sign-compare)
add_compiler_flag_if_supported(-Wno-reorder CXX)
add_compiler_flag_if_supported(-Wno-unused-function)
add_compiler_flag_if_supported(-Wno-unknown-pragmas)
add_compiler_flag_if_supported(-Wno-ps-abi)
if (RIOT_ENABLE_WARNINGS)
  add_compiler_flag_if_supported(-Wall)
else()
 if (RIOT_ENABLE_CUDA)
  add_compiler_flag_if_supported(-Wno-deprecated-gpu-targets)
 endif()
endif()


# TODO(JMM): For some reason, order still maters for including
# parthenon and singularity. Likely has to do with project
# includes other than Kokkos. MPI and OpenMP likely culprits.

# Apply Kokkos patch before building Parthenon
if(RIOT_BUILD_PARTHENON)
  set(KOKKOS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/parthenon/external/Kokkos")
  set(KOKKOS_PATCH "${CMAKE_CURRENT_SOURCE_DIR}/riot_kokkos.patch")

  if(EXISTS "${KOKKOS_DIR}" AND EXISTS "${KOKKOS_PATCH}")
    message(STATUS "Checking if Kokkos patch needs to be applied...")

    # Check if patch is already applied by looking for a marker file
    set(PATCH_MARKER "${KOKKOS_DIR}/.riot_patch_applied")

    if(NOT EXISTS "${PATCH_MARKER}")
      message(STATUS "Applying riot_kokkos.patch to ${KOKKOS_DIR}")
      execute_process(
        COMMAND git apply --directory=external/parthenon/external/Kokkos "${KOKKOS_PATCH}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE PATCH_RESULT
        OUTPUT_VARIABLE PATCH_OUTPUT
        ERROR_VARIABLE PATCH_ERROR
      )

      if(PATCH_RESULT EQUAL 0)
        message(STATUS "Kokkos patch applied successfully")
        # Create marker file to indicate patch has been applied
        file(WRITE "${PATCH_MARKER}" "Patch applied on ${CMAKE_CURRENT_TIMESTAMP}\n")
      else()
        message(WARNING "Failed to apply Kokkos patch. It may already be applied manually.\n${PATCH_ERROR}")
      endif()
    else()
      message(STATUS "Kokkos patch already applied (marker file exists)")
    endif()
  else()
    if(NOT EXISTS "${KOKKOS_DIR}")
      message(WARNING "Kokkos directory not found at ${KOKKOS_DIR}")
    endif()
    if(NOT EXISTS "${KOKKOS_PATCH}")
      message(WARNING "Kokkos patch file not found at ${KOKKOS_PATCH}")
    endif()
  endif()
endif()

# parthenon
# TODO(JMM): This should be renamed PARTHENON_BUILD_TESTING
if(RIOT_BUILD_PARTHENON)
  message("Configuring Parthenon")
  set(PARTHENON_ENABLE_CPP17 ON CACHE BOOL "" FORCE)
  set(PARTHENON_ENABLE_INIT_PACKING ON CACHE BOOL "" FORCE)
  set(PARTHENON_LINT_DEFAULT OFF CACHE BOOL "" FORCE)
  set(PARTHENON_DISABLE_EXAMPLES ON CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  add_subdirectory(external/parthenon parthenon)
  if (RIOT_ENABLE_WARNINGS)
    set(ENABLE_COMPILER_WARNINGS ON CACHE BOOL "" FORCE)
  endif()
else()
  find_package(parthenon REQUIRED)
endif()

# singularity common
if(RIOT_BUILD_SINGULARITY)
  message("Configuring singularity")
  set(SINGULARITY_USE_KOKKOS ON CACHE BOOL "" FORCE)
  set(SINGULARITY_USE_FORTRAN OFF CACHE BOOL "" FORCE)
  set(SINGULARITY_FORCE_SUBMODULE_MODE ON CACHE BOOL "" FORCE)
  set(SINGULARITY_USE_KOKKOSKERNELS ON CACHE BOOL "" FORCE)
  set(SINGULARITY_BUILD_CLOSURE ON CACHE BOOL "" FORCE)
  if (RIOT_BUILD_PYTHON)
     set(SINGULARITY_BUILD_PYTHON ON CACHE BOOL "" FORCE)
  else()
     set(SINGULARITY_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  endif()

  # Kokkos Kernels
  # TODO(JMM): singularity-eos now owns its own kokkos kernels, so we
  # don't need a copy, but it's useful to have our own to manage
  # Kokkos version mismatches... e.g., singularity-eos is on Kokkos <
  # 5, but Parthenon is on Kokkos 5.1
  if(SINGULARITY_USE_KOKKOSKERNELS)
    message("Adding Kokkos Kernels")
    set(KokkosKernels_ENABLE_SUPERNODAL_SPTRSV OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_ENABLE_TPL_BLAS OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_ENABLE_TPL_MKL OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_ENABLE_TPL_LAPACK OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_ENABLE_TPL_CUBLAS OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_ENABLE_TPL_CUSPARSE OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_ENABLE_TPL_CUSOLVER OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_ENABLE_TPL_MAGMA OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_COMPLEX_DOUBLE OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_COMPLEX_FLOAT OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_DOUBLE OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_EXECSPACE_OPENMP OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_EXECSPACE_SERIAL OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_EXECSPACE_THREADS OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_EXECSPACE_CUDA OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_FLOAT OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_LAYOUTLEFT OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_LAYOUTRIGHT OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_MEMSPACE_HOSTSPACE OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_OFFSET_INT OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_OFFSET_SIZE_T OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_ORDINAL_INT OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_ORDINAL_INT OFF CACHE BOOL "" FORCE)
    add_subdirectory(external/kokkos-kernels kokkos-kernels)
  endif()

  # singularity eos
  message("Configuring singularity-eos")
  add_subdirectory(external/singularity-eos singularity-eos)
  if (Kokkos_ENABLE_CUDA)
  endif()
else()
  find_package(singularity-eos REQUIRED)
endif()

# singularity opac
#message("Configuring singularity-opac")
#add_subdirectory(external/singularity-opac singularity-opac)

# NDI
if (RIOT_ENABLE_NDI)
  if(RIOT_BUILD_NDI)
    message("Fetching NDI")
    add_subdirectory(external/ndi ndi)
    # Silence all the warnings coming from NDI
    add_target_compiler_flag_if_supported(ndi PRIVATE "-Wno-incompatible-pointer-types" C)
    add_target_compiler_flag_if_supported(ndi PRIVATE "-Wno-missing-braces" C)
    add_target_compiler_flag_if_supported(ndi PRIVATE "-Wno-unused-result" C)
    add_target_compiler_flag_if_supported(ndi_float32 PRIVATE "-Wno-incompatible-pointer-types" C)
    add_target_compiler_flag_if_supported(ndi_float32 PRIVATE "-Wno-missing-braces" C)
    add_target_compiler_flag_if_supported(ndi_float32 PRIVATE "-Wno-unused-result" C)
    add_target_compiler_flag_if_supported(ndi_float64 PRIVATE "-Wno-incompatible-pointer-types" C)
    add_target_compiler_flag_if_supported(ndi_float64 PRIVATE "-Wno-missing-braces" C)
    add_target_compiler_flag_if_supported(ndi_float64 PRIVATE "-Wno-unused-result" C)
    add_target_compiler_flag_if_supported(ndi_C PRIVATE "-Wno-incompatible-pointer-types" C)
    add_target_compiler_flag_if_supported(ndi_C PRIVATE "-Wno-missing-braces" C)
    add_target_compiler_flag_if_supported(ndi_C PRIVATE "-Wno-unused-result" C)
  else()
    find_package(ndi REQUIRED)
  endif()

  # ndi2spiner tool (requires NDI and HDF5)
  if(SPINER_USE_HDF)
    add_subdirectory(tools/ndi2spiner)
  else()
    message(STATUS "Skipping ndi2spiner tool (requires HDF5 support)")
  endif()
endif() # RIOT_ENABLE_NDI

if(RIOT_BUILD_PYTHON)
  message("Python")
  find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
  if (Python_FOUND)
      message(STATUS "Python found: ${Python_EXECUTABLE}")
      message(STATUS "Python include directories: ${Python_INCLUDE_DIRS}")
      if (Python_NumPy_FOUND)
          message(STATUS "NumPy found. NumPy include directories: ${Python_NumPy_INCLUDE_DIRS}")
      else()
          message(WARNING "NumPy component not found.")
      endif()
  else()
      message(FATAL_ERROR "Python not found.")
  endif()

  # JMM: I think we want these even if we are building a code that
  #  depends on riot. But we only want it if Python is on.
  install(FILES
    ${CMAKE_SOURCE_DIR}/script/inputs/riot.py
    DESTINATION ${Python_SITELIB}
  )
endif()

if(RIOT_BUILD_CAD)
  message("CAD")
  find_package(OpenCASCADE REQUIRED CONFIG)
  if (OpenCASCADE_FOUND)
    message(STATUS "OpenCASCADE found: include directories: ${OpenCASCADE_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "OpenCASCADE not found.")
  endif()
endif()

# RIOT src
message("\nConfiguring src")
add_subdirectory(src)

# Allow for downstream plugins
set(PLUGINS_DIR "" CACHE PATH "Path to plugins directory")
if (PLUGINS_DIR)
  if (NOT IS_DIRECTORY "${PLUGINS_DIR}")
    message(FATAL_ERROR "PLUGINS_DIR is not a directory: ${PLUGINS_DIR}")
  endif()
  if (NOT EXISTS "${PLUGINS_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "No CMakeLists.txt found in: ${PLUGINS_DIR}")
  endif()
  add_subdirectory("${PLUGINS_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/plugins")
else()
  add_subdirectory(src/plugin_stubs)
endif()

# TODO(JMM): Testing?
if(RIOT_ENABLE_UNIT_TESTS OR RIOT_ENABLE_INTEGRATION_TESTS OR RIOT_ENABLE_REGRESSION_TESTS)
  message("\nConfiguring tests")
  if(RIOT_BUILD_CATCH2)
    add_subdirectory(external/Catch2 Catch2)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/external/Catch2/extras)
  else()
    find_package(Catch2 REQUIRED)
  endif()
  enable_testing()
  add_subdirectory(tst)
endif()

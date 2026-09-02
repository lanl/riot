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
# This file was made in part with generative AI.

set(RIOT_GOLD_DIRECTORY ${PROJECT_SOURCE_DIR}/tst/scripts/gold/files)
set(RIOT_GOLD_VERSION_FILE ${RIOT_GOLD_DIRECTORY}/current_version)
if (RIOT_ENABLE_REGRESSION_TESTS)
  if (RIOT_REGRESSION_GOLD_SYNC)
    if (EXISTS ${RIOT_GOLD_VERSION_FILE})
      file(READ ${RIOT_GOLD_VERSION_FILE} REGRESSION_GOLD_ON_DISK)
    else()
      set(REGRESSION_GOLD_ON_DISK 0)
    endif()
    set(RIOT_GOLD_TARBALL ${RIOT_GOLD_DIRECTORY}/riot_regression_gold_${RIOT_REGRESSION_GOLD_VER}.tgz)
    if (NOT (${REGRESSION_GOLD_ON_DISK} EQUAL ${RIOT_REGRESSION_GOLD_VER}))
      if (RIOT_REGRESSION_GOLD_LOCAL)
        # Use a gold tarball fetched ahead of time (e.g. on a login node with
        # internet access). Verify it against the expected hash, then stage it.
        if (NOT EXISTS ${RIOT_REGRESSION_GOLD_LOCAL})
          message(FATAL_ERROR "RIOT_REGRESSION_GOLD_LOCAL set but file not found: ${RIOT_REGRESSION_GOLD_LOCAL}")
        endif()
        message(STATUS "Using pre-fetched gold standard tarball: ${RIOT_REGRESSION_GOLD_LOCAL}")
        file(SHA512 ${RIOT_REGRESSION_GOLD_LOCAL} RIOT_GOLD_LOCAL_HASH)
        if (NOT "SHA512=${RIOT_GOLD_LOCAL_HASH}" STREQUAL "${RIOT_REGRESSION_GOLD_HASH}")
          message(FATAL_ERROR "Pre-fetched gold tarball hash mismatch.\n  expected: ${RIOT_REGRESSION_GOLD_HASH}\n  actual:   SHA512=${RIOT_GOLD_LOCAL_HASH}")
        endif()
        file(COPY_FILE ${RIOT_REGRESSION_GOLD_LOCAL} ${RIOT_GOLD_TARBALL})
      else()
        message(STATUS "Attempting to download latest gold standard files for regression tests. To disable set RIOT_REGRESSION_GOLD_SYNC=OFF.")
        file(
          DOWNLOAD https://github.com/lanl/riot/releases/download/regression-gold-${RIOT_REGRESSION_GOLD_VER}/riot_regression_gold_${RIOT_REGRESSION_GOLD_VER}.tgz ${RIOT_GOLD_TARBALL}
          EXPECTED_HASH ${RIOT_REGRESSION_GOLD_HASH}
          STATUS download_status
          )
        list(GET download_status 0 download_error)
        list(GET download_status 1 download_message)
        if(NOT download_error EQUAL 0)
          message(FATAL_ERROR "Download failed: ${download_message}")
        endif()
      endif()
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${RIOT_GOLD_TARBALL}
        WORKING_DIRECTORY ${RIOT_GOLD_DIRECTORY}
        )
      message(STATUS "  Gold standard ready. Now using version ${RIOT_REGRESSION_GOLD_VER}")
    else()
      message(STATUS "Gold standard is up-to-date (version ${RIOT_REGRESSION_GOLD_VER}). No download required.")
    endif()
  endif()
endif()

if(RIOT_ENABLE_UNIT_TESTS)
  add_subdirectory(unit)
endif()

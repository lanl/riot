set(RIOT_INSTALL_PYTHONDIR "python" CACHE STRING
  "Directory for Riot's bundled Python modules, relative to the install prefix")
set(RIOT_INSTALL_SOURCEDIR "${CMAKE_INSTALL_DATADIR}/riot/source" CACHE STRING
  "Directory for Riot's source snapshot, relative to the install prefix")

if(RIOT_BUILD_PYTHON)
  install(FILES "${PROJECT_SOURCE_DIR}/script/inputs/riot.py"
    DESTINATION "${RIOT_INSTALL_PYTHONDIR}"
    COMPONENT riot-bundle)

  if(TARGET singularity_eos)
    install(TARGETS singularity_eos
      LIBRARY DESTINATION "${RIOT_INSTALL_PYTHONDIR}"
        COMPONENT riot-bundle
      RUNTIME DESTINATION "${RIOT_INSTALL_PYTHONDIR}"
        COMPONENT riot-bundle)
  endif()
endif()

install(FILES "${PROJECT_SOURCE_DIR}/script/rshow.py"
  DESTINATION "${RIOT_INSTALL_PYTHONDIR}"
  COMPONENT riot-bundle)

install(DIRECTORY "${PROJECT_SOURCE_DIR}/script/rshowlib"
  DESTINATION "${RIOT_INSTALL_PYTHONDIR}"
  COMPONENT riot-bundle
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)

install(DIRECTORY
  "${PROJECT_SOURCE_DIR}/external/parthenon/scripts/python/packages/parthenon_tools/parthenon_tools"
  DESTINATION "${RIOT_INSTALL_PYTHONDIR}"
  COMPONENT riot-bundle
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)

install(PROGRAMS "${PROJECT_SOURCE_DIR}/cmake/rshow"
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  COMPONENT riot-bundle)

set(_riot_source_directories
  .github
  cmake
  doc
  inputs
  licenses
  script
  src
  tst)

foreach(_riot_source_directory IN LISTS _riot_source_directories)
  if(EXISTS "${PROJECT_SOURCE_DIR}/${_riot_source_directory}")
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/${_riot_source_directory}"
      DESTINATION "${RIOT_INSTALL_SOURCEDIR}"
      COMPONENT riot-bundle
      PATTERN ".git" EXCLUDE
      PATTERN ".venv*" EXCLUDE
      PATTERN ".cache" EXCLUDE
      PATTERN "build" EXCLUDE
      PATTERN "build-*" EXCLUDE
      PATTERN "_build" EXCLUDE
      PATTERN "_build-*" EXCLUDE
      PATTERN "CMakeFiles" EXCLUDE
      PATTERN "__pycache__" EXCLUDE
      PATTERN ".ipynb_checkpoints" EXCLUDE
      PATTERN "*.pyc" EXCLUDE
      PATTERN "*.swp" EXCLUDE
      PATTERN "*~" EXCLUDE)
  endif()
endforeach()

set(_riot_external_source_destination "${RIOT_INSTALL_SOURCEDIR}/external")

install(DIRECTORY "${PROJECT_SOURCE_DIR}/external/parthenon"
  DESTINATION "${_riot_external_source_destination}"
  COMPONENT riot-bundle
  PATTERN ".git" EXCLUDE
  PATTERN ".venv*" EXCLUDE
  PATTERN ".cache" EXCLUDE
  PATTERN "build" EXCLUDE
  PATTERN "build-*" EXCLUDE
  PATTERN "_build" EXCLUDE
  PATTERN "_build-*" EXCLUDE
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN ".ipynb_checkpoints" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN "*.swp" EXCLUDE
  PATTERN "*~" EXCLUDE
  PATTERN "Catch2" EXCLUDE
  PATTERN "Kokkos" EXCLUDE)

install(DIRECTORY "${PROJECT_SOURCE_DIR}/external/singularity-eos"
  DESTINATION "${_riot_external_source_destination}"
  COMPONENT riot-bundle
  PATTERN ".git" EXCLUDE
  PATTERN ".venv*" EXCLUDE
  PATTERN ".cache" EXCLUDE
  PATTERN "build" EXCLUDE
  PATTERN "build-*" EXCLUDE
  PATTERN "_build" EXCLUDE
  PATTERN "_build-*" EXCLUDE
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN ".ipynb_checkpoints" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN "*.swp" EXCLUDE
  PATTERN "*~" EXCLUDE
  PATTERN "eigen" EXCLUDE
  PATTERN "kokkos" EXCLUDE
  PATTERN "kokkos-kernels" EXCLUDE
  PATTERN "ports-of-call" EXCLUDE
  PATTERN "pybind11" EXCLUDE)

install(DIRECTORY "${PROJECT_SOURCE_DIR}/external/singularity-opac"
  DESTINATION "${_riot_external_source_destination}"
  COMPONENT riot-bundle
  PATTERN ".git" EXCLUDE
  PATTERN ".venv*" EXCLUDE
  PATTERN ".cache" EXCLUDE
  PATTERN "build" EXCLUDE
  PATTERN "build-*" EXCLUDE
  PATTERN "_build" EXCLUDE
  PATTERN "_build-*" EXCLUDE
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN ".ipynb_checkpoints" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN "*.swp" EXCLUDE
  PATTERN "*~" EXCLUDE
  PATTERN "herumi-fmath" EXCLUDE
  PATTERN "kokkos" EXCLUDE
  PATTERN "ports-of-call" EXCLUDE)

install(FILES
  "${PROJECT_SOURCE_DIR}/.clang-format"
  "${PROJECT_SOURCE_DIR}/.gitignore"
  "${PROJECT_SOURCE_DIR}/.gitlab-ci.yml"
  "${PROJECT_SOURCE_DIR}/.gitmodules"
  "${PROJECT_SOURCE_DIR}/CMakeLists.txt"
  "${PROJECT_SOURCE_DIR}/LICENSE"
  "${PROJECT_SOURCE_DIR}/README.md"
  "${PROJECT_SOURCE_DIR}/riot_kokkos.patch"
  "${PROJECT_SOURCE_DIR}/riot_logo.png"
  DESTINATION "${RIOT_INSTALL_SOURCEDIR}"
  COMPONENT riot-bundle)

install(PROGRAMS "${PROJECT_SOURCE_DIR}/cmake/riot-env.sh"
  DESTINATION .
  COMPONENT riot-bundle)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(_riot_install_dependencies riot)
  if(TARGET singularity_eos)
    list(APPEND _riot_install_dependencies singularity_eos)
  endif()

  add_custom_target(riot-install
    COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
      --prefix "${CMAKE_INSTALL_PREFIX}"
      --config $<CONFIG>
      --component riot-bundle
    DEPENDS ${_riot_install_dependencies}
    COMMENT "Installing the relocatable Riot bundle to ${CMAKE_INSTALL_PREFIX}"
    VERBATIM)
endif()

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

# This file was made with the assistance of generative AI

include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)

# add_compiler_flag_if_supported(FLAG [LANG])
#
# Adds a compiler flag to all targets if the compiler supports it.
# By default checks both C and CXX compilers and adds the flag only to
# compilers that support it. If LANG is specified (C or CXX), only checks
# that language.
#
# Examples:
#   add_compiler_flag_if_supported(-Wall)         # Adds to both C and CXX if supported
#   add_compiler_flag_if_supported(-Wno-reorder CXX)  # Adds only to CXX
#
macro(add_compiler_flag_if_supported FLAG)
  set(LANG "${ARGV1}")
  string(MAKE_C_IDENTIFIER "CXX_SUPPORTS${FLAG}" CXX_FLAG_VAR)
  string(MAKE_C_IDENTIFIER "C_SUPPORTS${FLAG}" C_FLAG_VAR)

  if("${LANG}" STREQUAL "")
    # Check both C and CXX
    check_cxx_compiler_flag("${FLAG}" ${CXX_FLAG_VAR})
    check_c_compiler_flag("${FLAG}" ${C_FLAG_VAR})
    if(${CXX_FLAG_VAR})
      add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${FLAG}>)
    endif()
    if(${C_FLAG_VAR})
      add_compile_options($<$<COMPILE_LANGUAGE:C>:${FLAG}>)
    endif()
  elseif("${LANG}" STREQUAL "CXX")
    # CXX only
    check_cxx_compiler_flag("${FLAG}" ${CXX_FLAG_VAR})
    if(${CXX_FLAG_VAR})
      add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${FLAG}>)
    endif()
  elseif("${LANG}" STREQUAL "C")
    # C only
    check_c_compiler_flag("${FLAG}" ${C_FLAG_VAR})
    if(${C_FLAG_VAR})
      add_compile_options($<$<COMPILE_LANGUAGE:C>:${FLAG}>)
    endif()
  endif()
endmacro()

# add_target_compiler_flag_if_supported(TARGET VISIBILITY FLAG [LANG])
#
# Adds a compiler flag to a specific target if the compiler supports it.
# By default checks both C and CXX compilers. If LANG is specified (C or CXX),
# only checks that language.
#
# Arguments:
#   TARGET: The target name
#   VISIBILITY: Visibility scope (PUBLIC, PRIVATE, INTERFACE)
#   FLAG: The compiler flag to add
#   LANG: (optional) Language to check (C or CXX), defaults to both
#
# Examples:
#   add_target_compiler_flag_if_supported(mylib PRIVATE -Wno-unused-result)
#   add_target_compiler_flag_if_supported(mylib PRIVATE -Wno-reorder CXX)
#
macro(add_target_compiler_flag_if_supported TARGET VISIBILITY FLAG)
  set(LANG "${ARGV3}")
  string(MAKE_C_IDENTIFIER "CXX_SUPPORTS${FLAG}" CXX_FLAG_VAR)
  string(MAKE_C_IDENTIFIER "C_SUPPORTS${FLAG}" C_FLAG_VAR)

  if("${LANG}" STREQUAL "")
    # Check both C and CXX
    check_cxx_compiler_flag("${FLAG}" ${CXX_FLAG_VAR})
    check_c_compiler_flag("${FLAG}" ${C_FLAG_VAR})
    if(${CXX_FLAG_VAR})
      target_compile_options(${TARGET} ${VISIBILITY} $<$<COMPILE_LANGUAGE:CXX>:${FLAG}>)
    endif()
    if(${C_FLAG_VAR})
      target_compile_options(${TARGET} ${VISIBILITY} $<$<COMPILE_LANGUAGE:C>:${FLAG}>)
    endif()
  elseif("${LANG}" STREQUAL "CXX")
    # CXX only
    check_cxx_compiler_flag("${FLAG}" ${CXX_FLAG_VAR})
    if(${CXX_FLAG_VAR})
      target_compile_options(${TARGET} ${VISIBILITY} $<$<COMPILE_LANGUAGE:CXX>:${FLAG}>)
    endif()
  elseif("${LANG}" STREQUAL "C")
    # C only
    check_c_compiler_flag("${FLAG}" ${C_FLAG_VAR})
    if(${C_FLAG_VAR})
      target_compile_options(${TARGET} ${VISIBILITY} $<$<COMPILE_LANGUAGE:C>:${FLAG}>)
    endif()
  endif()
endmacro()

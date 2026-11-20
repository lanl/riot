//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
// The stb single-header implementations are compiled here as C, on purpose.
//
// In CUDA builds, Kokkos sets CMAKE_CXX_COMPILER to nvcc_wrapper, so every .cpp file
// is compiled by nvcc, host-only ones included. With a GCC >= 14 host compiler,
// stb_image_resize2.h emits _Pragma("GCC unroll 0") before its loops. nvcc's front end
// rewrites those as standalone #pragma lines that land before a case label instead of
// the loop, and GCC then fails with "for, while or do statement expected". The C
// compiler is never nvcc, so it avoids the problem. It also keeps nvcc from compiling
// ~10k lines of stb.
//
// The stb headers wrap their declarations in extern "C" when included from C++, so
// riot_viz.cpp links against these symbols directly.

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

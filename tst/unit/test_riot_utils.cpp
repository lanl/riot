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
// This file was made in part with generative AI.

#include <string>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <parthenon/package.hpp>

#include "riot_utils/riot_utils.hpp"

using Catch::Approx;
using parthenon::Real;

TEST_CASE("RiotUtils::SafeSqrt clamps negatives to zero", "[riot_utils][safesqrt]") {
  SECTION("host") {
    CHECK(RiotUtils::SafeSqrt(0.0) == 0.0);
    CHECK(RiotUtils::SafeSqrt(4.0) == Approx(2.0));
    CHECK(RiotUtils::SafeSqrt(2.0) == Approx(std::sqrt(2.0)));
    // Invariant: negative inputs are treated as zero rather than producing NaN.
    CHECK(RiotUtils::SafeSqrt(-1.0) == 0.0);
    CHECK(RiotUtils::SafeSqrt(-1.0e30) == 0.0);
  }

  SECTION("device (Kokkos kernel)") {
    // Run the identical checks inside a parallel_reduce that counts failures.
    // A nonzero count means SafeSqrt misbehaved on the execution space.
    int n_failures = 0;
    Kokkos::parallel_reduce(
        "test SafeSqrt on device", 1,
        KOKKOS_LAMBDA(const int /*i*/, int &update) {
          if (RiotUtils::SafeSqrt(4.0) != 2.0) update += 1;
          if (RiotUtils::SafeSqrt(0.0) != 0.0) update += 1;
          if (RiotUtils::SafeSqrt(-1.0) != 0.0) update += 1;
        },
        n_failures);
    CHECK(n_failures == 0);
  }
}

TEST_CASE("RiotUtils::ToUpper / ToLower transform in place (host)",
          "[riot_utils][string]") {
  SECTION("ToUpper uppercases letters and leaves other characters alone") {
    std::string s = "Riot-123_hydro";
    RiotUtils::ToUpper(s);
    CHECK(s == "RIOT-123_HYDRO");
  }

  SECTION("ToLower lowercases letters and leaves other characters alone") {
    std::string s = "Riot-123_HYDRO";
    RiotUtils::ToLower(s);
    CHECK(s == "riot-123_hydro");
  }

  SECTION("empty string is a no-op") {
    std::string s;
    RiotUtils::ToUpper(s);
    CHECK(s.empty());
    RiotUtils::ToLower(s);
    CHECK(s.empty());
  }

  SECTION("round trip: ToLower after ToUpper is idempotent on the lowercased form") {
    std::string original = "MixedCase";
    std::string s = original;
    RiotUtils::ToUpper(s);
    RiotUtils::ToLower(s);
    CHECK(s == "mixedcase");
  }
}

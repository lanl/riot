//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
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
#ifndef RADIATION_DIFFUSION_POLYLOG_HPP_
#define RADIATION_DIFFUSION_POLYLOG_HPP_

#include <cmath>

namespace RadiationDiffusion {

namespace impl {
KOKKOS_INLINE_FUNCTION
constexpr int factorial(int n) {
  int ret{1};
  for (; n > 1; n--)
    ret *= n;
  return ret;
}

KOKKOS_INLINE_FUNCTION
constexpr double power(double val, int n) {
  double ret{1.0};
  for (int i = 0; i < n; ++i)
    ret *= val;
  return ret;
}

template <int N>
KOKKOS_INLINE_FUNCTION constexpr std::array<double, N> GetHarmonicNumbers() {
  std::array<double, N> H{};
  H[0] = 0.0; // Place holder for undefined H_0
  for (int i = 1; i < N; ++i) {
    H[i] = H[i - 1] + 1.0 / i;
  }
  return H;
}

// Returns an array containing B_n^-
// B_n^+ is the same except that B_1^+ = -B_1^-
template <int N>
KOKKOS_INLINE_FUNCTION constexpr std::array<double, N> GetBernoulliNumbers() {
  std::array<double, N> B{};
  for (int m = 0; m < N; ++m) {
    B[m] = (m == 0) ? 1.0 : 0.0;
    for (int k = 0; k < m; ++k) {
      B[m] -= (B[k] / (m - k + 1.0) * factorial(m)) / (factorial(k) * factorial(m - k));
      if (m > 1 && m % 2 == 1) B[m] = 0.0;
    }
  }
  return B;
}

// Returns the coefficients in the polylog expansion given by equation (49) in
// Volinga & Weingzierl (2004):
//
// C_1(j) = \delta_{0,j}
// C_n(j) = \sum_{k=0}^j \frac{j!}{(j-k)!} B^-_{j - k}  \frac{C_{n-1}(k)}{(j + 1)!}
//
// such that
//
// Li_n(x) = \sum_{j=0}^\infinity C_n(j) (-\ln(1 - x))^{j+1}
//
// which converges for x in [-1, 1/2]. (Note that we have absorbed a factor of (j+1)!
// into C_n(j) relative to the expression in the original paper)
template <int n, int N>
KOKKOS_INLINE_FUNCTION constexpr std::array<double, N> GetCn() {
  std::array<double, N> Cn{};
  if constexpr (n == 1) {
    Cn[0] = 1.0;
    for (int i = 1; i < N; ++i)
      Cn[i] = 0.0;
  } else {
    const std::array<double, N> B = GetBernoulliNumbers<N>();
    const auto Cnm1 = GetCn<n - 1, N>();
    for (int j = 0; j < N; ++j) {
      Cn[j] = 0.0;
      for (int k = 0; k <= j; ++k)
        Cn[j] += (Cnm1[k] * factorial(j) * B[j - k]) / factorial(j - k);
      Cn[j] /= factorial(j + 1);
    }
  }
  return Cn;
}

// Integral values of the Riemann zeta function taken from Mathematica
constexpr std::array<double, 10> integralZeta{-0.5,
                                              0.0, // Placeholder for infinite value
                                              1.6449340668482262,
                                              1.2020569031595942,
                                              1.0823232337111381,
                                              1.03692775514337,
                                              1.0173430619844488,
                                              1.008349277381923,
                                              1.0040773561979441,
                                              1.0020083928260821};

// Returns the polynomial coefficients of the expansion
//
// Li_n(e^{-a}) = \frac{(-a)^{n-1}}{(n-1)!}(H_{n-1} - ln(a))
//           + \sum_{m=0, m \neq n-1}^\infinity \zeta(n - m) (-a)^m m!
//
// where H_n is a harmonic number and we use the fact that
//
// \zeta(-n) = - B^+_{n + 1} / (n + 1) for n >= 1
//
// which is essentially the algorithm of R. E. Crandall (2006), which
// we took from Frellesvig, Tommasini, and Wever (2016). Note that the
// expansion parameter here is $a$ where $x = exp(-a)$, so we expect
// this to rapidly converge in the range x = [~1/2, 1].
//
// The last element of the array stores the coefficient of the ln(a)
// term in the expansion, hence why the returned array has a size
// one greater than the requested order N - 1
template <int npoly, int N>
KOKKOS_INLINE_FUNCTION constexpr std::array<double, N + 1> GetCrandallCoefficients() {
  std::array<double, N + 1> A{};
  std::array<double, N> H = GetHarmonicNumbers<N>();
  std::array<double, N> B = GetBernoulliNumbers<N>();
  A[npoly - 1] = power(-1.0, npoly - 1) / factorial(npoly - 1) * H[npoly - 1];
  A[N] = power(-1.0, npoly) / factorial(npoly - 1); // Logarithmic term
  for (int m = 0; m < N; ++m) {
    if (npoly - 1 == m) {
      // do nothing
    } else if (npoly - m >= 0) {
      A[m] = integralZeta[npoly - m] / factorial(m) * power(-1.0, m);
    } else {
      // Using zeta(-n) = -B^+_{n + 1}/(n + 1)
      const int l = m - npoly + 1;
      A[m] = -B[l] / (l * factorial(m)) * power(-1.0, m);
      // Correct for the fact that B[l] is B^-_l
      if (l == 1) A[m] *= -1.0;
    }
  }
  return A;
}

// Combines the coefficients of the two polylog expansions into a single
// array at compile time, padding the Bernoulli based expansion with zeros
// for the constant and logarithmic terms
template <int n, int N>
KOKKOS_INLINE_FUNCTION constexpr std::array<double, 2 * (N + 1)>
GetCombinedCoefficientArray() {
  constexpr auto Cn = GetCn<n, N - 1>();
  constexpr auto An = GetCrandallCoefficients<n, N>();
  std::array<double, 2 * (N + 1)> combined{};

  // The Bernoulli expansion coefficients
  combined[0] = 0.0; // Constant contribution from Bernoulli expansion is zero
  for (int i = 1; i < N; ++i)
    combined[i] = Cn[i - 1];
  combined[N] = 0.0; // No logarithmic contribution in Bernoulli expansion

  // The Crandall expansion coefficients
  for (int i = 0; i < N + 1; ++i)
    combined[i + N + 1] = An[i];

  return combined;
}

// Determines which polylog expansion to use based on an empirically chosen threshold
template <int nterms>
KOKKOS_INLINE_FUNCTION auto GetAlphaAndOffset(double x) {
  static constexpr double threshold = 0.46;
  const double mlnx = -log(x);
  const double x2 = x * x;
  const double x3 = x * x2;
  const double mlnmx =
      x > 1.e-3 ? -log(1.0 - x) : x + 0.5 * x2 + x3 / 3.0 + x3 * x * 0.25;
  const int offset = x < threshold ? 0 : nterms + 1;
  const double alpha = x < threshold ? mlnmx : mlnx;
  const double dalphadx = x < threshold ? 1.0 / (1.0 - x) : -1.0 / x;
  return std::make_tuple(alpha, dalphadx, offset);
}
} // namespace impl

// Calculates Li_n(x), the n-th classical polylog evaluated at x using an expansion in a
// small parameter to order nterms - 1
template <int n, int nterms>
KOKKOS_INLINE_FUNCTION double Li(double x) {
  static constexpr auto combined = impl::GetCombinedCoefficientArray<n, nterms>();
  const auto [alpha, dalphadx, offset] = impl::GetAlphaAndOffset<nterms>(x);
  double result{0.0};
  double z{1.0};
  for (int i = 0; i < nterms; ++i) {
    result += combined[offset + i] * z;
    z *= alpha;
  }

  if (offset > 0 && x < (1.0 - 1.e-12))
    result += combined[offset + nterms] * std::pow(alpha, n - 1) * log(alpha);
  return result;
}

// Calculates 15 / \pi^4 \int_0^x y^3 / (exp(y) - 1) dy in terms of polylogs. Gives
// exactly 1 as x -> \infinity
template <int nterms>
KOKKOS_INLINE_FUNCTION auto IncompleteBose3FromZero(double x) {
  static constexpr auto C1 = impl::GetCombinedCoefficientArray<1, nterms>();
  static constexpr auto C2 = impl::GetCombinedCoefficientArray<2, nterms>();
  static constexpr auto C3 = impl::GetCombinedCoefficientArray<3, nterms>();
  static constexpr auto C4 = impl::GetCombinedCoefficientArray<4, nterms>();
  static constexpr double pi4overfifteen = M_PI * M_PI * M_PI * M_PI / 15.0;

  const double x2 = x * x;
  const double x3 = x2 * x;
  const double emx = exp(-x);

  const auto [alpha, dalphademx, offset] = impl::GetAlphaAndOffset<nterms>(emx);
  double dalphadx = -dalphademx * emx;
  double result{pi4overfifteen};
  double deriv{0.0};
  double z{1.0};
  for (int i = 0; i < nterms; ++i) {
    const double dzdalpha = i > 0 ? i * std::pow(alpha, i - 1) : 0.0;
    const double fac = (x3 * C1[offset + i] + 3.0 * x2 * C2[offset + i] +
                        6.0 * x * C3[offset + i] + 6.0 * C4[offset + i]);
    const double dfacdx =
        (3.0 * x2 * C1[offset + i] + 6.0 * x * C2[offset + i] + 6.0 * C3[offset + i]);

    result -= fac * z;
    deriv -= dfacdx * z + fac * dzdalpha * dalphadx;

    z *= alpha;
  }

  if (offset > 0 && emx < (1.0 - 1.e-12))
    result -= (x3 * C1[offset + nterms] + 3.0 * x2 * C2[offset + nterms] * alpha +
               6.0 * x * C3[offset + nterms] * alpha * alpha +
               6.0 * C4[offset + nterms] * alpha * alpha * alpha) *
              log(alpha);
  return std::make_pair(result / pi4overfifteen, deriv / pi4overfifteen);
}

} // namespace RadiationDiffusion

#endif // RADIATION_DIFFUSION_POLYLOG_HPP_

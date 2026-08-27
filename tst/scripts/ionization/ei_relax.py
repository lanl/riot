# ========================================================================================
#  (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
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

# Regression test: Solve electron-ion equilibration using Landau-Spitzer model with Brysk Coulomb logarithm

import logging
import numpy as np
import scripts.utils.riot as riot
from scipy.integrate import solve_ivp
from glob import glob

file_id = "ionization/ei_relax"
logger = logging.getLogger("riot" + __name__[7:])  # set logger name

# constants
me = 9.10938e-28  # g
amu = 1.66054e-24  # g
planck = 6.6261e-27  # erg/s
hbar = 1.05457266e-27  # erg/s
kbev = 8.617333262e-5  # eV/K
kberg = 1.3807e-16  # erg/K
qe = 4.8e-10  # esu (statcoulombs)

# oxygen 16 at 1 g/cm^3
mion = 16.0 * amu
zion = 8.0
rho = 1.0

# initial conditions and final time
tele0 = 2000.0 / kbev  # K
tion0 = 1000.0 / kbev  # K
tmax = 2e-11  # s

# derived quantities
nion = rho / mion
ne = nion * zion
cve = 3.0 / 2.0 * kberg * ne
cvi = 3.0 / 2.0 * kberg * nion


def vboltzmann(m, tev):
    """mean velocity in boltzmann distribution for population"""
    tk = tev / kbev
    return np.sqrt(kberg * tk / m)


def urelthermal(teve, tevi):
    """relative thermal velocity of electrons and ions"""
    return vboltzmann(me, teve) - vboltzmann(mion, tevi)


def mu(me, mion):
    """reduced mass of electron-ion system"""
    return 1.0 / (1.0 / me + 1.0 / mion)


def bmin(teve, tevi):
    """minimum impact parameter"""
    return zion * qe**2 / mu(me, mion) / urelthermal(teve, tevi) ** 2


def edb(tev):
    """electron de broglie wavelength in cm"""
    tk = tev / kbev
    return planck / np.sqrt(2 * np.pi * me * kberg * tk)


def idb(teve):
    """ion de broglie wavelength in cm"""
    tk = teve / kbev
    return planck / np.sqrt(2 * np.pi * mion * kberg * tk)


def debyelength(teve):
    """debye screening length"""
    ne = nion * zion
    tk = teve / kbev
    return np.sqrt(kberg * tk / (4 * np.pi * ne * qe**2))


def coullog(teve, tevi):
    """coulomb logarithm basic"""
    return np.log(max(1, debyelength(teve) / bmin(teve, tevi)))


def coullog_brysk(teve, tevi):
    """coulomb logarithm a la brysk"""
    tke = teve / kbev
    tki = tevi / kbev
    ne = nion * zion
    kberg_ov_4piqe2 = kberg / (4.0 * np.pi * qe * qe)
    qe2_ov_3kberg = qe * qe / (3.0 * kberg)
    bmax = np.sqrt(kberg_ov_4piqe2 * tke / (ne + 1e-15))
    bmin = qe2_ov_3kberg * zion / tke
    bmin = max(bmin, 0.5 * hbar / np.sqrt(3.0 * kberg * tke * me))

    return np.log(max(1.0, bmax / (bmin + 1e-15)))


def tauei_brian(teve, tevi):
    """landau-spitzer electron-ion equilibration time scale a la xrage"""
    ne = nion * zion
    tke = teve / kbev
    tki = tevi / kbev
    tau_ei = (mion * kberg * tke) ** 1.5
    tau_ei /= 8.0 / 3.0
    tau_ei /= np.sqrt(2.0 * np.pi)
    tau_ei /= qe**4
    tau_ei /= np.sqrt(me * mion)
    tau_ei /= zion * zion
    tau_ei /= nion
    tau_ei /= coullog_brysk(teve, tevi)
    return tau_ei


def tauei_blancard(teve, tevi):
    """landau-spitzer electron-ion equilibration time scale a la blancard 2013"""
    ne = nion * zion
    tke = teve / kbev
    tki = tevi / kbev
    tau_ei = 8.0 * np.sqrt(2.0 * np.pi) * nion * zion * zion * qe**4
    tau_ei /= 3.0 * me * mion
    tau_ei *= coullog_brysk(teve, tevi)
    tau_ei *= (kberg * tke / me + kberg * tki / mion) ** (-1.5)
    tau_ei = 1.0 / tau_ei
    return tau_ei


def ke(teve, tevi):
    """spitzer-harm thermal conductivity"""
    ne = nion * zion
    tke = teve / kbev
    tki = tevi / kbev
    alf = 1.0 / (1.0 + 3.3 / zion)
    return (
        alf
        * (8.0 / np.pi) ** 1.5
        * kberg ** (7.0 / 2.0)
        / qe**4
        / me**0.5
        / zion
        / coullog(teve, tevi)
    )


def ode_rhs(t, y):
    """RHS of relaxation ODE system"""
    tke, tki = y
    teve = tke * kbev
    tevi = tki * kbev
    tau = tauei_blancard(teve, tevi)

    rhs = [-(tke - tki) / tau, -(tki - tke) / tau / (cvi / cve)]
    return rhs


def integrate_odes():
    """integrate the ODE system for electron and ion temperatures"""
    t_span, y0 = (0.0, tmax), (tele0, tion0)
    t = np.linspace(*t_span, 1024)
    sol = solve_ivp(
        lambda t, y: ode_rhs(t, y),
        t_span,
        y0,
        t_eval=t,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )
    return sol.t, sol.y[0, :], sol.y[1, :]


# Run riot
def run(**kwargs):
    logger.debug("Generating input " + __name__)
    riot.generate(file_id + ".py")
    logger.debug(f"Runnning test {__name__}")
    nranks = 1
    arguments = []
    riot.mpirun(nranks, file_id + ".rin", arguments)


# Analyze outputs
def analyze():
    import phdf
    from scipy.interpolate import CubicSpline

    logger.debug("Analyzing test " + __name__)
    analyze_status = True

    # compute semi-analytic solution and interpolants
    t_ana_raw, te_ana_raw, ti_ana_raw = integrate_odes()
    te_ana = CubicSpline(t_ana_raw, te_ana_raw)
    ti_ana = CubicSpline(t_ana_raw, ti_ana_raw)

    # Load dumps
    dumps = sorted(glob("build/src/ei_relax*.phdf"))
    err = 0.0
    for dump in dumps:
        d = phdf.phdf(dump)
        te = d.Get("c.c.bulk.electron_temperature", flatten=False)[0, 0, 0, 0]
        ti = d.Get("c.c.bulk.temperature", flatten=False)[0, 0, 0, 0]
        tea = te_ana(d.Time)
        tia = ti_ana(d.Time)
        te_err = (te - tea) / tea
        ti_err = (ti - tia) / tia
        local_err = 0.5 * (te_err * te_err + ti_err * ti_err) / len(dumps)
        if dump == dumps[0]:
            print(
                f"  {'d.Time':16s} {'te':16s} {'ti':16s} {'tea':16s} {'tia':16s} {'te_err':16s} {'ti_err':16s}"
            )
        print(
            f"{d.Time:16.8e} {te:16.8e} {ti:16.8e} {tea:16.8e} {tia:16.8e} {te_err:16.8e} {ti_err:16.8e}"
        )
        err += local_err

    err = np.sqrt(err)
    print(f"Total normalized error = {err:12.5e}")

    if err > 1.0e-6:
        logger.warning("error tolerance exceeded in ei_relax verification")
        analyze_status = False
    return analyze_status

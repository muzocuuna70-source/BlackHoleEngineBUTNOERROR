#!/usr/bin/env python3
"""
EVENT HORIZON - relativistic black hole simulation backend.

Pure-stdlib Python server (no pip installs). Serves blackhole.html and a JSON
physics API built on the exact Kerr metric (geometric units G = c = M = 1):

  /api/health                      server status
  /api/metrics   mass, spin, mdot  horizons, ISCO, photon orbits, Hawking T, ...
  /api/shadow    spin, inc, n      exact Bardeen shadow / critical curve
  /api/disk      mass, spin, mdot  Novikov-Thorne / Page-Thorne disk profiles + spectrum
  /api/orbit     spin, r0, speed, tilt, vr, tau   timelike Kerr geodesic (RK45 Dormand-Prince)
  /api/photons   spin, n, bmax     equatorial null-geodesic fan (lensing diagram)
  /api/infall    mass, r0, length  radial plunge: proper/coordinate time, tides, verdict
  /api/dilation  mass, spin        time-dilation table for static and orbiting clocks

Run:  python blackhole_server.py [--port 8765] [--no-browser]
"""
import argparse
import json
import math
import os
import sys
import threading
import time
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# ----------------------------------------------------------------------------
# Physical constants (SI, CODATA 2018)
# ----------------------------------------------------------------------------
G = 6.67430e-11
C = 299792458.0
HBAR = 1.054571817e-34
H_PLANCK = 6.62607015e-34
KB = 1.380649e-23
SIGMA_SB = 5.670374419e-8
M_SUN = 1.98847e30
M_PROTON = 1.67262192369e-27
SIGMA_THOMSON = 6.6524587321e-29
YEAR = 3.15576e7
AU = 1.495978707e11
LIGHT_YEAR = 9.4607304725808e15
AGE_UNIVERSE = 13.787e9 * YEAR

HERE = os.path.dirname(os.path.abspath(__file__))
HTML_FILE = os.path.join(HERE, "blackhole.html")
VERSION = "2.0.0"


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def safe_spin(a):
    a = clamp(float(a), -0.9999, 0.9999)
    if abs(a) < 1e-7:
        a = 1e-7
    return a


# ----------------------------------------------------------------------------
# Kerr spacetime (Boyer-Lindquist coordinates, M = 1)
# ----------------------------------------------------------------------------
class Kerr:
    def __init__(self, a):
        self.a = safe_spin(a)

    # --- characteristic radii ------------------------------------------------
    @property
    def r_plus(self):
        return 1.0 + math.sqrt(1.0 - self.a * self.a)

    @property
    def r_minus(self):
        return 1.0 - math.sqrt(1.0 - self.a * self.a)

    def ergosphere(self, theta):
        return 1.0 + math.sqrt(max(0.0, 1.0 - self.a * self.a * math.cos(theta) ** 2))

    def isco(self, prograde=True):
        a = self.a
        z1 = 1 + (1 - a * a) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
        z2 = math.sqrt(3 * a * a + z1 * z1)
        s = (-1 if prograde else 1) * (1 if a >= 0 else -1)
        return 3 + z2 + s * math.sqrt((3 - z1) * (3 + z1 + 2 * z2))

    def photon_orbit(self, prograde=True):
        s = -1 if prograde else 1
        return 2 * (1 + math.cos(2 / 3 * math.acos(s * self.a)))

    def marginally_bound(self, prograde=True):
        s = -1 if prograde else 1
        return 2 + s * self.a + 2 * math.sqrt(1 + s * self.a)

    # --- circular equatorial orbits (signed spin => prograde for a>0) -------
    def omega_kepler(self, r):
        return 1.0 / (r ** 1.5 + self.a)

    def u_t_circular(self, r):
        a = self.a
        d = r ** 1.5 - 3 * math.sqrt(r) + 2 * a
        if d <= 0:
            return float("inf")
        return (r ** 1.5 + a) / (r ** 0.75 * math.sqrt(d))

    def energy_circular(self, r):
        a = self.a
        d = r ** 0.75 * math.sqrt(max(1e-300, r ** 1.5 - 3 * math.sqrt(r) + 2 * a))
        return (r ** 1.5 - 2 * math.sqrt(r) + a) / d

    def angmom_circular(self, r):
        a = self.a
        d = r ** 0.75 * math.sqrt(max(1e-300, r ** 1.5 - 3 * math.sqrt(r) + 2 * a))
        return (r * r - 2 * a * math.sqrt(r) + a * a) / d

    # --- metric -------------------------------------------------------------
    def metric(self, r, th):
        a = self.a
        s, c = math.sin(th), math.cos(th)
        s2 = max(s * s, 1e-16)
        sig = r * r + a * a * c * c
        dlt = r * r - 2 * r + a * a
        A = (r * r + a * a) ** 2 - a * a * dlt * s2
        return {
            "gtt": -(1 - 2 * r / sig),
            "gtph": -2 * a * r * s2 / sig,
            "gphph": A * s2 / sig,
            "grr": sig / dlt,
            "gthth": sig,
            "sigma": sig,
            "delta": dlt,
            "A": A,
            "s2": s2,
        }

    def zamo(self, r, th):
        m = self.metric(r, th)
        alpha = math.sqrt(max(1e-300, m["sigma"] * m["delta"] / m["A"]))
        omega = 2 * self.a * r / m["A"]
        return alpha, omega, m

    def hamiltonian(self, r, th, pt, pr, pth, pph):
        """H = 1/2 g^{mu nu} p_mu p_nu in Boyer-Lindquist coordinates."""
        a = self.a
        s, c = math.sin(th), math.cos(th)
        s2 = max(s * s, 1e-16)
        sig = r * r + a * a * c * c
        dlt = r * r - 2 * r + a * a
        A = (r * r + a * a) ** 2 - a * a * dlt * s2
        sd = sig * dlt
        gtt = -A / sd
        gtph = -2 * a * r / sd
        gphph = (dlt - a * a * s2) / (sd * s2)
        return 0.5 * (gtt * pt * pt + 2 * gtph * pt * pph + gphph * pph * pph
                      + dlt / sig * pr * pr + pth * pth / sig)

    def geodesic_rhs(self, y, pt, pph):
        """y = [t, r, theta, phi, p_r, p_theta]; p_t, p_phi conserved."""
        a = self.a
        _, r, th, _, pr, pth = y
        s, c = math.sin(th), math.cos(th)
        s2 = max(s * s, 1e-16)
        sig = r * r + a * a * c * c
        dlt = r * r - 2 * r + a * a
        A = (r * r + a * a) ** 2 - a * a * dlt * s2
        sd = sig * dlt
        gtt = -A / sd
        gtph = -2 * a * r / sd
        gphph = (dlt - a * a * s2) / (sd * s2)
        dt = gtt * pt + gtph * pph
        dph = gtph * pt + gphph * pph
        dr = dlt / sig * pr
        dth = pth / sig
        er = 1e-6 * max(1.0, abs(r))
        eth = 1e-6
        dHr = (self.hamiltonian(r + er, th, pt, pr, pth, pph)
               - self.hamiltonian(r - er, th, pt, pr, pth, pph)) / (2 * er)
        dHth = (self.hamiltonian(r, th + eth, pt, pr, pth, pph)
                - self.hamiltonian(r, th - eth, pt, pr, pth, pph)) / (2 * eth)
        return [dt, dr, dth, dph, -dHr, -dHth]

    def carter_constant(self, r, th, pt, pth, pph, mu2):
        a = self.a
        c2 = math.cos(th) ** 2
        s2 = max(math.sin(th) ** 2, 1e-16)
        return pth * pth + c2 * (a * a * (mu2 - pt * pt) + pph * pph / s2)

    def initial_momentum(self, r, th, vr, vth, vph, null=False):
        """Covariant momentum of a particle with local 3-velocity in the ZAMO frame."""
        alpha, omega, m = self.zamo(r, th)
        v2 = vr * vr + vth * vth + vph * vph
        if null:
            n = math.sqrt(max(v2, 1e-30))
            vr, vth, vph = vr / n, vth / n, vph / n
            gam = 1.0
        else:
            if v2 >= 1:
                f = 0.999999 / math.sqrt(v2)
                vr, vth, vph = vr * f, vth * f, vph * f
                v2 = vr * vr + vth * vth + vph * vph
            gam = 1 / math.sqrt(1 - v2)
        ut = gam / alpha
        uph = gam * (omega / alpha + vph / math.sqrt(m["gphph"]))
        ur = gam * vr / math.sqrt(m["grr"])
        uth = gam * vth / math.sqrt(m["gthth"])
        pt = m["gtt"] * ut + m["gtph"] * uph
        pph = m["gtph"] * ut + m["gphph"] * uph
        return pt, m["grr"] * ur, m["gthth"] * uth, pph

    def local_circular_speed(self, r):
        alpha, omega, m = self.zamo(r, math.pi / 2)
        Om = self.omega_kepler(r)
        return (Om - omega) * math.sqrt(m["gphph"]) / alpha


# ----------------------------------------------------------------------------
# Adaptive Dormand-Prince RK45 integrator
# ----------------------------------------------------------------------------
DP_C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]
DP_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
DP_B5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
DP_B4 = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]


def rk45(f, y0, h0, lam_max, stop, rtol=1e-9, atol=1e-11, max_steps=200000, hmax=None):
    """Integrate y' = f(y). stop(y) -> reason string or None. Returns (samples, reason)."""
    y = list(y0)
    n = len(y)
    lam = 0.0
    h = h0
    out = [(lam, list(y))]
    reason = "max_lambda"
    steps = 0
    while lam < lam_max and steps < max_steps:
        steps += 1
        h = min(h, lam_max - lam)
        if hmax:
            h = min(h, hmax(y))
        k = [f(y)]
        for i in range(1, 7):
            yi = [y[j] + h * sum(DP_A[i][m] * k[m][j] for m in range(i)) for j in range(n)]
            k.append(f(yi))
        y5 = [y[j] + h * sum(DP_B5[m] * k[m][j] for m in range(7)) for j in range(n)]
        y4 = [y[j] + h * sum(DP_B4[m] * k[m][j] for m in range(7)) for j in range(n)]
        err = 0.0
        for j in range(n):
            sc = atol + rtol * max(abs(y[j]), abs(y5[j]))
            err = max(err, abs(y5[j] - y4[j]) / sc)
        if err <= 1.0 or h < 1e-12:
            lam += h
            y = y5
            out.append((lam, list(y)))
            r = stop(y)
            if r:
                reason = r
                break
        fac = 0.9 * (1.0 / max(err, 1e-10)) ** 0.2
        h *= clamp(fac, 0.2, 5.0)
    else:
        if steps >= max_steps:
            reason = "max_steps"
    return out, reason


def decimate(seq, n):
    if len(seq) <= n:
        return seq
    step = (len(seq) - 1) / (n - 1)
    return [seq[int(round(i * step))] for i in range(n)]


# ----------------------------------------------------------------------------
# Physics services
# ----------------------------------------------------------------------------
def units(mass_solar):
    M = mass_solar * M_SUN
    L = G * M / C ** 2
    T = L / C
    return M, L, T


@lru_cache(maxsize=256)
def compute_metrics(mass_solar, spin, mdot_edd):
    k = Kerr(spin)
    a = k.a
    M, Lg, Tg = units(mass_solar)
    rp, rm = k.r_plus, k.r_minus
    r_isco_p, r_isco_r = k.isco(True), k.isco(False)
    r_ph_p, r_ph_r = k.photon_orbit(True), k.photon_orbit(False)
    E_isco = k.energy_circular(r_isco_p)
    eff = 1 - E_isco
    kappa_geo = (rp - rm) / (2 * (rp * rp + a * a))
    kappa = kappa_geo * C ** 4 / (G * M)
    T_hawk = HBAR * kappa / (2 * math.pi * C * KB)
    area = 4 * math.pi * (rp * rp + a * a) * Lg * Lg
    entropy = KB * area * C ** 3 / (4 * G * HBAR)
    omega_h = a / (2 * rp) / Tg
    t_evap = 5120 * math.pi * G ** 2 * M ** 3 / (HBAR * C ** 4)
    L_edd = 4 * math.pi * G * M * M_PROTON * C / SIGMA_THOMSON
    mdot_edd_kg = L_edd / (max(eff, 1e-6) * C ** 2)
    mdot = mdot_edd * mdot_edd_kg
    lum = eff * mdot * C ** 2
    J = a * G * M * M / C
    E_rot = M * C ** 2 * (1 - math.sqrt(0.5 * (1 + math.sqrt(1 - a * a))))
    rho_avg = M / (4 / 3 * math.pi * (2 * Lg) ** 3)
    shadow_r = 3 * math.sqrt(3)
    surf_grav_horizon = kappa
    tidal_horizon = 2 * G * M / (rp * Lg) ** 3 * 2.0
    peak_T = disk_profile_raw(spin, mass_solar, mdot_edd, 400)["T_peak"]
    return {
        "mass_solar": mass_solar,
        "mass_kg": M,
        "spin": a,
        "gravitational_radius_m": Lg,
        "light_crossing_time_s": Tg,
        "schwarzschild_radius_m": 2 * Lg,
        "r_plus": rp,
        "r_minus": rm,
        "r_plus_m": rp * Lg,
        "ergosphere_equator": 2.0,
        "isco_prograde": r_isco_p,
        "isco_retrograde": r_isco_r,
        "photon_orbit_prograde": r_ph_p,
        "photon_orbit_retrograde": r_ph_r,
        "marginally_bound_prograde": k.marginally_bound(True),
        "marginally_bound_retrograde": k.marginally_bound(False),
        "isco_orbital_period_s": 2 * math.pi * (r_isco_p ** 1.5 + a) * Tg,
        "isco_orbital_speed_c": k.local_circular_speed(r_isco_p),
        "radiative_efficiency": eff,
        "horizon_angular_velocity_rad_s": omega_h,
        "horizon_area_m2": area,
        "surface_gravity_m_s2": surf_grav_horizon,
        "hawking_temperature_K": T_hawk,
        "bekenstein_hawking_entropy_J_K": entropy,
        "entropy_bits": entropy / (KB * math.log(2)),
        "evaporation_time_years": t_evap / YEAR,
        "evaporation_time_universe_ages": t_evap / AGE_UNIVERSE,
        "angular_momentum_kg_m2_s": J,
        "rotational_energy_J": E_rot,
        "rotational_energy_fraction": E_rot / (M * C * C),
        "eddington_luminosity_W": L_edd,
        "accretion_rate_kg_s": mdot,
        "accretion_rate_msun_yr": mdot * YEAR / M_SUN,
        "disk_luminosity_W": lum,
        "disk_luminosity_Lsun": lum / 3.828e26,
        "disk_peak_temperature_K": peak_T,
        "mean_density_kg_m3": rho_avg,
        "shadow_radius_schwarzschild_M": shadow_r,
        "tidal_accel_2m_at_horizon_m_s2": tidal_horizon,
    }


def page_thorne_shape(r, a, r_in):
    """Novikov-Thorne/Page-Thorne flux shape F(r)*(8 pi/3 mdot), geometric units."""
    if r <= r_in:
        return 0.0
    x = math.sqrt(r)
    x0 = math.sqrt(r_in)
    ac = math.acos(clamp(a, -1, 1))
    x1 = 2 * math.cos(ac / 3 - math.pi / 3)
    x2 = 2 * math.cos(ac / 3 + math.pi / 3)
    x3 = -2 * math.cos(ac / 3)
    br = x - x0 - 1.5 * a * math.log(x / x0)
    for xi, xj, xk in ((x1, x2, x3), (x2, x1, x3), (x3, x1, x2)):
        if abs(xi) < 1e-9:
            continue
        num = 3 * (xi - a) ** 2
        den = xi * (xi - xj) * (xi - xk)
        arg = (x - xi) / (x0 - xi)
        if den == 0 or arg <= 0:
            continue
        br -= num / den * math.log(arg)
    d = x ** 4 * (x ** 3 - 3 * x + 2 * a)
    if d <= 0:
        return 0.0
    return max(0.0, br / d)


def planck_nu(nu, T):
    x = H_PLANCK * nu / (KB * T)
    if x > 700:
        return 0.0
    return 2 * H_PLANCK * nu ** 3 / C ** 2 / math.expm1(x)


@lru_cache(maxsize=256)
def disk_profile_raw(spin, mass_solar, mdot_edd, n):
    k = Kerr(spin)
    a = k.a
    M, Lg, Tg = units(mass_solar)
    r_in = k.isco(True)
    eff = 1 - k.energy_circular(r_in)
    L_edd = 4 * math.pi * G * M * M_PROTON * C / SIGMA_THOMSON
    mdot = mdot_edd * L_edd / (max(eff, 1e-6) * C ** 2)
    scale = 3 * C ** 6 * mdot / (8 * math.pi * G * G * M * M)
    r_out = 60.0
    rs, F, T, Om, ut, g_face, vloc, tlike = [], [], [], [], [], [], [], []
    for i in range(n):
        r = r_in * (r_out / r_in) ** (i / (n - 1)) + 1e-9
        f = scale * page_thorne_shape(r, a, r_in)
        t = (f / SIGMA_SB) ** 0.25 if f > 0 else 0.0
        u = k.u_t_circular(r)
        rs.append(r)
        F.append(f)
        T.append(t)
        Om.append(k.omega_kepler(r) / Tg)
        ut.append(u)
        g_face.append(1 / u)
        vloc.append(k.local_circular_speed(r))
    T_peak = max(T)
    r_peak = rs[T.index(T_peak)]
    # spectrum: two-faced multicolour blackbody, nu L_nu
    nus = [10 ** (12 + 9 * i / 119) for i in range(120)]
    spec = []
    for nu in nus:
        s = 0.0
        for i in range(n - 1):
            if T[i] <= 0:
                continue
            R0, R1 = rs[i] * Lg, rs[i + 1] * Lg
            s += 4 * math.pi ** 2 * planck_nu(nu, T[i]) * 0.5 * (R0 + R1) * (R1 - R0)
        spec.append(nu * s)
    peak_nu = nus[spec.index(max(spec))] if max(spec) > 0 else 0
    return {
        "r": rs, "flux_W_m2": F, "T_K": T, "omega_rad_s": Om, "u_t": ut,
        "redshift_faceon": g_face, "v_local_c": vloc,
        "T_peak": T_peak, "r_T_peak": r_peak, "r_in": r_in, "r_out": r_out,
        "spectrum_nu_Hz": nus, "spectrum_nuLnu_W": spec, "spectrum_peak_Hz": peak_nu,
        "spectrum_peak_eV": H_PLANCK * peak_nu / 1.602176634e-19,
        "efficiency": eff,
    }


def band_name(nu):
    lam = C / nu if nu > 0 else 0
    if lam > 1e-3:
        return "radio/microwave"
    if lam > 7e-7:
        return "infrared"
    if lam > 3.8e-7:
        return "visible"
    if lam > 1e-8:
        return "ultraviolet"
    if lam > 1e-11:
        return "X-ray"
    return "gamma-ray"


@lru_cache(maxsize=512)
def compute_shadow(spin, inc_deg, n):
    """Bardeen (1973) critical curve seen by a distant observer at inclination inc."""
    k = Kerr(spin)
    a = k.a
    th = math.radians(clamp(inc_deg, 0.5, 179.5))
    r1 = k.photon_orbit(True) if a > 0 else k.photon_orbit(False)
    r2 = k.photon_orbit(False) if a > 0 else k.photon_orbit(True)
    upper, lower = [], []
    m = 4 * n
    for i in range(m + 1):
        r = r1 + (r2 - r1) * (0.5 - 0.5 * math.cos(math.pi * i / m))
        xi = -(r ** 3 - 3 * r * r + a * a * r + a * a) / (a * (r - 1))
        eta = r ** 3 * (4 * a * a - r * (r - 3) ** 2) / (a * a * (r - 1) ** 2)
        b2 = eta + a * a * math.cos(th) ** 2 - xi * xi / math.tan(th) ** 2
        if b2 < 0:
            continue
        alpha = -xi / math.sin(th)
        beta = math.sqrt(b2)
        upper.append([alpha, beta])
        lower.append([alpha, -beta])
    pts = upper + lower[::-1]
    if pts:
        pts.append(pts[0])
    xs = [p[0] for p in pts] or [0]
    ys = [p[1] for p in pts] or [0]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    area = 0.0
    for i in range(len(pts) - 1):
        area += pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1]
    area = abs(area) / 2
    return {
        "spin": a, "inclination_deg": math.degrees(th), "curve": decimate(pts, 2 * n),
        "width_M": width, "height_M": height, "centroid_shift_M": (max(xs) + min(xs)) / 2,
        "area_M2": area, "mean_radius_M": math.sqrt(area / math.pi),
        "asymmetry": 1 - width / height if height else 0,
        "schwarzschild_radius_M": 3 * math.sqrt(3),
    }


def compute_orbit(spin, r0, speed, tilt_deg, vr, tau_max, max_points):
    k = Kerr(spin)
    rp = k.r_plus
    r0 = max(r0, rp + 0.05)
    vc = k.local_circular_speed(r0) if r0 > k.photon_orbit(True) + 0.01 else 0.5
    tilt = math.radians(tilt_deg)
    vph = speed * vc * math.cos(tilt)
    vth = -speed * vc * math.sin(tilt)
    pt, pr, pth, pph = k.initial_momentum(r0, math.pi / 2, vr, vth, vph)
    E, Lz = -pt, pph
    Q = k.carter_constant(r0, math.pi / 2, pt, pth, pph, 1.0)
    y0 = [0.0, r0, math.pi / 2, 0.0, pr, pth]
    f = lambda y: k.geodesic_rhs(y, pt, pph)

    def stop(y):
        if y[1] < rp * 1.0005 + 1e-3:
            return "captured"
        if y[1] > 400:
            return "escaped"
        return None

    samples, reason = rk45(f, y0, 0.01, tau_max, stop, rtol=1e-10, atol=1e-12,
                           max_steps=60000, hmax=lambda y: 0.05 * y[1] ** 1.5)
    pts = []
    peri = []
    prev_dr = None
    H0 = k.hamiltonian(r0, math.pi / 2, pt, pr, pth, pph)
    for i, (lam, y) in enumerate(samples):
        t, r, th, ph, prr, pthh = y
        R = math.sqrt(r * r + k.a * k.a)
        pts.append([R * math.sin(th) * math.cos(ph), R * math.sin(th) * math.sin(ph),
                    r * math.cos(th), lam, t, r])
        dr = (k.metric(r, th)["delta"] / k.metric(r, th)["sigma"]) * prr
        if prev_dr is not None and prev_dr < 0 <= dr:
            peri.append(ph)
        prev_dr = dr
    last = samples[-1][1]
    H1 = k.hamiltonian(last[1], last[2], pt, last[4], last[5], pph)
    prec = []
    for i in range(1, len(peri)):
        prec.append(math.degrees(peri[i] - peri[i - 1] - 2 * math.pi))
    rs = [p[5] for p in pts]
    rmin, rmax = min(rs), max(rs)
    ecc = (rmax - rmin) / (rmax + rmin) if rmax + rmin > 0 else 0
    semi = 0.5 * (rmax + rmin)
    weak = math.degrees(6 * math.pi / (semi * (1 - ecc * ecc))) if semi > 0 and ecc < 1 else None
    return {
        "spin": k.a, "r0": r0, "E": E, "Lz": Lz, "Q": Q, "outcome": reason,
        "points": decimate(pts, max_points), "n_steps": len(samples),
        "proper_time_total": samples[-1][0], "coordinate_time_total": samples[-1][1][0],
        "r_min": rmin, "r_max": rmax, "eccentricity": ecc,
        "periapsis_precession_deg": prec, "weak_field_precession_deg": weak,
        "hamiltonian_drift": abs(H1 - H0),
        "circular_speed_local_c": vc,
        "isco": k.isco(True), "r_plus": rp,
    }


@lru_cache(maxsize=128)
def compute_photons(spin, n, bmax):
    k = Kerr(spin)
    rp = k.r_plus
    R0 = 30.0
    rays = []
    for i in range(n):
        b = -bmax + 2 * bmax * i / (n - 1) if n > 1 else 0
        x0 = -math.sqrt(max(R0 * R0 - b * b, 1))
        phi0 = math.atan2(b, x0)
        r0 = math.sqrt(x0 * x0 + b * b)
        nr, nph = math.cos(phi0), -math.sin(phi0)
        pt, pr, pth, pph = k.initial_momentum(r0, math.pi / 2, nr, 0.0, nph, null=True)
        f = lambda y, pt=pt, pph=pph: k.geodesic_rhs(y, pt, pph)

        def stop(y):
            if y[1] < rp * 1.001 + 1e-3:
                return "captured"
            if y[1] > R0 + 2 and y[4] > 0:
                return "escaped"
            return None

        samples, reason = rk45(f, [0, r0, math.pi / 2, phi0, pr, pth], 0.05, 400, stop,
                               rtol=1e-8, atol=1e-10, max_steps=8000,
                               hmax=lambda y: 0.1 * y[1])
        path = []
        for lam, y in decimate(samples, 160):
            R = math.sqrt(y[1] ** 2 + k.a ** 2)
            path.append([R * math.cos(y[3]), R * math.sin(y[3])])
        defl = None
        if reason == "escaped" and len(path) > 1:
            dx = path[-1][0] - path[-2][0]
            dy = path[-1][1] - path[-2][1]
            defl = math.degrees(math.atan2(dy, dx))
        rays.append({"b": b, "outcome": reason, "path": path, "deflection_deg": defl,
                     "L": pph, "E": -pt})
    return {"spin": k.a, "rays": rays, "r_plus": rp,
            "photon_orbit_prograde": k.photon_orbit(True),
            "photon_orbit_retrograde": k.photon_orbit(False),
            "ergosphere": 2.0, "isco": k.isco(True)}


def compute_infall(mass_solar, r0, length):
    """Radial plunge from rest at r0 (Schwarzschild), human-body tides."""
    M, Lg, Tg = units(mass_solar)
    r0 = max(r0, 2.05)
    E = math.sqrt(1 - 2 / r0)
    eta_h = math.acos(clamp(2 * 2 / r0 - 1, -1, 1))
    k = math.sqrt(r0 ** 3 / 8)
    tau_h = k * (eta_h + math.sin(eta_h))
    tau_sing = k * math.pi
    curve = []
    N = 240
    t_coord = 0.0
    r_prev = r0
    lethal = 1000.0
    for i in range(N + 1):
        eta = eta_h * i / N if i < N else eta_h * 0.9999
        r = r0 / 2 * (1 + math.cos(eta))
        tau = k * (eta + math.sin(eta))
        if i > 0:
            rm = 0.5 * (r + r_prev)
            dtdr = E / (1 - 2 / rm) / math.sqrt(max(1e-30, E * E - (1 - 2 / rm)))
            t_coord += dtdr * (r_prev - r)
        r_prev = r
        tid = 2 * G * M * length / (r * Lg) ** 3
        v_loc = math.sqrt(max(0.0, (2 / r - 2 / r0) / (1 - 2 / r0))) if r > 2 else 1
        curve.append({"r": r, "tau_s": tau * Tg, "t_s": t_coord * Tg,
                      "tidal_m_s2": tid, "v_local_c": min(v_loc, 1.0),
                      "dilation": math.sqrt(max(0, 1 - 2 / r))})
    r_spag_m = (2 * G * M * length / lethal) ** (1 / 3)
    r_spag = r_spag_m / Lg
    survives = r_spag < 2
    # remaining time from horizon to singularity (max proper time from horizon = pi M)
    return {
        "mass_solar": mass_solar, "r0": r0, "r0_m": r0 * Lg, "length_m": length,
        "proper_time_to_horizon_s": tau_h * Tg,
        "proper_time_to_singularity_s": tau_sing * Tg,
        "horizon_to_singularity_s": (tau_sing - tau_h) * Tg,
        "max_horizon_to_singularity_s": math.pi * Tg,
        "spaghettification_radius_M": r_spag,
        "spaghettification_radius_km": r_spag_m / 1000,
        "lethal_tidal_threshold_m_s2": lethal,
        "survives_horizon_crossing": survives,
        "curve": curve,
    }


def compute_dilation(mass_solar, spin):
    k = Kerr(spin)
    M, Lg, Tg = units(mass_solar)
    rows = []
    for r in [1.01 * k.r_plus, 1.1 * k.r_plus, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 20, 50, 100, 1000]:
        if r <= k.r_plus:
            continue
        m = k.metric(r, math.pi / 2)
        static = math.sqrt(-m["gtt"]) if m["gtt"] < 0 else None
        alpha, omega, _ = k.zamo(r, math.pi / 2)
        orb = None
        if r > k.photon_orbit(True) * 1.0001:
            u = k.u_t_circular(r)
            orb = 1 / u if math.isfinite(u) else None
        rows.append({"r": r, "r_km": r * Lg / 1000, "static": static, "zamo": alpha,
                     "orbiting": orb, "frame_drag_rad_s": omega / Tg,
                     "inside_ergosphere": r < 2.0})
    return {"spin": k.a, "rows": rows}


# ----------------------------------------------------------------------------
# HTTP layer
# ----------------------------------------------------------------------------
def qf(q, key, default, lo, hi):
    try:
        v = float(q.get(key, [default])[0])
        if not math.isfinite(v):
            v = default
    except (TypeError, ValueError):
        v = default
    return clamp(v, lo, hi)


def qi(q, key, default, lo, hi):
    return int(round(qf(q, key, default, lo, hi)))


def jsonable(o):
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return float(f"{o:.9g}")
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


START = time.time()
STATS = {"requests": 0}


def route(path, q):
    if path == "/api/health":
        return {"ok": True, "version": VERSION, "uptime_s": time.time() - START,
                "python": sys.version.split()[0], "requests": STATS["requests"],
                "engine": "Kerr metric, Boyer-Lindquist Hamiltonian, RK45 Dormand-Prince"}
    if path == "/api/metrics":
        m = qf(q, "mass", 4.3e6, 1e-3, 1e12)
        return compute_metrics(m, round(qf(q, "spin", 0.9, -0.9999, 0.9999), 5),
                               round(qf(q, "mdot", 0.1, 1e-6, 10), 6))
    if path == "/api/shadow":
        return compute_shadow(round(qf(q, "spin", 0.9, -0.9999, 0.9999), 5),
                              round(qf(q, "inc", 80, 0, 180), 3), qi(q, "n", 180, 16, 1000))
    if path == "/api/disk":
        d = dict(disk_profile_raw(round(qf(q, "spin", 0.9, -0.9999, 0.9999), 5),
                                  qf(q, "mass", 4.3e6, 1e-3, 1e12),
                                  round(qf(q, "mdot", 0.1, 1e-6, 10), 6),
                                  qi(q, "n", 160, 20, 600)))
        d["spectrum_peak_band"] = band_name(d["spectrum_peak_Hz"])
        return d
    if path == "/api/orbit":
        return compute_orbit(qf(q, "spin", 0.9, -0.9999, 0.9999), qf(q, "r0", 10, 1, 300),
                             qf(q, "speed", 1.0, 0, 1.6), qf(q, "tilt", 25, -90, 90),
                             qf(q, "vr", 0.0, -0.9, 0.9), qf(q, "tau", 3000, 10, 50000),
                             qi(q, "points", 3000, 50, 20000))
    if path == "/api/photons":
        return compute_photons(round(qf(q, "spin", 0.9, -0.9999, 0.9999), 4),
                               qi(q, "n", 41, 3, 151), round(qf(q, "bmax", 9, 1, 30), 3))
    if path == "/api/infall":
        return compute_infall(qf(q, "mass", 4.3e6, 1e-3, 1e12), qf(q, "r0", 20, 2.05, 1e6),
                              qf(q, "length", 2.0, 0.01, 1000))
    if path == "/api/dilation":
        return compute_dilation(qf(q, "mass", 4.3e6, 1e-3, 1e12),
                                qf(q, "spin", 0.9, -0.9999, 0.9999))
    return None


class Handler(BaseHTTPRequestHandler):
    server_version = "EventHorizon/" + VERSION

    def log_message(self, fmt, *args):
        if "--verbose" in sys.argv:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        STATS["requests"] += 1
        u = urlparse(self.path)
        if u.path in ("/", "/index.html", "/blackhole.html"):
            try:
                with open(HTML_FILE, "rb") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
            except OSError:
                self._send(404, b"blackhole.html not found next to the server", "text/plain")
            return
        if u.path.startswith("/api/"):
            t0 = time.perf_counter()
            try:
                res = route(u.path, parse_qs(u.query))
            except Exception as e:  # report numerical failures to the client
                self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
                return
            if res is None:
                self._send(404, b'{"error":"unknown endpoint"}', "application/json")
                return
            res = dict(res)
            res["compute_ms"] = (time.perf_counter() - t0) * 1000
            self._send(200, json.dumps(jsonable(res)).encode(), "application/json")
            return
        self._send(404, b"not found", "text/plain")


def self_test():
    k = Kerr(0.0)
    assert abs(k.isco() - 6) < 1e-6 and abs(k.photon_orbit() - 3) < 1e-6
    k = Kerr(0.9999)
    assert k.isco() < 1.2
    s = compute_shadow(1e-7, 90, 64)
    assert abs(s["mean_radius_M"] - 3 * math.sqrt(3)) < 0.05, s["mean_radius_M"]
    o = compute_orbit(0.0, 10, 1.0, 0, 0, 300, 200)
    assert abs(o["r_max"] - 10) < 1e-3 and abs(o["r_min"] - 10) < 1e-3, (o["r_min"], o["r_max"])
    return True


def main():
    ap = argparse.ArgumentParser(description="Event Horizon black hole simulation server")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    self_test()
    print(f"  physics self-test passed ({(time.perf_counter() - t0) * 1000:.0f} ms)")
    if args.test:
        return
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"  EVENT HORIZON v{VERSION}  ->  {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  shutting down")
        srv.server_close()


if __name__ == "__main__":
    main()

"""The quality term.

    g(d, beta) = (d / f_x) / cos(beta)          m per pixel on the surface
    b_m        = f_x * t_exp * v_perp / d       px, motion smear
    c          = k_dof * |d - d_f| / d          px, defocus
    b_tot      = sqrt(b_0^2 + b_m^2 + c^2)      px, effective PSF width
    w_min      = k_r * b_tot * g(d, beta)       m, finest resolvable feature
    Q_def      = clip(w_t / w_min, 0, Q_max)

Quadrature composition turns GSD, blur and defocus into one scalar without
three arbitrary weights, and w_min is exactly what the printed line-pair panel
measures, so the model is falsifiable.
"""

from dataclasses import dataclass

import numpy as np

_GH_T, _GH_W = np.polynomial.hermite.hermgauss(5)
GH_NODES = np.sqrt(2.0) * _GH_T
GH_WEIGHTS = _GH_W / np.sqrt(np.pi)


@dataclass
class QualityParams:

    f_x: float = 1431.85          # [M] camera_calibration.yaml
    f_y: float = 1431.85          # [M]
    c_x: float = 970.57           # [M]
    c_y: float = 540.0            # [M]
    width: int = 1920             # [M]
    height: int = 1080            # [M]

    t_exp: float = 0.00184        # [M] s, upper bound
    b_0: float = 1.461            # [M] px, PSF width at best focus
    k_dof: float = 2.768          # [M] px per unit relative defocus
    d_f: float = 1.567            # [M] m, focus distance
    k_r: float = 1.234            # [M] median over 9 standoffs, sd 0.231

    # Where MTF50 stays within 70 per cent of its peak of 0.1282 cy/px.
    d_min: float = 1.02           # [M] m, DoF band
    d_max: float = 3.40           # [M] m
    cap_to_concave: bool = True
    beta_max_deg: float = 60.0    # [T] incidence beyond which a patch is not
                                  #     usefully imaged
    Q_max: float = 2.0            # quality is clipped: twice the required
                                  # width is already enough

    def half_fov_deg(self):
        return (np.degrees(2 * np.arctan(self.width / (2 * self.f_x))) / 2,
                np.degrees(2 * np.arctan(self.height / (2 * self.f_y))) / 2)


_BAND_CACHE = {}


def effective_band(p: QualityParams, v_perp=0.5, w_t=0.002):
    """(d_min, d_max) actually used, after the concavity cap."""
    if not p.cap_to_concave:
        return p.d_min, p.d_max
    key = (p.d_min, p.d_max, p.d_f, p.k_dof, p.b_0, p.t_exp, p.k_r,
           p.f_x, v_perp, w_t)
    if key not in _BAND_CACHE:
        band = concave_band(p, v_perp, w_t)
        hi = p.d_max if band is None else min(p.d_max, band[1])
        lo = p.d_min if band is None else max(p.d_min, band[0])
        if hi <= lo:
            raise ValueError(
                f"the concave region {band} does not overlap the DoF band "
                f"[{p.d_min}, {p.d_max}]; check k_dof and d_f")
        _BAND_CACHE[key] = (lo, hi)
    return _BAND_CACHE[key]


def gsd(d, beta, p: QualityParams):
    """Metres per pixel on the surface."""
    return (d / p.f_x) / np.cos(beta)


def blur_px(d, v_perp, p: QualityParams):
    """Total effective PSF width in pixels."""
    b_m = p.f_x * p.t_exp * v_perp / d
    c = p.k_dof * np.abs(d - p.d_f) / d
    return np.sqrt(p.b_0 ** 2 + b_m ** 2 + c ** 2)


def w_min(d, beta, v_perp, p: QualityParams):
    """Finest resolvable feature, metres. This is what the panel measures."""
    return p.k_r * blur_px(d, v_perp, p) * gsd(d, beta, p)


def q_def(d, beta, v_perp, w_t, p: QualityParams):
    """Nominal quality at an exact depth."""
    d = np.maximum(np.asarray(d, float), 1e-3)
    return np.clip(w_t / w_min(d, beta, v_perp, p), 0.0, p.Q_max)


def sigma_d(n_hat, p_view, p_nav, Sigma_pp, sigma_s_over_s):
    """Standard deviation of achieved depth. planner_design.md 4.4."""
    n = np.asarray(n_hat, float)
    pos = np.asarray(p_view, float) - np.asarray(p_nav, float)
    var_pos = float(n @ np.asarray(Sigma_pp, float) @ n)
    scale = float(n @ pos) * float(sigma_s_over_s)
    return float(np.sqrt(max(var_pos, 0.0) + scale ** 2))


def expected_q(d, beta, v_perp, w_t, sd, p: QualityParams):
    """E[Q_def] over the depth distribution, 5-node Gauss-Hermite.

    Returns the nominal value when sd is negligible, so a caller can compare
    E[Q] against Q(E[d]) without a special case.
    """
    if sd <= 1e-9:
        return float(q_def(d, beta, v_perp, w_t, p))
    depths = d + GH_NODES * sd
    return float(np.sum(GH_WEIGHTS * q_def(depths, beta, v_perp, w_t, p)))


# ------------------------------------------------------------------ checks
def peak_depth(p: QualityParams, v_perp=0.5, w_t=0.002, n=4001):
    """Depth at which Q_def peaks, by dense evaluation over the band."""
    d = np.linspace(max(p.d_min * 0.5, 0.05), p.d_max * 1.5, n)
    q = q_def(d, 0.0, v_perp, w_t, p)
    return float(d[int(np.argmax(q))])


def curvature_at(d0, p: QualityParams, v_perp=0.5, w_t=0.002, h=1e-3):
    """Second derivative of Q_def, by central difference."""
    f = lambda x: float(q_def(x, 0.0, v_perp, w_t, p))
    return (f(d0 + h) - 2 * f(d0) + f(d0 - h)) / h ** 2


def concave_band(p: QualityParams, v_perp=0.5, w_t=0.002, n=401):
    """Depth range over which Q_def is concave.
    """
    d = np.linspace(max(p.d_min * 0.6, 0.05), p.d_max * 1.2, n)
    k = np.array([curvature_at(x, p, v_perp, w_t) for x in d])
    neg = np.where(k < 0)[0]
    if len(neg) == 0:
        return None
    return float(d[neg[0]]), float(d[neg[-1]])


def assert_concave_near_peak(p: QualityParams, v_perp=0.5, w_t=0.002):
    """4.3 in executable form.
    """
    d0 = peak_depth(p, v_perp, w_t)
    q2 = curvature_at(d0, p, v_perp, w_t)
    if not (p.d_min <= d0 <= p.d_max):
        raise AssertionError(
            f"Q_def peaks at {d0:.3f} m, outside the DoF band "
            f"[{p.d_min}, {p.d_max}]; every candidate would sit on the same "
            "side of the peak and the concavity argument would not apply")
    if q2 >= 0:
        raise AssertionError(
            f"Q_def is convex at its peak ({q2:.3f}); uncertainty would "
            "appear to improve quality. Check k_dof and d_f.")
    return d0, q2

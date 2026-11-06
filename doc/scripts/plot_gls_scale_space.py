# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Figure of the page maths_spheres_page: Growing Least Squares parameters at a point of a 2D curve with two
characteristic scales (a circle of radius 3 with small undulations), as functions of the scale t. The scale
derivatives of the geometric variation are computed by finite differences."""

import numpy as np
import ponca_numpy as pn

R, A, M = 3.0, 0.04, 24  # radius of the circle, amplitude and frequency of the undulations


def curve(theta):
    r = R + A * np.cos(M * theta)
    dr = -A * M * np.sin(M * theta)
    pos = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    tangent = np.stack([dr * np.cos(theta) - r * np.sin(theta), dr * np.sin(theta) + r * np.cos(theta)], axis=1)
    nrm = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)  # outward normal
    return pos, nrm / np.linalg.norm(nrm, axis=1, keepdims=True)


def gls_at(pts, nrm, p, t):
    w, x = pn.weights(pts, p, t)
    m = w > 0
    return pn.gls_parameters(pn.oriented_sphere_fit(x[m], nrm[m], w[m]))


def main():
    plt = pn.setup_style()
    pts, nrm = curve(np.linspace(-np.pi, np.pi, 20000, endpoint=False))
    p = curve(np.array([0.0]))[0][0]  # top of an undulation
    ts = np.geomspace(0.03, 4.0, 90)
    tau, kappa, nu = [], [], []
    for t in ts:
        h = 1e-3 * t
        tm, em, km = gls_at(pts, nrm, p, t - h)
        tp, ep, kp = gls_at(pts, nrm, p, t + h)
        t0, e0, k0 = gls_at(pts, nrm, p, t)
        dtau, deta, dkappa = (tp - tm) / (2 * h), (ep - em) / (2 * h), (kp - km) / (2 * h)
        tau.append(t0 / t)
        kappa.append(k0)
        nu.append(dtau**2 + t**2 * deta @ deta + (t**2 * dkappa) ** 2)

    r0 = R + A
    kappa_peak = (1 + A * M**2 / r0) / r0  # curvature of the curve at the evaluation point
    wavelength = 2 * np.pi * R / M
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(ts, kappa, color=pn.SERIES[0])
    axes[0].axhline(kappa_peak, color=pn.SERIES[1], linestyle="--", linewidth=1.5)
    axes[0].axhline(1 / R, color=pn.SERIES[2], linestyle="--", linewidth=1.5)
    axes[0].text(ts[0], kappa_peak * 0.88, "curvature of the undulation", color=pn.INK_SECONDARY, va="top")
    axes[0].text(ts[-1], 1 / R + 0.15, "curvature of the circle, 1/R", color=pn.INK_SECONDARY, ha="right")
    axes[0].set_ylabel(r"$\kappa$")
    axes[0].set_title("GLS parameters at the top of an undulation, as functions of the scale")
    axes[1].plot(ts, tau, color=pn.SERIES[0])
    axes[1].set_ylabel(r"$\tau / t$")
    axes[2].plot(ts, nu, color=pn.SERIES[0])
    axes[2].set_ylabel(r"geometric variation $\nu$")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("scale $t$ (log scale)")
    for ax in axes:
        ax.axvline(wavelength, color=pn.INK_SECONDARY, linewidth=0.8)
    axes[2].text(wavelength * 1.05, max(nu) * 0.9, "wavelength of the undulations", color=pn.INK_SECONDARY)
    fig.tight_layout()
    pn.save(fig, "maths_gls_scale_space.png")


if __name__ == "__main__":
    main()

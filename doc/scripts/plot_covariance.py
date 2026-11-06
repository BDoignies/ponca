# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Figure of the page maths_covariance_page.

(a) Covariance analysis of a weighted 2D neighborhood: barycenter, principal axes and fitted line.
(b, c) Distance between the evaluation point and the PCA plane, and surface variation, measured on a densely sampled
       unit sphere with constant weights, compared with their second order expansions t^2 / (4R) and t^2 / (8R^2).
"""

import numpy as np
import ponca_numpy as pn


def panel_illustration(ax, plt):
    rng = np.random.default_rng(3)
    s = rng.uniform(-1.1, 1.1, 500)
    pts = np.stack([s, 1.1 * s**2], axis=1) + rng.normal(0, 0.015, (500, 2))
    p = np.array([0.0, 0.0])
    t = 0.85
    w, x = pn.weights(pts, p, t)
    b, evals, evecs = pn.covariance_analysis(x[w > 0], w[w > 0])
    n, d = evecs[:, 0], evecs[:, 1]
    n, d = n * np.sign(n[1]), d * np.sign(d[0])  # orient the axes for the drawing only
    order = np.argsort(w)
    sc = ax.scatter(pts[order, 0], pts[order, 1], c=w[order], cmap=pn.sequential_colormap(), vmin=0, vmax=1,
                    s=10, edgecolors="none")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label=r"weight $w_i$")
    ax.add_patch(plt.Circle(p, t, fill=False, linestyle="--", color=pn.INK_SECONDARY, linewidth=1))
    c = p + b
    ax.plot(*np.stack([c - 1.1 * d, c + 1.1 * d]).T, color=pn.SERIES[1], label=r"fitted line, normal $\mathbf{v}_0$")
    ax.annotate("", xy=c + 2 * np.sqrt(evals[1]) * d, xytext=c,
                arrowprops=dict(arrowstyle="->", color=pn.INK, linewidth=1.5))
    ax.text(*(c + 2 * np.sqrt(evals[1]) * d + np.array([-0.05, 0.08])), r"$2\sqrt{\lambda_1}\,\mathbf{v}_1$",
            fontsize=10)
    ax.annotate("", xy=c + 0.35 * n, xytext=c, arrowprops=dict(arrowstyle="->", color=pn.INK, linewidth=1.5))
    ax.text(*(c + 0.35 * n + np.array([0.03, 0.0])), r"$\mathbf{v}_0$", fontsize=10)
    foot = c + d * (d @ (p - c))  # projection of p on the line
    ax.plot(*np.stack([p, foot]).T, color=pn.SERIES[2], linewidth=3, label=r"distance $\delta$ to the line")
    ax.text(*(0.5 * (p + foot) + np.array([0.05, -0.02])), r"$\delta$", color=pn.INK, fontsize=11)
    ax.plot(*p, "o", color=pn.INK, markersize=6, label=r"evaluation point $\mathbf{p}$")
    ax.plot(*c, "s", color=pn.SERIES[1], markersize=6, label=r"barycenter $\mathbf{p} + \mathbf{b}$")
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.9, 1.25)
    ax.set_title("(a) Covariance analysis of a neighborhood")
    ax.legend(loc="lower right", fontsize=8)


def sphere_measurements():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(600000, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)  # unit sphere, R = 1
    p = np.array([0.0, 0.0, 1.0])
    ts = np.linspace(0.05, 0.8, 16)
    delta, sigma = [], []
    for t in ts:
        w, x = pn.weights(pts, p, t, "ConstantWeightKernel")
        b, evals, evecs = pn.covariance_analysis(x[w > 0], w[w > 0])
        delta.append(abs(evecs[:, 0] @ b))
        sigma.append(pn.surface_variation(evals))
    return ts, np.array(delta), np.array(sigma)


def main():
    plt = pn.setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    panel_illustration(axes[0], plt)
    ts, delta, sigma = sphere_measurements()
    tt = np.linspace(0, 0.82, 200)
    axes[1].plot(tt, tt**2 / 4, color=pn.SERIES[0], label=r"expansion $t^2 / (4R)$")
    axes[1].plot(ts, delta, "o", color=pn.SERIES[1], markersize=6, label="measured")
    axes[1].set_title(r"(b) Distance $\delta$ to the PCA plane")
    axes[2].plot(tt, tt**2 / 8, color=pn.SERIES[0], label=r"expansion $t^2 / (8R^2)$")
    axes[2].plot(ts, sigma, "o", color=pn.SERIES[1], markersize=6, label="measured")
    axes[2].set_title(r"(c) Surface variation $3\lambda_0 / (\lambda_0 + \lambda_1 + \lambda_2)$")
    for ax in axes[1:]:
        ax.set_xlabel(r"scale $t$ (unit sphere, constant weights)")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_xlim(0, 0.82)
    fig.tight_layout()
    pn.save(fig, "maths_covariance.png")


if __name__ == "__main__":
    main()

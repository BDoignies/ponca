# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Figure of the page maths_spheres_page: scalar fields fitted by the covariance plane fit and the three algebraic
sphere fits on the same noisy 2D neighborhood (sampled on an ellipse)."""

import numpy as np
import ponca_numpy as pn


def ellipse(theta, a=1.6, b=1.0):
    pos = np.stack([a * np.cos(theta), b * np.sin(theta)], axis=1)
    nrm = np.stack([np.cos(theta) / a, np.sin(theta) / b], axis=1)
    return pos, nrm / np.linalg.norm(nrm, axis=1, keepdims=True)


def main():
    plt = pn.setup_style()
    rng = np.random.default_rng(7)
    pts, nrm = ellipse(rng.uniform(-np.pi, np.pi, 220))
    pts += rng.normal(0, 0.015, pts.shape)
    flipped = nrm * np.where(rng.uniform(size=len(nrm)) < 0.5, -1, 1)[:, None]
    p = ellipse(np.array([0.25]))[0][0]
    t = 0.9
    w, x = pn.weights(pts, p, t)
    m = w > 0

    xs = np.linspace(p[0] - 1.6, p[0] + 1.2, 260)
    ys = np.linspace(p[1] - 1.5, p[1] + 1.2, 260)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1) - p

    fields = []
    b, evals, evecs = pn.covariance_analysis(x[m], w[m])
    fields.append(("CovariancePlaneFit (unsigned)", (grid - b) @ evecs[:, 0], None))
    u = pn.normalize(pn.sphere_fit(x[m], w[m]))
    fields.append(("SphereFit (unsigned)", pn.potential(u, grid), None))
    u = pn.normalize(pn.oriented_sphere_fit(x[m], nrm[m], w[m]))
    fields.append(("OrientedSphereFit (APSS)", pn.potential(u, grid), nrm))
    u = pn.normalize(pn.unoriented_sphere_fit(x[m], flipped[m], w[m]))
    fields.append(("UnorientedSphereFit (unsigned)", pn.potential(u, grid), flipped))

    curve = ellipse(np.linspace(-np.pi, np.pi, 400))[0]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), sharey=True)
    levels = np.linspace(-0.6, 0.6, 25)
    for ax, (title, values, normals) in zip(axes, fields):
        cf = ax.contourf(gx, gy, np.clip(values, -0.6, 0.6).reshape(gx.shape), levels=levels,
                         cmap=pn.diverging_colormap(), extend="both")
        ax.contour(gx, gy, values.reshape(gx.shape), levels=[0], colors=pn.INK, linewidths=2)
        ax.plot(*curve.T, color=pn.INK_SECONDARY, linewidth=1, linestyle=":")
        ax.scatter(*pts.T, s=6, color=pn.INK, zorder=3)
        if normals is not None:
            ax.quiver(*pts[m].T, *normals[m].T, color=pn.INK, scale=18, width=0.004, zorder=3)
        ax.add_patch(plt.Circle(p, t, fill=False, linestyle="--", color=pn.INK, linewidth=1))
        ax.plot(*p, "o", color=pn.SERIES[3], markeredgecolor=pn.INK, markersize=8, zorder=4)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys[0], ys[-1])
        ax.grid(False)
    fig.colorbar(cf, ax=axes, fraction=0.012, pad=0.01, label="Pratt-normalized scalar field")
    pn.save(fig, "maths_sphere_fits.png")


if __name__ == "__main__":
    main()

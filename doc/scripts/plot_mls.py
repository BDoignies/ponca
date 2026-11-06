# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Figure of the page maths_mls_page: implicit function of the MLS surface defined by oriented sphere fits on a
sparse noisy 2D point cloud, its zero isoline, and the successive positions of the MLS projection of a few points."""

import numpy as np
import ponca_numpy as pn


def shape(theta):
    r = 1 + 0.25 * np.cos(2 * theta) + 0.12 * np.sin(3 * theta)
    dr = -0.5 * np.sin(2 * theta) + 0.36 * np.cos(3 * theta)
    pos = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    tangent = np.stack([dr * np.cos(theta) - r * np.sin(theta), dr * np.sin(theta) + r * np.cos(theta)], axis=1)
    nrm = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    return pos, nrm / np.linalg.norm(nrm, axis=1, keepdims=True)


def main():
    plt = pn.setup_style()
    rng = np.random.default_rng(11)
    pts, nrm = shape(np.sort(rng.uniform(-np.pi, np.pi, 70)))
    pts += rng.normal(0, 0.012, pts.shape)
    t = 0.55

    xs = np.linspace(-1.75, 1.75, 150)
    ys = np.linspace(-1.55, 1.55, 135)
    gx, gy = np.meshgrid(xs, ys)
    field = np.array([pn.mls_function(np.array([a, b]), pts, nrm, t) for a, b in zip(gx.ravel(), gy.ravel())])
    field = field.reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    levels = np.linspace(-0.4, 0.4, 25)
    cf = ax.contourf(gx, gy, np.clip(field, -0.4, 0.4), levels=levels, cmap=pn.diverging_colormap(), extend="both")
    ax.contour(gx, gy, field, levels=[0], colors=pn.INK, linewidths=2)
    ax.scatter(*pts.T, s=12, color=pn.INK, zorder=3, label="samples")
    ax.quiver(*pts.T, *nrm.T, color=pn.INK, scale=25, width=0.003, zorder=3)
    angles = np.array([0.35, 1.2, 1.9, 2.7, 3.6, 4.3, 5.1, 5.8])
    offsets = np.array([0.32, -0.3, 0.3, -0.28, 0.33, -0.3, 0.3, -0.25])
    anchors, normals = shape(angles)
    starts = anchors + offsets[:, None] * normals  # positions within reach of the samples
    for k, q in enumerate(starts):
        traj = pn.mls_projection(q, pts, nrm, t, n_iter=10)
        ax.plot(*traj.T, "-o", color=pn.SERIES[1], markersize=4, linewidth=1.5, zorder=4,
                label="MLS projection iterates" if k == 0 else None)
        ax.plot(*q, "o", markerfacecolor=pn.SURFACE, markeredgecolor=pn.SERIES[1], markersize=8, zorder=5,
                label="starting positions" if k == 0 else None)
    ax.plot([], [], color=pn.INK, linewidth=2, label="MLS surface (zero isoline)")
    ax.text(0, 0.1, "undefined\n(too few neighbors)", ha="center", va="center", color=pn.INK_SECONDARY, fontsize=9)
    fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02, label="implicit function $f(\\mathbf{q})$")
    ax.set_aspect("equal")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.grid(False)
    ax.legend(loc="lower left", fontsize=8, frameon=True, facecolor=pn.SURFACE)
    ax.set_title("MLS surface of oriented sphere fits and projections")
    pn.save(fig, "maths_mls.png")


if __name__ == "__main__":
    main()

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Figure of the page maths_curvature_page: quadratic Monge patch fitted to noisy samples of a saddle, with its
normal and principal directions at the evaluation point."""

import numpy as np
import ponca_numpy as pn

K1, K2 = 1.2, -0.6  # principal curvatures of the sampled surface z = (K1 x^2 + K2 y^2) / 2


def main():
    plt = pn.setup_style()
    rng = np.random.default_rng(5)
    r, a = np.sqrt(rng.uniform(0, 1, 500)), rng.uniform(0, 2 * np.pi, 500)
    xy = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
    pts = np.column_stack([xy, 0.5 * (K1 * xy[:, 0] ** 2 + K2 * xy[:, 1] ** 2)]) + rng.normal(0, 0.01, (500, 3))
    p, t = np.zeros(3), 1.0
    w, x = pn.weights(pts, p, t)
    m = w > 0
    frame, c, kappas, dirs = pn.monge_patch(x[m], w[m], normal_hint=np.array([0, 0, 1.0]))

    # Fitted height field over the tangent plane, mapped to the global frame
    gu, gv = np.meshgrid(np.linspace(-0.9, 0.9, 25), np.linspace(-0.9, 0.9, 25))
    gh = c[0] * gu**2 + c[1] * gv**2 + c[2] * gu * gv + c[3] * gu + c[4] * gv + c[5]
    surf = np.einsum("ij,jkl->ikl", frame, np.stack([gh, gu, gv]))

    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_wireframe(*surf, color=pn.SERIES[0], linewidth=0.6, alpha=0.8)
    ax.scatter(*pts[m].T, s=4, color=pn.INK, depthshade=False)
    arrows = [(frame[:, 0], pn.INK, "normal"),
              (dirs[:, 0], pn.SERIES[1], rf"$k_{{\min}}$ = {kappas[0]:.2f} (exact {K2})"),
              (dirs[:, 1], pn.SERIES[2], rf"$k_{{\max}}$ = {kappas[1]:.2f} (exact {K1})")]
    for v, color, label in arrows:
        ax.quiver(*p, *(0.7 * v), color=color, linewidth=2.5, arrow_length_ratio=0.15, label=label)
    ax.set_title("Monge patch fitted to a noisy saddle (t = 1)")
    ax.legend(loc="upper left", fontsize=9)
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1, 1, 0.6))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.98, 0.98, 0.98, 1.0))
    pn.save(fig, "maths_monge_patch.png")


if __name__ == "__main__":
    main()

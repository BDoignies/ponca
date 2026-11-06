# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Figure of the page maths_curvature_page: mean curvature estimated on a noisy unit sphere by three estimators, as
a function of the scale (mean and standard deviation over random evaluation points)."""

import numpy as np
import ponca_numpy as pn


def main():
    plt = pn.setup_style()
    rng = np.random.default_rng(2)
    n = 150000
    nrm = rng.normal(size=(n, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    pts = nrm + rng.normal(0, 0.004, (n, 3))                 # unit sphere, noisy positions
    nrm_noisy = nrm + rng.normal(0, 0.03, (n, 3))
    nrm_noisy /= np.linalg.norm(nrm_noisy, axis=1, keepdims=True)
    evaluation = rng.choice(n, 60, replace=False)
    ts = np.geomspace(0.03, 0.7, 14)

    results = {"gls": [], "monge": [], "pca": []}
    for t in ts:
        gls, monge, pca = [], [], []
        for i in evaluation:
            p = pts[i]
            w, x = pn.weights(pts, p, t)
            m = w > 0
            gls.append(pn.gls_parameters(pn.oriented_sphere_fit(x[m], nrm_noisy[m], w[m]))[2])
            k = pn.monge_patch(x[m], w[m], restricted=True, normal_hint=nrm[i])[2]
            monge.append(-k.mean())                          # sign flipped: Monge convention, outward normal
            wc, xc = pn.weights(pts, p, t, "ConstantWeightKernel")
            mc = wc > 0
            b, evals, evecs = pn.covariance_analysis(xc[mc], wc[mc])
            pca.append(4 * abs(evecs[:, 0] @ b) / t**2)      # H ~ 4 delta / t^2 (constant weights)
        for key, values in zip(("gls", "monge", "pca"), (gls, monge, pca)):
            results[key].append((np.mean(values), np.std(values)))

    titles = {"gls": r"GLS $\kappa$ (oriented sphere fit)", "monge": r"Monge patch, $-H$ (restricted)",
              "pca": r"Distance to the PCA plane, $4\delta / t^2$"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for k, (ax, key) in enumerate(zip(axes, ("gls", "monge", "pca"))):
        mean, std = np.array(results[key]).T
        ax.axhline(1.0, color=pn.INK_SECONDARY, linewidth=1)
        ax.fill_between(ts, mean - std, mean + std, color=pn.SERIES[k], alpha=0.2, linewidth=0,
                        label="mean $\\pm$ standard deviation")
        ax.plot(ts, mean, "-o", color=pn.SERIES[k], markersize=5)
        ax.set_xscale("log")
        ax.set_xticks([0.03, 0.05, 0.1, 0.2, 0.5])
        ax.set_xticklabels(["0.03", "0.05", "0.1", "0.2", "0.5"])
        ax.minorticks_off()
        ax.set_title(titles[key], fontsize=10)
        ax.set_xlabel("scale $t$ (log scale)")
        ax.text(0.031, 1.03, "exact value", color=pn.INK_SECONDARY, va="bottom", fontsize=9)
    axes[0].set_ylim(0.4, 1.6)
    axes[0].set_ylabel("mean curvature estimate")
    fig.suptitle("Mean curvature of a noisy unit sphere: estimates over 60 points", fontsize=11)
    fig.tight_layout()
    pn.save(fig, "maths_curvature_scale.png")


if __name__ == "__main__":
    main()

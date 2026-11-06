# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Figure of the page maths_weights_page: the weight kernels of Ponca and their first derivative."""

import numpy as np
import ponca_numpy as pn


def main():
    plt = pn.setup_style()
    x = np.linspace(0, 1.4, 800)
    styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, (name, (phi, dphi, compact)) in enumerate(pn.KERNELS.items()):
        label = name.replace("WeightKernel", "") + ("" if compact else " (not compact)")
        axes[0].plot(x, phi(x), color=pn.SERIES[k], linestyle=styles[k], label=label)
        axes[1].plot(x, dphi(x), color=pn.SERIES[k], linestyle=styles[k], label=label)
    axes[0].set_title(r"Kernel $\varphi(x)$")
    axes[1].set_title(r"First derivative $\varphi'(x)$")
    for ax in axes:
        ax.set_xlabel(r"normalized distance $x = \|\mathbf{q} - \mathbf{p}\| / t$")
        ax.axvline(1, color=pn.INK_SECONDARY, linewidth=0.8)
        ax.set_xlim(0, 1.4)
    axes[0].set_ylim(-0.05, 1.1)
    axes[1].legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    pn.save(fig, "maths_kernels.png")


if __name__ == "__main__":
    main()

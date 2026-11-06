# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Figure of the page maths_spatial_page: cells of a 2D kd-tree built with Ponca's splitting rule (midpoint of the
bounding box, along its largest dimension), and the leaves visited by a range query."""

import numpy as np
import ponca_numpy as pn


def leaves(node):
    if node["leaf"]:
        return [node]
    return leaves(node["children"][0]) + leaves(node["children"][1])


def main():
    plt = pn.setup_style()
    rng = np.random.default_rng(4)
    pts = np.vstack([rng.uniform(0, 1, (300, 2)), rng.normal([0.72, 0.68], 0.05, (220, 2))])
    tree = pn.build_kdtree(pts, leaf_size=12)
    q, r = np.array([0.62, 0.58]), 0.13
    result, visited = pn.range_query(tree, pts, q, r)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    all_leaves = leaves(tree)
    for leaf in all_leaves:
        lo, hi = leaf["box"]
        is_visited = any(leaf is v for v in visited)
        ax.add_patch(plt.Rectangle(lo, *(hi - lo), fill=is_visited, facecolor="#cde2fb",
                                   edgecolor=pn.INK_SECONDARY, linewidth=0.8))
    ax.scatter(*pts.T, s=6, color=pn.INK_SECONDARY, zorder=3)
    ax.scatter(*pts[result].T, s=14, color=pn.SERIES[1], zorder=4, label=f"{len(result)} points in the ball")
    ax.add_patch(plt.Circle(q, r, fill=False, color=pn.INK, linewidth=1.5, zorder=5))
    ax.plot(*q, "*", color=pn.INK, markersize=12, zorder=6, label="query position")
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor="#cde2fb", edgecolor=pn.INK_SECONDARY,
                               label=f"visited leaves: {len(visited)} of {len(all_leaves)}"))
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor=pn.SURFACE)
    ax.set_title("Bounding boxes of the leaves of a kd-tree (12 points per leaf)")
    pn.save(fig, "maths_kdtree.png")


if __name__ == "__main__":
    main()

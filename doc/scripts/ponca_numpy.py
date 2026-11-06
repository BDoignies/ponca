# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Reference NumPy implementations of the formulas of the mathematical background of the Ponca documentation.

The functions of this module re-implement, in plain NumPy and in a few lines each, the formulas described in the pages
`doc/src/maths*.mdoc`. They are written for clarity rather than efficiency, do not call the library, and are only used
by the scripts of this folder to generate the figures of the documentation (see `generate_figures.py`).

Conventions (as in the documentation): the evaluation position is `p`, the scale is `t`, and the fits work in local
coordinates `x = q - p`. Positions are arrays of shape `(N, d)`.
"""

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Weight kernels (Ponca/src/Fitting/Filters/weightKernel.h)
# ----------------------------------------------------------------------------------------------------------------------


def _compact(x, values):
    """Set to 0 the values of a compact kernel for x >= 1."""
    return np.where(x < 1, values, 0.0)


# [kernels]
# Each kernel is a pair (phi, dphi) of functions of the normalized distance x = |q - p| / t, and a compactness flag.
KERNELS = {
    "SmoothWeightKernel": (lambda x: _compact(x, (x**2 - 1) ** 2),
                           lambda x: _compact(x, 4 * x * (x**2 - 1)), True),
    "ConstantWeightKernel": (lambda x: _compact(x, np.ones_like(x)),
                             lambda x: np.zeros_like(x), True),
    "WendlandWeightKernel": (lambda x: _compact(x, (1 - x) ** 4 * (4 * x + 1)),
                             lambda x: _compact(x, 20 * x * (x - 1) ** 3), True),
    "CompactExpWeightKernel": (lambda x: _compact(x, np.exp(-x**2 / np.maximum(1 - x**2, 1e-12))),
                               lambda x: _compact(x, -2 * x * np.exp(-x**2 / np.maximum(1 - x**2, 1e-12))
                                                  / np.maximum(1 - x**2, 1e-12) ** 2), True),
    "GaussianWeightKernel": (lambda x: np.exp(-x**2 / 2),
                             lambda x: -x * np.exp(-x**2 / 2), False),
}
# [kernels]


def weights(points, p, t, kernel="SmoothWeightKernel"):
    """Weights w_i = phi(|q_i - p| / t) of all the points, and their local coordinates x_i = q_i - p."""
    x = points - p
    phi = KERNELS[kernel][0]
    return phi(np.linalg.norm(x, axis=1) / t), x


# ----------------------------------------------------------------------------------------------------------------------
# Covariance analysis (Ponca/src/Fitting/Basket/tools/covariance.h, fit/covariancePlaneFit.h)
# ----------------------------------------------------------------------------------------------------------------------

# [covariance]
def covariance_analysis(x, w):
    """Weighted barycenter b, eigenvalues (increasing) and eigenvectors (columns) of the weighted covariance."""
    W = w.sum()
    b = (w[:, None] * x).sum(axis=0) / W
    C = (w[:, None, None] * x[:, :, None] * x[:, None, :]).sum(axis=0) / W - np.outer(b, b)
    evals, evecs = np.linalg.eigh(C)
    return b, evals, evecs
# [covariance]


def surface_variation(evals):
    """Surface variation as computed by Ponca: d * lambda_0 / sum(lambda)."""
    return len(evals) * evals[0] / evals.sum()


# ----------------------------------------------------------------------------------------------------------------------
# Algebraic spheres (Ponca/src/Fitting/Basket/primitive/algebraicSphere.h)
# A sphere is a tuple u = (uc, ul, uq) defining s(x) = uc + ul.x + uq |x|^2 in local coordinates.
# ----------------------------------------------------------------------------------------------------------------------


def potential(u, x):
    """Value of the algebraic scalar field at the positions x (shape (N, d) or (d,))."""
    uc, ul, uq = u
    return uc + x @ ul + uq * np.sum(x * x, axis=-1)


def gradient(u, x):
    """Gradient of the algebraic scalar field at the positions x."""
    uc, ul, uq = u
    return ul + 2 * uq * x


def pratt_norm(u):
    """Pratt norm sqrt(|ul|^2 - 4 uc uq): the norm of the gradient of the field on the sphere."""
    uc, ul, uq = u
    return np.sqrt(ul @ ul - 4 * uc * uq)


def normalize(u):
    """Pratt normalization of the sphere."""
    n = pratt_norm(u)
    return (u[0] / n, u[1] / n, u[2] / n)


def center_radius(u):
    """Center and radius of the sphere (local coordinates); None for a plane."""
    uc, ul, uq = u
    if abs(uq) < 1e-12:
        return None, np.inf
    c = -ul / (2 * uq)
    return c, np.sqrt(c @ c - uc / uq)


# [project]
def project(u, q):
    """Closed-form orthogonal projection of q onto the sphere, following the gradient of the field."""
    uc, ul, uq = u
    s, g = potential(u, q), gradient(u, q)
    ng = np.linalg.norm(g)
    if abs(uq) < 1e-12:                                    # plane: one step along the gradient
        return q - s * g / ng**2
    tau = -(ng - np.sqrt(ng**2 - 4 * uq * s)) / (2 * uq * ng)
    return q + tau * g
# [project]


# [sphere_fit]
def sphere_fit(x, w):
    """Ponca::SphereFit: minimizes sum_i w_i s(x_i)^2 under the Pratt constraint (generalized eigenproblem)."""
    d = x.shape[1]
    X = np.hstack([np.ones((len(x), 1)), x, np.sum(x * x, axis=1, keepdims=True)])  # rows [1, x^T, |x|^2]
    A = (w[:, None, None] * X[:, :, None] * X[:, None, :]).sum(axis=0)
    C = np.eye(d + 2)                                       # Pratt constraint u^T C u = |ul|^2 - 4 uc uq
    C[0, 0] = C[-1, -1] = 0
    C[0, -1] = C[-1, 0] = -2
    evals, evecs = np.linalg.eig(np.linalg.solve(C, A))     # A u = lambda C u
    evals = evals.real
    candidates = np.where(evals > 0)[0]
    u = evecs[:, candidates[np.argmin(evals[candidates])]].real  # smallest positive eigenvalue
    return (u[0], u[1:-1], u[-1])
# [sphere_fit]


# [oriented_sphere_fit]
def oriented_sphere_fit(x, n, w):
    """Ponca::OrientedSphereFit (APSS): fits the gradient of the field to the normals, then the constant term."""
    W = w.sum()
    mx = (w[:, None] * x).sum(axis=0) / W                   # <x>
    mn = (w[:, None] * n).sum(axis=0) / W                   # <n>
    mxn = (w * np.sum(x * n, axis=1)).sum() / W             # <x.n>
    mxx = (w * np.sum(x * x, axis=1)).sum() / W             # <x.x>
    uq = 0.5 * (mxn - mx @ mn) / (mxx - mx @ mx)
    ul = mn - 2 * uq * mx
    uc = -(ul @ mx + uq * mxx)
    return (uc, ul, uq)
# [oriented_sphere_fit]


# [unoriented_sphere_fit]
def unoriented_sphere_fit(x, n, w):
    """Ponca::UnorientedSphereFit: maximizes sum_i w_i (n_i . grad s(x_i))^2 with <|grad s|^2> = 1."""
    W, d = w.sum(), x.shape[1]
    B = np.hstack([n, np.sum(n * x, axis=1, keepdims=True)])  # n_i . grad s(x_i) = [n_i, n_i.x_i] . [ul, 2 uq]
    A = (w[:, None, None] * B[:, :, None] * B[:, None, :]).sum(axis=0)
    mx = (w[:, None] * x).sum(axis=0) / W
    mxx = (w * np.sum(x * x, axis=1)).sum() / W
    Q = np.eye(d + 1)                                       # <|grad s|^2> = v^T Q v, v = [ul, 2 uq]
    Q[:d, d] = Q[d, :d] = mx
    Q[d, d] = mxx
    evals, evecs = np.linalg.eig(np.linalg.solve(Q, A))
    v = evecs[:, np.argmax(evals.real)].real                # largest eigenvalue
    ul, uq = v[:d], 0.5 * v[d]
    uc = -(ul @ mx + uq * mxx)
    return (uc, ul, uq)
# [unoriented_sphere_fit]


# [gls]
def gls_parameters(u):
    """GLS parameters (tau, eta, kappa) of a sphere fitted at the evaluation position (origin of the local frame)."""
    pn = pratt_norm(u)
    return u[0] / pn, u[1], 2 * u[2] / pn
# [gls]


# ----------------------------------------------------------------------------------------------------------------------
# MLS projection (Ponca/src/Fitting/ComputeScheme/mlsEvaluationScheme.h)
# ----------------------------------------------------------------------------------------------------------------------

# [mls]
def mls_projection(q, points, normals, t, n_iter=10, eps=1e-10, kernel="SmoothWeightKernel"):
    """Iterates oriented sphere fits and projections; returns the successive positions (global coordinates)."""
    trajectory = [np.array(q, dtype=float)]
    for _ in range(n_iter):
        p = trajectory[-1]
        w, x = weights(points, p, t, kernel)
        if np.count_nonzero(w) < 2 * points.shape[1]:       # not enough neighbors: the fit is not stable
            break
        u = oriented_sphere_fit(x[w > 0], normals[w > 0], w[w > 0])
        new_p = p + project(u, np.zeros_like(p))            # p is the origin of the local frame
        trajectory.append(new_p)
        if np.linalg.norm(new_p - p) <= eps * min(np.linalg.norm(new_p), np.linalg.norm(p)):
            break
    return np.array(trajectory)
# [mls]


def mls_function(q, points, normals, t, kernel="SmoothWeightKernel"):
    """Implicit function of the MLS surface: Pratt-normalized potential of the fit centered at q, evaluated at q."""
    w, x = weights(points, q, t, kernel)
    mask = w > 0
    if np.count_nonzero(mask) < 2 * points.shape[1]:
        return np.nan
    u = normalize(oriented_sphere_fit(x[mask], normals[mask], w[mask]))
    return u[0]


# ----------------------------------------------------------------------------------------------------------------------
# Monge patches (Ponca/src/Fitting/Basket/fit/mongePatch.h, extensions/weingarten.h)
# ----------------------------------------------------------------------------------------------------------------------

# [monge]
def monge_patch(x, w, restricted=False, normal_hint=None):
    """Quadratic height field over the PCA plane (3D).

    Returns the frame (columns: normal, u, v), the coefficients [h_uu, h_vv, h_uv, (h_u, h_v,) h_c], the principal
    curvatures (increasing) and the principal directions (columns, global frame). The PCA normal has an arbitrary
    orientation, unless `normal_hint` is given."""
    b, evals, evecs = covariance_analysis(x, w)
    if normal_hint is not None and evecs[:, 0] @ normal_hint < 0:
        evecs[:, 0] = -evecs[:, 0]
    h, uu, vv = (x @ evecs).T                               # coordinates (h, u, v): normal first, as in Ponca
    if restricted:
        M = np.stack([uu**2, vv**2, uu * vv, np.ones_like(uu)], axis=1)
    else:
        M = np.stack([uu**2, vv**2, uu * vv, uu, vv, np.ones_like(uu)], axis=1)
    sw = np.sqrt(w)
    c = np.linalg.lstsq(M * sw[:, None], h * sw, rcond=None)[0]  # weighted least squares
    huu, hvv, huv = c[0], c[1], c[2]
    hu, hv = (0.0, 0.0) if restricted else (c[3], c[4])
    # Fundamental forms at (u, v) = (0, 0), with the normal (1, -hu, -hv) / |.| (Ponca convention)
    E, F, G = 1 + hu**2, hu * hv, 1 + hv**2
    nrm = np.sqrt(1 + hu**2 + hv**2)
    L, M_, N = 2 * huu / nrm, huv / nrm, 2 * hvv / nrm
    weingarten = np.linalg.solve(np.array([[E, F], [F, G]]), np.array([[L, M_], [M_, N]]))
    kappas, dirs = np.linalg.eig(weingarten)
    order = np.argsort(kappas.real)
    directions = evecs[:, 1:] @ dirs.real[:, order]         # (u, v) coordinates to the global frame
    return evecs, c, kappas.real[order], directions / np.linalg.norm(directions, axis=0)
# [monge]


# ----------------------------------------------------------------------------------------------------------------------
# kd-tree (Ponca/src/SpatialPartitioning/KdTree/kdTree.hpp)
# ----------------------------------------------------------------------------------------------------------------------

# [kdtree]
def build_kdtree(points, indices=None, leaf_size=64, depth=0, max_depth=32):
    """Recursive midpoint split of the bounding box along its largest dimension."""
    if indices is None:
        indices = np.arange(len(points))
    lo, hi = points[indices].min(axis=0), points[indices].max(axis=0)
    if len(indices) <= leaf_size or depth >= max_depth:
        return {"leaf": True, "indices": indices, "box": (lo, hi)}
    dim = int(np.argmax(hi - lo))
    split = 0.5 * (lo[dim] + hi[dim])
    left = indices[points[indices, dim] < split]
    right = indices[points[indices, dim] >= split]
    return {"leaf": False, "dim": dim, "split": split, "box": (lo, hi),
            "children": [build_kdtree(points, left, leaf_size, depth + 1, max_depth),
                         build_kdtree(points, right, leaf_size, depth + 1, max_depth)]}
# [kdtree]


def range_query(node, points, q, r, visited=None):
    """Range query; prunes the far child when the query is farther than r from the splitting plane."""
    if visited is None:
        visited = []
    if node["leaf"]:
        visited.append(node)
        idx = node["indices"]
        return [i for i in idx if np.sum((points[i] - q) ** 2) < r * r], visited
    offset = q[node["dim"]] - node["split"]
    near, far = (node["children"][0], node["children"][1]) if offset < 0 else (node["children"][1],
                                                                               node["children"][0])
    result, visited = range_query(near, points, q, r, visited)
    if offset * offset < r * r:
        more, visited = range_query(far, points, q, r, visited)
        result += more
    return result, visited


# ----------------------------------------------------------------------------------------------------------------------
# Figure style (palette of the documentation figures)
# ----------------------------------------------------------------------------------------------------------------------

#: Categorical colors, always used in this order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def setup_style():
    """Matplotlib settings shared by all the figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 4, "mathtext.fontset": "cm",
    })
    return plt


def diverging_colormap():
    """Blue (negative) to red (positive) colormap with a neutral gray midpoint, for signed scalar fields."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "ponca_diverging", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3aaa9", "#e34948", "#9c2121"])


def sequential_colormap():
    """Single-hue (blue) colormap, light to dark, for magnitudes."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("ponca_sequential", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95",
                                                                  "#0d366b"])


def save(fig, name):
    """Save a figure in doc/images/maths/."""
    import os
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "maths")
    os.makedirs(folder, exist_ok=True)
    path = os.path.normpath(os.path.join(folder, name))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print("wrote", path)

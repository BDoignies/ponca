# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Generates all the figures of the mathematical background of the documentation in doc/images/maths/.

Usage (requires numpy and matplotlib):

    python doc/scripts/generate_figures.py

The figures are committed in the repository: they only need to be regenerated when the scripts change. The scripts are
deterministic (fixed random seeds). They rely on `ponca_numpy.py`, which re-implements the formulas of the
documentation in NumPy, independently of the C++ library.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plot_covariance
import plot_curvature_scale
import plot_gls_scale_space
import plot_kdtree
import plot_monge_patch
import plot_mls
import plot_sphere_fits
import plot_weight_kernels

if __name__ == "__main__":
    for module in (plot_weight_kernels, plot_covariance, plot_sphere_fits, plot_gls_scale_space, plot_mls,
                   plot_monge_patch, plot_curvature_scale, plot_kdtree):
        module.main()

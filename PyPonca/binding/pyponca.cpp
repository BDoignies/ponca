/*
This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <Ponca/Ponca>

#include "Common/RegisterCommon.h"
// #include "Mangling.h"
// #include "Common.h"
#include "Fitting/Fitting.h"
#include "SpatialParitioning/SpatialPartitioning.h"

namespace nb = nanobind;

NB_MODULE(_pyponca, m)
{
    // TODO
    m.doc() = "Ponca python module";
    auto internal = m.def_submodule("internal");
    
    RegisterPointClouds(m);
    RegisterManglingUtils(internal);
    RegisterComputeObjects(m);
    RegisterKDTree(m);
}

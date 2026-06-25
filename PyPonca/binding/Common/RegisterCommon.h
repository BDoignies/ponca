/*
This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/
#pragma once

#include <nanobind/nanobind.h>
namespace nb = nanobind;

using namespace nb::literals;

#include "Mangling.h"
#include "pypoint.h"
#include "ParallelExecutionMode.h"

/**
 * \brief Register a PyPoncaPointCloud depending on the point type
 * 
 * \param m The module to register the class within
 */
template<typename _Scalar, unsigned int _Dim>
inline void RegisterPyPointCloud(nb::module_& m)
{
    using PointCloud = PyPointCloud<_Scalar, _Dim>;
    using Point = typename PointCloud::Point;

    const std::string pointName = ManglePoint<Point>() + "PN";
    const std::string className = "PointCloud" + pointName;
    
    // Only the constructor and the mangledName are exposed. Nothing else should 
    // be necessary in client code. 
    // MangledName is necessary for dispatching to the correct PointCloud type 
    auto cls = nb::class_<PointCloud>(m, className.c_str());
    
    cls.def(nb::init<const PyVectorArray<Point>&>());
    cls.def(nb::init<const PyVectorArray<Point>&, const PyVectorArray<Point>&>());
    // cls.attr("pointName") = PointCloud::MangledName;
}

/**
 * \brief Register all point clouds
 * 
 * \param m The module to register the class within
 */
inline void RegisterPointClouds(nb::module_& m)
{
    RegisterPyPointCloud<double, 3>(m);
    RegisterPyPointCloud<float, 3>(m);
    RegisterPyPointCloud<double, 2>(m);
    RegisterPyPointCloud<float, 2>(m);

#ifndef _OPENMP
    std::cout << "PyPonca compiled without OMP support. Any CPU parrallel code will instead run sequentially." << std::endl;
#endif

    nb::enum_<ParallelExecutionMode::Device>(m, "Device").value("CPU", ParallelExecutionMode::Device::CPU);
    nb::enum_<ParallelExecutionMode::Storage>(m, "Storage").value("THREAD_LOCAL", ParallelExecutionMode::Storage::THREAD_LOCAL).value("OBJECT", ParallelExecutionMode::Storage::OBJECT);
    nb::class_<ParallelExecutionMode>(m, "ParallelExecutionMode")
        .def(nb::init<ParallelExecutionMode::Storage, ParallelExecutionMode::Device, int>(), "storage"_a = ParallelExecutionMode::Storage::OBJECT, "device"_a = ParallelExecutionMode::Device::CPU, "threads"_a = -1);
}

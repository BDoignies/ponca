/*
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/
#include "FittingList.h"

// This file is the main binding code for the Fitting module. 
//
// The philosophy is to avoid lambdas as much as possible and rely
// on plain function instead. Lambdas may or may not be supported by 
// some device code. In CUDA for instance, only one layer of lambda
// is supported.  

/**
 * \brief Adds a computation to the PyComputeObject
 * 
 * The main purpose of this function is to fill the outputDimension
 * which the PyCo can not do (it only sees the id);
 * 
 * \param self The instance of the object
 * \param id The computation to be performed
 * \param data The input array. 
 */
template<typename PyCo>
void addComputation(PyCo& self, Computation id, nb::ndarray<typename PyCo::Scalar> data)
{
    ComputationDescriptor<typename PyCo::Scalar> descriptor;
    descriptor.id = static_cast<size_t>(id);
    descriptor.outputDims = ComputeOutputDimension<PyCo>(id, nb::ndarray<>(data));
    descriptor.inputData  = data;
    
    self.addComputation(std::move(descriptor));
}

/**
 * \brief Performs the computation 
 * 
 * \param self The instance of the object
 * \param cloud The point cloud 
 */
template<typename PyCo, typename Cloud>
auto compute(PyCo& self, const Cloud& cloud)
{
    return self.compute(cloud, &PerformComputation<typename PyCo::ComputeObject>);
}

/**
 * \brief Binds all compute object given by factories
 *
 * \tparam P The point type. Should be compatible with PyPoncaPointCloud
 * \tparam NF The neighbor filter
 * \tparam Diff The type of differentiation
 */
template <typename PointCloud, typename _NF, Ponca::DiffType Diff>
void RegisterComputeObjects(nb::module_& m, std::set<std::string>& list)
{
    using P       = typename PointCloud::Point;
    using Scalar  = typename P::Scalar;
    using NF      = typename _NF::NF;
    using Factory = Ponca::Factory<P, NF, Diff>;

    // Compute mangling informations
    const std::string mangledName = PointCloud::PointName + _NF::name;

    // General properties
    Factory::foreach ([&](const auto& x) {
        using T                   = decltype(x.object);
        using PyCo                = PyComputeObject<T>;
        const std::string newname = x.name + mangledName;
        list.insert(x.name);

        auto pyco = nb::class_<PyCo>(m, newname.c_str());
        pyco.def(nb::init<>());
        pyco.def("setNeighborFilter", &PyCo::setNeighborFilter);
        pyco.def("addComputation", &addComputation<PyCo>);
        pyco.def("compute", &compute<PyCo, PointCloud>);
    });
}

template <typename Scalar, unsigned int Dim, template <class> class NF>
void RegisterComputeObjects(nb::module_& m, std::set<std::string>& list)
{
    using namespace Ponca;
    using PointCloud = PyPointCloud<Scalar, Dim>;
    using Point      = typename PointCloud::Point;

    RegisterComputeObjects<PointCloud, NF<Point>, Ponca::FitSpaceDer>(m, list);
}

template <typename Scalar, unsigned int Dim>
void RegisterComputeObjects(nb::module_& m, std::set<std::string>& list)
{
    RegisterComputeObjects<Scalar, Dim, SWFilter>(m, list);
    RegisterComputeObjects<Scalar, Dim, CWFilter>(m, list);
    RegisterComputeObjects<Scalar, Dim, NWFilter>(m, list);

    auto result = nb::class_<ComputationResult<Scalar>>(m, "");
    result.def_rw("data", &ComputationResult<Scalar>::resultData, nb::rv_policy::reference);
}

/**
 * \brief Register all instances of ComputeObjects
 *
 * \param m The module to register instances within
 */
void RegisterFitting(nb::module_& m, nb::module_& internal)
{
    // Due to registry, this is not moved to another location
    std::set<std::string> computeObjectList;
    RegisterComputeObjects<double, 2>(m, computeObjectList);
    RegisterComputeObjects<double, 3>(m, computeObjectList);
    RegisterComputeObjects<float, 2>(m, computeObjectList);
    RegisterComputeObjects<float, 3>(m, computeObjectList);

    nb::enum_<Computation>(m, "Computation")
        .value("POTENTIAL", Computation::POTENTIAL)
        .value("PROJECTION", Computation::PROJECTION)
        .export_values();

    m.attr("ComputeObjectList") = nb::cast(computeObjectList);
}

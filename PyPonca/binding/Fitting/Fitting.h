/*
This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/
#pragma once

#include "utils.h"

using namespace nb::literals;

/**
 * \brief Helper class to store all created classes
 *
 * Because the factory will iterate over the Baskets depending
 * on their compute capability, we need a way to store the
 * 'class' being created and add functions latter.
 *
 * This class stores the nb::object as well as the 'name' of the
 * method and its mangled name.
 */
struct ComputeObjectRegistry
{
    /**
     * \brief Helper struct to hold informations
     */
    struct Entry
    {
        std::string baseName;
        std::string fullName;
        nb::object object;
    };

    /**
     * \brief Add a new object to the registry
     */
    void AddEntry(const std::string& base, const std::string& full, nb::object&& o)
    {
        classes.insert(base);
        entries[full] = Entry{base, full, std::move(o)};
    }

    std::set<std::string> classes;        // List of compute object
    std::map<std::string, Entry> entries; // Maps mangled name to the actual object
};

// Without parameters
#define BIND_SCALAR_GETTER(name) BindGetterFunctions<PyCo>(#name, pyco, [](const T& object) { return object. name (); });
#define BIND_VECTOR_GETTER(name) BindGetterFunctions<PyCo>(#name, pyco, [](const T& object) { return object. name (); });

// With vector parameters
#define BIND_SCALAR_GETTER_V(name) BindGetterFunctions<PyCo>(#name, pyco, [](const T& object, const auto& v) { return object. name (v); });
#define BIND_VECTOR_GETTER_V(name) BindGetterFunctions<PyCo>(#name, pyco, [](const T& object, const auto& v) { return object. name (v); });

#define BIND_FUNCTIONS_FOR(Provider, ...) Factory::template Filter<Provider>().foreach( \
    [&] (auto& x) \
    { \
        using T = std::remove_cv_t<decltype(x.object)>; \
        using PyCo = PyComputeObjectArray<T, PointCloud>; \
        const std::string name = x.name + mangledName;  \
        auto pyco = nb::cast<nb::class_<PyCo>>(registry.entries[name].object); \
        __VA_ARGS__; \
    }); 


/**
 * \brief Binds all compute object given by factories
 *
 * \tparam P The point type. Should be compatible with PyPoncaPointCloud
 * \tparam NF The neighbor filter
 * \tparam Diff The type of differentiation
 */
template <typename PointCloud, typename _NF, Ponca::DiffType Diff>
void RegisterComputeObjects(nb::module_& m, ComputeObjectRegistry& registry)
{
    using P       = typename PointCloud::Point;
    using Scalar  = typename P::Scalar;
    using NF      = typename _NF::NF;
    using Factory = Ponca::Factory<P, NF, Diff>;

    static constexpr unsigned int Dim = P::Dim;
    // Compute mangling informations
    const std::string mangledName = PointCloud::PointName + _NF::name;

    // General properties
    Factory::foreach ([&](const auto& x) {
        using T                   = decltype(x.object);
        using PyCo                = PyComputeObjectArray<T, PointCloud>;
        const std::string newname = x.name + mangledName;

        auto pyco = nb::class_<PyCo>(m, newname.c_str());
        pyco.def(nb::init<ParallelExecutionMode>(), "mode"_a = ParallelExecutionMode{});
        pyco.def("setNeighborFilter", &PyCo::setNeighborFilter);
        pyco.def("attach", nb::overload_cast<const PointCloud&>(&PyCo::attach));
        pyco.def("attach", nb::overload_cast<const PyKDTree<PointCloud>*>(&PyCo::attach));
        registry.AddEntry(x.name, newname, std::move(pyco));
    });
 
    BIND_FUNCTIONS_FOR(Ponca::ProjectionOperatorProvider, {
        BIND_VECTOR_GETTER_V(project);
    });

    BIND_FUNCTIONS_FOR(Ponca::ImplicitPrimitiveProvider, {
        BIND_SCALAR_GETTER(potential);
        BIND_VECTOR_GETTER_V(potential);
    });

    BIND_FUNCTIONS_FOR(Ponca::AlgebraicSphereProvider, {
        BIND_SCALAR_GETTER(prattNorm);
        BIND_SCALAR_GETTER(prattNorm2);
        BIND_SCALAR_GETTER(radius);
        BIND_VECTOR_GETTER(center);
    });
    
    BIND_FUNCTIONS_FOR(Ponca::MeanPositionProvider, {
        BIND_SCALAR_GETTER(barycenter);
        BIND_SCALAR_GETTER(barycenterDistance);
    });

    BIND_FUNCTIONS_FOR(Ponca::MeanNormalProvider, {
        BIND_VECTOR_GETTER(meanNormalVector);
    });


    BIND_FUNCTIONS_FOR(Ponca::MeanCurvatureProvider, {
        BIND_VECTOR_GETTER(kMean);
    });

    BIND_FUNCTIONS_FOR(Ponca::GLSParamProvider, {
        BIND_SCALAR_GETTER(fitness);
        BIND_SCALAR_GETTER(tau);
        BIND_SCALAR_GETTER(tau_normalized);
        BIND_VECTOR_GETTER(eta);
        BIND_VECTOR_GETTER(eta_normalized);
        BIND_VECTOR_GETTER(kappa);
        BIND_VECTOR_GETTER(kappa_normalized);
        // TODO: "Supper getter" to retrieve everything at once?  
    });

    BIND_FUNCTIONS_FOR(Ponca::PrincipalCurvaturesProvider, {
        BIND_SCALAR_GETTER(kmin);
        BIND_SCALAR_GETTER(kmax);
        BIND_SCALAR_GETTER(GaussianCurvature);
        BIND_VECTOR_GETTER(kminDirection);
        BIND_VECTOR_GETTER(kmaxDirection);
        // TODO: "Supper getter" to retrieve everything at once?  
    });

    BIND_FUNCTIONS_FOR(Ponca::TangentPlaneBasisProvider, {
        BIND_VECTOR_GETTER_V(worldToTangentPlane);
        BIND_VECTOR_GETTER_V(tangentPlaneToWorld);
    });

    // TODO List
    // TODO: DECLARE_FACTORY_CONCEPT(HeightField)
    // TODO: DECLARE_FACTORY_CONCEPT(GeomVar)
    
    // TODO: Other bindings
}

template <typename Scalar, unsigned int Dim, template <class> class NF>
void RegisterComputeObjects(ComputeObjectRegistry& registry, nb::module_& m)
{
    using namespace Ponca;
    using PointCloud = PyPointCloud<Scalar, Dim>;
    using Point      = typename PointCloud::Point;

    RegisterComputeObjects<PointCloud, NF<Point>, Ponca::FitSpaceDer>(m, registry);
}

template <typename Scalar, unsigned int Dim>
void RegisterComputeObjects(ComputeObjectRegistry& registry, nb::module_& m)
{
    RegisterComputeObjects<Scalar, Dim, SWFilter>(registry, m);
    RegisterComputeObjects<Scalar, Dim, CWFilter>(registry, m);
    RegisterComputeObjects<Scalar, Dim, NWFilter>(registry, m);
}

/**
 * \brief Register all instances of ComputeObjects
 *
 * \param m The module to register instances within
 */
void RegisterComputeObjects(nb::module_& m)
{
    // Due to registry, this is not moved to another location
    ComputeObjectRegistry registry;
    RegisterComputeObjects<double, 2>(registry, m);
    RegisterComputeObjects<double, 3>(registry, m);
    RegisterComputeObjects<float, 2>(registry, m);
    RegisterComputeObjects<float, 3>(registry, m);

    // Make available the list of class names so that python code can create
    // the necessary dispatchers
    m.attr("ComputeObjectList") = std::move(registry.classes);
}

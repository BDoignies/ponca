/*
This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/
#pragma once

#include "../Common/pypoint.h"
#include "../Common/pytypes.h"

#include "ComputeObjectArray.h"
#include "Filters.h"

#include <nanobind/stl/map.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

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
        entries[full] = Entry{ base, full, std::move(o) };
    }

    std::set<std::string> classes;        // List of compute object
    std::map<std::string, Entry> entries; // Maps mangled name to the actual object
};

/**
 * \brief Copy data and advance the pointer
 * 
 * Depending on the given type, this function may write 
 * multiple scalars. 
 */
template<typename Point, typename _Scalar, typename _T>
void CopyAndAdvance(_Scalar* dest, const _T& value)
{
    using T = std::remove_cvref_t<_T>;
    using Scalar = std::remove_cvref_t<_Scalar>;
    using VectorType = typename Point::VectorType;

    if constexpr (std::is_same_v<T, Scalar>)
    {
        *dest = value;
    }    
    else if constexpr (std::is_same_v<T, VectorType>)
    {
        for (size_t i = 0; i < Point::Dim; ++i)
        {
            *dest = value[i];
            dest++;
        }
    }
    else
    {
        // We use sizeof(_T) to trick the compiler into thinking there is dependency.
        // In C++23, static_assert(false) should work properly.
        static_assert(!sizeof(_T), "Return type not supported for CopyAndAdvance. Please add the corresponding branch");
    }

}

/**
 * \brief Binds one or several function to a sigle python equivalent.
 * 
 * For now, this functions may only bind function that accepts a Point::VectoryType argument or none. 
 * 
 * \tparam withParam Wether the function accepts a vector argument or not
 * \tparam outputDim Total output dimension
 * \tparam PyCOArray A PyComputeArray obejct
 * \tparam PyCo The current nb::class_
 * \tparam Fs... A list of function
 * 
 * \param name The name of the function
 * \param pyco The nb::class_ 
 * \param fs The list of function to apply
 */
template<bool withParam, unsigned int outputDim, typename PyCOArray, typename PyCo, typename... Fs>
void BindFunctions(const std::string& name, PyCo& pyco, Fs&&... fs)
{
    using CO     = typename PyCOArray::CO;
    using Point  = typename PyCOArray::Point;
    using Scalar = typename Point::Scalar;
    using VectorType = typename Point::VectorType;

    if constexpr (withParam)
    {
        // A simple location
        pyco.def(name.c_str(), [&](PyCOArray& object, const PyVector<Point>& _p) { 
            auto v = PyVectorToVector<Point>(_p); 
            return object.DispatchFunction([&](Scalar* dest, CO& co, size_t i) { 
                (CopyAndAdvance<Point>(dest, fs(co, v)), ...);
            }, outputDim, 1); 
        }); 

        // An array of location (one for each center)
        pyco.def(name.c_str(), [&](PyCOArray& object, const PyVectorArray<Point>& _p) { 
            if (object.GetPosCount() != _p.shape(0)) 
                throw std::runtime_error("Shape mismatch between argument and center count");
            
            return object.DispatchFunction([&](Scalar* dest, CO& co, size_t i) { 
                (CopyAndAdvance<Point>(dest, fs(co, PyVectorArrayIndex<Point>(_p, i))), ...);
            }, outputDim, 1); 
         });

        // An array of array: For each center, an array of positions
        pyco.def(name.c_str(), [&](PyCOArray& object, const PyVectorVectorArray<Point>& _p) { 
            if (object.GetPosCount() != _p.shape(0)) 
                throw std::runtime_error("Shape mismatch between argument and center count");
            
            return object.DispatchFunction([&](Scalar* dest, CO& co, size_t i) { 
                for (unsigned int j = 0; j < _p.shape(1); ++j)
                    (CopyAndAdvance<Point>(dest, fs(co, PyVectorVectorArrayIndex<Point>(_p, i, j))), ...);
            }, outputDim, _p.shape(1)); 
        });
    }
    else
    {
        // No parameters
        pyco.def(name.c_str(), [&](PyCOArray& object) { 
            return object.DispatchFunction([&](Scalar* dest, CO& co, size_t i) { 
                (CopyAndAdvance<Point>(dest, fs(co)), ...);
            }, outputDim, 1); 
        }); 
    }
}

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
    using P = typename PointCloud::Point;
    using Scalar     = typename P::Scalar;
    using NF         = typename _NF::NF;
    using Factory    = Ponca::Factory<P, NF, Diff>;

    static constexpr unsigned int Dim = P::Dim;
    // Compute mangling informations
    const std::string mangledName = PointCloud::PointName + _NF::name;

    // General properties
    Factory::foreach ([&](const auto& x) {
        using T                   = decltype(x.object);
        using PyCo = PyComputeObjectArray<T, PointCloud>;
        const std::string newname = x.name + mangledName;

        auto pyco               = nb::class_<PyCo>(m, newname.c_str());
        pyco.def(nb::init<ParallelExecutionMode>(), "mode"_a = ParallelExecutionMode{});
        pyco.def("setNeighborFilter", &PyCo::setNeighborFilter);
        pyco.def("attach", nb::overload_cast<const PointCloud&>(&PyCo::attach));
        pyco.def("attach", nb::overload_cast<const PyKDTree<PointCloud>*>(&PyCo::attach));
        registry.AddEntry(x.name, newname, std::move(pyco));
    });

    Factory::template Filter<Ponca::ProjectionOperatorProvider>().foreach([&](auto& x) {
        using T    = decltype(x.object);
        using PyCo = PyComputeObjectArray<T, PointCloud>;
        const std::string name = x.name + mangledName;
        
        auto pyco = nb::cast<nb::class_<PyCo>>(registry.entries[name].object);
        BindFunctions<true , 1, PyCo>("project", pyco, [](T& object, auto p) { return object.project(p); });
    });

    Factory::template Filter<Ponca::ImplicitPrimitiveProvider>().foreach([&](auto& x) {
        using T    = decltype(x.object);
        using PyCo = PyComputeObjectArray<T, PointCloud>;
        const std::string name = x.name + mangledName;
        
        auto pyco = nb::cast<nb::class_<PyCo>>(registry.entries[name].object);
        // With and without parameters 
        BindFunctions<true , 1, PyCo>("potential", pyco, [](T& object, auto p) { return object.potential(p); });
        BindFunctions<false, 1, PyCo>("potential", pyco, [](T& object)         { return object.potential();  });  
    });

    // TODO: Other bindings
}

template<typename Scalar, unsigned int Dim, template <class> class NF>
void RegisterComputeObjects(ComputeObjectRegistry registry, nb::module_& m)
{
    using namespace Ponca;
    using PointCloud = PyPointCloud<Scalar, Dim>;
    using Point  = typename PointCloud::Point;
    
    RegisterComputeObjects<PointCloud, NF<Point>, Ponca::FitSpaceDer>(m, registry);
}

template<typename Scalar, unsigned int Dim>
void RegisterComputeObjects(ComputeObjectRegistry registry, nb::module_& m)
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

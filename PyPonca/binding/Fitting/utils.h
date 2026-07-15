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

/**
 * \brief Copy data and advance the pointer
 *
 * Depending on the given type, this function may write
 * multiple scalars.
 */
template <typename Point, typename _Scalar, typename _T>
void CopyAndAdvance(_Scalar* dest, const _T& value)
{
    using T          = std::remove_cvref_t<_T>;
    using Scalar     = std::remove_cvref_t<_Scalar>;
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

template<typename Point, typename Co, typename Func>
concept ParamlessFunction = requires(Func&& f, Co& o) { f(o); };

template<typename Point, typename Co, typename Func>
concept VectorFunction = requires(Func&& f,Co& o, typename Point::VectorType v) { f(o, v); };

template<typename Point, typename Co, typename Func>
constexpr unsigned int ComputeOutputDimension()
{
    using Scalar = typename Point::Scalar;
    using VectorType = typename Point::VectorType;

    Func f;
    // If a function satisfies both condition, we can expect the return type to be the same and all path 
    // will return the same value
    if constexpr (ParamlessFunction<Point, Co, Func>) 
    {
        using ReturnType = std::remove_cv_t<decltype(f(Co{}))>;
        if constexpr (std::is_convertible_v<ReturnType, Scalar>) return 1;
        else if constexpr (std::is_convertible_v<ReturnType, VectorType>) return Point::Dim;
        else static_assert(!sizeof(ReturnType), "ReturnType not supported for ComputeOutputDimensions");
    }
    else if constexpr (VectorFunction<Point, Co, Func>)
    {
        using ReturnType = std::remove_cv_t<decltype(f(Co{}, VectorType{}))>;
        if constexpr (std::is_convertible_v<ReturnType, Scalar>) return 1;
        else if constexpr (std::is_convertible_v<ReturnType, VectorType>) return Point::Dim;
        else static_assert(!sizeof(ReturnType), "ReturnType not supported for ComputeOutputDimensions");
    }

    return 0;
}

/**
 * \brief Binds one or several getter function to a sigle python equivalent.
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
template <typename PyCOArray, typename PyCo, typename... Fs>
void BindGetterFunctions(const std::string& name, PyCo& pyco, Fs&&... fs)
{
    using CO         = typename PyCOArray::CO;
    using Point      = typename PyCOArray::Point;
    using Scalar     = typename Point::Scalar;
    using VectorType = typename Point::VectorType;

    constexpr bool AcceptsVector  = (VectorFunction<Point, CO, Fs> && ...);
    constexpr bool Paramless      = (ParamlessFunction<Point, CO, Fs> && ...);
    static_assert(AcceptsVector || Paramless, "Can not detect input type of functions");

    constexpr unsigned int outputDim = (ComputeOutputDimension<Point, CO, Fs>() + ...);
    static_assert(outputDim != 0, "Function must return something (check that ComputeOutputDimension supports the appropriate inputs).");
    
    if constexpr (AcceptsVector)
    {
        // A simple location
        pyco.def(name.c_str(), [&](PyCOArray& object, const PyVector<Point>& _p) {
            auto v = PyVectorToVector<Point>(_p);
            return object.DispatchFunction(
                [&](Scalar* dest, CO& co, size_t i) { (CopyAndAdvance<Point>(dest, fs(co, v)), ...); }, outputDim, 1);
        });

        // An array of location (one for each center)
        pyco.def(name.c_str(), [&](PyCOArray& object, const PyVectorArray<Point>& _p) {
            if (object.GetPosCount() != _p.shape(0))
                throw std::runtime_error("Shape mismatch between argument and center count");

            return object.DispatchFunction(
                [&](Scalar* dest, CO& co, size_t i) {
                    (CopyAndAdvance<Point>(dest, fs(co, PyVectorArrayIndex<Point>(_p, i))), ...);
                },
                outputDim, 1);
        });

        // An array of array: For each center, an array of positions
        pyco.def(name.c_str(), [&](PyCOArray& object, const PyVectorVectorArray<Point>& _p) {
            if (object.GetPosCount() != _p.shape(0))
                throw std::runtime_error("Shape mismatch between argument and center count");

            return object.DispatchFunction(
                [&](Scalar* dest, CO& co, size_t i) {
                    for (unsigned int j = 0; j < _p.shape(1); ++j)
                        (CopyAndAdvance<Point>(dest, fs(co, PyVectorVectorArrayIndex<Point>(_p, i, j))), ...);
                },
                outputDim, _p.shape(1));
        });
    }

    if constexpr (Paramless)
    {
        // No parameters
        pyco.def(name.c_str(), [&](PyCOArray& object) {
            return object.DispatchFunction(
                [&](Scalar* dest, CO& co, size_t i) { (CopyAndAdvance<Point>(dest, fs(co)), ...); }, outputDim, 1);
        });
    }
}

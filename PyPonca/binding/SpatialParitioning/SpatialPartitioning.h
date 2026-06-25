/*
This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <type_traits>
#include <variant>

namespace nb = nanobind;

#include <Ponca/SpatialPartitioning>
#include "../Common/pypoint.h"

/**
 * \brief Abstracts away kdtree for python binding
 * 
 * This function aims at providing some kind of runtime dispatch 
 * between Sparse and Dense KDTree. In the future, it may also
 * be used for CPU/GPU trees. 
 */
template<typename PointCloud>
struct PyKDTree
{
    using Point  = typename PointCloud::Point;
    using Dense  = Ponca::KdTreeDense<Point>;
    using Sparse = Ponca::KdTreeSparse<Point>;

    PyKDTree() : isDense(true) {}
    
    /**
     * \brief Constructor from a pointcloud
     * 
     * \param cloud The pointcloud
     * \param dense Storage type
     */
    PyKDTree(const PointCloud& cloud, bool dense = true) : isDense(dense) 
    { 
        if (isDense)
            data = Dense(cloud);
        else 
            data = Sparse(cloud);
    }

    /**
     * \brief Constructor from a PyVectorArray
     * 
     * \param cloud The pointcloud
     * \param dense Storage type
     */
    PyKDTree(const PyVectorArray<Point>& cloud, bool dense = true) : isDense(dense) 
    {
        PointCloud tmp(cloud);
        if (isDense)
            data = Dense(tmp);
        else 
            data = Sparse(tmp);
    }

    /**
     * \brief Run a function with a dispatch depending on kdtree type
     * 
     * \tparam Func Function type
     *  
     * \param f The function 
     */
    template<typename Func>
    void Run(Func&& f) const
    {
        if (isDense)
            f(std::get<DenseIndex>(data));
        else
            f(std::get<SparseIndex>(data));
    }

    // Should be const but copy operator makes it impossible...
    bool isDense;
    static constexpr unsigned int DenseIndex  = 0;
    static constexpr unsigned int SparseIndex = 1;
    std::variant<Dense, Sparse> data;
};

template<typename Scalar, unsigned int Dim>
void RegisterKDTree(nb::module_& m)
{
    using IndexType = long long int;
    using PointCloud = PyPointCloud<Scalar, Dim>;
    using Point  = typename PointCloud::Point;
 
    using KdTree = PyKDTree<PointCloud>;

    const std::string treeName = "KdTree" + PointCloud::PointName; 
    nb::class_<KdTree>(m, treeName.c_str())
        .def(nb::init<const PointCloud&, bool>(), "pc"_a, "dense"_a = true)
        .def(nb::init<const PyVectorArray<Point>&, bool>(), "pos"_a, "dense"_a = true)
        .def("rangeNeighbors", [](const KdTree& pytree, const PyVector<Point>& pos, Scalar radius) 
        {
            // Overload for a single position
            // Unfortunately we do not have access to the number of indices that will be returned. 
            // We use a vector for insertion. But we have to copy the data to obtain a dlpack array.
            std::vector<IndexType> result;
            pytree.Run([&](const auto& tree) {
                for (const auto& i : tree.rangeNeighbors(PyVectorToVector<Point>(pos), radius))
                    result.push_back(i);
            });
            
            IndexType* newData = new IndexType[result.size()];
            std::memcpy(newData, result.data(), result.size() * sizeof(IndexType));

            return nb::ndarray<nb::array_api>(
                newData, { result.size() }, PyArrayDeleter(newData), { 1 }, nb::dtype<IndexType>(), nb::device::cpu::value
            );
        })
        .def("rangeNeighbors", [](const KdTree& pytree, const PyVectorArray<Point>& v, const PyScalarArray<Point>& _t)
        {
            // Overload for multiple positions
            // Unfortunately we do not have access to the number of indices that will be returned. 
            // We use a vector for insertion. But we have to copy the data to obtain a dlpack array.
            std::vector<std::vector<IndexType>> results(v.shape(0));
            PyScalarArray<Point> rs = PyScalarArray<Point>(broadcastArrayTo<Scalar, 1>(_t, { v.shape(0) }));

            IndexType maxSize = 0;
            for (size_t i = 0; i < v.shape(0); ++i)
            {
                const auto pos = PyVectorArrayIndex<Point>(v, i);
                const auto radius = PyScalarArrayIndex<Point>(rs, i);
                
                pytree.Run([&](const auto& tree) {
                    for (const auto& j : tree.rangeNeighbors(pos, radius))
                        results[i].push_back(j);
                });
            
                maxSize = std::max(maxSize, (IndexType)results[i].size());
            }

            IndexType* newData = new IndexType[maxSize * v.shape(0)];
            for (size_t i = 0; i < v.shape(0); ++i)
            {
                std::memcpy(newData + i * maxSize, results[i].data(), results[i].size() * sizeof(IndexType));
                // We set a negative index when this is not covered. 
                for (size_t j = results[i].size(); j < maxSize; ++j)
                    *(newData + j + i * maxSize) = -1;
            }
            return nb::ndarray<nb::array_api>(
                newData, { v.shape(0), (size_t)maxSize }, PyArrayDeleter(newData), { maxSize, 1 }, nb::dtype<IndexType>(), nb::device::cpu::value
            );
        });
}

void RegisterKDTree(nb::module_& m)
{
    RegisterKDTree<double, 2>(m);
    RegisterKDTree<double, 3>(m);
    RegisterKDTree<float, 2>(m);
    RegisterKDTree<float, 3>(m);
}
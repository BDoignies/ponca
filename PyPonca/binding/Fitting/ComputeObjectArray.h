/*
This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/
#pragma once

#include <functional>
#include <optional>

#include "../Common/ParallelExecutionMode.h"
#include "../Common/pytypes.h"
#include "../SpatialParitioning/SpatialPartitioning.h"

#ifdef _OPENMP
    #include "omp.h"
#endif

/**
 * \brief Array of compute objects
 * 
 * This class aims at avoiding loops in python. Depending on the ParallelExecutionMode given by the user, 
 * it can store an array of ComputeObjects that can be used multiple times for fast computation once 
 * fitting is performed. 
 * 
 * Unlike the traditionnal ponca api, here 'setNeighborFilter' and 'attach' stores the informations and 
 * use them only when the post-finalize computation are needed. SetNeighborFilter accepts an array of 
 * location (and distances). Attach will store the point cloud for latter use. 
 * 
 * Once a computation is required (e.g. potential()), this class will iterate over each point given to setNeighborFilter, 
 * builds a Compute object, call the compute method and return the results. Computation can happend in 
 * parallel and might be optionnal if it was already performed and the object are stored. 
 * 
 * \tparam _CO The compute object type
 * \tparam
 */
template<typename _CO, typename PointCloud>
struct PyComputeObjectArray
{
    using CO         = _CO;
    using Point      = typename CO::DataPoint;
    using Scalar     = typename Point::Scalar;
    using VectorType = typename Point::VectorType;

    /**
     * \brief Ctor
     * 
     * \param mode The execution mode for computation
     */
    PyComputeObjectArray(ParallelExecutionMode mode = ParallelExecutionMode{}) :
        m_executionMode(mode) 
    {
        m_hasFilter     = false;
        m_hasPointCloud = false;
        m_hasCompute    = false;
        m_kdtree = nullptr;
    }
    
    /**
     * \brief Returns the number of position given through setNeighborFilter
     */
    size_t GetPosCount() const
    {
        return m_pos.shape(0);
    }

    /**
     * \brief Sets the neighborFilter informations
     * 
     * Nothing will be set for now, informations are stored for further used
     * 
     * \param _pos The position array
     * \param _t The scalar (radius) array
     */
    void setNeighborFilter(const PyVectorArray<Point>& _pos, const PyScalarArray<Point>& _t)
    {
        if (_pos.device_type() != _t.device_type())
            throw std::runtime_error("Incompatible devices between pos and t.");
            
        m_pos = _pos;
        m_t   = PyScalarArray<Point>(broadcastArrayTo<Scalar, 1>(_t, { _pos.shape(0) }));
        m_hasFilter  = true;
        m_hasCompute = false;
    }

    /**
     * \brief Attachs the pointcloud for future computation
     * 
     * The compute method will not be called now. This function only stores the point cloud. 
     * 
     * This function will overwrite the kdtree set by attach(KDTree) if this function
     * was called earlier.
     * 
     * \param cloud The pointcloud
     */
    void attach(const PointCloud& cloud)
    {
        if (!m_hasFilter)
            throw std::runtime_error("No filter set. Please call setNeighborFilter first.");

        if (m_pos.device_type() != cloud.device_type())
            throw std::runtime_error("Device mismatch between pointcloud and filter centers.");

        switch(m_executionMode.device)
        {
        case ParallelExecutionMode::Device::CPU:
            if (m_pos.device_type() != nb::device::cpu::value)
                throw std::runtime_error("Device mismatch between pointcloud and execution device.");
            break;
        default:
            throw std::runtime_error("Unsupported device");
        }

        m_pointcloud = cloud;
        m_hasPointCloud = true;
        m_hasCompute = false;
        m_kdtree = nullptr; // Reset kdtree 
    }

    /**
     * \brief Attachs the kdtree for future computation
     * 
     * The compute method will not be called now. This function only stores the the kdtree. 
     * 
     * This function will overwrite the pointcloud set by attach(PointCloud) if this function
     * was called earlier.
     * 
     * \param tree The kdtree. It is stored as a pointer to avoid copies...
     */
    void attach(const PyKDTree<PointCloud>* tree)
    {
        if (!m_hasFilter)
            throw std::runtime_error("No filter set. Please call setNeighborFilter first.");

        if (m_executionMode.device != ParallelExecutionMode::Device::CPU)
            throw std::runtime_error("Only CPU computing is supported for KDTrees");

        switch(m_executionMode.device)
        {
        case ParallelExecutionMode::Device::CPU:
            if (m_pos.device_type() != nb::device::cpu::value)
                throw std::runtime_error("Device mismatch between pointcloud and execution device.");
            break;
        default:
            throw std::runtime_error("Unsupported device");
        }

        m_kdtree = tree;
        m_hasPointCloud = true;
        m_hasCompute = false;
    }

    /**
     * \brief Dispatch a function depending on the device
     *
     * The function signature should be:
     *      - Scalar* : a pointer to output data. 
     *      - Object& : a reference to the compute object
     *      - int : current index (for indexing into args)
     *      - args... : arguments given to the function.
     * 
     * \tparam Func Function Type
     * \tparam Args Arguments type to function
     * 
     * \param f Function
     * \param funcDim Output dimension of the function
     * \param argsDim If arguments to the function are arrays, the number of elements
     * \param args Arguments to the function
     */
    template<typename Func, class... Args>
    nb::ndarray<nb::array_api> DispatchFunction(Func&& f, size_t funcDim, size_t argsDim, const Args&... args)
    {
        if (!m_hasFilter)
            throw std::runtime_error("No filter set. Please call setNeighborFilter first.");
        if (!m_hasPointCloud)
            throw std::runtime_error("No pointcloud attached. Please call attach first.");
        
        switch(m_executionMode.device)
        {
        case ParallelExecutionMode::Device::CPU:
            return computeCPU(std::move(f), funcDim, argsDim, std::forward(args)...);
        default:
            throw std::runtime_error("Unsupported device");
        }
    }
private:
    /**
     * \brief Performs the computation
     * 
     * The function signature should be:
     *      - Scalar* : a pointer to output data. 
     *      - Object& : a reference to the compute object
     *      - int : current index (for indexing into args)
     *      - args... : arguments given to the function.
     *
     * \tparam Func Function Type
     * \tparam Args Arguments type to function
     * 
     * \param f Function
     * \param funcDim Output dimension of the function
     * \param argsDim If arguments to the function are arrays, the number of elements
     * \param args Arguments to the function
     */
    template<typename Func, class... Args>
    nb::ndarray<nb::array_api> computeCPU(Func&& f, size_t funcDim, size_t argsDim, const Args&... args)
    {
        int numThreads = m_executionMode.numberOfThreads;
#ifdef _OPENMP
        if (numThreads < 0) numThreads = omp_get_max_threads();
#endif
        switch(m_executionMode.storage)
        {
        case ParallelExecutionMode::Storage::THREAD_LOCAL:
            m_computeObjects.resize(m_executionMode.numberOfThreads);
            break;
        case ParallelExecutionMode::Storage::OBJECT:
            if (!m_hasCompute)
                m_computeObjects.resize(m_pos.shape(0));
            break;
        }

        Scalar* data = new Scalar[m_pos.shape(0) * funcDim * argsDim];
#ifdef _OPENMP
        #pragma omp parallel for num_threads(numThreads) if(numThreads > 1)
#endif
        for (size_t i = 0; i < m_pos.shape(0); ++i)
        {
            CO* object = nullptr;
            switch (m_executionMode.storage)
            {
            case ParallelExecutionMode::Storage::THREAD_LOCAL:
                {
#ifdef _OPENMP
                    object = &m_computeObjects[omp_get_thread_num()];
#else
                    object = &m_computeObjects[0];
#endif
                }
                break;
            case ParallelExecutionMode::Storage::OBJECT:
                {
                    object = &m_computeObjects[i];
                }    
                break;
            }
            
            // Perform computation if we can not reuse indices
            if (m_executionMode.storage != ParallelExecutionMode::Storage::OBJECT || !m_hasCompute)
            {
                object->setNeighborFilter({ PyVectorArrayIndex<Point>(m_pos, i), PyScalarArrayIndex<Point>(m_t, i) });
                if (!m_kdtree)
                {
                    object->compute(m_pointcloud.begin(), m_pointcloud.end());
                }
                else
                {
                    m_kdtree->Run([&](const auto& tree) {
                        auto neighbors = tree.rangeNeighbors(PyVectorArrayIndex<Point>(m_pos, i), PyScalarArrayIndex<Point>(m_t, i));
                        object->computeWithIds(neighbors, tree.points());
                    });
                }
            }
            f(data + i * funcDim * argsDim, *object, i, args...);
        }

        return nb::ndarray<nb::array_api>(
            data, { m_pos.shape(0), funcDim, argsDim }, PyArrayDeleter(data), 
            {int64_t(funcDim * argsDim), int64_t(funcDim), 1}, nb::dtype<Scalar>(), 
            nb::device::cpu::value
        );
    }

    // Execution information
    std::vector<CO> m_computeObjects;
    ParallelExecutionMode m_executionMode;
    
    // Current status
    bool m_hasFilter;
    bool m_hasPointCloud;
    bool m_hasCompute;

    // Input provided by the user
    PointCLoud m_pointcloud;
    const PyKDTree<PointCloud>* m_kdtree;

    PyVectorArray<Point> m_pos;
    PyScalarArray<Point> m_t;
};
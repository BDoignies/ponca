/*
This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/
#pragma once

/**
 * \brief Information about how to execute parallel code
 */
struct ParallelExecutionMode
{
    enum class Device
    {
        CPU 
    };

    /**
     * \brief Decide if objects are only thread local or bound to lifetime of object
     */
    enum class Storage
    {
        THREAD_LOCAL, 
        OBJECT
    };

    ParallelExecutionMode() : storage(Storage::OBJECT), device(Device::CPU), numberOfThreads(-1) {}

    ParallelExecutionMode(Storage s, Device d, int nT) : storage(s), device(d), numberOfThreads(nT)
    { }

    Device device = Device::CPU;
    Storage storage = Storage::THREAD_LOCAL;

    int numberOfThreads = 1;
};
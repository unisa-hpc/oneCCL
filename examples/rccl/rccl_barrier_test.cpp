/*
 * Copyright 2016-2020 Intel Corporation
 * Copyright 2026 Contributors - AMD ROCm/RCCL backend support
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <sycl/sycl.hpp>
#include <oneapi/ccl.hpp>
#include <mpi.h>

sycl::device get_device_for_rank(int rank) {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
        std::cerr << "No GPU devices found!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return devices[rank % devices.size()];
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ccl::init();

    sycl::device dev = get_device_for_rank(rank);
    sycl::context ctx(dev);
    sycl::queue q(ctx, dev, sycl::property::queue::in_order());

    std::cout << "Rank " << rank << " using device: "
              << dev.get_info<sycl::info::device::name>() << std::endl;

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type addr;

    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        addr = kvs->get_address();
        MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(addr);
    }

    auto ccl_dev = ccl::create_device(q.get_device());
    auto ccl_ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, ccl_dev, ccl_ctx, kvs);
    auto ccl_stream = ccl::create_stream(q);

    // Test barrier synchronization
    const int num_iterations = 10;
    bool success = true;

    for (int iter = 0; iter < num_iterations; iter++) {
        // Each rank waits a different amount of time to test barrier synchronization
        std::this_thread::sleep_for(std::chrono::milliseconds(rank * 10));

        try {
            auto event = ccl::barrier(comm, ccl_stream);
            event.wait();

            if (rank == 0) {
                std::cout << "Barrier iteration " << iter << " completed" << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Rank " << rank << ": Barrier failed at iteration " << iter
                      << ": " << e.what() << std::endl;
            success = false;
            break;
        }
    }

    int local_ok = success ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "RCCL Barrier Test: " << (global_ok ? "PASSED" : "FAILED") << std::endl;
    }

    MPI_Finalize();

    return global_ok ? 0 : 1;
}

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
              << dev.get_info<sycl::info::device::name>()
              << " (vendor: " << dev.get_info<sycl::info::device::vendor>() << ")"
              << std::endl;

    // Test KVS creation and address exchange
    bool success = true;

    try {
        ccl::shared_ptr_class<ccl::kvs> kvs;
        ccl::kvs::address_type addr;

        if (rank == 0) {
            kvs = ccl::create_main_kvs();
            addr = kvs->get_address();
            std::cout << "Rank 0: Created main KVS with address size " << addr.size() << std::endl;
            MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
            kvs = ccl::create_kvs(addr);
            std::cout << "Rank " << rank << ": Created KVS from address" << std::endl;
        }

        // Create communicator to verify KVS works correctly
        auto ccl_dev = ccl::create_device(q.get_device());
        auto ccl_ctx = ccl::create_context(q.get_context());
        auto comm = ccl::create_communicator(size, rank, ccl_dev, ccl_ctx, kvs);

        std::cout << "Rank " << rank << ": Communicator created successfully" << std::endl;

        // Verify communicator properties
        if (comm.rank() != rank) {
            std::cerr << "Rank " << rank << ": Communicator rank mismatch!" << std::endl;
            success = false;
        }
        if (comm.size() != size) {
            std::cerr << "Rank " << rank << ": Communicator size mismatch!" << std::endl;
            success = false;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Rank " << rank << ": Exception: " << e.what() << std::endl;
        success = false;
    }

    int local_ok = success ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "RCCL KVS Test: " << (global_ok ? "PASSED" : "FAILED") << std::endl;
    }

    MPI_Finalize();

    return global_ok ? 0 : 1;
}

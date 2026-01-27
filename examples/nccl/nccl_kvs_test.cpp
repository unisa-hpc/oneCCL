/*
 * Copyright 2016-2020 Intel Corporation
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
#include <iomanip>
#include <mpi.h>
#include <sycl/sycl.hpp>
#include "oneapi/ccl.hpp"

void print_address_bytes(const ccl::kvs::address_type& addr, int rank) {
    std::cout << "Rank " << rank << " address bytes (first 32): ";
    for (size_t i = 0; i < 32 && i < addr.size(); i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << (static_cast<unsigned int>(static_cast<unsigned char>(addr[i])));
    }
    std::cout << std::dec << std::endl;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int size = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "=== OneCCL NCCL Backend - KVS Test ===" << std::endl;
        std::cout << "Number of ranks: " << size << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Initialize CCL
    ccl::init();

    {
        // Setup SYCL device
        sycl::device dev{ sycl::gpu_selector_v };
        sycl::context ctx{ dev };
        sycl::queue q{ ctx, dev };

        std::cout << "Rank " << rank << "/" << size << " SYCL device: "
                  << dev.get_info<sycl::info::device::name>()
                  << " | platform: "
                  << dev.get_platform().get_info<sycl::info::platform::name>()
                  << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);

        // KVS initialization
        ccl::shared_ptr_class<ccl::kvs> kvs;
        ccl::kvs::address_type main_addr;

        if (rank == 0) {
            std::cout << "\n--- KVS Creation ---" << std::endl;
            kvs = ccl::create_main_kvs();
            main_addr = kvs->get_address();
            std::cout << "Rank 0: Main KVS created successfully" << std::endl;
            print_address_bytes(main_addr, 0);
        }

        // Broadcast the KVS address to all ranks
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            kvs = ccl::create_kvs(main_addr);
            std::cout << "Rank " << rank << ": KVS created from received address" << std::endl;
            print_address_bytes(main_addr, rank);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Verify all ranks have the same address
        ccl::kvs::address_type local_addr = kvs->get_address();

        bool addresses_match = true;
        for (size_t i = 0; i < main_addr.size(); i++) {
            if (local_addr[i] != main_addr[i]) {
                addresses_match = false;
                break;
            }
        }

        if (addresses_match) {
            std::cout << "Rank " << rank << ": Address verification PASSED" << std::endl;
        } else {
            std::cerr << "Rank " << rank << ": Address verification FAILED" << std::endl;
        }

        // Print KVS info
        std::cout << "Rank " << rank << " KVS ID: " << kvs->get_id() << std::endl;
        std::cout << "Rank " << rank << " Address size: " << kvs->get_address().size() << " bytes" << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);

        // Test creating CCL device and context
        auto ccl_dev = ccl::create_device(q.get_device());
        auto ccl_ctx = ccl::create_context(q.get_context());

        std::cout << "Rank " << rank << ": CCL device and context created successfully" << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "\n=== KVS Test Completed Successfully ===" << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}

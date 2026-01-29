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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int size = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* init ccl */
    ccl::init();

    {
        /* create sycl queue */
        sycl::device dev{ sycl::gpu_selector_v };
        sycl::context ctx{ dev };
        sycl::queue q{ ctx, dev };

        /* create kvs */
        ccl::shared_ptr_class<ccl::kvs> kvs;
        ccl::kvs::address_type main_addr;

        if (rank == 0) {
            kvs = ccl::create_main_kvs();
            main_addr = kvs->get_address();
            MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
            kvs = ccl::create_kvs(main_addr);
        }

        /* check address */
        ccl::kvs::address_type local_addr = kvs->get_address();

        bool addresses_match = true;
        for (size_t i = 0; i < main_addr.size(); i++) {
            if (local_addr[i] != main_addr[i]) {
                addresses_match = false;
                break;
            }
        }

        /* create device and context */
        auto ccl_dev = ccl::create_device(q.get_device());
        auto ccl_ctx = ccl::create_context(q.get_context());

        int local_ok = addresses_match ? 1 : 0;
        int global_ok = 0;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << (global_ok ? "PASSED" : "FAILED") << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}

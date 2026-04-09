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
#include <mpi.h>
#include <sycl/sycl.hpp>
#include <oneapi/ccl.hpp>

// Helper to select a GPU device based on rank
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

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ccl::init();

    /* create sycl queue */
    sycl::device dev = get_device_for_rank(rank);
    sycl::context ctx(dev);
    sycl::queue q(ctx, dev, sycl::property::queue::in_order());

    /* create kvs */
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

    /* create communicator */
    auto ccl_dev = ccl::create_device(q.get_device());
    auto ccl_ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, ccl_dev, ccl_ctx, kvs);
    auto stream = ccl::create_stream(q);

    bool success = true;
    const int num_iterations = 5;

    /*
     * Test: call barrier multiple times, each preceded by GPU work.
     * After each barrier, verify all ranks have completed their GPU work
     * by collecting a per-rank flag through MPI.
     */
    const size_t count = 256;
    int* d_buf = sycl::malloc_device<int>(count, q);

    for (int iter = 0; iter < num_iterations; ++iter) {
        /* each rank fills its device buffer with (rank + iter) */
        int fill_val = rank + iter;
        q.fill(d_buf, fill_val, count).wait();

        /* invoke barrier */
        ccl::barrier(comm, stream).wait();
        q.wait();

        /* verify the device buffer still holds the expected value */
        std::vector<int> host_buf(count);
        q.memcpy(host_buf.data(), d_buf, count * sizeof(int)).wait();

        for (size_t i = 0; i < count; ++i) {
            if (host_buf[i] != fill_val) {
                std::cerr << "Rank " << rank << ": ERROR at iteration " << iter
                          << ", index " << i << ": got " << host_buf[i]
                          << ", expected " << fill_val << std::endl;
                success = false;
                break;
            }
        }

        /* confirm all ranks passed this iteration */
        int local_ok = success ? 1 : 0;
        int global_ok = 0;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (!global_ok) {
            success = false;
            break;
        }
    }

    sycl::free(d_buf, q);

    int local_ok = success ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << (global_ok ? "PASSED" : "FAILED") << std::endl;
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
}

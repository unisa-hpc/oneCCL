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
#include <cmath>
#include <sycl/sycl.hpp>
#include <oneapi/ccl.hpp>
#include <mpi.h>

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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* init ccl */
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

    /* create stream */
    auto ccl_stream = ccl::create_stream(q);

    /* create buffers */
    const size_t count = 1024;

    std::vector<float> send_data(count, static_cast<float>(rank + 1));  // rank 0 -> 1.0, rank 1 -> 2.0
    std::vector<float> recv_data(count, 0.0f);

    float* d_send = sycl::malloc_device<float>(count, q);
    float* d_recv = sycl::malloc_device<float>(count, q);

    q.memcpy(d_send, send_data.data(), count * sizeof(float)).wait();
    q.memset(d_recv, 0, count * sizeof(float)).wait();

    /* invoke allreduce */
    auto event = ccl::allreduce(d_send,
                                 d_recv,
                                 count,
                                 ccl::datatype::float32,
                                 ccl::reduction::sum,
                                 comm,
                                 ccl_stream);

    event.wait();
    q.wait();

    q.memcpy(recv_data.data(), d_recv, count * sizeof(float)).wait();

    /* check result */
    float expected = 0.0f;
    for (int i = 0; i < size; i++) {
        expected += static_cast<float>(i + 1);
    }

    bool success = true;
    int errors = 0;
    for (size_t i = 0; i < count && errors < 5; i++) {
        if (std::fabs(recv_data[i] - expected) > 1e-5f) {
            std::cerr << "Rank " << rank << ": ERROR at index " << i
                      << ": got " << recv_data[i] << ", expected " << expected << std::endl;
            success = false;
            errors++;
        }
    }

    int local_ok = success ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    sycl::free(d_send, q);
    sycl::free(d_recv, q);

    if (rank == 0) {
        std::cout << (global_ok ? "PASSED" : "FAILED") << std::endl;
    }

    MPI_Finalize();

    return global_ok ? 0 : 1;
}

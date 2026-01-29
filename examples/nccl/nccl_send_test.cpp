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

    if (size < 2) {
        if (rank == 0) {
            std::cout << "Need at least 2 ranks to run SEND test" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    ccl::init();

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

    /* create buffers */
    const size_t count = 1024;
    std::vector<float> send_data(count);
    std::vector<float> recv_data(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        send_data[i] = static_cast<float>(i + 1);
    }

    float* d_send = sycl::malloc_device<float>(count, q);
    float* d_recv = sycl::malloc_device<float>(count, q);

    if (rank == 0) {
        q.memcpy(d_send, send_data.data(), count * sizeof(float)).wait();
    }
    q.memset(d_recv, 0, count * sizeof(float)).wait();

    /* invoke send/recv */
    if (rank == 0) {
        ccl::send(d_send, count, ccl::datatype::float32, 1, comm, stream).wait();
    }
    else if (rank == 1) {
        ccl::recv(d_recv, count, ccl::datatype::float32, 0, comm, stream).wait();
    }

    q.wait();

    bool success = true;
    if (rank == 1) {
        q.memcpy(recv_data.data(), d_recv, count * sizeof(float)).wait();
        int errors = 0;
        for (size_t i = 0; i < count && errors < 5; ++i) {
            float expected = static_cast<float>(i + 1);
            if (std::fabs(recv_data[i] - expected) > 1e-5f) {
                std::cerr << "Rank " << rank << ": ERROR at index " << i
                          << ": got " << recv_data[i] << ", expected " << expected << std::endl;
                success = false;
                errors++;
            }
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

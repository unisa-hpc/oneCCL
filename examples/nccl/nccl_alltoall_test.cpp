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
    const size_t count = 8;
    const size_t total = count * static_cast<size_t>(size);

    std::vector<int> send_data(total);
    std::vector<int> recv_data(total, -1);

    for (int peer = 0; peer < size; ++peer) {
        for (size_t i = 0; i < count; ++i) {
            send_data[(peer * count) + i] = (rank * 1000) + (peer * 100) + static_cast<int>(i);
        }
    }

    int* d_send = sycl::malloc_device<int>(total, q);
    int* d_recv = sycl::malloc_device<int>(total, q);

    q.memcpy(d_send, send_data.data(), total * sizeof(int)).wait();
    q.memset(d_recv, 0, total * sizeof(int)).wait();

    /* invoke alltoall */
    ccl::alltoall(d_send, d_recv, count, ccl::datatype::int32, comm, stream).wait();
    q.wait();

    q.memcpy(recv_data.data(), d_recv, total * sizeof(int)).wait();

    bool success = true;
    int errors = 0;
    for (int peer = 0; peer < size; ++peer) {
        for (size_t i = 0; i < count; ++i) {
            int expected = (peer * 1000) + (rank * 100) + static_cast<int>(i);
            int value = recv_data[(peer * count) + i];
            if (value != expected) {
                std::cerr << "Rank " << rank << ": ERROR from peer " << peer
                          << " at index " << i << ": got " << value
                          << ", expected " << expected << std::endl;
                success = false;
                if (++errors >= 5) {
                    break;
                }
            }
        }
        if (errors >= 5) {
            break;
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

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
#include <cmath>
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

    const size_t count_per_rank = 256;
    const size_t total_count = count_per_rank * size;

    // Initialize send buffer: each chunk destined for rank r contains value = rank * 1000 + r
    std::vector<float> send_data(total_count);
    std::vector<float> recv_data(total_count, 0.0f);

    for (int dest = 0; dest < size; dest++) {
        for (size_t i = 0; i < count_per_rank; i++) {
            send_data[dest * count_per_rank + i] = static_cast<float>(rank * 1000 + dest);
        }
    }

    float* d_send = sycl::malloc_device<float>(total_count, q);
    float* d_recv = sycl::malloc_device<float>(total_count, q);

    q.memcpy(d_send, send_data.data(), total_count * sizeof(float)).wait();
    q.memset(d_recv, 0, total_count * sizeof(float)).wait();

    auto event = ccl::alltoall(d_send,
                                d_recv,
                                count_per_rank,
                                ccl::datatype::float32,
                                comm,
                                ccl_stream);

    event.wait();
    q.wait();

    q.memcpy(recv_data.data(), d_recv, total_count * sizeof(float)).wait();

    // Verify: chunk from rank src should contain value = src * 1000 + rank
    bool success = true;
    int errors = 0;
    for (int src = 0; src < size && errors < 5; src++) {
        float expected = static_cast<float>(src * 1000 + rank);
        for (size_t i = 0; i < count_per_rank && errors < 5; i++) {
            size_t idx = src * count_per_rank + i;
            if (std::fabs(recv_data[idx] - expected) > 1e-5f) {
                std::cerr << "Rank " << rank << ": ERROR at index " << idx
                          << " (from rank " << src << "): got " << recv_data[idx]
                          << ", expected " << expected << std::endl;
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
        std::cout << "RCCL Alltoall Test: " << (global_ok ? "PASSED" : "FAILED") << std::endl;
    }

    MPI_Finalize();

    return global_ok ? 0 : 1;
}

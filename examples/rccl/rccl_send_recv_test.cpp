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

    if (size < 2) {
        if (rank == 0) {
            std::cerr << "This test requires at least 2 ranks" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

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

    const size_t count = 1024;
    const int tag = 42;
    bool success = true;

    float* d_buf = sycl::malloc_device<float>(count, q);
    std::vector<float> host_buf(count);

    // Ring pattern: rank i sends to rank (i+1) % size and receives from rank (i-1+size) % size
    int dest = (rank + 1) % size;
    int src = (rank - 1 + size) % size;

    // Initialize send buffer with rank-specific data
    float send_value = static_cast<float>(rank * 100);
    for (size_t i = 0; i < count; i++) {
        host_buf[i] = send_value + static_cast<float>(i);
    }
    q.memcpy(d_buf, host_buf.data(), count * sizeof(float)).wait();

    // Each rank sends and receives simultaneously
    std::vector<ccl::event> events;

    // Send to next rank
    events.push_back(ccl::send(d_buf,
                                count,
                                ccl::datatype::float32,
                                dest,
                                comm,
                                ccl_stream));

    // Use a separate buffer for receiving
    float* d_recv_buf = sycl::malloc_device<float>(count, q);
    q.memset(d_recv_buf, 0, count * sizeof(float)).wait();

    // Receive from previous rank
    events.push_back(ccl::recv(d_recv_buf,
                                count,
                                ccl::datatype::float32,
                                src,
                                comm,
                                ccl_stream));

    // Wait for all operations
    for (auto& e : events) {
        e.wait();
    }
    q.wait();

    // Verify received data
    q.memcpy(host_buf.data(), d_recv_buf, count * sizeof(float)).wait();

    float expected_base = static_cast<float>(src * 100);
    int errors = 0;
    for (size_t i = 0; i < count && errors < 5; i++) {
        float expected = expected_base + static_cast<float>(i);
        if (std::fabs(host_buf[i] - expected) > 1e-5f) {
            std::cerr << "Rank " << rank << ": ERROR at index " << i
                      << ": got " << host_buf[i] << ", expected " << expected << std::endl;
            success = false;
            errors++;
        }
    }

    sycl::free(d_buf, q);
    sycl::free(d_recv_buf, q);

    int local_ok = success ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "RCCL Send/Recv Test: " << (global_ok ? "PASSED" : "FAILED") << std::endl;
    }

    MPI_Finalize();

    return global_ok ? 0 : 1;
}

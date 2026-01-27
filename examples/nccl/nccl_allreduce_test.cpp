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

// Helper per selezionare il device GPU in base al rank
sycl::device get_device_for_rank(int rank) {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
        std::cerr << "No GPU devices found!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return devices[rank % devices.size()];
}

int main(int argc, char* argv[]) {
    // ========================================
    // Step 1: Inizializzazione MPI
    // ========================================
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "=== OneCCL NCCL Communicator Test ===" << std::endl;
    std::cout << "Rank " << rank << "/" << size << " started" << std::endl;

    // ========================================
    // Step 2: Inizializzazione CCL
    // ========================================
    ccl::init();

    // ========================================
    // Step 3: Setup Device SYCL/CUDA
    // ========================================
    sycl::device dev = get_device_for_rank(rank);
    sycl::context ctx(dev);
    sycl::queue q(ctx, dev, sycl::property::queue::in_order());

    std::cout << "Rank " << rank << "/" << size
              << " SYCL device: " << dev.get_info<sycl::info::device::name>()
              << " | platform: " << dev.get_platform().get_info<sycl::info::platform::name>()
              << std::endl;

    // ========================================
    // Step 4: Setup KVS (Key-Value Store)
    // ========================================
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type addr;

    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        addr = kvs->get_address();
        std::cout << "Rank 0: Main KVS created" << std::endl;
    }

    MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        kvs = ccl::create_kvs(addr);
        std::cout << "Rank " << rank << ": KVS created from address" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ========================================
    // Step 5: Creazione Communicator CCL (NCCL backend)
    // ========================================
    std::cout << "Rank " << rank << ": Creating communicator..." << std::endl;

    auto ccl_dev = ccl::create_device(q.get_device());
    auto ccl_ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, ccl_dev, ccl_ctx, kvs);

    std::cout << "Rank " << rank << ": Communicator created successfully!" << std::endl;
    std::cout << "  - comm.rank() = " << comm.rank() << std::endl;
    std::cout << "  - comm.size() = " << comm.size() << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    // ========================================
    // Step 6: Test Allreduce
    // ========================================
    const size_t count = 1024;

    // Buffer host
    std::vector<float> send_data(count, static_cast<float>(rank + 1));  // rank 0 -> 1.0, rank 1 -> 2.0
    std::vector<float> recv_data(count, 0.0f);

    // Buffer device
    float* d_send = sycl::malloc_device<float>(count, q);
    float* d_recv = sycl::malloc_device<float>(count, q);

    // Copia dati su device
    q.memcpy(d_send, send_data.data(), count * sizeof(float)).wait();
    q.memset(d_recv, 0, count * sizeof(float)).wait();

    std::cout << "Rank " << rank << ": Calling allreduce with " << count << " elements..." << std::endl;
    std::cout << "  - send_data[0] = " << send_data[0] << std::endl;

    // Crea stream CCL
    auto ccl_stream = ccl::create_stream(q);

    // Esegui allreduce (sum)
    auto event = ccl::allreduce(d_send,
                                 d_recv,
                                 count,
                                 ccl::datatype::float32,
                                 ccl::reduction::sum,
                                 comm,
                                 ccl_stream);

    // Attendi completamento
    event.wait();

    // Sincronizza la queue SYCL
    q.wait();

    // Copia risultato su host
    q.memcpy(recv_data.data(), d_recv, count * sizeof(float)).wait();

    // ========================================
    // Step 7: Verifica Risultato
    // ========================================
    // Per 2 rank: sum = 1.0 + 2.0 = 3.0
    // Per N rank: sum = 1 + 2 + ... + N = N*(N+1)/2
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

    if (success) {
        std::cout << "Rank " << rank << ": ALLREDUCE SUCCESS!" << std::endl;
        std::cout << "  - recv_data[0] = " << recv_data[0] << " (expected " << expected << ")" << std::endl;
    } else {
        std::cerr << "Rank " << rank << ": ALLREDUCE FAILED!" << std::endl;
    }

    // ========================================
    // Step 8: Cleanup
    // ========================================
    sycl::free(d_send, q);
    sycl::free(d_recv, q);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n=== Test Completed ===" << std::endl;
    }

    MPI_Finalize();

    return success ? 0 : 1;
}

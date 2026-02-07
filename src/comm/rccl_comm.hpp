/*
 Copyright 2016-2020 Intel Corporation
 Copyright 2026 Contributors - AMD ROCm/RCCL backend support

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
#pragma once

#ifdef CCL_ENABLE_RCCL

// HIP headers require platform definition - we're targeting AMD GPUs
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include "comm/comm_interface.hpp"
#include "common/api_wrapper/rccl_api_wrapper.hpp"
#include <hip/hip_runtime.h>

namespace ccl {

class rccl_kvs_impl;

class alignas(CACHELINE_SIZE) rccl_comm : public ccl::comm_interface {
public:
    rccl_comm(rccl_comm& src) = delete;
    rccl_comm(rccl_comm&& src) = default;
    rccl_comm& operator=(rccl_comm& src) = delete;
    rccl_comm& operator=(rccl_comm&& src) = default;
    ~rccl_comm();

    static rccl_comm* create(device_t device,
                             context_t context,
                             size_t size,
                             size_t rank,
                             std::shared_ptr<ccl::kvs_interface> kvs_interface);

private:
    rccl_comm(device_t device,
              context_t context,
              size_t size,
              size_t rank,
              ncclUniqueId rccl_id,
              std::shared_ptr<ccl::kvs> kvs,
              const ccl::rccl_kvs_impl* kvs_impl);

public:
    int rank() const override {
        return comm_rank;
    }

    int size() const override {
        return comm_size;
    }

    device_ptr_t get_device() const override {
        return device_ptr;
    }

    context_ptr_t get_context() const override {
        return context_ptr;
    }

    // collective operation declarations
    ccl::event barrier(const ccl::stream::impl_value_t& stream,
                       const ccl::barrier_attr& attr,
                       const ccl::vector_class<ccl::event>& deps = {}) override {
        return barrier_impl(stream, attr, deps);
    }

    ccl::event barrier_impl(const ccl::stream::impl_value_t& stream,
                            const ccl::barrier_attr& attr,
                            const ccl::vector_class<ccl::event>& deps = {});

    COMM_INTERFACE_COLL_DEFINITION__VOID_REQUIRED

    COMM_IMPL_DECLARATION_VOID_REQUIRED

    ccl::comm_interface_ptr split(int color, int key, bool split_external_use) override {
        CCL_THROW("split is not supported for RCCL backend yet");
    }

private:
    rccl_comm* get_impl() {
        return this;
    }

    hipStream_t get_hip_stream(const ccl::stream::impl_value_t& stream);
    ncclDataType_t get_rccl_datatype(ccl::datatype dtype);
    ncclRedOp_t get_rccl_reduction(ccl::reduction reduction);

    device_ptr_t device_ptr;
    context_ptr_t context_ptr;

    size_t comm_rank;
    size_t comm_size;

    ncclComm_t rccl_comm_handle = nullptr;

    // while we only use the impl, keep the original object to avoid early destruction
    std::shared_ptr<ccl::kvs> kvs;
    const ccl::rccl_kvs_impl* kvs_impl;
};

} // namespace ccl

#endif // CCL_ENABLE_RCCL

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

#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/aliases.hpp"
#include "common/global/global.hpp"
#include "common/api_wrapper/rccl_api_wrapper.hpp"
#include "kvs_impl.hpp"

namespace ccl {

class rccl_kvs_impl : public base_kvs_impl {
public:
    // constructor for rank 0 (main KVS) - generates the ncclUniqueId
    rccl_kvs_impl();

    // constructor for non-root ranks - receives the address containing ncclUniqueId
    rccl_kvs_impl(const kvs::address_type& addr);

    ncclUniqueId get_rccl_id() const;

    // get/set are not needed for the RCCL backend (the KVS is only used to distribute the ID)
    vector_class<char> get(const string_class& key) override {
        CCL_THROW("get() is not needed for RCCL backend");
    }

    void set(const string_class& key, const vector_class<char>& data) override {
        CCL_THROW("set() is not needed for RCCL backend");
    }

private:
    ncclUniqueId rccl_id;

    // helpers to convert between ncclUniqueId and address_type
    static kvs::address_type convert_id_to_addr(const ncclUniqueId& id);
    static ncclUniqueId convert_addr_to_id(const kvs::address_type& addr);
};

} // namespace ccl

#endif // CCL_ENABLE_RCCL

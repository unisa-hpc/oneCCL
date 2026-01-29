/*
 Copyright 2016-2020 Intel Corporation

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
#ifdef CCL_ENABLE_NCCL

#include "nccl_kvs_impl.hpp"
#include "common/log/log.hpp"
#include <cstring>

namespace ccl {

/* convert ncclUniqueId (128 bytes) -> address_type */
kvs::address_type nccl_kvs_impl::convert_id_to_addr(const ncclUniqueId& id) {
    kvs::address_type addr{ 0 };

    static_assert(sizeof(ncclUniqueId) <= kvs::address_max_size,
                  "ncclUniqueId too large for kvs::address_type");

    memcpy(addr.data(), &id, sizeof(ncclUniqueId));

    return addr;
}

/* convert address_type -> ncclUniqueId */
ncclUniqueId nccl_kvs_impl::convert_addr_to_id(const kvs::address_type& addr) {
    ncclUniqueId id;
    memcpy(&id, addr.data(), sizeof(ncclUniqueId));
    return id;
}

/* constructor for rank 0 (main KVS) */
nccl_kvs_impl::nccl_kvs_impl() : base_kvs_impl() {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::nccl,
                     "unexpected backend");

    auto status = ncclGetUniqueId(&nccl_id);
    CCL_THROW_IF_NOT(status == ncclSuccess,
                     "ncclGetUniqueId failed: ", ncclGetErrorString(status));

    LOG_DEBUG("NCCL KVS: generated unique ID");

    addr = convert_id_to_addr(nccl_id);
}

/* constructor for non-root ranks */
nccl_kvs_impl::nccl_kvs_impl(const kvs::address_type& input_addr)
    : base_kvs_impl(input_addr) {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::nccl,
                     "unexpected backend");

    nccl_id = convert_addr_to_id(input_addr);

    LOG_DEBUG("NCCL KVS: received unique ID from address");
}

ncclUniqueId nccl_kvs_impl::get_nccl_id() const {
    return nccl_id;
}

} // namespace ccl

#endif // CCL_ENABLE_NCCL

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
#ifdef CCL_ENABLE_RCCL

// HIP headers require platform definition - we're targeting AMD GPUs
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include "comm/rccl_comm.hpp"
#include "common/event/impls/stub_event.hpp"
#include "rccl_kvs_impl.hpp"
#include "common/log/log.hpp"
#include "oneapi/ccl/api_functions.hpp"

#if defined(CCL_ENABLE_SYCL)
#include <sycl/sycl.hpp>
#endif

namespace ccl {

rccl_comm::rccl_comm(device_t device,
                     context_t context,
                     size_t size,
                     size_t rank,
                     ncclUniqueId rccl_id,
                     std::shared_ptr<ccl::kvs> kvs,
                     const ccl::rccl_kvs_impl* kvs_impl)
        : device_ptr(std::make_shared<ccl::device>(device)),
          context_ptr(std::make_shared<ccl::context>(context)),
          comm_rank(rank),
          comm_size(size),
          kvs(kvs),
          kvs_impl(kvs_impl) {

    LOG_DEBUG("RCCL COMM: initializing communicator for rank ", rank, "/", size);

    ncclResult_t status = rcclCommInitRank(&rccl_comm_handle, size, rccl_id, rank);
    CCL_THROW_IF_NOT(status == ncclSuccess,
                     "rcclCommInitRank failed: ", rcclGetErrorString(status));

    LOG_INFO("RCCL COMM: communicator initialized successfully for rank ", rank, "/", size);
}

rccl_comm::~rccl_comm() {
    LOG_DEBUG("RCCL COMM: destroying communicator for rank ", comm_rank);

    if (rccl_comm_handle != nullptr) {
        ncclResult_t status = rcclCommDestroy(rccl_comm_handle);
        if (status != ncclSuccess) {
            LOG_WARN("RCCL COMM: rcclCommDestroy failed: ", rcclGetErrorString(status));
        }
    }
}

rccl_comm* rccl_comm::create(device_t device,
                             context_t context,
                             size_t size,
                             size_t rank,
                             std::shared_ptr<ccl::kvs_interface> kvs_interface) {
    auto kvs_inst = std::dynamic_pointer_cast<ccl::kvs>(kvs_interface);
    CCL_THROW_IF_NOT(kvs_inst != nullptr, "only ccl::kvs is allowed with RCCL backend");

    auto kvs_impl = ccl::get_kvs_impl_typed<rccl_kvs_impl>(kvs_inst);

    ncclUniqueId rccl_id = kvs_impl->get_rccl_id();

    return new rccl_comm(device, context, size, rank, rccl_id, std::move(kvs_inst), kvs_impl);
}

/* barrier */
ccl::event rccl_comm::barrier_impl(const ccl::stream::impl_value_t& stream,
                                   const ccl::barrier_attr& attr,
                                   const ccl::vector_class<ccl::event>& deps) {
    // RCCL does not have a native barrier, simulate with allreduce on a single element
    LOG_DEBUG("RCCL COMM: barrier (simulated with allreduce)");

    hipStream_t hip_stream = get_hip_stream(stream);

    // Allocate device memory via SYCL to avoid direct HIP link dependency
    auto sycl_queue = stream->get_native_stream();
    int* d_dummy = sycl::malloc_device<int>(1, sycl_queue);
    CCL_THROW_IF_NOT(d_dummy != nullptr,
                     "sycl::malloc_device for barrier dummy buffer failed");

    sycl_queue.memset(d_dummy, 0, sizeof(int)).wait();

    ncclResult_t status = rcclAllReduce(d_dummy, d_dummy, 1, ncclInt, ncclSum,
                                        rccl_comm_handle, hip_stream);

    // synchronize and free
    sycl_queue.wait();
    sycl::free(d_dummy, sycl_queue);

    CCL_THROW_IF_NOT(status == ncclSuccess,
                     "RCCL barrier (allreduce) failed: ", rcclGetErrorString(status));

    return std::unique_ptr<ccl::event_impl>(new ccl::stub_event_impl());
}

/* allreduce */
ccl::event rccl_comm::allreduce_impl(const void* send_buf,
                                     void* recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     ccl::reduction reduction,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allreduce_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("RCCL COMM: allreduce count=", count, " dtype=", static_cast<int>(dtype),
              " reduction=", static_cast<int>(reduction));

    ncclDataType_t rccl_dtype = get_rccl_datatype(dtype);
    ncclRedOp_t rccl_op = get_rccl_reduction(reduction);
    hipStream_t hip_stream = get_hip_stream(stream);

    ncclResult_t status = rcclAllReduce(send_buf, recv_buf, count, rccl_dtype, rccl_op,
                                        rccl_comm_handle, hip_stream);
    CCL_THROW_IF_NOT(status == ncclSuccess,
                     "rcclAllReduce failed: ", rcclGetErrorString(status));

    LOG_DEBUG("RCCL COMM: allreduce completed successfully");

    return std::unique_ptr<ccl::event_impl>(new ccl::stub_event_impl());
}

/* extract hipStream_t from SYCL queue */
hipStream_t rccl_comm::get_hip_stream(const ccl::stream::impl_value_t& stream) {
#if defined(CCL_ENABLE_SYCL)
    try {
        auto sycl_queue = stream->get_native_stream();
        auto hip_stream = sycl::get_native<sycl::backend::ext_oneapi_hip>(sycl_queue);

        LOG_DEBUG("RCCL COMM: extracted HIP stream: ", hip_stream);
        return hip_stream;
    }
    catch (const std::exception& e) {
        CCL_THROW("Failed to extract HIP stream from SYCL queue: ", e.what());
    }
#else
    CCL_THROW("SYCL support not enabled, cannot extract HIP stream");
#endif
}

/* convert ccl::datatype -> ncclDataType_t */
ncclDataType_t rccl_comm::get_rccl_datatype(ccl::datatype dtype) {
    switch (dtype) {
        case ccl::datatype::int8:
            return ncclInt8;
        case ccl::datatype::uint8:
            return ncclUint8;
        case ccl::datatype::int32:
            return ncclInt32;
        case ccl::datatype::uint32:
            return ncclUint32;
        case ccl::datatype::int64:
            return ncclInt64;
        case ccl::datatype::uint64:
            return ncclUint64;
        case ccl::datatype::float16:
            return ncclFloat16;
        case ccl::datatype::float32:
            return ncclFloat32;
        case ccl::datatype::float64:
            return ncclFloat64;
        case ccl::datatype::bfloat16:
            return ncclBfloat16;
        default:
            CCL_THROW("Unsupported datatype for RCCL: ", static_cast<int>(dtype));
    }
}

/* convert ccl::reduction -> ncclRedOp_t */
ncclRedOp_t rccl_comm::get_rccl_reduction(ccl::reduction reduction) {
    switch (reduction) {
        case ccl::reduction::sum:
            return ncclSum;
        case ccl::reduction::prod:
            return ncclProd;
        case ccl::reduction::min:
            return ncclMin;
        case ccl::reduction::max:
            return ncclMax;
        default:
            CCL_THROW("Unsupported reduction operation for RCCL: ", static_cast<int>(reduction));
    }
}

// stub implementations for collectives not yet supported
#define RCCL_COMM_STUB_IMPL(name) \
    CCL_THROW(#name " is not implemented for RCCL backend yet");

/* allgather */
ccl::event rccl_comm::allgather_impl(const void* send_buf,
                                     void* recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allgather_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    RCCL_COMM_STUB_IMPL(allgather);
}

ccl::event rccl_comm::allgather_impl(const void* send_buf,
                                     const ccl::vector_class<void*>& recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allgather_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    RCCL_COMM_STUB_IMPL(allgather);
}

/* allgatherv */
ccl::event rccl_comm::allgatherv_impl(const void* send_buf,
                                      size_t send_count,
                                      void* recv_buf,
                                      const ccl::vector_class<size_t>& recv_counts,
                                      ccl::datatype dtype,
                                      const ccl::stream::impl_value_t& stream,
                                      const ccl::allgatherv_attr& attr,
                                      const ccl::vector_class<ccl::event>& deps) {
    RCCL_COMM_STUB_IMPL(allgatherv);
}

ccl::event rccl_comm::allgatherv_impl(const void* send_buf,
                                      size_t send_count,
                                      const ccl::vector_class<void*>& recv_bufs,
                                      const ccl::vector_class<size_t>& recv_counts,
                                      ccl::datatype dtype,
                                      const ccl::stream::impl_value_t& stream,
                                      const ccl::allgatherv_attr& attr,
                                      const ccl::vector_class<ccl::event>& deps) {
    RCCL_COMM_STUB_IMPL(allgatherv);
}

/* alltoall */
ccl::event rccl_comm::alltoall_impl(const void* send_buf,
                                    void* recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::alltoall_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("RCCL COMM: alltoall count=", count, " dtype=", static_cast<int>(dtype));

    ncclDataType_t rccl_dtype = get_rccl_datatype(dtype);
    hipStream_t hip_stream = get_hip_stream(stream);

#if CCL_RCCL_ALLTOALL_SUPPORTED
    int version = 0;
    ncclResult_t version_status = rcclGetVersion(&version);
    if (version_status == ncclSuccess && version >= NCCL_VERSION(2, 28, 0)) {
        auto alltoall_fn = rcclGetAllToAll();
        if (alltoall_fn) {
            LOG_DEBUG("RCCL COMM: using rcclAlltoAll native implementation");
            ncclResult_t status =
                alltoall_fn(send_buf, recv_buf, count, rccl_dtype, rccl_comm_handle, hip_stream);
            CCL_THROW_IF_NOT(status == ncclSuccess,
                             "rcclAlltoAll failed: ", rcclGetErrorString(status));
            return std::unique_ptr<ccl::event_impl>(new ccl::stub_event_impl());
        }
        else {
            LOG_DEBUG("RCCL COMM: rcclAlltoAll symbol not found, fallback to send/recv");
        }
    }
    else if (version_status != ncclSuccess) {
        LOG_DEBUG("RCCL COMM: rcclGetVersion failed, fallback to send/recv: ",
                  rcclGetErrorString(version_status));
    }
#endif
    // fallback: implement alltoall with send/recv inside a group
    LOG_DEBUG("RCCL COMM: using send/recv fallback implementation for alltoall");
    const size_t dtype_size = ccl::get_datatype_size(dtype);
    const size_t block_size = count * dtype_size;
    const char* send_base = static_cast<const char*>(send_buf);
    char* recv_base = static_cast<char*>(recv_buf);

    ncclResult_t status = rcclGroupStart();
    CCL_THROW_IF_NOT(status == ncclSuccess,
                     "rcclGroupStart failed: ", rcclGetErrorString(status));

    ncclResult_t first_error = ncclSuccess;
    for (size_t peer = 0; peer < comm_size; ++peer) {
        const void* send_ptr = send_base + (peer * block_size);
        void* recv_ptr = recv_base + (peer * block_size);

        status = rcclSend(send_ptr, count, rccl_dtype, static_cast<int>(peer), rccl_comm_handle,
                          hip_stream);
        if (status != ncclSuccess && first_error == ncclSuccess) {
            first_error = status;
        }

        status = rcclRecv(recv_ptr, count, rccl_dtype, static_cast<int>(peer), rccl_comm_handle,
                          hip_stream);
        if (status != ncclSuccess && first_error == ncclSuccess) {
            first_error = status;
        }
    }

    status = rcclGroupEnd();
    if (first_error == ncclSuccess) {
        first_error = status;
    }

    CCL_THROW_IF_NOT(first_error == ncclSuccess,
                     "rcclAllToAll (send/recv) failed: ", rcclGetErrorString(first_error));

    return std::unique_ptr<ccl::event_impl>(new ccl::stub_event_impl());
}

/* alltoallv */
ccl::event rccl_comm::alltoallv_impl(const void* send_buf,
                                     const ccl::vector_class<size_t>& send_counts,
                                     void* recv_buf,
                                     const ccl::vector_class<size_t>& recv_counts,
                                     ccl::datatype dtype,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::alltoallv_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    RCCL_COMM_STUB_IMPL(alltoallv);
}

/* broadcast */
ccl::event rccl_comm::broadcast_impl(void* buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     int root,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::broadcast_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    RCCL_COMM_STUB_IMPL(broadcast);
}

ccl::event rccl_comm::broadcast_impl(void* send_buf,
                                     void* recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     int root,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::broadcast_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    RCCL_COMM_STUB_IMPL(broadcast);
}

/* reduce */
ccl::event rccl_comm::reduce_impl(const void* send_buf,
                                  void* recv_buf,
                                  size_t count,
                                  ccl::datatype dtype,
                                  ccl::reduction reduction,
                                  int root,
                                  const ccl::stream::impl_value_t& stream,
                                  const ccl::reduce_attr& attr,
                                  const ccl::vector_class<ccl::event>& deps) {
    RCCL_COMM_STUB_IMPL(reduce);
}

/* reduce_scatter */
ccl::event rccl_comm::reduce_scatter_impl(const void* send_buf,
                                          void* recv_buf,
                                          size_t recv_count,
                                          ccl::datatype dtype,
                                          ccl::reduction reduction,
                                          const ccl::stream::impl_value_t& stream,
                                          const ccl::reduce_scatter_attr& attr,
                                          const ccl::vector_class<ccl::event>& deps) {
    RCCL_COMM_STUB_IMPL(reduce_scatter);
}

/* recv */
ccl::event rccl_comm::recv_impl(void* recv_buf,
                                size_t recv_count,
                                ccl::datatype dtype,
                                int peer,
                                const ccl::stream::impl_value_t& stream,
                                const ccl::pt2pt_attr& attr,
                                const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("RCCL COMM: recv count=", recv_count, " dtype=", static_cast<int>(dtype),
              " peer=", peer);

    ncclDataType_t rccl_dtype = get_rccl_datatype(dtype);
    hipStream_t hip_stream = get_hip_stream(stream);

    ncclResult_t status =
        rcclRecv(recv_buf, recv_count, rccl_dtype, peer, rccl_comm_handle, hip_stream);
    CCL_THROW_IF_NOT(status == ncclSuccess,
                     "rcclRecv failed: ", rcclGetErrorString(status));

    return std::unique_ptr<ccl::event_impl>(new ccl::stub_event_impl());
}

/* send */
ccl::event rccl_comm::send_impl(void* send_buf,
                                size_t send_count,
                                ccl::datatype dtype,
                                int peer,
                                const ccl::stream::impl_value_t& stream,
                                const ccl::pt2pt_attr& attr,
                                const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("RCCL COMM: send count=", send_count, " dtype=", static_cast<int>(dtype),
              " peer=", peer);

    ncclDataType_t rccl_dtype = get_rccl_datatype(dtype);
    hipStream_t hip_stream = get_hip_stream(stream);

    ncclResult_t status =
        rcclSend(send_buf, send_count, rccl_dtype, peer, rccl_comm_handle, hip_stream);
    CCL_THROW_IF_NOT(status == ncclSuccess,
                     "rcclSend failed: ", rcclGetErrorString(status));

    return std::unique_ptr<ccl::event_impl>(new ccl::stub_event_impl());
}

} // namespace ccl

#endif // CCL_ENABLE_RCCL

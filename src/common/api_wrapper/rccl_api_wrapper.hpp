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

#include "oneapi/ccl/config.h"

#if defined(CCL_ENABLE_RCCL)

// HIP headers require platform definition - we're targeting AMD GPUs
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include <rccl/rccl.h>
#include <string>
#include <vector>

namespace ccl {

// RCCL version check for AllToAll support (similar to NCCL 2.28.0)
// RCCL typically follows NCCL versioning
#if defined(NCCL_VERSION_CODE) && defined(NCCL_VERSION)
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
#define CCL_RCCL_ALLTOALL_SUPPORTED 1
#else
#define CCL_RCCL_ALLTOALL_SUPPORTED 0
#endif
#else
#define CCL_RCCL_ALLTOALL_SUPPORTED 0
#endif

typedef struct rccl_lib_ops {
    decltype(ncclGetVersion)* rcclGetVersion_ptr;
    decltype(ncclGetErrorString)* rcclGetErrorString_ptr;
    decltype(ncclGetUniqueId)* rcclGetUniqueId_ptr;
    decltype(ncclCommInitRank)* rcclCommInitRank_ptr;
    decltype(ncclCommDestroy)* rcclCommDestroy_ptr;
    decltype(ncclCommAbort)* rcclCommAbort_ptr;
    decltype(ncclGroupStart)* rcclGroupStart_ptr;
    decltype(ncclGroupEnd)* rcclGroupEnd_ptr;
    decltype(ncclAllReduce)* rcclAllReduce_ptr;
    decltype(ncclSend)* rcclSend_ptr;
    decltype(ncclRecv)* rcclRecv_ptr;
#if CCL_RCCL_ALLTOALL_SUPPORTED
    decltype(ncclAlltoAll)* rcclAlltoAll_ptr;
#endif
} rccl_lib_ops_t;

static std::vector<std::string> rccl_fn_names = {
    "ncclGetVersion",
    "ncclGetErrorString",
    "ncclGetUniqueId",
    "ncclCommInitRank",
    "ncclCommDestroy",
    "ncclCommAbort",
    "ncclGroupStart",
    "ncclGroupEnd",
    "ncclAllReduce",
    "ncclSend",
    "ncclRecv",
};

extern ccl::rccl_lib_ops_t rccl_lib_ops;

// RCCL uses the same API as NCCL, so we map to the same function names
// but through our rccl_lib_ops structure
#define rcclGetVersion     ccl::rccl_lib_ops.rcclGetVersion_ptr
#define rcclGetErrorString ccl::rccl_lib_ops.rcclGetErrorString_ptr
#define rcclGetUniqueId    ccl::rccl_lib_ops.rcclGetUniqueId_ptr
#define rcclCommInitRank   ccl::rccl_lib_ops.rcclCommInitRank_ptr
#define rcclCommDestroy    ccl::rccl_lib_ops.rcclCommDestroy_ptr
#define rcclCommAbort      ccl::rccl_lib_ops.rcclCommAbort_ptr
#define rcclGroupStart     ccl::rccl_lib_ops.rcclGroupStart_ptr
#define rcclGroupEnd       ccl::rccl_lib_ops.rcclGroupEnd_ptr
#define rcclAllReduce      ccl::rccl_lib_ops.rcclAllReduce_ptr
#define rcclSend           ccl::rccl_lib_ops.rcclSend_ptr
#define rcclRecv           ccl::rccl_lib_ops.rcclRecv_ptr

#if CCL_RCCL_ALLTOALL_SUPPORTED
decltype(ncclAlltoAll)* rcclGetAllToAll();
#endif

bool rccl_api_init();
void rccl_api_fini();
std::string get_rccl_lib_path();

} //namespace ccl

#endif // CCL_ENABLE_RCCL

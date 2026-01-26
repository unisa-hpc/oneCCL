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
#pragma once

#include "oneapi/ccl/config.h"

#if defined(CCL_ENABLE_NCCL)

#include <nccl.h>
#include <string>
#include <vector>

namespace ccl {

typedef struct nccl_lib_ops {
    decltype(ncclGetVersion)* ncclGetVersion_ptr;
    decltype(ncclGetErrorString)* ncclGetErrorString_ptr;
    decltype(ncclGetUniqueId)* ncclGetUniqueId_ptr;
    decltype(ncclCommInitRank)* ncclCommInitRank_ptr;
    decltype(ncclCommDestroy)* ncclCommDestroy_ptr;
    decltype(ncclCommAbort)* ncclCommAbort_ptr;
    decltype(ncclGroupStart)* ncclGroupStart_ptr;
    decltype(ncclGroupEnd)* ncclGroupEnd_ptr;
    decltype(ncclAllReduce)* ncclAllReduce_ptr;
} nccl_lib_ops_t;

static std::vector<std::string> nccl_fn_names = {
    "ncclGetVersion",
    "ncclGetErrorString",
    "ncclGetUniqueId",
    "ncclCommInitRank",
    "ncclCommDestroy",
    "ncclCommAbort",
    "ncclGroupStart",
    "ncclGroupEnd",
    "ncclAllReduce",
};

extern ccl::nccl_lib_ops_t nccl_lib_ops;

#define ncclGetVersion     ccl::nccl_lib_ops.ncclGetVersion_ptr
#define ncclGetErrorString ccl::nccl_lib_ops.ncclGetErrorString_ptr
#define ncclGetUniqueId    ccl::nccl_lib_ops.ncclGetUniqueId_ptr
#define ncclCommInitRank   ccl::nccl_lib_ops.ncclCommInitRank_ptr
#define ncclCommDestroy    ccl::nccl_lib_ops.ncclCommDestroy_ptr
#define ncclCommAbort      ccl::nccl_lib_ops.ncclCommAbort_ptr
#define ncclGroupStart     ccl::nccl_lib_ops.ncclGroupStart_ptr
#define ncclGroupEnd       ccl::nccl_lib_ops.ncclGroupEnd_ptr
#define ncclAllReduce      ccl::nccl_lib_ops.ncclAllReduce_ptr

bool nccl_api_init();
void nccl_api_fini();
std::string get_nccl_lib_path();

} //namespace ccl

#endif // CCL_ENABLE_NCCL

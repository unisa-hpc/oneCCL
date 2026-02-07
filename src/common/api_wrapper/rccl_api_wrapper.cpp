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
#include <sys/stat.h>
#include <dlfcn.h>

// HIP headers require platform definition - we're targeting AMD GPUs
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include "common/api_wrapper/api_wrapper.hpp"
#include "common/api_wrapper/rccl_api_wrapper.hpp"
#include "common/log/log.hpp"

#if defined(CCL_ENABLE_RCCL)

namespace ccl {

lib_info_t rccl_lib_info;
rccl_lib_ops_t rccl_lib_ops;

static bool path_exists(const std::string& path) {
    struct stat buffer {};
    return (stat(path.c_str(), &buffer) == 0);
}

std::string get_rccl_lib_path() {
    // lib_path specifies the name and full path to the RCCL library -
    // it should be an absolute and validated path pointing to the
    // desired librccl library

    // the order of searching for librccl is:
    // * CCL_RCCL_LIBRARY_PATH (rccl_lib_path env)
    // * RCCL_ROOT or RCCL_HOME (/lib or /lib64)
    // * ROCM_PATH (/lib or /lib64)
    // * LD_LIBRARY_PATH

    auto rccl_lib_path = ccl::global_data::env().rccl_lib_path;
    if (!rccl_lib_path.empty()) {
        LOG_DEBUG("RCCL lib path (CCL_RCCL_LIBRARY_PATH): ", rccl_lib_path);
        return rccl_lib_path;
    }

    const char* rccl_root = getenv("RCCL_ROOT");
    if (!rccl_root) {
        rccl_root = getenv("RCCL_HOME");
    }

    if (rccl_root) {
        std::string root(rccl_root);
        std::string lib_path = root + "/lib/librccl.so";
        if (path_exists(lib_path)) {
            LOG_DEBUG("RCCL lib path (RCCL_ROOT/RCCL_HOME): ", lib_path);
            return lib_path;
        }
        std::string lib64_path = root + "/lib64/librccl.so";
        if (path_exists(lib64_path)) {
            LOG_DEBUG("RCCL lib path (RCCL_ROOT/RCCL_HOME): ", lib64_path);
            return lib64_path;
        }
    }

    // Try ROCM_PATH
    const char* rocm_path = getenv("ROCM_PATH");
    if (!rocm_path) {
        rocm_path = getenv("ROCM_ROOT");
    }
    if (!rocm_path) {
        // Default ROCm installation path
        rocm_path = "/opt/rocm";
    }

    if (rocm_path) {
        std::string root(rocm_path);
        std::string lib_path = root + "/lib/librccl.so";
        if (path_exists(lib_path)) {
            LOG_DEBUG("RCCL lib path (ROCM_PATH): ", lib_path);
            return lib_path;
        }
        std::string lib64_path = root + "/lib64/librccl.so";
        if (path_exists(lib64_path)) {
            LOG_DEBUG("RCCL lib path (ROCM_PATH): ", lib64_path);
            return lib64_path;
        }
    }

    rccl_lib_path = "librccl.so";
    LOG_DEBUG("RCCL lib path (LD_LIBRARY_PATH): ", rccl_lib_path);
    return rccl_lib_path;
}

bool rccl_api_init() {
    rccl_lib_info.ops = &rccl_lib_ops;
    rccl_lib_info.fn_names = rccl_fn_names;
    rccl_lib_info.path = get_rccl_lib_path();

    int error = load_library(rccl_lib_info);
    if (error != CCL_LOAD_LB_SUCCESS) {
        print_error(error, rccl_lib_info);
        return false;
    }

    int version = 0;
    auto status = rcclGetVersion(&version);
    if (status == ncclSuccess) {
        LOG_INFO("RCCL version: ", version);
    }
    else {
        LOG_WARN("RCCL version query failed: ", rcclGetErrorString(status));
    }

#if CCL_RCCL_ALLTOALL_SUPPORTED
    if (status == ncclSuccess && version >= NCCL_VERSION(2, 28, 0)) {
        dlerror();
        void* symbol = dlsym(rccl_lib_info.handle, "ncclAlltoAll");
        if (symbol) {
            rccl_lib_ops.rcclAlltoAll_ptr = reinterpret_cast<decltype(ncclAlltoAll)*>(symbol);
        }
        else {
            LOG_DEBUG("RCCL symbol ncclAlltoAll not found: ", dlerror());
        }
    }
#endif

    return true;
}

void rccl_api_fini() {
    LOG_DEBUG("close RCCL lib: handle: ", rccl_lib_info.handle);
    close_library(rccl_lib_info);
}

#if CCL_RCCL_ALLTOALL_SUPPORTED
decltype(ncclAlltoAll)* rcclGetAllToAll() {
    return rccl_lib_ops.rcclAlltoAll_ptr;
}
#endif

} //namespace ccl

#endif //CCL_ENABLE_RCCL

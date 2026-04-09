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
#include <sys/stat.h>
#include <dlfcn.h>

#include "common/api_wrapper/api_wrapper.hpp"
#include "common/api_wrapper/nccl_api_wrapper.hpp"
#include "common/log/log.hpp"

#if defined(CCL_ENABLE_NCCL)

namespace ccl {

lib_info_t nccl_lib_info;
nccl_lib_ops_t nccl_lib_ops;

static bool path_exists(const std::string& path) {
    struct stat buffer {};
    return (stat(path.c_str(), &buffer) == 0);
}

std::string get_nccl_lib_path() {
    // lib_path specifies the name and full path to the NCCL library -
    // it should be an absolute and validated path pointing to the
    // desired libnccl library

    // the order of searching for libnccl is:
    // * CCL_NCCL_LIBRARY_PATH (nccl_lib_path env)
    // * NCCL_ROOT or NCCL_HOME (/lib or /lib64)
    // * LD_LIBRARY_PATH

    auto nccl_lib_path = ccl::global_data::env().nccl_lib_path;
    if (!nccl_lib_path.empty()) {
        LOG_DEBUG("NCCL lib path (CCL_NCCL_LIBRARY_PATH): ", nccl_lib_path);
        return nccl_lib_path;
    }

    const char* nccl_root = getenv("NCCL_ROOT");
    if (!nccl_root) {
        nccl_root = getenv("NCCL_HOME");
    }

    if (nccl_root) {
        std::string root(nccl_root);
        std::string lib_path = root + "/lib/libnccl.so";
        if (path_exists(lib_path)) {
            LOG_DEBUG("NCCL lib path (NCCL_ROOT/NCCL_HOME): ", lib_path);
            return lib_path;
        }
        std::string lib64_path = root + "/lib64/libnccl.so";
        if (path_exists(lib64_path)) {
            LOG_DEBUG("NCCL lib path (NCCL_ROOT/NCCL_HOME): ", lib64_path);
            return lib64_path;
        }
    }

    nccl_lib_path = "libnccl.so";
    LOG_DEBUG("NCCL lib path (LD_LIBRARY_PATH): ", nccl_lib_path);
    return nccl_lib_path;
}

bool nccl_api_init() {
    nccl_lib_info.ops = &nccl_lib_ops;
    nccl_lib_info.fn_names = nccl_fn_names;
    nccl_lib_info.path = get_nccl_lib_path();

    int error = load_library(nccl_lib_info);
    if (error != CCL_LOAD_LB_SUCCESS) {
        print_error(error, nccl_lib_info);
        return false;
    }

    int version = 0;
    auto status = ncclGetVersion(&version);
    if (status == ncclSuccess) {
        LOG_INFO("NCCL version: ", version);
    }
    else {
        LOG_WARN("NCCL version query failed: ", ncclGetErrorString(status));
    }

#if CCL_NCCL_ALLTOALL_SUPPORTED
    if (status == ncclSuccess && version >= NCCL_VERSION(2, 28, 0)) {
        dlerror();
        void* symbol = dlsym(nccl_lib_info.handle, "ncclAlltoAll");
        if (symbol) {
            nccl_lib_ops.ncclAlltoAll_ptr = reinterpret_cast<decltype(ncclAlltoAll)*>(symbol);
        }
        else {
            LOG_DEBUG("NCCL symbol ncclAlltoAll not found: ", dlerror());
        }
    }
#endif

    return true;
}

void nccl_api_fini() {
    LOG_DEBUG("close NCCL lib: handle: ", nccl_lib_info.handle);
    close_library(nccl_lib_info);
}

#if CCL_NCCL_ALLTOALL_SUPPORTED
decltype(ncclAlltoAll)* ncclGetAllToAll() {
    return nccl_lib_ops.ncclAlltoAll_ptr;
}
#endif

} //namespace ccl

#endif //CCL_ENABLE_NCCL

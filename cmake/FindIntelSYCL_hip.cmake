#===============================================================================
# Copyright 2019 Intel Corporation
# Copyright 2026 Contributors - AMD ROCm/RCCL backend support
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# FindIntelSYCL_hip.cmake
# This module finds SYCL with HIP/AMD backend support for use with DPC++
# compiled with AMD GPU support (e.g., from intel/llvm with HIP backend)

file(GLOB sycl_headers "lib/clang/*/include")

list(APPEND dpcpp_root_hints
            ${DPCPP_ROOT}
            $ENV{DPCPP_ROOT})
set(original_cmake_prefix_path ${CMAKE_PREFIX_PATH})
if(dpcpp_root_hints)
    list(INSERT CMAKE_PREFIX_PATH 0 ${dpcpp_root_hints})
else()
    message("DPCPP_ROOT prefix path hint is not defined")
endif()

message("Searching for SYCL with HIP/AMD backend")

include(CheckCXXCompilerFlag)
include(FindPackageHandleStandardArgs)

# Check if the compiler supports -fsycl
unset(INTEL_SYCL_SUPPORTED CACHE)
check_cxx_compiler_flag("-fsycl" INTEL_SYCL_SUPPORTED)

get_filename_component(INTEL_SYCL_BINARY_DIR ${CMAKE_CXX_COMPILER} PATH)

# Try to find Intel SYCL version.hpp header
find_path(INTEL_SYCL_INCLUDE_DIRS
    NAMES CL/sycl/version.hpp sycl/version.hpp
    PATHS
      ${sycl_root_hints}
      "${INTEL_SYCL_BINARY_DIR}/.."
      "${INTEL_SYCL_BINARY_DIR}/../opt/compiler"
    PATH_SUFFIXES
        include
        include/sycl
        "${sycl_headers}"
    NO_DEFAULT_PATH)

find_library(INTEL_SYCL_LIBRARIES
    NAMES "sycl"
    PATHS
        ${sycl_root_hints}
        "${INTEL_SYCL_BINARY_DIR}/.."
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH)

# Find ROCm/HIP installation
# ROCm typically installs to /opt/rocm or a user-specified ROCM_PATH
set(ROCM_PATH_HINTS
    ${ROCM_PATH}
    $ENV{ROCM_PATH}
    ${ROCM_ROOT}
    $ENV{ROCM_ROOT}
    ${HIP_PATH}
    $ENV{HIP_PATH}
    /opt/rocm
    /opt/rocm/hip
)

find_path(HIP_INCLUDE_DIRS
    NAMES hip/hip_runtime.h
    PATHS ${ROCM_PATH_HINTS}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)

# Also try default paths
if(NOT HIP_INCLUDE_DIRS)
    find_path(HIP_INCLUDE_DIRS NAMES hip/hip_runtime.h)
endif()

find_library(HIP_RUNTIME_LIBRARY
    NAMES amdhip64 hip_hcc
    PATHS ${ROCM_PATH_HINTS}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH)

# Also try default paths
if(NOT HIP_RUNTIME_LIBRARY)
    find_library(HIP_RUNTIME_LIBRARY NAMES amdhip64 hip_hcc)
endif()

if(HIP_INCLUDE_DIRS AND HIP_RUNTIME_LIBRARY)
    set(HIP_FOUND TRUE)
    set(HIP_LIBRARIES ${HIP_RUNTIME_LIBRARY})
    message(STATUS "HIP found:")
    message(STATUS "  HIP include dir: ${HIP_INCLUDE_DIRS}")
    message(STATUS "  HIP runtime library: ${HIP_RUNTIME_LIBRARY}")
else()
    set(HIP_FOUND FALSE)
    message(WARNING "HIP toolkit not found. AMD GPU support may be limited.")
endif()

# Find RCCL library (ROCm Collective Communications Library)
find_path(RCCL_INCLUDE_DIRS
    NAMES rccl/rccl.h rccl.h
    PATHS
        ${RCCL_ROOT}
        $ENV{RCCL_ROOT}
        ${RCCL_HOME}
        $ENV{RCCL_HOME}
        ${ROCM_PATH_HINTS}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)

# Also try default paths
if(NOT RCCL_INCLUDE_DIRS)
    find_path(RCCL_INCLUDE_DIRS NAMES rccl/rccl.h rccl.h)
endif()

find_library(RCCL_LIBRARIES
    NAMES rccl
    PATHS
        ${RCCL_ROOT}
        $ENV{RCCL_ROOT}
        ${RCCL_HOME}
        $ENV{RCCL_HOME}
        ${ROCM_PATH_HINTS}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH)

# Also try default paths
if(NOT RCCL_LIBRARIES)
    find_library(RCCL_LIBRARIES NAMES rccl)
endif()

if(RCCL_INCLUDE_DIRS AND RCCL_LIBRARIES)
    set(RCCL_FOUND TRUE CACHE BOOL "RCCL library found" FORCE)
    message(STATUS "RCCL found:")
    message(STATUS "  RCCL include dir: ${RCCL_INCLUDE_DIRS}")
    message(STATUS "  RCCL library: ${RCCL_LIBRARIES}")
else()
    set(RCCL_FOUND FALSE CACHE BOOL "RCCL library found" FORCE)
    message(WARNING "RCCL library not found. Set RCCL_ROOT or ROCM_PATH environment variable.")
endif()

find_package_handle_standard_args(IntelSYCL_hip
    FOUND_VAR IntelSYCL_hip_FOUND
    REQUIRED_VARS
        INTEL_SYCL_LIBRARIES
        INTEL_SYCL_INCLUDE_DIRS
        INTEL_SYCL_SUPPORTED)

if(IntelSYCL_hip_FOUND AND NOT TARGET Intel::SYCL_hip)
    add_library(Intel::SYCL_hip UNKNOWN IMPORTED)

    # Collect all include directories
    set(SYCL_HIP_INCLUDE_DIRS "${INTEL_SYCL_INCLUDE_DIRS}")
    if(HIP_FOUND AND HIP_INCLUDE_DIRS)
        list(APPEND SYCL_HIP_INCLUDE_DIRS "${HIP_INCLUDE_DIRS}")
    endif()
    if(RCCL_FOUND AND RCCL_INCLUDE_DIRS)
        list(APPEND SYCL_HIP_INCLUDE_DIRS "${RCCL_INCLUDE_DIRS}")
    endif()

    message(STATUS "IntelSYCL_hip_FOUND: TRUE")
    message(STATUS "SYCL_HIP_INCLUDE_DIRS: ${SYCL_HIP_INCLUDE_DIRS}")

    # AMD GPU architecture selection
    # Common architectures:
    #   gfx906  - Radeon VII
    #   gfx908  - MI100
    #   gfx90a  - MI200/MI210/MI250
    #   gfx942  - MI300
    #   gfx1030 - RX 6800/6900 (RDNA 2)
    #   gfx1100 - RX 7900 (RDNA 3)
    if(NOT DEFINED CCL_AMD_GPU_ARCH)
        set(CCL_AMD_GPU_ARCH "gfx90a" CACHE STRING "AMD GPU architecture for SYCL offloading")
    endif()
    message(STATUS "AMD GPU architecture: ${CCL_AMD_GPU_ARCH}")

    # Set the SYCL flags for AMD/HIP backend
    # -fsycl enables SYCL mode
    # -fsycl-targets=amdgcn-amd-amdhsa targets AMD GPUs
    # -D__HIP_PLATFORM_AMD__ is required by HIP headers to select AMD platform
    # -Xsycl-target-backend --offload-arch specifies the GPU architecture
    set(INTEL_SYCL_HIP_FLAGS "-fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${CCL_AMD_GPU_ARCH} -D__HIP_PLATFORM_AMD__")

    # Build import libraries list
    set(imp_libs
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${INTEL_SYCL_HIP_FLAGS}>
        ${COMPUTE_BACKEND_NAME})

    # Add HIP and RCCL libraries if found
    if(HIP_FOUND)
        list(APPEND imp_libs ${HIP_LIBRARIES})
    endif()

    if(RCCL_FOUND)
        list(APPEND imp_libs ${RCCL_LIBRARIES})
    endif()

    set_target_properties(Intel::SYCL_hip PROPERTIES
        INTERFACE_LINK_LIBRARIES "${imp_libs}"
        INTERFACE_INCLUDE_DIRECTORIES "${SYCL_HIP_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${INTEL_SYCL_LIBRARIES}")

    set(INTEL_SYCL_FLAGS "${INTEL_SYCL_HIP_FLAGS}")

    mark_as_advanced(
        INTEL_SYCL_FLAGS
        INTEL_SYCL_LIBRARIES
        INTEL_SYCL_INCLUDE_DIRS
        HIP_INCLUDE_DIRS
        HIP_LIBRARIES
        RCCL_INCLUDE_DIRS
        RCCL_LIBRARIES)
endif()

cmake_minimum_required(VERSION 4.4.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
project(diff-solver VERSION 1.0.0 LANGUAGES CXX CUDA)

option(DIFF_SOLVER_BUILD_APP "Build the optional Differentiable Solver executable" OFF)

# ============================================================================
# C++ Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)


# ============================================================================
# CUDA Configuration
# ============================================================================
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (MSVC)
    add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif ()


# ============================================================================
# Dependencies
# ============================================================================
find_package(CUDAToolkit 13.2 REQUIRED)

function(download_if_needed url out_path)
    if (EXISTS "${out_path}")
        message(STATUS "Using cached file: ${out_path}")
        return()
    endif ()

    get_filename_component(out_dir "${out_path}" DIRECTORY)
    file(MAKE_DIRECTORY "${out_dir}")

    message(STATUS "Downloading ${url}")
    file(DOWNLOAD
            "${url}"
            "${out_path}"
            TLS_VERIFY ON
            STATUS dl_status
    )

    list(GET dl_status 0 dl_code)
    list(GET dl_status 1 dl_msg)
    if (NOT dl_code EQUAL 0)
        message(FATAL_ERROR "Failed to download ${url}: [${dl_code}] ${dl_msg}")
    endif ()
endfunction()


# ============================================================================
# Optional App
# ============================================================================
if (DIFF_SOLVER_BUILD_APP)
    set(DIFF_SOLVER_UTIL_GIT_TAG "bb261075e7c8a4722b23f42c3be01aa24024731a")
    set(DIFF_SOLVER_UTIL_VENDOR_DIR "${CMAKE_CURRENT_BINARY_DIR}/vendor/util/${KEYFRAME_UTIL_GIT_TAG}")
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/util/${DIFF_SOLVER_UTIL_GIT_TAG}/xcli.ixx"
            "${DIFF_SOLVER_UTIL_VENDOR_DIR}/xcli.ixx"
    )
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/util/${DIFF_SOLVER_UTIL_GIT_TAG}/xcli.cpp"
            "${DIFF_SOLVER_UTIL_VENDOR_DIR}/xcli.cpp"
    )

    add_library(xayah-util-xcli)
    target_sources(
            xayah-util-xcli
            PRIVATE
            "${DIFF_SOLVER_UTIL_VENDOR_DIR}/xcli.cpp"
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            "${DIFF_SOLVER_UTIL_VENDOR_DIR}/xcli.ixx"
    )
    set_property(TARGET xayah-util-xcli PROPERTY CXX_MODULE_STD ON)

    add_executable(diff-solver main.cpp)
    target_link_libraries(diff-solver PRIVATE xayah-diffsim-projectile xayah-util-xcli)
    if (MSVC)
        target_link_options(diff-solver PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()
    set_property(TARGET diff-solver PROPERTY CXX_MODULE_STD ON)
endif ()

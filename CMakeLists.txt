cmake_minimum_required(VERSION 4.4.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
project(diff-solver VERSION 1.0.0 LANGUAGES CXX CUDA)

option(DIFF_SOLVER_BUILD_EXAMPLES "Build the examples" OFF)
option(DIFF_SOLVER_BUILD_SPECTRA_PROVIDERS "Build the Spectra Providers" OFF)

# ============================================================================
# Project Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_CXX_MODULE_STD ON)

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
find_package(CUDAToolkit 13.3 REQUIRED)


# ============================================================================
# Solver Modules
# ============================================================================
add_subdirectory(solvers)


# ============================================================================
# Spectra Providers
# ============================================================================
if (DIFF_SOLVER_BUILD_EXAMPLES)
    if (DIFF_SOLVER_BUILD_SPECTRA_PROVIDERS)
        find_package(SpectraSDK 1 CONFIG REQUIRED)
    endif ()
    add_subdirectory(examples)
endif ()

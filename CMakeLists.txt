cmake_minimum_required(VERSION 4.4.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
project(diff-solver VERSION 1.0.0 LANGUAGES CXX CUDA)

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
find_package(CUDAToolkit 13.2 EXACT REQUIRED)


# ============================================================================
# Solver Modules
# ============================================================================
add_library(xayah-solver)
target_sources(
        xayah-solver
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        solver.ixx
)
set_property(TARGET xayah-solver PROPERTY CXX_MODULE_STD ON)


# ============================================================================
# Spring Chain Modules
# ============================================================================
add_library(xayah-spring-chain)
target_sources(
        xayah-spring-chain
        PRIVATE
        spring_chain/data.cpp
        spring_chain/operators.cpp
        spring_chain/model.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        spring_chain/data.ixx
        spring_chain/operators.ixx
        spring_chain/model.ixx
)
set_property(TARGET xayah-spring-chain PROPERTY CXX_MODULE_STD ON)


# ============================================================================
# Control Optimization Application
# ============================================================================
add_executable(diff-solver main.cpp)
target_link_libraries(diff-solver PRIVATE xayah-solver xayah-spring-chain)
set_property(TARGET diff-solver PROPERTY CXX_MODULE_STD ON)
if (MSVC)
    target_link_options(diff-solver PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
endif ()

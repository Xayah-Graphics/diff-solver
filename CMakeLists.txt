cmake_minimum_required(VERSION 4.4.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
project(diff-solver VERSION 1.0.0 LANGUAGES CXX CUDA)

option(DIFF_SOLVER_BUILD_PROJECT_PLUGIN "Build the Spectra Cloth project plugin" ON)

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
# Cloth CUDA Kernels and Modules
# ============================================================================
add_library(xayah-cloth-cuda-kernels STATIC cloth/kernels.cu)
target_include_directories(xayah-cloth-cuda-kernels PRIVATE cloth)
target_link_libraries(xayah-cloth-cuda-kernels PRIVATE CUDA::cudart)

add_library(xayah-cloth)
target_sources(
        xayah-cloth
        PRIVATE
        cloth/runtime.cpp
        cloth/operators.cpp
        cloth/model.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        cloth/runtime.ixx
        cloth/data.ixx
        cloth/operators.ixx
        cloth/model.ixx
)
target_link_libraries(xayah-cloth PUBLIC xayah-solver PRIVATE xayah-cloth-cuda-kernels CUDA::cudart)
set_property(TARGET xayah-cloth PROPERTY CXX_MODULE_STD ON)


# ============================================================================
# Cloth Gradient Verification and Hanging Demo
# ============================================================================
add_executable(diff-cloth cloth/main.cpp)
target_link_libraries(diff-cloth PRIVATE xayah-solver xayah-cloth CUDA::cudart)
set_property(TARGET diff-cloth PROPERTY CXX_MODULE_STD ON)
if (MSVC)
    target_link_options(diff-cloth PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
endif ()


# ============================================================================
# Spectra Cloth Project Plugin
# ============================================================================
if (DIFF_SOLVER_BUILD_PROJECT_PLUGIN)
    add_library(xayah-cloth-visualization-cuda STATIC cloth/cloth.visualization.cu)
    target_include_directories(xayah-cloth-visualization-cuda PRIVATE cloth)
    target_link_libraries(xayah-cloth-visualization-cuda PRIVATE CUDA::cudart)

    add_library(cloth_project_plugin SHARED cloth/cloth.project.cpp)
    target_sources(
            cloth_project_plugin
            PRIVATE
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            cloth/cloth.plugin.ixx
            cloth/cloth.project.ixx
    )
    target_link_libraries(cloth_project_plugin PRIVATE xayah-cloth xayah-cloth-visualization-cuda CUDA::cudart)
    set_property(TARGET cloth_project_plugin PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(cloth_project_plugin PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()
endif ()

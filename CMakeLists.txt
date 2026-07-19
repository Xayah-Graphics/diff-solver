cmake_minimum_required(VERSION 4.4.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
project(diff-solver VERSION 1.0.0 LANGUAGES CXX CUDA)

option(DIFF_SOLVER_BUILD_EXAMPLES "Build the examples" OFF)
option(DIFF_SOLVER_BUILD_SPECTRA_PLUGINS "Build the Spectra example plugins" OFF)

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

add_library(xayah-cuda cuda.cpp)
target_sources(
        xayah-cuda
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        cuda.ixx
)
target_link_libraries(xayah-cuda PUBLIC CUDA::cudart)
set_property(TARGET xayah-cuda PROPERTY CXX_MODULE_STD ON)


# ============================================================================
# Cloth Solver
# ============================================================================
add_library(xayah-cloth)
target_sources(
        xayah-cloth
        PRIVATE
        solvers/cloth/operators.cpp
        solvers/cloth/operators.cu
        solvers/cloth/operators.h
        solvers/cloth/model.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        solvers/cloth/data.ixx
        solvers/cloth/operators.ixx
        solvers/cloth/model.ixx
)
target_include_directories(xayah-cloth PRIVATE solvers/cloth)
target_link_libraries(xayah-cloth PUBLIC xayah-solver xayah-cuda PRIVATE CUDA::cudart)
set_property(TARGET xayah-cloth PROPERTY CXX_MODULE_STD ON)


# ============================================================================
# Smoke Solver
# ============================================================================
add_library(xayah-smoke)
target_sources(
        xayah-smoke
        PRIVATE
        solvers/smoke/operators.cpp
        solvers/smoke/operators.cu
        solvers/smoke/operators.h
        solvers/smoke/model.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        solvers/smoke/data.ixx
        solvers/smoke/operators.ixx
        solvers/smoke/model.ixx
)
target_include_directories(xayah-smoke PRIVATE solvers/smoke)
target_link_libraries(xayah-smoke PUBLIC xayah-solver xayah-cuda PRIVATE CUDA::cudart)
set_property(TARGET xayah-smoke PROPERTY CXX_MODULE_STD ON)

if (DIFF_SOLVER_BUILD_EXAMPLES)
    # Cloth Stretch-Stiffness Inverse Example
    add_executable(
            diff-cloth-stretch-stiffness-inverse
            examples/cloth/stretch_stiffness_inverse/main.cpp
            examples/cloth/stretch_stiffness_inverse/task.cpp
            examples/cloth/stretch_stiffness_inverse/task.cu
    )
    target_sources(
            diff-cloth-stretch-stiffness-inverse
            PRIVATE
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/cloth/stretch_stiffness_inverse/task.ixx
    )
    target_include_directories(diff-cloth-stretch-stiffness-inverse PRIVATE examples/cloth/stretch_stiffness_inverse)
    target_link_libraries(diff-cloth-stretch-stiffness-inverse PRIVATE xayah-cloth CUDA::cudart)
    set_property(TARGET diff-cloth-stretch-stiffness-inverse PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(diff-cloth-stretch-stiffness-inverse PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()

    # Cloth Wind-Trajectory Optimization Example
    add_executable(
            diff-cloth-wind-trajectory-optimization
            examples/cloth/wind_trajectory_optimization/main.cpp
            examples/cloth/wind_trajectory_optimization/task.cpp
            examples/cloth/wind_trajectory_optimization/task.cu
    )
    target_sources(
            diff-cloth-wind-trajectory-optimization
            PRIVATE
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/cloth/wind_trajectory_optimization/task.ixx
    )
    target_include_directories(diff-cloth-wind-trajectory-optimization PRIVATE examples/cloth/wind_trajectory_optimization)
    target_link_libraries(diff-cloth-wind-trajectory-optimization PRIVATE xayah-cloth CUDA::cudart)
    set_property(TARGET diff-cloth-wind-trajectory-optimization PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(diff-cloth-wind-trajectory-optimization PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()

    # Cloth Forward Example
    add_executable(
            diff-cloth-forward
            examples/cloth/forward/main.cpp
            examples/cloth/forward/simulation.cpp
            examples/cloth/forward/simulation.cu
    )
    target_sources(
            diff-cloth-forward
            PRIVATE
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/cloth/forward/simulation.ixx
    )
    target_include_directories(diff-cloth-forward PRIVATE examples/cloth/forward)
    target_link_libraries(diff-cloth-forward PRIVATE xayah-cloth CUDA::cudart)
    set_property(TARGET diff-cloth-forward PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(diff-cloth-forward PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()

    # Smoke Wind-Trajectory Optimization Example
    add_executable(
            diff-smoke-wind-trajectory-optimization
            examples/smoke/wind_trajectory_optimization/main.cpp
            examples/smoke/wind_trajectory_optimization/task.cpp
            examples/smoke/wind_trajectory_optimization/task.cu
    )
    target_sources(
            diff-smoke-wind-trajectory-optimization
            PRIVATE
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/smoke/wind_trajectory_optimization/task.ixx
    )
    target_include_directories(diff-smoke-wind-trajectory-optimization PRIVATE examples/smoke/wind_trajectory_optimization)
    target_link_libraries(diff-smoke-wind-trajectory-optimization PRIVATE xayah-smoke CUDA::cudart)
    set_property(TARGET diff-smoke-wind-trajectory-optimization PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(diff-smoke-wind-trajectory-optimization PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()

    # Smoke Forward Example
    add_executable(
            diff-smoke-forward
            examples/smoke/forward/main.cpp
            examples/smoke/forward/simulation.cpp
            examples/smoke/forward/simulation.cu
    )
    target_sources(
            diff-smoke-forward
            PRIVATE
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/smoke/forward/simulation.ixx
    )
    target_include_directories(diff-smoke-forward PRIVATE examples/smoke/forward)
    target_link_libraries(diff-smoke-forward PRIVATE xayah-smoke CUDA::cudart)
    set_property(TARGET diff-smoke-forward PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(diff-smoke-forward PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()
endif ()


# ============================================================================
# Spectra Plugin Adapter
# ============================================================================
if (DIFF_SOLVER_BUILD_SPECTRA_PLUGINS)
    add_library(xayah-spectra-plugin)
    target_sources(
            xayah-spectra-plugin
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            plugin.ixx
    )
    set_property(TARGET xayah-spectra-plugin PROPERTY CXX_MODULE_STD ON)

    # Cloth Stretch-Stiffness Inverse Example Plugin
    add_library(
            cloth-stretch-stiffness-inverse-plugin SHARED
            examples/cloth/stretch_stiffness_inverse/project.cpp
            examples/cloth/stretch_stiffness_inverse/project.cu
            examples/cloth/stretch_stiffness_inverse/task.cpp
            examples/cloth/stretch_stiffness_inverse/task.cu
    )
    target_sources(
            cloth-stretch-stiffness-inverse-plugin
            PRIVATE
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/cloth/stretch_stiffness_inverse/task.ixx
    )
    target_include_directories(cloth-stretch-stiffness-inverse-plugin PRIVATE examples/cloth/stretch_stiffness_inverse)
    target_link_libraries(cloth-stretch-stiffness-inverse-plugin PRIVATE xayah-spectra-plugin xayah-cloth CUDA::cudart)
    set_property(TARGET cloth-stretch-stiffness-inverse-plugin PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(cloth-stretch-stiffness-inverse-plugin PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()

    # Cloth Wind-Trajectory Optimization Example Plugin
    add_library(
            cloth-wind-trajectory-optimization-plugin SHARED
            examples/cloth/wind_trajectory_optimization/project.cpp
            examples/cloth/wind_trajectory_optimization/project.cu
            examples/cloth/wind_trajectory_optimization/task.cpp
            examples/cloth/wind_trajectory_optimization/task.cu
    )
    target_sources(
            cloth-wind-trajectory-optimization-plugin
            PRIVATE
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/cloth/wind_trajectory_optimization/task.ixx
    )
    target_include_directories(cloth-wind-trajectory-optimization-plugin PRIVATE examples/cloth/wind_trajectory_optimization)
    target_link_libraries(cloth-wind-trajectory-optimization-plugin PRIVATE xayah-spectra-plugin xayah-cloth CUDA::cudart)
    set_property(TARGET cloth-wind-trajectory-optimization-plugin PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(cloth-wind-trajectory-optimization-plugin PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()

    # Cloth Forward Example Plugin
    add_library(
            cloth-forward-plugin SHARED
            examples/cloth/forward/project.cpp
            examples/cloth/forward/project.cu
            examples/cloth/forward/simulation.cpp
            examples/cloth/forward/simulation.cu
    )
    target_sources(
            cloth-forward-plugin
            PRIVATE
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/cloth/forward/simulation.ixx
    )
    target_include_directories(cloth-forward-plugin PRIVATE examples/cloth/forward)
    target_link_libraries(cloth-forward-plugin PRIVATE xayah-spectra-plugin xayah-cloth CUDA::cudart)
    set_property(TARGET cloth-forward-plugin PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(cloth-forward-plugin PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()

    # Smoke Wind-Trajectory Optimization Example Plugin
    add_library(
            smoke-wind-trajectory-optimization-plugin SHARED
            examples/smoke/wind_trajectory_optimization/project.cpp
            examples/smoke/wind_trajectory_optimization/project.cu
            examples/smoke/wind_trajectory_optimization/task.cpp
            examples/smoke/wind_trajectory_optimization/task.cu
    )
    target_sources(
            smoke-wind-trajectory-optimization-plugin
            PRIVATE
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/smoke/wind_trajectory_optimization/task.ixx
    )
    target_include_directories(smoke-wind-trajectory-optimization-plugin PRIVATE examples/smoke/wind_trajectory_optimization)
    target_link_libraries(smoke-wind-trajectory-optimization-plugin PRIVATE xayah-spectra-plugin xayah-smoke CUDA::cudart)
    set_property(TARGET smoke-wind-trajectory-optimization-plugin PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(smoke-wind-trajectory-optimization-plugin PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()

    # Smoke Forward Example Plugin
    add_library(
            smoke-forward-plugin SHARED
            examples/smoke/forward/project.cpp
            examples/smoke/forward/simulation.cpp
            examples/smoke/forward/simulation.cu
    )
    target_sources(
            smoke-forward-plugin
            PRIVATE
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            examples/smoke/forward/simulation.ixx
    )
    target_include_directories(smoke-forward-plugin PRIVATE examples/smoke/forward)
    target_link_libraries(smoke-forward-plugin PRIVATE xayah-spectra-plugin xayah-smoke CUDA::cudart)
    set_property(TARGET smoke-forward-plugin PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(smoke-forward-plugin PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()
endif ()

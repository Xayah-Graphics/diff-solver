cmake_minimum_required(VERSION 4.4.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
project(diff-solver VERSION 1.0.0 LANGUAGES CXX CUDA)

option(DIFF_SOLVER_BUILD_SPECTRA_PLUGINS "Build the Spectra example plugins" ON)

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
# Cloth CUDA Kernels and Modules
# ============================================================================
add_library(xayah-cloth-cuda-kernels STATIC cloth/kernels.cu)
target_include_directories(xayah-cloth-cuda-kernels PRIVATE cloth)
target_link_libraries(xayah-cloth-cuda-kernels PRIVATE CUDA::cudart)

add_library(xayah-cloth)
target_sources(
        xayah-cloth
        PRIVATE
        cloth/operators.cpp
        cloth/model.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        cloth/data.ixx
        cloth/operators.ixx
        cloth/model.ixx
)
target_link_libraries(xayah-cloth PUBLIC xayah-solver xayah-cuda PRIVATE xayah-cloth-cuda-kernels CUDA::cudart)
set_property(TARGET xayah-cloth PROPERTY CXX_MODULE_STD ON)


# ============================================================================
# Smoke CUDA Kernels and Modules
# ============================================================================
add_library(xayah-smoke-cuda-kernels STATIC smoke/kernels.cu)
target_include_directories(xayah-smoke-cuda-kernels PRIVATE smoke)
target_link_libraries(xayah-smoke-cuda-kernels PRIVATE CUDA::cudart)

add_library(xayah-smoke)
target_sources(
        xayah-smoke
        PRIVATE
        smoke/operators.cpp
        smoke/model.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        smoke/data.ixx
        smoke/operators.ixx
        smoke/model.ixx
)
target_link_libraries(xayah-smoke PUBLIC xayah-solver xayah-cuda PRIVATE xayah-smoke-cuda-kernels CUDA::cudart)
set_property(TARGET xayah-smoke PROPERTY CXX_MODULE_STD ON)

add_executable(diff-smoke smoke/main.cpp)
target_link_libraries(diff-smoke PRIVATE xayah-smoke)
set_property(TARGET diff-smoke PROPERTY CXX_MODULE_STD ON)
if (MSVC)
    target_link_options(diff-smoke PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
endif ()


# ============================================================================
# Cloth Stretch-Stiffness Inverse Example
# ============================================================================
set(STRETCH_STIFFNESS_INVERSE_DIR examples/cloth/stretch_stiffness_inverse)

add_library(xayah-example-cloth-stretch-stiffness-inverse-cuda STATIC ${STRETCH_STIFFNESS_INVERSE_DIR}/task.cu)
target_include_directories(xayah-example-cloth-stretch-stiffness-inverse-cuda PRIVATE ${STRETCH_STIFFNESS_INVERSE_DIR})
target_link_libraries(xayah-example-cloth-stretch-stiffness-inverse-cuda PRIVATE CUDA::cudart)

add_library(xayah-example-cloth-stretch-stiffness-inverse)
target_sources(
        xayah-example-cloth-stretch-stiffness-inverse
        PRIVATE
        ${STRETCH_STIFFNESS_INVERSE_DIR}/task.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        ${STRETCH_STIFFNESS_INVERSE_DIR}/task.ixx
)
target_link_libraries(xayah-example-cloth-stretch-stiffness-inverse PUBLIC xayah-cloth PRIVATE xayah-example-cloth-stretch-stiffness-inverse-cuda CUDA::cudart)
set_property(TARGET xayah-example-cloth-stretch-stiffness-inverse PROPERTY CXX_MODULE_STD ON)

add_executable(diff-cloth-stretch-stiffness-inverse ${STRETCH_STIFFNESS_INVERSE_DIR}/main.cpp)
target_link_libraries(diff-cloth-stretch-stiffness-inverse PRIVATE xayah-example-cloth-stretch-stiffness-inverse)
set_property(TARGET diff-cloth-stretch-stiffness-inverse PROPERTY CXX_MODULE_STD ON)
if (MSVC)
    target_link_options(diff-cloth-stretch-stiffness-inverse PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
endif ()


# ============================================================================
# Cloth Wind-Trajectory Optimization Example
# ============================================================================
set(WIND_TRAJECTORY_OPTIMIZATION_DIR examples/cloth/wind_trajectory_optimization)

add_library(xayah-example-cloth-wind-trajectory-optimization-cuda STATIC ${WIND_TRAJECTORY_OPTIMIZATION_DIR}/task.cu)
target_include_directories(xayah-example-cloth-wind-trajectory-optimization-cuda PRIVATE ${WIND_TRAJECTORY_OPTIMIZATION_DIR})
target_link_libraries(xayah-example-cloth-wind-trajectory-optimization-cuda PRIVATE CUDA::cudart)

add_library(xayah-example-cloth-wind-trajectory-optimization)
target_sources(
        xayah-example-cloth-wind-trajectory-optimization
        PRIVATE
        ${WIND_TRAJECTORY_OPTIMIZATION_DIR}/task.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        ${WIND_TRAJECTORY_OPTIMIZATION_DIR}/task.ixx
)
target_link_libraries(xayah-example-cloth-wind-trajectory-optimization PUBLIC xayah-cloth PRIVATE xayah-example-cloth-wind-trajectory-optimization-cuda CUDA::cudart)
set_property(TARGET xayah-example-cloth-wind-trajectory-optimization PROPERTY CXX_MODULE_STD ON)

add_executable(diff-cloth-wind-trajectory-optimization ${WIND_TRAJECTORY_OPTIMIZATION_DIR}/main.cpp)
target_link_libraries(diff-cloth-wind-trajectory-optimization PRIVATE xayah-example-cloth-wind-trajectory-optimization)
set_property(TARGET diff-cloth-wind-trajectory-optimization PROPERTY CXX_MODULE_STD ON)
if (MSVC)
    target_link_options(diff-cloth-wind-trajectory-optimization PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
endif ()


# ============================================================================
# Smoke Wind-Trajectory Optimization Example
# ============================================================================
set(SMOKE_WIND_TRAJECTORY_OPTIMIZATION_DIR examples/smoke/wind_trajectory_optimization)

add_library(xayah-example-smoke-wind-trajectory-optimization-cuda STATIC ${SMOKE_WIND_TRAJECTORY_OPTIMIZATION_DIR}/task.cu)
target_include_directories(xayah-example-smoke-wind-trajectory-optimization-cuda PRIVATE ${SMOKE_WIND_TRAJECTORY_OPTIMIZATION_DIR})
target_link_libraries(xayah-example-smoke-wind-trajectory-optimization-cuda PRIVATE CUDA::cudart)

add_library(xayah-example-smoke-wind-trajectory-optimization)
target_sources(
        xayah-example-smoke-wind-trajectory-optimization
        PRIVATE
        ${SMOKE_WIND_TRAJECTORY_OPTIMIZATION_DIR}/task.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        ${SMOKE_WIND_TRAJECTORY_OPTIMIZATION_DIR}/task.ixx
)
target_link_libraries(xayah-example-smoke-wind-trajectory-optimization PUBLIC xayah-smoke PRIVATE xayah-example-smoke-wind-trajectory-optimization-cuda CUDA::cudart)
set_property(TARGET xayah-example-smoke-wind-trajectory-optimization PROPERTY CXX_MODULE_STD ON)

add_executable(diff-smoke-wind-trajectory-optimization ${SMOKE_WIND_TRAJECTORY_OPTIMIZATION_DIR}/main.cpp)
target_link_libraries(diff-smoke-wind-trajectory-optimization PRIVATE xayah-example-smoke-wind-trajectory-optimization)
set_property(TARGET diff-smoke-wind-trajectory-optimization PROPERTY CXX_MODULE_STD ON)
if (MSVC)
    target_link_options(diff-smoke-wind-trajectory-optimization PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
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
    add_library(xayah-example-cloth-stretch-stiffness-inverse-spectra-cuda STATIC ${STRETCH_STIFFNESS_INVERSE_DIR}/project.cu)
    target_include_directories(xayah-example-cloth-stretch-stiffness-inverse-spectra-cuda PRIVATE ${STRETCH_STIFFNESS_INVERSE_DIR})
    target_link_libraries(xayah-example-cloth-stretch-stiffness-inverse-spectra-cuda PRIVATE CUDA::cudart)

    add_library(cloth-stretch-stiffness-inverse-plugin SHARED ${STRETCH_STIFFNESS_INVERSE_DIR}/project.cpp)
    target_link_libraries(cloth-stretch-stiffness-inverse-plugin PRIVATE xayah-spectra-plugin xayah-example-cloth-stretch-stiffness-inverse xayah-example-cloth-stretch-stiffness-inverse-spectra-cuda CUDA::cudart)
    set_property(TARGET cloth-stretch-stiffness-inverse-plugin PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(cloth-stretch-stiffness-inverse-plugin PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()

    # Cloth Wind-Trajectory Optimization Example Plugin
    add_library(xayah-example-cloth-wind-trajectory-optimization-spectra-cuda STATIC ${WIND_TRAJECTORY_OPTIMIZATION_DIR}/project.cu)
    target_include_directories(xayah-example-cloth-wind-trajectory-optimization-spectra-cuda PRIVATE ${WIND_TRAJECTORY_OPTIMIZATION_DIR})
    target_link_libraries(xayah-example-cloth-wind-trajectory-optimization-spectra-cuda PRIVATE CUDA::cudart)

    add_library(cloth-wind-trajectory-optimization-plugin SHARED ${WIND_TRAJECTORY_OPTIMIZATION_DIR}/project.cpp)
    target_link_libraries(cloth-wind-trajectory-optimization-plugin PRIVATE xayah-spectra-plugin xayah-example-cloth-wind-trajectory-optimization xayah-example-cloth-wind-trajectory-optimization-spectra-cuda CUDA::cudart)
    set_property(TARGET cloth-wind-trajectory-optimization-plugin PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(cloth-wind-trajectory-optimization-plugin PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()

    # Smoke Wind-Trajectory Optimization Example Plugin
    add_library(xayah-example-smoke-wind-trajectory-optimization-spectra-cuda STATIC ${SMOKE_WIND_TRAJECTORY_OPTIMIZATION_DIR}/project.cu)
    target_include_directories(xayah-example-smoke-wind-trajectory-optimization-spectra-cuda PRIVATE ${SMOKE_WIND_TRAJECTORY_OPTIMIZATION_DIR})
    target_link_libraries(xayah-example-smoke-wind-trajectory-optimization-spectra-cuda PRIVATE CUDA::cudart)

    add_library(smoke-wind-trajectory-optimization-plugin SHARED ${SMOKE_WIND_TRAJECTORY_OPTIMIZATION_DIR}/project.cpp)
    target_link_libraries(smoke-wind-trajectory-optimization-plugin PRIVATE xayah-spectra-plugin xayah-example-smoke-wind-trajectory-optimization xayah-example-smoke-wind-trajectory-optimization-spectra-cuda CUDA::cudart)
    set_property(TARGET smoke-wind-trajectory-optimization-plugin PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(smoke-wind-trajectory-optimization-plugin PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()
endif ()

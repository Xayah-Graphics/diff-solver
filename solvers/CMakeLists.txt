add_library(xayah-solver)
target_sources(
        xayah-solver
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        solver.ixx
)

add_library(xayah-cuda cuda.cpp)
target_sources(
        xayah-cuda
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        cuda.ixx
)
target_link_libraries(xayah-cuda PRIVATE CUDA::cudart)

if (MSVC)
    set(DIFF_SOLVER_CXX_STD_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib" PARENT_SCOPE)
endif ()

add_subdirectory(cloth)
add_subdirectory(smoke)
add_subdirectory(fluid)

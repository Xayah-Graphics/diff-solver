if (NOT TARGET xayah-cuda)
    message(FATAL_ERROR "xayah-smoke requires the xayah-cuda target")
endif ()

add_library(xayah-smoke)
target_sources(
        xayah-smoke
        PUBLIC
        FILE_SET cuda_headers TYPE HEADERS
        BASE_DIRS "${PROJECT_SOURCE_DIR}/solvers"
        FILES cuda_types.h
        PRIVATE
        operators.cpp
        operators.cu
        operators.h
        model.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        data.ixx
        operators.ixx
        model.ixx
)
target_include_directories(xayah-smoke PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(xayah-smoke PUBLIC xayah-cuda PRIVATE CUDA::cudart)

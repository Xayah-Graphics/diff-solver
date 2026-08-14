if (NOT TARGET xayah-cuda)
    message(FATAL_ERROR "xayah-cloth requires the xayah-cuda target")
endif ()

add_library(xayah-cloth)
target_sources(
        xayah-cloth
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
target_include_directories(xayah-cloth PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(xayah-cloth PUBLIC xayah-cuda PRIVATE CUDA::cudart)

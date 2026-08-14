if (NOT TARGET xayah-cuda)
    message(FATAL_ERROR "xayah-fluid requires the xayah-cuda target")
endif ()

add_library(xayah-fluid)
target_sources(
        xayah-fluid
        PRIVATE
        cuda_common.h
        sph.cpp
        sph.cu
        sph.h
        grid.cpp
        grid.cu
        grid.h
        pbf.cpp
        pbf.cu
        pbf.h
        pcisph.cpp
        pcisph.cu
        pcisph.h
        wcsph.cpp
        wcsph.cu
        wcsph.h
        iisph.cpp
        iisph.cu
        iisph.h
        dfsph.cpp
        dfsph.cu
        dfsph.h
        flip.cpp
        apic.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        data.ixx
        sph.ixx
        grid.ixx
        pbf.ixx
        pcisph.ixx
        wcsph.ixx
        iisph.ixx
        dfsph.ixx
        flip.ixx
        apic.ixx
)
target_include_directories(xayah-fluid PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(xayah-fluid PUBLIC xayah-cuda PRIVATE CUDA::cudart)

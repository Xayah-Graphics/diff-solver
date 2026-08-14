if (NOT TARGET xayah-cloth)
    message(FATAL_ERROR "cloth/forward requires the xayah-cloth target")
endif ()

add_library(xayah-cloth-forward)
target_sources(
        xayah-cloth-forward
        PRIVATE
        simulation.cpp
        simulation.cu
        simulation.h
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        simulation.ixx
)
target_include_directories(xayah-cloth-forward PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(xayah-cloth-forward PUBLIC xayah-cloth PRIVATE CUDA::cudart)

if (DIFF_SOLVER_BUILD_EXAMPLES)
    add_executable(diff-cloth-forward main.cpp)
    target_link_libraries(diff-cloth-forward PRIVATE xayah-cloth-forward)
    if (MSVC)
        target_link_options(diff-cloth-forward PRIVATE "/WHOLEARCHIVE:${DIFF_SOLVER_CXX_STD_ARCHIVE}")
    endif ()
endif ()

if (DIFF_SOLVER_BUILD_SPECTRA_PROVIDERS)
    if (NOT TARGET SpectraSDK::sdk)
        message(FATAL_ERROR "cloth/forward Provider requires the SpectraSDK::sdk target")
    endif ()

    spectra_add_provider(
            cloth-forward-provider
            MODULE xayah.examples.cloth.forward.provider
            TYPE xayah::cloth::examples::forward::visualization::Provider
    )
    target_sources(
            cloth-forward-provider
            PRIVATE
            project.cu
            project.h
            PRIVATE
            FILE_SET provider_module TYPE CXX_MODULES FILES
            project.cpp
    )
    target_include_directories(cloth-forward-provider PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(cloth-forward-provider PRIVATE xayah-cloth-forward)

    set(bundle_directory "${PROJECT_BINARY_DIR}/spectra/cloth-forward")
    set_target_properties(
            cloth-forward-provider
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${bundle_directory}"
            RUNTIME_OUTPUT_DIRECTORY "${bundle_directory}"
    )
    add_custom_target(
            cloth-forward-bundle
            COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${CMAKE_CURRENT_SOURCE_DIR}/spectra" "${bundle_directory}"
            SOURCES
            spectra/cloth-forward.spectra
            spectra/assets/flag.ply
            VERBATIM
    )
    add_dependencies(cloth-forward-provider cloth-forward-bundle)
    if (MSVC)
        target_link_options(cloth-forward-provider PRIVATE "/WHOLEARCHIVE:${DIFF_SOLVER_CXX_STD_ARCHIVE}")
    endif ()
endif ()

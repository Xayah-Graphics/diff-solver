if (NOT TARGET xayah-smoke)
    message(FATAL_ERROR "smoke/forward requires the xayah-smoke target")
endif ()

add_library(xayah-smoke-forward)
target_sources(
        xayah-smoke-forward
        PRIVATE
        simulation.cpp
        simulation.cu
        simulation.h
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        simulation.ixx
)
target_include_directories(xayah-smoke-forward PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(xayah-smoke-forward PUBLIC xayah-smoke PRIVATE CUDA::cudart)

if (DIFF_SOLVER_BUILD_EXAMPLES)
    add_executable(diff-smoke-forward main.cpp)
    target_link_libraries(diff-smoke-forward PRIVATE xayah-smoke-forward CUDA::cudart)
    if (MSVC)
        target_link_options(diff-smoke-forward PRIVATE "/WHOLEARCHIVE:${DIFF_SOLVER_CXX_STD_ARCHIVE}")
    endif ()
endif ()

if (DIFF_SOLVER_BUILD_SPECTRA_PROVIDERS)
    if (NOT TARGET SpectraSDK::sdk)
        message(FATAL_ERROR "smoke/forward Provider requires the SpectraSDK::sdk target")
    endif ()

    spectra_add_provider(
            smoke-forward-provider
            MODULE xayah.examples.smoke.forward.provider
            TYPE xayah::smoke::examples::forward::visualization::Provider
    )
    target_sources(
            smoke-forward-provider
            PRIVATE
            project.cu
            project.h
            PRIVATE
            FILE_SET provider_module TYPE CXX_MODULES FILES
            project.cpp
    )
    target_include_directories(smoke-forward-provider PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(smoke-forward-provider PRIVATE xayah-smoke-forward)

    set(bundle_directory "${PROJECT_BINARY_DIR}/spectra/smoke-forward")
    set_target_properties(
            smoke-forward-provider
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${bundle_directory}"
            RUNTIME_OUTPUT_DIRECTORY "${bundle_directory}"
    )
    add_custom_target(
            smoke-forward-bundle
            COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${CMAKE_CURRENT_SOURCE_DIR}/spectra" "${bundle_directory}"
            SOURCES
            spectra/smoke-forward.spectra
            VERBATIM
    )
    add_dependencies(smoke-forward-provider smoke-forward-bundle)
    if (MSVC)
        target_link_options(smoke-forward-provider PRIVATE "/WHOLEARCHIVE:${DIFF_SOLVER_CXX_STD_ARCHIVE}")
    endif ()
endif ()

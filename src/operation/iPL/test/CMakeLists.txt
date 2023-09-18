# aux_source_directory(. SourceFiles)
# add_executable(iPLTest ${iPL_SOURCE}/PlacerDB.cc ${SourceFiles})
# set(CMAKE_BUILD_TYPE "Debug")

set(SANITIZER OFF)
if(SANITIZER)
    ADD_COMPILE_OPTIONS("-fsanitize=address")
    LINK_LIBRARIES("-fsanitize=address")
endif()

set(PERFTOOL OFF)
if(PERFTOOL)
    link_libraries(profiler)
endif()
add_executable(iPLTest
    ${iPL_SOURCE}/PlacerDB.cc
    ${iPL_API}/PLAPI.cc
    ${iPL_TEST}/APITest.cc
    # ${iPL_TEST}/ReportCongTest.cc
    ${iPL_TEST}/DCTTest.cc
    # ${iPL_TEST}/ComputationCheck.cc
    # ${iPL_TEST}/GlogTest.cc
    # ${iPL_TEST}/CongEvalAPITest.cc
    # ${iPL_TEST}/GridManagerTest.cc
    # ${iPL_TEST}/MPTest.cc
)
set(OPENMP ON)
if(OPENMP)
    # Find OpenMP package
    find_package(OpenMP REQUIRED)
    # Add compiler flags for OpenMP support
    if(OpenMP_CXX_FOUND)
        target_compile_options(iPLTest PUBLIC ${OpenMP_CXX_FLAGS})
    endif()
endif()

target_link_libraries(iPLTest
    PUBLIC
    ipl-api_external_libs
    ipl-test_external_libs
    ipl-api
    ipl-source_external_libs
    ipl-source

    tool_api_icts
    icts_api
    icts_router
    icts_slew_aware
    icts_evaluator
    ito_api

    tool_api_ipl
    tool_manager
    file_manager_placement
)

add_executable(MacroPlacer
    ${iPL_TEST}/MP.cc)

target_link_libraries(MacroPlacer
    PUBLIC
    ipl-api
    ipl-source
    ipl-api_external_libs
)

add_executable(MPLseTest
    ${iPL_TEST}/MPLseTest.cc)


target_link_libraries(MPLseTest
    PUBLIC
    ipl-api
    ipl-source
    ipl-api_external_libs
)
add_executable(ployclip
    ${iPL_TEST}/ploygonclip.cc)
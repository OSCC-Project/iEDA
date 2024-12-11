include(${HOME_CMAKE}/operation/idb.cmake)
include(${HOME_CMAKE}/operation/iplf.cmake)

include(cmake/external_lib.cmake)
include(cmake/idrc_api_external_lib.cmake)

add_subdirectory(api)
add_subdirectory(source)
# add_subdirectory(test)


# ADD_EXECUTABLE(run_drc ${IDRC_TESTS}/run_drc.cpp)
# TARGET_LINK_LIBRARIES(run_drc DRC)

# ADD_EXECUTABLE(conflict_graph ${IDRC_TESTS}/conflict_graph.cpp)
# TARGET_LINK_LIBRARIES(conflict_graph DRC)

# add_executable(test_boost ${IDRC_TESTS}/test_boost.cpp)
# target_link_libraries(test_boost DRC)
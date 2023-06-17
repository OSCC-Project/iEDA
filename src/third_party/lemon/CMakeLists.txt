include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/config.h)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lemon.pc.in
               ${CMAKE_CURRENT_BINARY_DIR}/lemon.pc @ONLY)

set(LEMON_SOURCES
    arg_parser.cc
    base.cc
    color.cc
    lp_base.cc
    lp_skeleton.cc
    random.cc
    bits/windows.cc)

add_library(lemon ${LEMON_SOURCES})

target_link_libraries(lemon ${GLPK_LIBRARIES} ${COIN_LIBRARIES}
                      ${ILOG_LIBRARIES} ${SOPLEX_LIBRARIES})

if(UNIX)
  set_target_properties(
    lemon
    PROPERTIES OUTPUT_NAME emon
               VERSION ${LEMON_VERSION}
               SOVERSION ${LEMON_VERSION})
endif()

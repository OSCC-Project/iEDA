find_package(Eigen3 REQUIRED)

add_library(ircx_compare_spef
  compare/Comparator.cc
  compare/CouplingCapComparator.cc
  config/CompareSpefConfig.cc
  compare/NetSelector.cc
  compare/PathPairGenerator.cc
  report/ReportWriter.cc
  compare/ResistanceSolver.cc
  compare/ResultSorter.cc
  reader/SpefReader.cc
)

target_link_libraries(ircx_compare_spef
  PUBLIC
    ircx_headers
  PRIVATE
    log
    ircx_spef
    fort
    OpenMP::OpenMP_CXX
)

target_include_directories(ircx_compare_spef
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(ircx_compare_spef SYSTEM
  PRIVATE
    ${EIGEN3_INCLUDE_DIR}
)

add_library(ircx_tool_compare_spef
  CompareSpefTool.cc
)

target_link_libraries(ircx_tool_compare_spef
  PUBLIC
    ircx_compare_spef
    ircx_headers
  PRIVATE
    log
)

target_include_directories(ircx_tool_compare_spef
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

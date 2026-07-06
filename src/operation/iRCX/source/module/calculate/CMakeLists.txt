set(IRCX_CALCULATE_SRC
  resistance/ResistanceCalc.cc
  resistance/ViaResistanceModel.cc
  resistance/WireResistanceModel.cc
  capacitance/CapacitanceCalc.cc
  capacitance/CapTableQuery.cc
  capacitance/EdgeCapAccumulator.cc
)

add_library(ircx_calculate ${IRCX_CALCULATE_SRC})

target_link_libraries(ircx_calculate
  PUBLIC
    ircx_cap_table
    ircx_database_design
    ircx_headers
    OpenMP::OpenMP_CXX
  PRIVATE
    ircx_topology
    log
)

target_include_directories(ircx_calculate
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

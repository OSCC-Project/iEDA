set(ICTS_DATABASE_ADAPTER_FAST_STA_CLOCK_STATE ${ICTS_DATABASE_ADAPTER_FAST_STA}/clock_state)
set(ICTS_DATABASE_ADAPTER_FAST_STA_CLOCK_TREE ${ICTS_DATABASE_ADAPTER_FAST_STA}/clock_tree)
set(ICTS_DATABASE_ADAPTER_FAST_STA_LIBERTY ${ICTS_DATABASE_ADAPTER_FAST_STA}/liberty)
set(ICTS_DATABASE_ADAPTER_FAST_STA_CLOCK_NET_PARASITIC ${ICTS_DATABASE_ADAPTER_FAST_STA}/clock_net_parasitic)
set(ICTS_DATABASE_ADAPTER_FAST_STA_TIMING ${ICTS_DATABASE_ADAPTER_FAST_STA}/timing)
set(ICTS_DATABASE_ADAPTER_FAST_STA_POWER ${ICTS_DATABASE_ADAPTER_FAST_STA}/power)
set(ICTS_DATABASE_ADAPTER_FAST_STA_CLOCK_SIZING ${ICTS_DATABASE_ADAPTER_FAST_STA}/clock_sizing)
set(ICTS_DATABASE_ADAPTER_FAST_STA_SEGMENT_CHAR ${ICTS_DATABASE_ADAPTER_FAST_STA}/segment_char)
set(ICTS_DATABASE_ADAPTER_FAST_STA_REPORT ${ICTS_DATABASE_ADAPTER_FAST_STA}/report)

set(
  ICTS_DATABASE_ADAPTER_FAST_STA_INTERNAL_INCLUDE_DIRS
  ${ICTS_DATABASE_ADAPTER_FAST_STA}
  ${ICTS_DATABASE_ADAPTER_FAST_STA_CLOCK_STATE}
  ${ICTS_DATABASE_ADAPTER_FAST_STA_CLOCK_TREE}
  ${ICTS_DATABASE_ADAPTER_FAST_STA_LIBERTY}
  ${ICTS_DATABASE_ADAPTER_FAST_STA_CLOCK_NET_PARASITIC}
  ${ICTS_DATABASE_ADAPTER_FAST_STA_TIMING}
  ${ICTS_DATABASE_ADAPTER_FAST_STA_POWER}
  ${ICTS_DATABASE_ADAPTER_FAST_STA_CLOCK_SIZING}
  ${ICTS_DATABASE_ADAPTER_FAST_STA_SEGMENT_CHAR}
  ${ICTS_DATABASE_ADAPTER_FAST_STA_REPORT}
)

add_library(
  icts_source_database_adapter_fast_sta_liberty
  ${ICTS_DATABASE_ADAPTER_FAST_STA_LIBERTY}/FastStaLiberty.cc
)

add_library(
  icts_source_database_adapter_fast_sta_liberty_model
  ${ICTS_DATABASE_ADAPTER_FAST_STA_LIBERTY}/FastStaLibertyModel.cc
)

target_link_libraries(
  icts_source_database_adapter_fast_sta_liberty
  PRIVATE
  icts_source_database_io
  icts_source_utils_logger
  absl::inlined_vector
  liberty
)

target_include_directories(
  icts_source_database_adapter_fast_sta_liberty
  PRIVATE
  ${ICTS_DATABASE_ADAPTER_FAST_STA_INTERNAL_INCLUDE_DIRS}
  ${ICTS_DATABASE}
  ${HOME_DATABASE}/manager/parser
  ${HOME_UTILITY}/stdBase/include
)

target_include_directories(
  icts_source_database_adapter_fast_sta_liberty_model
  PRIVATE
  ${ICTS_DATABASE_ADAPTER_FAST_STA_INTERNAL_INCLUDE_DIRS}
)

add_library(
  icts_source_database_adapter_fast_sta_clock_net_parasitic
  ${ICTS_DATABASE_ADAPTER_FAST_STA_CLOCK_NET_PARASITIC}/FastStaParasitics.cc
)

target_link_libraries(
  icts_source_database_adapter_fast_sta_clock_net_parasitic
  PRIVATE
  icts_source_database_config
  icts_source_database_design
  icts_source_database_io
  icts_source_database_routing
  log
)

target_include_directories(
  icts_source_database_adapter_fast_sta_clock_net_parasitic
  PRIVATE
  ${ICTS_DATABASE_ADAPTER_FAST_STA_INTERNAL_INCLUDE_DIRS}
  ${ICTS_DATABASE}
)

add_library(
  icts_source_database_adapter_fast_sta_clock_tree
  ${ICTS_DATABASE_ADAPTER_FAST_STA_CLOCK_TREE}/FastStaClockTree.cc
)

target_link_libraries(
  icts_source_database_adapter_fast_sta_clock_tree
  PRIVATE
  icts_source_database_adapter_fast_sta_clock_net_parasitic
  icts_source_database_design
  icts_source_database_routing
  log
)

target_include_directories(
  icts_source_database_adapter_fast_sta_clock_tree
  PRIVATE
  ${ICTS_DATABASE_ADAPTER_FAST_STA_INTERNAL_INCLUDE_DIRS}
  ${ICTS_DATABASE}
)

add_library(
  icts_source_database_adapter_fast_sta_timing
  ${ICTS_DATABASE_ADAPTER_FAST_STA_TIMING}/FastStaDmpCeff.cc
  ${ICTS_DATABASE_ADAPTER_FAST_STA_TIMING}/FastStaDmpCeffEquations.cc
  ${ICTS_DATABASE_ADAPTER_FAST_STA_TIMING}/FastStaDmpCeffShared.cc
  ${ICTS_DATABASE_ADAPTER_FAST_STA_TIMING}/FastStaDmpCeffSolver.cc
  ${ICTS_DATABASE_ADAPTER_FAST_STA_TIMING}/FastStaTiming.cc
)

target_link_libraries(
  icts_source_database_adapter_fast_sta_timing
  PRIVATE
  icts_source_database_adapter_fast_sta_clock_net_parasitic
  icts_source_database_adapter_fast_sta_liberty_model
  absl::fixed_array
)

target_include_directories(
  icts_source_database_adapter_fast_sta_timing
  PRIVATE
  ${ICTS_DATABASE_ADAPTER_FAST_STA_INTERNAL_INCLUDE_DIRS}
)

add_library(
  icts_source_database_adapter_fast_sta_power
  ${ICTS_DATABASE_ADAPTER_FAST_STA_POWER}/FastStaPower.cc
)

target_link_libraries(
  icts_source_database_adapter_fast_sta_power
  PRIVATE
  icts_source_database_adapter_fast_sta_liberty_model
)

target_include_directories(
  icts_source_database_adapter_fast_sta_power
  PRIVATE
  ${ICTS_DATABASE_ADAPTER_FAST_STA_INTERNAL_INCLUDE_DIRS}
)

add_library(
  icts_source_database_adapter_fast_sta_clock_sizing
  ${ICTS_DATABASE_ADAPTER_FAST_STA_CLOCK_SIZING}/FastStaIncremental.cc
)

target_link_libraries(
  icts_source_database_adapter_fast_sta_clock_sizing
  PRIVATE
  icts_source_database_adapter_fast_sta_liberty
  log
)

target_include_directories(
  icts_source_database_adapter_fast_sta_clock_sizing
  PRIVATE
  ${ICTS_DATABASE_ADAPTER_FAST_STA_INTERNAL_INCLUDE_DIRS}
)

add_library(
  icts_source_database_adapter_fast_sta_clock_state
  ${ICTS_DATABASE_ADAPTER_FAST_STA_CLOCK_STATE}/FastStaBuilder.cc
)

target_link_libraries(
  icts_source_database_adapter_fast_sta_clock_state
  PRIVATE
  icts_source_database_adapter_fast_sta_clock_tree
  icts_source_database_adapter_fast_sta_liberty
  icts_source_database_config
  icts_source_database_design
  icts_source_database_io
  icts_source_database_routing
  log
)

target_include_directories(
  icts_source_database_adapter_fast_sta_clock_state
  PRIVATE
  ${ICTS_DATABASE_ADAPTER_FAST_STA_INTERNAL_INCLUDE_DIRS}
  ${ICTS_DATABASE}
)

add_library(
  icts_source_database_adapter_fast_sta_segment_char
  ${ICTS_DATABASE_ADAPTER_FAST_STA_SEGMENT_CHAR}/FastStaChar.cc
)

target_link_libraries(
  icts_source_database_adapter_fast_sta_segment_char
  PRIVATE
  icts_source_database_adapter_fast_sta_liberty
  icts_source_database_adapter_fast_sta_power
  icts_source_database_adapter_fast_sta_timing
  icts_source_database_io
  log
)

target_include_directories(
  icts_source_database_adapter_fast_sta_segment_char
  PRIVATE
  ${ICTS_DATABASE_ADAPTER_FAST_STA_INTERNAL_INCLUDE_DIRS}
  ${ICTS_DATABASE}
)

add_library(
  icts_source_database_adapter_fast_sta_report
  ${ICTS_DATABASE_ADAPTER_FAST_STA_REPORT}/FastStaReport.cc
)

target_link_libraries(
  icts_source_database_adapter_fast_sta_report
  PRIVATE
  icts_source_database_adapter_fast_sta_clock_state
  log
)

target_include_directories(
  icts_source_database_adapter_fast_sta_report
  PRIVATE
  ${ICTS_DATABASE_ADAPTER_FAST_STA_INTERNAL_INCLUDE_DIRS}
)

add_library(icts_source_database_adapter_fast_sta ${ICTS_DATABASE_ADAPTER_FAST_STA}/FastSta.cc)

target_link_libraries(
  icts_source_database_adapter_fast_sta
  PRIVATE
  icts_source_database_adapter_fast_sta_clock_sizing
  icts_source_database_adapter_fast_sta_clock_state
  icts_source_database_adapter_fast_sta_segment_char
  icts_source_database_design
  log
)

target_include_directories(
  icts_source_database_adapter_fast_sta
  PUBLIC
  ${ICTS_DATABASE_ADAPTER_FAST_STA}
  PRIVATE
  ${ICTS_DATABASE_ADAPTER_FAST_STA_INTERNAL_INCLUDE_DIRS}
  ${ICTS_DATABASE}
)

if(DEBUG_ICTS_SOURCE_DATABASE_ADAPTER_FAST_STA)
  foreach(
    target_name
    icts_source_database_adapter_fast_sta
    icts_source_database_adapter_fast_sta_clock_net_parasitic
    icts_source_database_adapter_fast_sta_clock_sizing
    icts_source_database_adapter_fast_sta_clock_state
    icts_source_database_adapter_fast_sta_clock_tree
    icts_source_database_adapter_fast_sta_liberty
    icts_source_database_adapter_fast_sta_liberty_model
    icts_source_database_adapter_fast_sta_power
    icts_source_database_adapter_fast_sta_report
    icts_source_database_adapter_fast_sta_segment_char
    icts_source_database_adapter_fast_sta_timing
  )
    if(MSVC)
      target_compile_options(${target_name} PRIVATE /Zi /Od /Ob0)
      target_link_options(${target_name} PRIVATE /DEBUG)
    else()
      target_compile_options(${target_name} PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
    endif()
  endforeach()
endif()

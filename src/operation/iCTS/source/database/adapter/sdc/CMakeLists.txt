add_library(
  icts_source_database_adapter_sdc
  ${ICTS_SOURCE}/database/adapter/sdc/SdcClockReader.cc
  ${ICTS_SOURCE}/database/adapter/sdc/clock_parser/SdcClockParser.cc
  ${ICTS_SOURCE}/database/adapter/sdc/clock_parser/SdcClockCommands.cc
  ${ICTS_SOURCE}/database/adapter/sdc/clock_parser/SdcClockEvaluator.cc
  ${ICTS_SOURCE}/database/adapter/sdc/clock_parser/SdcClockValue.cc
  ${ICTS_SOURCE}/database/adapter/sdc/clock_trace/ClockTraceResolver.cc
  ${ICTS_SOURCE}/database/adapter/sdc/clock_trace/ClockTracePins.cc
  ${ICTS_SOURCE}/database/adapter/sdc/clock_trace/ClockTraceReport.cc
  ${ICTS_SOURCE}/database/adapter/sdc/clock_trace/ClockTraceResolve.cc
)

target_include_directories(
  icts_source_database_adapter_sdc
  PUBLIC
  ${ICTS_DATABASE_ADAPTER_SDC}
  ${ICTS_DATABASE_ADAPTER}
  ${ICTS_DATABASE}
  PRIVATE
  ${ICTS_DATABASE_ADAPTER_SDC}/clock_parser
  ${ICTS_DATABASE_ADAPTER_SDC}/clock_trace
  ${ICTS_UTILS}
  ${HOME_DATABASE}/data/design
  ${HOME_DATABASE}/data/design/db_design
  ${HOME_DATABASE}/data/design/db_layout
  ${HOME_DATABASE}/manager/builder
)

target_link_libraries(
  icts_source_database_adapter_sdc
  PRIVATE
  idm
  icts_source_database_config
  icts_source_utils_logger
  log
  liberty
)

if(DEBUG_ICTS_SOURCE_DATABASE_ADAPTER_SDC)
  if(MSVC)
    target_compile_options(icts_source_database_adapter_sdc PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_database_adapter_sdc PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_database_adapter_sdc PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()

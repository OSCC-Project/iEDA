if(TEST_SHELL)
  add_definitions(-DTEST_SHELL)
endif()

# ADD_COMPILE_OPTIONS("-fsanitize=address") LINK_LIBRARIES("-fsanitize=address")

add_library(ircx_headers INTERFACE)

target_compile_features(ircx_headers INTERFACE cxx_std_20)

target_include_directories(ircx_headers
  INTERFACE
    ${HOME_UTILITY}
    ${HOME_UTILITY}/log
    ${HOME_UTILITY}/stdBase
    ${HOME_UTILITY}/stdBase/include
    ${HOME_OPERATION}/iRCX
    ${HOME_OPERATION}/iRCX/source/database/config
    ${HOME_OPERATION}/iRCX/source/database/design
    ${HOME_OPERATION}/iRCX/source/database/io
    ${HOME_OPERATION}/iRCX/source/flow
    ${HOME_OPERATION}/iRCX/source/flow/extraction
    ${HOME_OPERATION}/iRCX/source/flow/report
    ${HOME_OPERATION}/iRCX/source/flow/setup
    ${HOME_OPERATION}/iRCX/source/module
    ${HOME_OPERATION}/iRCX/source/module/environment
    ${HOME_OPERATION}/iRCX/source/module/calculate
    ${HOME_OPERATION}/iRCX/source/module/calculate/capacitance
    ${HOME_OPERATION}/iRCX/source/module/calculate/resistance
    ${HOME_OPERATION}/iRCX/source/module/process
    ${HOME_OPERATION}/iRCX/source/module/report
    ${HOME_OPERATION}/iRCX/source/module/topology
    ${HOME_OPERATION}/iRCX/source/parser
    ${HOME_OPERATION}/iRCX/source/parser/cap_table
    ${HOME_OPERATION}/iRCX/source/parser/itf
    ${HOME_OPERATION}/iRCX/source/parser/itf_builder
    ${HOME_OPERATION}/iRCX/source/parser/itf_data
    ${HOME_OPERATION}/iRCX/source/parser/mapping
    ${HOME_OPERATION}/iRCX/source/utils
)

target_include_directories(ircx_headers
  SYSTEM INTERFACE
    ${HOME_THIRDPARTY}
)

# add_subdirectory(ics55) # disabled: vendor-specific install
add_subdirectory(source/parser)
add_subdirectory(source/database)
add_subdirectory(source/module/topology)
add_subdirectory(source/module/environment)
add_subdirectory(source/module/process)
add_subdirectory(source/module/calculate)
add_subdirectory(source/module/report)
add_subdirectory(source/tool/compare_spef)
add_subdirectory(source/tool/dump_net_shape)
add_subdirectory(source/tool/plot_spef)
add_subdirectory(source/flow/extraction)
add_subdirectory(source/flow/report)
add_subdirectory(source/flow)
add_subdirectory(spef_parser)
add_subdirectory(api)

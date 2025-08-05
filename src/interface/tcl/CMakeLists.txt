add_subdirectory(tcl_util)
add_subdirectory(tcl_config)
add_subdirectory(tcl_idb)
add_subdirectory(tcl_flow)
add_subdirectory(tcl_icts)
add_subdirectory(tcl_idrc)
add_subdirectory(tcl_instance)
add_subdirectory(tcl_irt)
add_subdirectory(tcl_ifp)
add_subdirectory(tcl_ipdn)
add_subdirectory(tcl_ipnp)
if(BUILD_GUI)
add_subdirectory(tcl_gui)
endif()
add_subdirectory(tcl_ipl)
add_subdirectory(tcl_ito)
add_subdirectory(tcl_ista)
add_subdirectory(tcl_ipw)
add_subdirectory(tcl_report)
add_subdirectory(tcl_ino)
add_subdirectory(tcl_feature)
add_subdirectory(tcl_eval)
add_subdirectory(tcl_eco)
add_subdirectory(tcl_vec)

if(CONTEST)
add_subdirectory(tcl_contest)
endif()

AUX_SOURCE_DIRECTORY(./ SRC)

add_library(ieda_tcl ${SRC})

target_link_libraries(ieda_tcl 
    PUBLIC 
        flow 
        tool_manager
        tcl 
        shell-cmd 
        tcl_config
        tcl_idb
        tcl_flow
        tcl_icts
        tcl_idrc
        tcl_irt 
        tcl_ifp 
        tcl_ipdn
        tcl_inst 
        tcl_ipl
        tcl_ito
        tcl_ista
        tcl_ipw
        tcl_report
        tcl_ino
        tcl_feature
        tcl_eval
        tcl_eco
        tcl_vec
        tcl_ipnp
)

target_include_directories(ieda_tcl 
    PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tcl_ipnp

    ${HOME_UTILITY}/tcl

    ${HOME_PLATFORM}/flow
    ${HOME_PLATFORM}/data_manager
    ${HOME_PLATFORM}/tool_manager
    ${HOME_PLATFORM}/tool_manager/tool_api
)

if(BUILD_GUI)
  target_link_libraries(ieda_tcl PUBLIC tcl_gui)

  target_include_directories(ieda_tcl 
    PUBLIC 
    ${HOME_INTERFACE}/tcl/tcl_gui
  )
endif()

if(CONTEST)
  target_link_libraries(ieda_tcl PUBLIC tcl_contest)

  target_include_directories(ieda_tcl 
    PUBLIC 
    ${HOME_INTERFACE}/tcl/tcl_contest
  )
endif()


include(${HOME_CMAKE}/operation/idb.cmake)

add_library(idm
    idm.cpp
    idm_init.cpp
    idm_checker.cpp
    idm_design.cpp
    idm_design_inst.cpp
    idm_design_inst_replace.cpp
    idm_design_net.cpp
    idm_design_pdn.cpp
    idm_design_blockage.cpp
    idm_layout.cpp
    idm_save.cpp
    # idm_pa.cpp
    idm_transform.cpp
    config/dm_config.cpp
    config/dm_cts_config.cpp
    checker/check_connection.cpp
)

target_link_libraries(idm 
    PUBLIC
    flow 
    file_manager_cts 
    # file_manager_drc 
    file_manager_placement 
    tool_manager 
    tool_api_ista 
    )

target_include_directories(idm 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/config
        ${CMAKE_CURRENT_SOURCE_DIR}/checker

        ${HOME_PLATFORM}/flow
        ${HOME_PLATFORM}/flow/config

        ${HOME_PLATFORM}/tool_manager
        ${HOME_PLATFORM}/tool_manager/tool_api
        ${HOME_PLATFORM}/tool_manager/tool_api/irt_io

        ${HOME_PLATFORM}/file_manager

        ##json
        ${HOME_THIRDPARTY}/json
        ${HOME_UTILITY}/json

        ##auxiliary
        ${HOME_UTILITY}
        ${HOME_UTILITY}/json

        ##operation
        ${HOME_OPERATION}

)
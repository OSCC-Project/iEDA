include(${HOME_CMAKE}/operation/idb.cmake)

add_subdirectory(tool_api/irt_io)
add_subdirectory(tool_api/ifp_io)
add_subdirectory(tool_api/ieval_io)
add_subdirectory(tool_api/ipdn_io)
add_subdirectory(tool_api/ista_io)
add_subdirectory(tool_api/ipw_io)
add_subdirectory(tool_api/icts_io)
add_subdirectory(tool_api/idrc_io)
add_subdirectory(tool_api/ipl_io)
add_subdirectory(tool_api/ito_io)
add_subdirectory(tool_api/ino_io)
add_subdirectory(tool_api/ipnp_io)

add_library(tool_manager
    tool_manager.cpp
)

target_link_libraries(tool_manager PUBLIC icts_source)

target_link_libraries(tool_manager PUBLIC)

target_link_libraries(tool_manager
    PUBLIC
        tool_api_icts
        tool_api_idrc
        tool_api_ieval
        tool_api_ifp
        tool_api_ipdn
        tool_api_ipl
        tool_api_irt
        tool_api_ista
        tool_api_ipw
        tool_api_ito
        tool_api_ino
        tool_api_ipnp

        ipl-api
        ivec_db
)

target_link_libraries(tool_manager PUBLIC idm)

target_link_libraries(tool_manager PUBLIC ito_source)

target_link_libraries(tool_manager PUBLIC ino_source)

target_include_directories(tool_manager
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_PLATFORM}/
    ${HOME_PLATFORM}/data_manager
    ${HOME_PLATFORM}/data_manager/config
    ${HOME_PLATFORM}/flow/config
    ${HOME_PLATFORM}/evaluation
    ${HOME_UTILITY}/json
    ${HOME_OPERATION}
)

if(BUILD_GUI)
    target_link_libraries(tool_manager PUBLIC iGUI)
    target_include_directories(tool_manager PUBLIC ${HOME_INTERFACE}/gui/interface)
endif()

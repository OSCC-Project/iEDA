include_directories(      
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HOME_PLATFORM}
        ${HOME_PLATFORM}/tool_manager/tool_api
        ${HOME_OPERATION}
)

add_subdirectory(base)
add_subdirectory(icts_fm)
add_subdirectory(idrc_fm)
add_subdirectory(ipl_fm)
add_subdirectory(irt_fm)


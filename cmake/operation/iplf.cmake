set(IPLF_HOME ${HOME_PLATFORM})

include_directories(
    ${IPLF_HOME}/tool_manager
    ${IPLF_HOME}/tool_manager/tool_api
    ${IPLF_HOME}/data_manager
    ${IPLF_HOME}/data_manager/module
)

include(${HOME_CMAKE}/operation/idb.cmake)

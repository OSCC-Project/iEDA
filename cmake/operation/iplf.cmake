SET(IPLF_HOME ${HOME_PLATFORM})

INCLUDE_DIRECTORIES(
    ${IPLF_HOME}/tool_manager
    ${IPLF_HOME}/tool_manager/tool_api
    ${IPLF_HOME}/data_manager
    ${IPLF_HOME}/data_manager/module
)

include(${HOME_CMAKE}/operation/idb.cmake)

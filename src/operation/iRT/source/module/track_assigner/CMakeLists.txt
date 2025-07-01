if(DEBUG_IRT_TRACK_ASSIGNER)
    message(STATUS "RT: DEBUG_IRT_TRACK_ASSIGNER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_TRACK_ASSIGNER")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_track_assigner
    ## module track_assigner
    ${IRT_MODULE}/track_assigner/TrackAssigner.cpp
)

target_link_libraries(irt_track_assigner
    PUBLIC
    irt_data_manager
    irt_module
    irt_toolkit
)

target_include_directories(irt_track_assigner
    PUBLIC
    ${IRT_MODULE}/track_assigner/ta_data_manager
    ${IRT_MODULE}/track_assigner
)

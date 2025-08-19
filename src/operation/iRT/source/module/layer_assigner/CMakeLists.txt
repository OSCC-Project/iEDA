if(DEBUG_IRT_LAYER_ASSIGNER)
    message(STATUS "RT: DEBUG_IRT_LAYER_ASSIGNER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_LAYER_ASSIGNER")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_layer_assigner
    ## module layer_assigner
    ${IRT_MODULE}/layer_assigner/LayerAssigner.cpp
)

target_link_libraries(irt_layer_assigner
    PUBLIC
    irt_data_manager
    irt_module
    irt_toolkit
)

target_include_directories(irt_layer_assigner
    PUBLIC
    ${IRT_MODULE}/layer_assigner/la_data_manager
    ${IRT_MODULE}/layer_assigner
)

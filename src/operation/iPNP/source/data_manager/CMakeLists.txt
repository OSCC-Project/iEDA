add_library(data_manager
    GridManager.cpp
    iPNPIdbWrapper.cpp
)

target_link_libraries(data_manager
    PUBLIC
    idb
    idm
    synthesis
    
)

# target_include_directories(
#     PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}
# )
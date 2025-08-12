add_library(data_manager
    GridManager.cpp
    SingleTemplate.cpp
    TemplateLib.cpp
    iPNPIdbWrapper.cpp
)

target_link_libraries(data_manager
    PUBLIC
    idm
    idb
    synthesis
)

# target_include_directories(
#     PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}
# )
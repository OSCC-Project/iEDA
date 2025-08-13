add_library(pnp-data
    GridManager.cpp
    SingleTemplate.cpp
    TemplateLib.cpp
    iPNPIdbWrapper.cpp
)

target_link_libraries(pnp-data
    PUBLIC
    idb
    synthesis
)

# target_include_directories(
#     PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}
# )
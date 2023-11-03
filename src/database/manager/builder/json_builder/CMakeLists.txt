# SET(CMAKE_BUILD_TYPE "Debug")
add_library(json_builder
    json_write.cpp
)

target_include_directories(json_builder 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HOME_DATABASE}/data/design
        ${HOME_DATABASE}/data/design/db_design
        ${HOME_DATABASE}/data/design/db_layout
        ${HOME_DATABASE}/manager/service/def_service
        ${HOME_DATABASE}/manager/service/lef_service
        ${HOME_DATABASE}/manager/parser/json
)

target_link_libraries(json_builder PRIVATE json-parser idb)
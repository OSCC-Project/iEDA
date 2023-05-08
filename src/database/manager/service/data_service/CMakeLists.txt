add_library(data_service
data_service.cpp
)

target_link_libraries(data_service PRIVATE idb)

target_include_directories(data_service 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HOME_DATABASE}/data/design
        ${HOME_DATABASE}/data/design/db_design
        ${HOME_DATABASE}/data/design/db_layout
)
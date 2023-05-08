add_library(data_process
    header.cpp
    layout_write.cpp
    layout_read.cpp
)

target_include_directories(data_process 
PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HOME_DATABASE}/data
        ${HOME_DATABASE}/manager/service/def_service
)


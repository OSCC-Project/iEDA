add_library(verilog_builder
    verilog_read.cpp
    verilog_write.cpp
)

target_include_directories(verilog_builder 
     PUBLIC 
             ${CMAKE_CURRENT_SOURCE_DIR}
             ${HOME_DATABASE}/basic
             ${HOME_DATABASE}/data/design
             ${HOME_DATABASE}/data/design/db_design
             ${HOME_DATABASE}/data/design/db_layout
             ${HOME_DATABASE}/manager/service/def_service
             ${HOME_DATABASE}/manager/service/lef_service
             ${HOME_DATABASE}/manager/parser
)

target_link_libraries(verilog_builder 
    PRIVATE
        verilog 
        def_service 
        idb
        time
)
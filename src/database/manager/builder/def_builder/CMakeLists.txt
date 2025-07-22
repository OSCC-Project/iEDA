add_library(def_builder
    def_read.cpp
    def_write.cpp
)

target_include_directories(def_builder 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HOME_DATABASE}/data/design
        ${HOME_DATABASE}/data/design/db_design
        ${HOME_DATABASE}/data/design/db_layout
        ${HOME_DATABASE}/manager/service/def_service
        ${HOME_DATABASE}/manager/service/lef_service
        ${HOME_UTILITY}/string
)

target_include_directories(def_builder 
    SYSTEM PUBLIC 
        ${HOME_THIRDPARTY}/lefdef/def
        ${HOME_THIRDPARTY}/lefdef/def/def
        ${HOME_THIRDPARTY}/lefdef/def/defzlib
)

target_link_libraries(def_builder PRIVATE  def defzlib str solver_geometry_boost)

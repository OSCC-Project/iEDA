file(GLOB_RECURSE DB_SRC "*.cpp")
if(BUILD_STATIC_LIB)
  add_library(idb ${DB_SRC})
else()
  add_library(idb SHARED ${DB_SRC})
endif()

target_include_directories(idb 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/db_design
        ${CMAKE_CURRENT_SOURCE_DIR}/db_layout
        ${CMAKE_CURRENT_SOURCE_DIR}/db_property
        ${HOME_DATABASE}/basic/geometry
        ${HOME_UTILITY}/string
)

target_link_libraries(idb PRIVATE str geometry_db)
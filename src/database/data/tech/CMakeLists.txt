file(GLOB_RECURSE DB_SRC "*.cpp")
if(BUILD_STATIC_LIB)
  add_library(tech_db ${DB_SRC})
else()
  add_library(tech_db SHARED ${DB_SRC})
endif()

target_link_libraries(tech_db PRIVATE str geometry_db)

target_include_directories(tech_db  
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HOME_DATABASE}/basic/geometry
        ${HOME_DATABASE}/data/design
        ${HOME_DATABASE}/data/design/db_design
        ${HOME_DATABASE}/data/design/db_layout
        ${HOME_UTILITY}/string
)


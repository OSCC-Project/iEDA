file(GLOB_RECURSE DB_SRC "*.cpp")
if(BUILD_STATIC_LIB)
  add_library(geometry_db ${DB_SRC})
else()
  add_library(geometry_db SHARED ${DB_SRC})
endif()

target_include_directories(geometry_db 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)




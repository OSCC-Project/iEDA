file(GLOB_RECURSE EVAL_DB_SRC "*.cpp")
if(BUILD_STATIC_LIB)
  add_library(eval_db ${EVAL_DB_SRC})
else()
  add_library(eval_db SHARED ${EVAL_DB_SRC})
endif()

target_include_directories(eval_db 
    PUBLIC 
        ${EVAL_DATA}
)

file(GLOB_RECURSE EVAL_DB_SRC "*.cpp")
if(BUILD_STATIC_LIB)
  add_library(eval_pro_db ${EVAL_DB_SRC})
else()
  add_library(eval_pro_db SHARED ${EVAL_DB_SRC})
endif()

target_include_directories(eval_pro_db 
    PUBLIC 
        ${EVAL_PRO_DATA}
)

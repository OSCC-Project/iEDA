file(GLOB_RECURSE FEATURE_DB_SRC "*.cpp")
if(BUILD_STATIC_LIB)
  add_library(feature_db ${FEATURE_DB_SRC})
else()
  add_library(feature_db SHARED ${FEATURE_DB_SRC})
endif()

target_include_directories(feature_db 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)

install(TARGETS feature_db
    LIBRARY DESTINATION ${ECC_INSTALL_LIB_DIR}
    RUNTIME DESTINATION ${ECC_INSTALL_LIB_DIR})

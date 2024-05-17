file(GLOB_RECURSE FEATURE_BUILDER_SRC "*.cpp")
if(BUILD_STATIC_LIB)
  add_library(feature_builder ${FEATURE_BUILDER_SRC})
else()
  add_library(feature_builder SHARED ${FEATURE_BUILDER_SRC})
endif()

target_include_directories(feature_builder 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(feature_builder 
    PRIVATE 
        idb
        flow_config
        idm
        irt_interface
        ipl-source
        ipl-api
        icts_api
        ista-engine
        idrc_pro_api
        eval_api
        ieda_report_evaluator
        feature_db
)
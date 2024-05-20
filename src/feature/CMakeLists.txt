add_subdirectory(database)
add_subdirectory(parser)
add_subdirectory(builder)

add_library(ieda_feature
  feature_manager.cpp
)

target_link_libraries(ieda_feature 
    PUBLIC 
        idb
        feature_parser   
        feature_db
        feature_builder
)

target_include_directories(ieda_feature
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
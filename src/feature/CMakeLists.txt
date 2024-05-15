add_subdirectory(database)
add_subdirectory(parser)

add_library(ieda_feature
  feature_manager.cpp
)

target_link_libraries(ieda_feature 
    PUBLIC 
        feature_parser   
        feature_db
        idb
)

target_include_directories(ieda_feature
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

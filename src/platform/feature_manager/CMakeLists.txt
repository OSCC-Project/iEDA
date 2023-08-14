add_library(feature_manager
  feature_manager.cpp
)

target_link_libraries(feature_manager PUBLIC feature_parser idb)

target_include_directories(feature_manager
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

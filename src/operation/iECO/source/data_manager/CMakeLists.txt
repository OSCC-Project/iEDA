add_library(ieco_dm 
    ieco_dm.cpp
)

target_link_libraries(ieco_dm
    PUBLIC
    ieco_data
)

target_include_directories(ieco_dm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

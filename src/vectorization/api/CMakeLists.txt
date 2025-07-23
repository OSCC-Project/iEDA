add_library(ivec_api
    vec_api.cpp
)

target_link_libraries(ivec_api
    PUBLIC
    vectorization
    ivec_layout_db
)

target_include_directories(ivec_api
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
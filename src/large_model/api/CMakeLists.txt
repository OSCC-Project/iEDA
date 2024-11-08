add_library(ilm_api
    lm_api.cpp
)

target_link_libraries(ilm_api
    PUBLIC
    large_model
)

target_include_directories(ilm_api
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set(CMAKE_BUILD_TYPE "debug")
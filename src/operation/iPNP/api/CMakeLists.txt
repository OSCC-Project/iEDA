add_library(iPNPApi 
iPNPApi.cpp
)

target_link_libraries(iPNPApi 
    PUBLIC

)

target_include_directories(iPNPApi
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)


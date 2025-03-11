cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)

aux_source_directory(./ SRC)
add_library(inference ${SRC})

target_link_libraries(inference sta onnxruntime)

target_include_directories(inference
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
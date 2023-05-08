aux_source_directory(. PY_ISTA_SRC)

add_library(py_ista ${PY_ISTA_SRC})

target_link_libraries(py_ista
    PUBLIC
        tool_manager
        sta
        pybind11::pybind11

)
target_include_directories(py_ista
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
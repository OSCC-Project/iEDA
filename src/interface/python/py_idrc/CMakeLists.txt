aux_source_directory(. PY_IDRC_SRC)

add_library(py_idrc ${PY_IDRC_SRC})

target_link_libraries(py_idrc
    PUBLIC
        tool_manager
        pybind11::pybind11
)
target_include_directories(py_idrc
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

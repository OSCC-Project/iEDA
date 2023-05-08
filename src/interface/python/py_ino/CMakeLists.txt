aux_source_directory(. PY_INO_SRC)

add_library(py_ino ${PY_INO_SRC})

target_link_libraries(py_ino
    PUBLIC
        tool_manager
        pybind11::pybind11
)
target_include_directories(py_ino
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

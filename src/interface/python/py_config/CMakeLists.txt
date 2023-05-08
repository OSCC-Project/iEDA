aux_source_directory(. PY_CONFIG_SRC)

add_library(py_config ${PY_CONFIG_SRC})

target_link_libraries(py_config
    PUBLIC
        flow
        tool_manager
        pybind11::pybind11
)
target_include_directories(py_config
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

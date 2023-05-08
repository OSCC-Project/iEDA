aux_source_directory(. PY_IPDN_SRC)

add_library(py_ipdn ${PY_IPDN_SRC})

target_link_libraries(py_ipdn
    PUBLIC
        tool_manager
        ipdn_api
        pybind11::pybind11
)
target_include_directories(py_ipdn
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
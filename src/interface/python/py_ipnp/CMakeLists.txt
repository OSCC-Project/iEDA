aux_source_directory(. PY_IPNP_SRC)

add_library(py_ipnp ${PY_IPNP_SRC})

target_link_libraries(py_ipnp
    PUBLIC
        tool_manager
        ipnp_api
        pybind11::pybind11
)
target_include_directories(py_ipnp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
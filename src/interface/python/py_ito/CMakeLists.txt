aux_source_directory(. PY_ITO_SRC)

add_library(py_ito ${PY_ITO_SRC})

target_link_libraries(py_ito
    PUBLIC
        tool_manager
        pybind11::pybind11

)
target_include_directories(py_ito
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
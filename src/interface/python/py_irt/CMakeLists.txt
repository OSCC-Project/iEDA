aux_source_directory(. PY_IRT_SRC)

add_library(py_irt ${PY_IRT_SRC})

target_link_libraries(py_irt
    PUBLIC
        tool_manager
        irt_interface
        tcl_util
        pybind11::pybind11

)
target_include_directories(py_irt
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_UTILITY}/json
)
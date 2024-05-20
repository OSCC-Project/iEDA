aux_source_directory(. PY_REPORT_SRC)

add_library(py_report ${PY_REPORT_SRC})

target_link_libraries(py_report
PUBLIC
        tool_manager
        pybind11::pybind11
        ieda_report

)
target_include_directories(py_report
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
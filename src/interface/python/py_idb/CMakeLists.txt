aux_source_directory(. PY_IDB_SRC)

add_library(py_idb ${PY_IDB_SRC})

target_link_libraries(py_idb
    PUBLIC
        idm
        pybind11::pybind11
)
target_include_directories(py_idb
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}


)

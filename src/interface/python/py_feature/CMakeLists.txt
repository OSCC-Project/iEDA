aux_source_directory(. PY_FEATURE_SRC)

add_library(py_feature ${PY_FEATURE_SRC})

target_link_libraries(py_feature
    PUBLIC
        feature_manager
        idm
        idb
        pybind11::pybind11

)
target_include_directories(py_feature
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
aux_source_directory(. PY_IMP_SRC)

add_library(py_imp ${PY_IMP_SRC})

target_link_libraries(py_imp
    PUBLIC
    imp_api
    idm
    pybind11::pybind11
)

target_include_directories(py_imp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}
    ${HOME_PLATFORM}/data_manager
    ${HOME_DATABASE}/data/design)
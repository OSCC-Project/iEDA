# add_subdirectory(${HOME_THIRDPARTY}/pybind11 binary_dir)

SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# DEFINE PYTHON MODULE & CMAKE TARGET
pybind11_add_module(ieda_py)

add_subdirectory(py_idb)
add_subdirectory(py_config)
add_subdirectory(py_flow)
add_subdirectory(py_icts)
add_subdirectory(py_idrc)
add_subdirectory(py_ifp)
add_subdirectory(py_ino)
add_subdirectory(py_instance)
add_subdirectory(py_ipdn)
add_subdirectory(py_ipl)
add_subdirectory(py_irt)
add_subdirectory(py_ista)
add_subdirectory(py_ipw)
add_subdirectory(py_ito)
add_subdirectory(py_report)
add_subdirectory(py_feature)
add_subdirectory(py_eval)
add_subdirectory(py_vec)
add_subdirectory(py_ipnp)

# ADD SOURCE FILES
# file(GLOB_RECURSE PYTHON_INTERFACE_SRC CONFIGURE_DEPENDS ./*.cpp ./*.cc)
file(GLOB PYTHON_INTERFACE_SRC CONFIGURE_DEPENDS *.cpp .h)

# SET SOURCE FILES
target_sources(ieda_py PUBLIC ${PYTHON_INTERFACE_SRC})

target_link_libraries(ieda_py
    PRIVATE
    py_idb
    py_config
    py_flow
    py_icts
    py_idrc
    py_ifp
    py_ino
    py_inst
    py_ipdn
    py_ipl
    py_irt
    py_ista
    py_ipw
    py_ito
    py_report
    py_feature
    py_eval
    py_vec
    py_ipnp
)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
pybind11_add_module(py_dm PyDataManager.cpp)

target_link_libraries(py_dm 
    PUBLIC
        imp-datamanager
)
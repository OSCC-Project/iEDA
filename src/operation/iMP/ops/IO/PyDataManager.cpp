#include <pybind11/pybind11.h>

#include <iostream>
#include <string>

namespace py = pybind11;

void wrapperIdb(std::string& idb_json)
{
}

PYBIND11_MODULE(py_dm, m)
{
  m.def("wrapperIdb", &wrapperIdb, "A function which wrapper data from idb");
}
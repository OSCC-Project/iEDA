#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

namespace python_interface {

void register_idb(pybind11::module& m);

void register_idb_op(pybind11::module& m);
}  // namespace python_interface
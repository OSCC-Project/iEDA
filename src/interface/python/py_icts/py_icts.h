#pragma once
#include <string>

namespace python_interface {
bool ctsAutoRun(const std::string& cts_config);
void ctsReport(const std::string& path);
}  // namespace python_interface
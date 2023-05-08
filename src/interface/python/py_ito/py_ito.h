#pragma once

#include <string>

namespace python_interface {
bool toAutoRun(const std::string& config);
bool toRunDrv(const std::string& config);

bool toRunHold(const std::string& config);
bool toRunSetup(const std::string& config);
}  // namespace python_interface
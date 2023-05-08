#pragma once
#include <string>

namespace python_interface {
bool fpPlaceInst(const std::string& inst_name, int llx, int lly, const std::string& orient, const std::string& cellmaster,
                 const std::string& source);
}
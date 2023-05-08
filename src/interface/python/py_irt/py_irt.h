#pragma once

#include <tcl_util.h>

#include <iRT/api/RTAPI.hpp>
namespace python_interface {

bool destroyRT();
bool initRT(std::string& config, std::map<std::string, std::string>& config_dict);
bool runDR();
bool runGR();
bool runRT();

}  // namespace python_interface

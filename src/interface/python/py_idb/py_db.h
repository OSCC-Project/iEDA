#pragma once

#include <set>
#include <string>
#include <vector>

namespace python_interface {

bool initIdb(const std::string& config_path);
bool initTechLef(const std::string& techlef_path);
bool initLef(const std::vector<std::string>& lef_paths);
bool initDef(const std::string& def_path);
bool initVerilog(const std::string& verilog_path);
bool saveDef(const std::string& def_name);
bool saveNetList(const std::string& netlist_path, std::set<std::string> exclude_cell_names = {});

}  // namespace python_interface
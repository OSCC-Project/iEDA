#pragma once

#include <string>
#include <vector>

namespace python_interface {
bool flow_init(const std::string& flow_config);

bool db_init(const std::string& config_path, const std::string& tech_lef_path, const std::vector<std::string>& lef_paths,
             const std::string& def_path, const std::string& verilog_path, const std::string& output_dir_path,
             const std::vector<std::string>& lib_paths, const std::string& sdc_path);
}  // namespace python_interface
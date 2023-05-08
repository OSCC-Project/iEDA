#pragma once

#include <set>
#include <string>
namespace python_interface {
bool staRun(const std::string& output);

bool staInit(const std::string& output);

bool staReport(const std::string& output);
bool setDesignWorkSpace(const std::string& design_workspace);

bool readVerilog(const std::string& file_name);

bool readLiberty(const std::string& file_name);

bool linkDesign(const std::string& cell_name);

bool readSpef(const std::string& file_name);

bool readSdc(const std::string& file_name);
bool reportTiming(int digits, const std::string& delay_type, std::set<std::string> exclude_cell_names, bool derate);

}  // namespace python_interface
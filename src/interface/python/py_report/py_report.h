#pragma once

#include <string>
#include <vector>

#include "report_manager.h"
namespace python_interface {
bool reportDbSummary(const std::string& path);
bool reportWireLength(const std::string& path);
bool reportCong(const std::string& path);
bool reportDanglingNet(const std::string& path);
bool reportRoute(const std::string& path, const std::string& netname, bool summary);
bool reportPlaceDistribution(const std::vector<std::string>& prefixes);
bool reportPrefixedInst(const std::string& prefix, int level, int num_threshold);

bool reportDRC(const std::string& filename);
}  // namespace python_interface
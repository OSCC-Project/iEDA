#include "py_report.h"

#include <report_manager.h>

namespace python_interface {
bool reportDbSummary(const std::string& path)
{
  return rptInst->reportDBSummary(path);
}
bool reportWireLength(const std::string& path)
{
  return rptInst->reportWL(path);
}

bool reportCong(const std::string& path)
{
  return rptInst->reportCongestion(path);
}
bool reportDanglingNet(const std::string& path)
{
  return rptInst->reportDanglingNet(path);
}

bool reportRoute(const std::string& path, const std::string& netname, bool summary)
{
  return rptInst->reportRoute(path, netname, summary);
}

bool reportPlaceDistribution(const std::vector<std::string>& prefixes)
{
  return rptInst->reportPlaceDistribution(prefixes);
}

bool reportPrefixedInst(const std::string& prefix, int level, int num_threshold){
  return rptInst->reportInstLevel(prefix, level,  num_threshold);
}

bool reportDRC(const std::string& filename){
  return rptInst->reportDRC(filename);
}
}  // namespace python_interface
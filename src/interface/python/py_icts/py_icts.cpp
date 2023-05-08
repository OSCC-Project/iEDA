#include "py_icts.h"

#include <tool_manager.h>

#include <CTSAPI.hpp>
namespace python_interface {
bool ctsAutoRun(const std::string& cts_config)
{
  bool cts_run_ok = iplf::tmInst->autoRunCTS(cts_config);
  return cts_run_ok;
}

void ctsReport(const std::string& path) { CTSAPIInst.report(path); }

}  // namespace python_interface
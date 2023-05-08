#include "py_ito.h"
#include <tool_manager.h>

#include "ToApi.hpp"

namespace python_interface {
bool toAutoRun(const std::string& config)
{
  bool run_ok = iplf::tmInst->autoRunTO(config);
  return run_ok;
}

bool toRunDrv(const std::string& config)
{
  bool run_ok = iplf::tmInst->RunTODrv(config);
  return run_ok;
}

bool toRunHold(const std::string& config)
{
  bool run_ok = iplf::tmInst->RunTOHold(config);
  return run_ok;
}

bool toRunSetup(const std::string& config)
{
  bool run_ok = iplf::tmInst->RunTOSetup(config);
  return run_ok;
}
}  // namespace python_interface
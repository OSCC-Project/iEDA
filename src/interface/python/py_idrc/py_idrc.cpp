#include "py_idrc.h"

#include <tool_manager.h>

namespace python_interface {

bool drcAutoRun(const std::string& config)
{
  bool run_ok = iplf::tmInst->autoRunDRC(config);
  return run_ok;
}

}  // namespace python_interface
#include "py_ino.h"

#include <tool_manager.h>
namespace python_interface {

bool noRunFixFanout(const std::string& config)
{
  bool run_result = iplf::tmInst->RunNOFixFanout(config);
  return run_result;
}

}  // namespace python_interface
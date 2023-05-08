#include "py_flow.h"
#include <flow.h>

namespace python_interface {
bool flowAutoRun()
{
  iplf::plfInst->runFlow();
  return true;
}
bool flowExit()
{
  std::exit(0);
  return true;
}

}  // namespace python_interface
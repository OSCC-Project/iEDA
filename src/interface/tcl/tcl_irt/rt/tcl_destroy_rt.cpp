#include "RTAPI.hpp"
#include "tcl_rt.h"
#include "tcl_util.h"

namespace tcl {

TclDestroyRT::TclDestroyRT(const char* cmd_name) : TclCmd(cmd_name)
{
}

unsigned TclDestroyRT::exec()
{
  if (!check()) {
    return 0;
  }

  RTAPI_INST.destroyRT();

  return 1;
}

}  // namespace tcl

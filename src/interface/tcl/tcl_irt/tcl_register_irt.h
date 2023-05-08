#pragma once

#include "tcl_egr.h"
#include "tcl_rt.h"

using namespace ieda;

namespace tcl {

int registerCmdRT()
{
  registerTclCmd(TclRunEGR, "run_egr");
  registerTclCmd(TclDestroyRT, "destroy_rt");
  registerTclCmd(TclInitRT, "init_rt");
  registerTclCmd(TclRunRT, "run_rt");
  return EXIT_SUCCESS;
}

}  // namespace tcl
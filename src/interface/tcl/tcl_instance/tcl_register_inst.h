#pragma once
/**
 * @file tcl_register_inst.h
 * @author yell
 * @brief
 * @version 0.1
 * @date 2022-10-28
 */
#include "UserShell.hh"
#include "tcl_inst.h"

using namespace ieda;

namespace tcl {

int registerCmdInstance()
{
  registerTclCmd(TclFpPlaceInst, "place_instance");

  return EXIT_SUCCESS;
}

}  // namespace tcl
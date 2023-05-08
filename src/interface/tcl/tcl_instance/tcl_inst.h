#pragma once

#include <any>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "ScriptEngine.hh"
#include "tcl_definition.h"

namespace tcl {

using ieda::ScriptEngine;
using ieda::TclCmd;
using ieda::TclCmds;
using ieda::TclDoubleListOption;
using ieda::TclDoubleOption;
using ieda::TclIntListOption;
using ieda::TclIntOption;
using ieda::TclOption;
using ieda::TclStringListOption;
using ieda::TclStringOption;
using ieda::TclSwitchOption;

class TclFpPlaceInst : public TclCmd
{
 public:
  explicit TclFpPlaceInst(const char* cmd_name);
  ~TclFpPlaceInst() override = default;

  unsigned check();
  unsigned exec();
};

}  // namespace tcl

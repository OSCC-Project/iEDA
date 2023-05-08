#pragma once

#include <iostream>
#include <string>

#include "../../tcl_definition.h"
#include "ScriptEngine.hh"

using ieda::TclCmd;
using ieda::TclOption;
using ieda::TclStringOption;

namespace tcl {

DEFINE_CMD_CLASS(ReportWL);
DEFINE_CMD_CLASS(ReportCong);

}  // namespace tcl
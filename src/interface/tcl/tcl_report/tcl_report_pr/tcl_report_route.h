#pragma once

#include "../../tcl_definition.h"
#include "report_manager.h"
#include "tcl/ScriptEngine.hh"

namespace tcl {

using ieda::TclCmd;
using ieda::TclOption;
using ieda::TclStringOption;

class CmdReportRoute : public TclCmd
{
 public:
  explicit CmdReportRoute(const char* cmd_name);
  ~CmdReportRoute() override = default;

  unsigned check() override { return 1; };
  unsigned exec() override;
};
}  // namespace tcl
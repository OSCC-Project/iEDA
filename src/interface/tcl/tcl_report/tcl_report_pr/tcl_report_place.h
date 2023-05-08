#pragma once

#include "../../tcl_definition.h"
#include "tcl/ScriptEngine.hh"

namespace tcl {

using ieda::TclCmd;

class CmdReportPlaceDistro : public TclCmd
{
 public:
  explicit CmdReportPlaceDistro(const char* cmd_name);
  ~CmdReportPlaceDistro() override = default;

  unsigned check() override { return 1; };
  unsigned exec() override;
};

class CmdReportPrefixedInst : public TclCmd
{
 public:
  explicit CmdReportPrefixedInst(const char* cmd_name);
  ~CmdReportPrefixedInst() override = default;
  unsigned check() override { return 1; }
  unsigned exec() override;
};

}  // namespace tcl
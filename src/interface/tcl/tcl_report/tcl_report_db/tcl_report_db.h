#pragma once
/**
 * @File Name: tcl_placer.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include <iostream>
#include <string>

#include "../../tcl_definition.h"
#include "ScriptEngine.hh"

using ieda::TclCmd;
using ieda::TclOption;
using ieda::TclStringOption;

namespace tcl {

class CmdReportDbSummary : public TclCmd
{
 public:
  explicit CmdReportDbSummary(const char* cmd_name);
  ~CmdReportDbSummary() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

DEFINE_CMD_CLASS(ReportDanglingNet);

}  // namespace tcl
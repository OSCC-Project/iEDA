#pragma once
/**
 * @File Name: tcl_flow.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */

#include <iostream>

#include "ScriptEngine.hh"
#include "tcl_definition.h"

using ieda::TclCmd;
using ieda::TclOption;
using ieda::TclStringOption;

namespace tcl {

class CmdFlowAutoRun : public TclCmd
{
 public:
  explicit CmdFlowAutoRun(const char* cmd_name);
  ~CmdFlowAutoRun() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdFlowExit : public TclCmd
{
 public:
  explicit CmdFlowExit(const char* cmd_name);
  ~CmdFlowExit() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

}  // namespace tcl
#pragma once
/**
 * @File Name: tcl_config.h
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
using ieda::TclStringListOption;
using ieda::TclStringOption;

namespace tcl {

class CmdFlowInitConfig : public TclCmd
{
 public:
  explicit CmdFlowInitConfig(const char* cmd_name);
  ~CmdFlowInitConfig() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdDbConfigSetting : public TclCmd
{
 public:
  explicit CmdDbConfigSetting(const char* cmd_name);
  ~CmdDbConfigSetting() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

}  // namespace tcl
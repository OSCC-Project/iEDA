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

#include "ScriptEngine.hh"
#include "tcl_definition.h"

using ieda::TclCmd;
using ieda::TclOption;
using ieda::TclStringOption;

namespace tcl {

class CmdNORunFixFanout : public TclCmd
{
 public:
  explicit CmdNORunFixFanout(const char* cmd_name);
  ~CmdNORunFixFanout() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

}  // namespace tcl
#pragma once
/**
 * @File Name: tcl_db.h
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
using ieda::TclStringListOption;
using ieda::TclStringOption;

namespace tcl {

class CmdIdbSetNet : public TclCmd
{
 public:
  explicit CmdIdbSetNet(const char* cmd_name);
  ~CmdIdbSetNet() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

////Blockage operate
class CmdIdbClearBlockage : public TclCmd
{
 public:
  explicit CmdIdbClearBlockage(const char* cmd_name);
  ~CmdIdbClearBlockage() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdIdbClearBlockageExceptPgNet : public TclCmd
{
 public:
  explicit CmdIdbClearBlockageExceptPgNet(const char* cmd_name);
  ~CmdIdbClearBlockageExceptPgNet() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};


DEFINE_CMD_CLASS(IdbDeleteInstance);
DEFINE_CMD_CLASS(IdbDeleteNet);
DEFINE_CMD_CLASS(IdbCreateInstance);
DEFINE_CMD_CLASS(IdbCreateNet);
}  // namespace tcl
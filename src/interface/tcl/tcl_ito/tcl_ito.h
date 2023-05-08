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

class CmdTOAutoRun : public TclCmd
{
 public:
  explicit CmdTOAutoRun(const char* cmd_name);
  ~CmdTOAutoRun() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdTORunDrv : public TclCmd
{
 public:
  explicit CmdTORunDrv(const char* cmd_name);
  ~CmdTORunDrv() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdTORunHold : public TclCmd
{
 public:
  explicit CmdTORunHold(const char* cmd_name);
  ~CmdTORunHold() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdTORunSetup : public TclCmd
{
 public:
  explicit CmdTORunSetup(const char* cmd_name);
  ~CmdTORunSetup() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};


DEFINE_CMD_CLASS(TOApiInitConfig);
DEFINE_CMD_CLASS(TOApiRunFlow);
DEFINE_CMD_CLASS(TOApiOptDRV);
DEFINE_CMD_CLASS(TOApiOptSetup);
DEFINE_CMD_CLASS(TOApiOptHold);
DEFINE_CMD_CLASS(TOApiSaveDef);
DEFINE_CMD_CLASS(TOApiReportTiming);
}  // namespace tcl
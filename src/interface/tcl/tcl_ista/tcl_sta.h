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

class CmdSTARun : public TclCmd
{
 public:
  explicit CmdSTARun(const char* cmd_name);
  ~CmdSTARun() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdBuildClockTree : public TclCmd
{
 public:
  explicit CmdBuildClockTree(const char* cmd_name);
  ~CmdBuildClockTree() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdSTAInit : public TclCmd
{
 public:
  explicit CmdSTAInit(const char* cmd_name);
  ~CmdSTAInit() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdSTAReport : public TclCmd
{
 public:
  explicit CmdSTAReport(const char* cmd_name);
  ~CmdSTAReport() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

}  // namespace tcl
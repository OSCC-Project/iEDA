#pragma once
/**
 * @File Name: tcl_gui.h
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
using ieda::TclIntOption;
using ieda::TclOption;
using ieda::TclStringOption;

namespace tcl {

class CmdGuiStart : public TclCmd
{
 public:
  explicit CmdGuiStart(const char* cmd_name);
  ~CmdGuiStart() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdGuiShow : public TclCmd
{
 public:
  explicit CmdGuiShow(const char* cmd_name);
  ~CmdGuiShow() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdGuiHide : public TclCmd
{
 public:
  explicit CmdGuiHide(const char* cmd_name);
  ~CmdGuiHide() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdGuiShowDrc : public TclCmd
{
 public:
  explicit CmdGuiShowDrc(const char* cmd_name);
  ~CmdGuiShowDrc() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdGuiShowClockTree : public TclCmd
{
 public:
  explicit CmdGuiShowClockTree(const char* cmd_name);
  ~CmdGuiShowClockTree() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdGuiShowPlacement : public TclCmd
{
 public:
  explicit CmdGuiShowPlacement(const char* cmd_name);
  ~CmdGuiShowPlacement() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

}  // namespace tcl
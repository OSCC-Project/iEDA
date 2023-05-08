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

class CmdInitIdb : public TclCmd
{
 public:
  explicit CmdInitIdb(const char* cmd_name);
  ~CmdInitIdb() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdInitTechLef : public TclCmd
{
 public:
  explicit CmdInitTechLef(const char* cmd_name);
  ~CmdInitTechLef() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdInitLef : public TclCmd
{
 public:
  explicit CmdInitLef(const char* cmd_name);
  ~CmdInitLef() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdInitDef : public TclCmd
{
 public:
  explicit CmdInitDef(const char* cmd_name);
  ~CmdInitDef() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdInitVerilog : public TclCmd
{
 public:
  explicit CmdInitVerilog(const char* cmd_name);
  ~CmdInitVerilog() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdSaveDef : public TclCmd
{
 public:
  explicit CmdSaveDef(const char* cmd_name);
  ~CmdSaveDef() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdSaveNetlist : public TclCmd
{
 public:
  explicit CmdSaveNetlist(const char* cmd_name);
  ~CmdSaveNetlist() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdSaveGDS : public TclCmd
{
 public:
  explicit CmdSaveGDS(const char* cmd_name);
  ~CmdSaveGDS() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

}  // namespace tcl
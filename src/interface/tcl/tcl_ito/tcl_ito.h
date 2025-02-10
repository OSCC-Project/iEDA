// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
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

class CmdTORunDrvSpecialNet : public TclCmd
{
 public:
  explicit CmdTORunDrvSpecialNet(const char* cmd_name);
  ~CmdTORunDrvSpecialNet() override = default;

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

class CmdTOBuffering : public TclCmd
{
 public:
  explicit CmdTOBuffering(const char* cmd_name);
  ~CmdTOBuffering() override = default;

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
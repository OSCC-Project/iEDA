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
 * @File Name: tcl_eval.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */

#include <iostream>

#include "ScriptEngine.hh"
#include "timing_io.h"
#include "tcl_definition.h"

using ieda::TclCmd;
using ieda::TclIntOption;
using ieda::TclOption;
using ieda::TclStringOption;

namespace tcl {

class CmdEvalInit : public TclCmd
{
 public:
  explicit CmdEvalInit(const char* cmd_name);
  ~CmdEvalInit() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdEvalTimingRun final : public TclCmd
{
 public:
  explicit CmdEvalTimingRun(const char* cmd_name);
  ~CmdEvalTimingRun() override = default;

  unsigned check() override;
  unsigned exec() override;
};

class CmdEvalWirelengthRun : public TclCmd
{
 public:
  explicit CmdEvalWirelengthRun(const char* cmd_name);
  ~CmdEvalWirelengthRun() override = default;

  unsigned check() override;
  unsigned exec() override;
};

class CmdEvalDensityRun : public TclCmd
{
public:
  explicit CmdEvalDensityRun(const char* cmd_name);
  ~CmdEvalDensityRun() override = default;

  unsigned check() override;
  unsigned exec() override;
};

class CmdEvalEgrConfig : public TclCmd
{
 public:
  explicit CmdEvalEgrConfig(const char* cmd_name);
  ~CmdEvalEgrConfig() override = default;
  unsigned check() override;
  unsigned exec() override;
};

}  // namespace tcl
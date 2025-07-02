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
/**
 * @file ShellCmd.hh
 * @author Jianrong Su
 * @brief 
 * @version 1.0
 * @date 2025-06-30
 */

#pragma once

#include "ScriptEngine.hh"
#include "iPNP.hh"
#include "PNPConfig.hh"

namespace ipnp {

using ieda::ScriptEngine;
using ieda::TclCmd;
using ieda::TclCmds;
using ieda::TclDoubleListOption;
using ieda::TclDoubleOption;
using ieda::TclEncodeResult;
using ieda::TclIntListOption;
using ieda::TclIntOption;
using ieda::TclOption;
using ieda::TclStringListListOption;
using ieda::TclStringListOption;
using ieda::TclStringOption;
using ieda::TclSwitchOption;

// Function to register commands

class CmdRunPnp : public TclCmd {
public:
  explicit CmdRunPnp(const char* cmd_name);
  ~CmdRunPnp() override = default;

  unsigned check() override;
  unsigned exec() override;
};

class CmdAddVIA1 : public TclCmd {
public:
  explicit CmdAddVIA1(const char* cmd_name);
  ~CmdAddVIA1() override = default;

  unsigned check() override;
  unsigned exec() override;
};

} // namespace ipnp

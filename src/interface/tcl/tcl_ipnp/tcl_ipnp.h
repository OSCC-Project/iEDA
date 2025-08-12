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

#include "ScriptEngine.hh"
#include "tcl_definition.h"

using ieda::TclCmd;

namespace tcl {

class CmdRunPnp : public TclCmd
{
 public:
  explicit CmdRunPnp(const char* cmd_name);
  ~CmdRunPnp() override = default;

  unsigned check() override;
  unsigned exec() override;
};

class CmdAddVIA1 : public TclCmd
{
 public:
  explicit CmdAddVIA1(const char* cmd_name);
  ~CmdAddVIA1() override = default;

  unsigned check() override;
  unsigned exec() override;
};

}  // namespace tcl

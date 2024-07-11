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

#include <any>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "ScriptEngine.hh"
#include "tcl_definition.h"

namespace tcl {

using ieda::ScriptEngine;
using ieda::TclCmd;
using ieda::TclCmds;
using ieda::TclDoubleListOption;
using ieda::TclDoubleOption;
using ieda::TclIntListOption;
using ieda::TclIntOption;
using ieda::TclOption;
using ieda::TclStringListOption;
using ieda::TclStringOption;
using ieda::TclSwitchOption;

class TclPdnAddIO : public TclCmd
{
 public:
  explicit TclPdnAddIO(const char* cmd_name);
  ~TclPdnAddIO() override = default;

  unsigned check();
  unsigned exec();
};

class TclPdnGlobalConnect : public TclCmd
{
 public:
  explicit TclPdnGlobalConnect(const char* cmd_name);
  ~TclPdnGlobalConnect() override = default;

  unsigned check();
  unsigned exec();
};

class TclPdnPlacePort : public TclCmd
{
 public:
  explicit TclPdnPlacePort(const char* cmd_name);
  ~TclPdnPlacePort() override = default;

  unsigned check();
  unsigned exec();
};

class TclPdnCreateGrid : public TclCmd
{
 public:
  explicit TclPdnCreateGrid(const char* cmd_name);
  ~TclPdnCreateGrid() override = default;

  unsigned check();
  unsigned exec();
};

class TclPdnCreateStripe : public TclCmd
{
 public:
  explicit TclPdnCreateStripe(const char* cmd_name);
  ~TclPdnCreateStripe() override = default;

  unsigned check();
  unsigned exec();
};

class TclPdnConnectLayer : public TclCmd
{
 public:
  explicit TclPdnConnectLayer(const char* cmd_name);
  ~TclPdnConnectLayer() override = default;

  unsigned check();
  unsigned exec();
};

class TclPdnConnectMacro : public TclCmd
{
 public:
  explicit TclPdnConnectMacro(const char* cmd_name);
  ~TclPdnConnectMacro() override = default;

  unsigned check();
  unsigned exec();
};

class TclPdnConnectIOPin : public TclCmd
{
 public:
  explicit TclPdnConnectIOPin(const char* cmd_name);
  ~TclPdnConnectIOPin() override = default;

  unsigned check();
  unsigned exec();
};

class TclPdnConnectStripe : public TclCmd
{
 public:
  explicit TclPdnConnectStripe(const char* cmd_name);
  ~TclPdnConnectStripe() override = default;

  unsigned check();
  unsigned exec();
};

class TclPdnAddSegmentStripe : public TclCmd
{
 public:
  explicit TclPdnAddSegmentStripe(const char* cmd_name);
  ~TclPdnAddSegmentStripe() override = default;

  unsigned check();
  unsigned exec();
};

class TclPdnAddSegmentVia : public TclCmd
{
 public:
  explicit TclPdnAddSegmentVia(const char* cmd_name);
  ~TclPdnAddSegmentVia() override = default;

  unsigned check();
  unsigned exec();
};

class TclRunPNP : public TclCmd
{
 public:
  explicit TclRunPNP(const char* cmd_name);
  ~TclRunPNP() override = default;

  unsigned check();
  unsigned exec();
};

}  // namespace tcl

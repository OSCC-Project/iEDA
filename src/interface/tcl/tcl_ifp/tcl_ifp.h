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

class TclFpInit : public TclCmd
{
 public:
  explicit TclFpInit(const char* cmd_name);
  ~TclFpInit() override = default;

  unsigned check();
  unsigned exec();
};

class TclFpMakeTracks : public TclCmd
{
 public:
  explicit TclFpMakeTracks(const char* cmd_name);
  ~TclFpMakeTracks() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class TclFpPlacePins : public TclCmd
{
 public:
  explicit TclFpPlacePins(const char* cmd_name);
  ~TclFpPlacePins() override = default;

  unsigned check();
  unsigned exec();
};

class TclFpPlacePort : public TclCmd
{
 public:
  explicit TclFpPlacePort(const char* cmd_name);
  ~TclFpPlacePort() override = default;

  unsigned check();
  unsigned exec();
};

class TclFpPlaceIOFiller : public TclCmd
{
 public:
  explicit TclFpPlaceIOFiller(const char* cmd_name);
  ~TclFpPlaceIOFiller() override = default;

  unsigned check();
  unsigned exec();
};

class TclFpAutoPlaceIO : public TclCmd
{
 public:
  explicit TclFpAutoPlaceIO(const char* cmd_name);
  ~TclFpAutoPlaceIO() override = default;

  unsigned check();
  unsigned exec();
};

class TclFpAddPlacementBlockage : public TclCmd
{
 public:
  explicit TclFpAddPlacementBlockage(const char* cmd_name);
  ~TclFpAddPlacementBlockage() override = default;

  unsigned check();
  unsigned exec();
};

class TclFpAddPlacementHalo : public TclCmd
{
 public:
  explicit TclFpAddPlacementHalo(const char* cmd_name);
  ~TclFpAddPlacementHalo() override = default;

  unsigned check();
  unsigned exec();
};

class TclFpAddRoutingBlockage : public TclCmd
{
 public:
  explicit TclFpAddRoutingBlockage(const char* cmd_name);
  ~TclFpAddRoutingBlockage() override = default;

  unsigned check();
  unsigned exec();
};

class TclFpAddRoutingHalo : public TclCmd
{
 public:
  explicit TclFpAddRoutingHalo(const char* cmd_name);
  ~TclFpAddRoutingHalo() override = default;

  unsigned check();
  unsigned exec();
};

class TclFpTapCell : public TclCmd
{
 public:
  explicit TclFpTapCell(const char* cmd_name);
  ~TclFpTapCell() override = default;

  unsigned check();
  unsigned exec();
};

}  // namespace tcl

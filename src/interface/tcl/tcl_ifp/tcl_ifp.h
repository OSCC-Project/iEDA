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

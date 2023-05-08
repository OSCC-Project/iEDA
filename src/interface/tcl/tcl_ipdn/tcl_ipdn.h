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

}  // namespace tcl

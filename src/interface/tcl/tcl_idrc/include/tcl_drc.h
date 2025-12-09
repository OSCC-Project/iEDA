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

#include "tcl_definition.h"
#include "tcl_util.h"

namespace tcl {

class TclCheckDef : public TclCmd
{
 public:
  explicit TclCheckDef(const char* cmd_name);
  ~TclCheckDef() override = default;

  unsigned check() override { return 1; };

  unsigned exec() override;

 private:
  std::vector<std::pair<std::string, ValueType>> _config_list;
};

class TclCmpDRC : public TclCmd
{
 public:
  explicit TclCmpDRC(const char* cmd_name);
  ~TclCmpDRC() override = default;

  unsigned check() override { return 1; };

  unsigned exec() override;

 private:
  std::vector<std::pair<std::string, ValueType>> _config_list;
};

class TclDestroyDRC : public TclCmd
{
 public:
  explicit TclDestroyDRC(const char* cmd_name);
  ~TclDestroyDRC() override = default;

  unsigned check() override { return 1; };

  unsigned exec() override;

 private:
  std::vector<std::pair<std::string, ValueType>> _config_list;
};

class TclInitDRC : public TclCmd
{
 public:
  explicit TclInitDRC(const char* cmd_name);
  ~TclInitDRC() override = default;

  unsigned check() override { return 1; };

  unsigned exec() override;

 private:
  std::vector<std::pair<std::string, ValueType>> _config_list;
};

class CmdDRCAutoRun : public TclCmd
{
 public:
  explicit CmdDRCAutoRun(const char* cmd_name);
  ~CmdDRCAutoRun() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdDRCSaveDetailFile : public TclCmd
{
 public:
  explicit CmdDRCSaveDetailFile(const char* cmd_name);
  ~CmdDRCSaveDetailFile() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

}  // namespace tcl

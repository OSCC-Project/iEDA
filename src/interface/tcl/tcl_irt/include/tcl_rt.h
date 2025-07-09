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

#include "tcl_util.h"

namespace tcl {

class TclClearDef : public TclCmd
{
 public:
  explicit TclClearDef(const char* cmd_name);
  ~TclClearDef() override = default;

  unsigned check() override { return 1; };

  unsigned exec() override;

 private:
  std::vector<std::pair<std::string, ValueType>> _config_list;
};

class TclDestroyRT : public TclCmd
{
 public:
  explicit TclDestroyRT(const char* cmd_name);
  ~TclDestroyRT() override = default;

  unsigned check() override { return 1; };

  unsigned exec() override;

 private:
  std::vector<std::pair<std::string, ValueType>> _config_list;
};

class TclInitRT : public TclCmd
{
 public:
  explicit TclInitRT(const char* cmd_name);
  ~TclInitRT() override = default;

  unsigned check() override { return 1; };

  unsigned exec() override;

 private:
  std::vector<std::pair<std::string, ValueType>> _config_list;
};

class TclOutputDBJson : public TclCmd
{
 public:
  explicit TclOutputDBJson(const char* cmd_name);
  ~TclOutputDBJson() override = default;

  unsigned check() override { return 1; };

  unsigned exec() override;

 private:
  std::vector<std::pair<std::string, ValueType>> _config_list;
};

class TclRunEGR : public TclCmd
{
 public:
  explicit TclRunEGR(const char* cmd_name);
  ~TclRunEGR() override = default;

  unsigned check() override { return 1; };

  unsigned exec() override;

 private:
  std::vector<std::pair<std::string, ValueType>> _config_list;
};

class TclRunRT : public TclCmd
{
 public:
  explicit TclRunRT(const char* cmd_name);
  ~TclRunRT() override = default;

  unsigned check() override { return 1; };

  unsigned exec() override;

 private:
  std::vector<std::pair<std::string, ValueType>> _config_list;
};

}  // namespace tcl

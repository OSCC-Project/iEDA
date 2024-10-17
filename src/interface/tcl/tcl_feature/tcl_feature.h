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
 * @File Name: tcl_flow.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */

#include <iostream>

#include "../tcl_definition.h"
#include "ScriptEngine.hh"

using ieda::TclCmd;
using ieda::TclIntOption;
using ieda::TclOption;
using ieda::TclStringOption;

namespace tcl {
class CmdFeatureSummary : public TclCmd
{
 public:
  explicit CmdFeatureSummary(const char* cmd_name);
  ~CmdFeatureSummary() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdFeatureTool : public TclCmd
{
 public:
  explicit CmdFeatureTool(const char* cmd_name);
  ~CmdFeatureTool() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdFeatureEvalMap : public TclCmd
{
 public:
  explicit CmdFeatureEvalMap(const char* cmd_name);
  ~CmdFeatureEvalMap() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdFeatureRoute : public TclCmd
{
 public:
  explicit CmdFeatureRoute(const char* cmd_name);
  ~CmdFeatureRoute() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdFeatureRouteRead : public TclCmd
{
 public:
  explicit CmdFeatureRouteRead(const char* cmd_name);
  ~CmdFeatureRouteRead() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdFeatureEvalSummary : public TclCmd
{
 public:
  explicit CmdFeatureEvalSummary(const char* cmd_name);
  ~CmdFeatureEvalSummary() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdFeatureCongMap : public TclCmd
{
 public:
  explicit CmdFeatureCongMap(const char* cmd_name);
  ~CmdFeatureCongMap() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

}  // namespace tcl
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
 * @File Name: tcl_register.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include "ScriptEngine.hh"
#include "UserShell.hh"
#include "tcl_feature.h"

using namespace ieda;

namespace tcl {

int registerCmdFeature()
{
  registerTclCmd(CmdFeatureSummary, "feature_summary");
  registerTclCmd(CmdFeatureTool, "feature_tool");
  registerTclCmd(CmdFeatureEvalMap, "feature_eval_map");
  registerTclCmd(CmdFeatureRoute, "feature_route");
  registerTclCmd(CmdFeatureRouteRead, "feature_route_read");
  registerTclCmd(CmdFeatureCongMap, "feature_cong_map");

  return EXIT_SUCCESS;
}

}  // namespace tcl
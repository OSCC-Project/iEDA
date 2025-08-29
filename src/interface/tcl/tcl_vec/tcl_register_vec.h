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
 * @File Name: tcl_eco.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2024-08-26
 *
 */
#include "ScriptEngine.hh"
#include "UserShell.hh"
#include "tcl_vec.h"

using namespace ieda;

namespace tcl {

int registerCmdVectorization()
{
  registerTclCmd(CmdVecLayoutPatchs, "layout_patchs");
  registerTclCmd(CmdVecLayoutGraph, "layout_graph");
  registerTclCmd(CmdVecFeature, "generate_vectors");
  registerTclCmd(CmdReadVecNets, "read_vectors_nets");
  registerTclCmd(CmdReadVecNetsPattern, "read_vectors_nets_patterns");

  return EXIT_SUCCESS;
}

}  // namespace tcl
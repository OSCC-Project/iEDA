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
 * @File Name: tcl_placer.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include <iostream>
#include <string>

#include "ScriptEngine.hh"
#include "tcl_definition.h"

using ieda::TclCmd;
using ieda::TclOption;
using ieda::TclStringOption;
using ieda::TclSwitchOption;

namespace tcl {

DEFINE_CMD_CLASS(PlacerAutoRun);
DEFINE_CMD_CLASS(PlacerFiller);
DEFINE_CMD_CLASS(PlacerIncrementalFlow);
DEFINE_CMD_CLASS(PlacerIncrementalLG);
DEFINE_CMD_CLASS(PlacerCheckLegality);
DEFINE_CMD_CLASS(PlacerReport);

DEFINE_CMD_CLASS(PlacerInit);
DEFINE_CMD_CLASS(PlacerDestroy);
DEFINE_CMD_CLASS(PlacerRunMP);
DEFINE_CMD_CLASS(PlacerRunGP);
DEFINE_CMD_CLASS(PlacerRunLG);
DEFINE_CMD_CLASS(PlacerRunDP);

}  // namespace tcl

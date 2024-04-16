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
 * @File Name: tcl_register_fp.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-10-31
 *
 */
// #include "ScriptEngine.hh"
#include "UserShell.hh"
#include "tcl_ifp.h"

using namespace ieda;

namespace tcl {

int registerCmdFP()
{
  registerTclCmd(TclFpInit, "init_floorplan");
  registerTclCmd(TclFpMakeTracks, "gern_track");
  registerTclCmd(TclFpPlacePins, "auto_place_pins");
  registerTclCmd(TclFpPlacePort, "place_port");
  registerTclCmd(TclFpAutoPlaceIO, "auto_place_io");
  registerTclCmd(TclFpPlaceIOFiller, "place_io_filler");
  registerTclCmd(TclFpAddPlacementBlockage, "add_placement_blockage");
  registerTclCmd(TclFpAddPlacementHalo, "add_placement_halo");
  registerTclCmd(TclFpAddRoutingBlockage, "add_routing_blockage");
  registerTclCmd(TclFpAddRoutingHalo, "add_routing_halo");
  registerTclCmd(TclFpTapCell, "tapcell");

  return EXIT_SUCCESS;
}

}  // namespace tcl
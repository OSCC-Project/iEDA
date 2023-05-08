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
  registerTclCmd(TclFpPlaceIOFiller, "place_io_filler");
  registerTclCmd(TclFpAddPlacementBlockage, "add_placement_blockage");
  registerTclCmd(TclFpAddPlacementHalo, "add_placement_halo");
  registerTclCmd(TclFpAddRoutingBlockage, "add_routing_blockage");
  registerTclCmd(TclFpAddRoutingHalo, "add_routing_halo");
  registerTclCmd(TclFpTapCell, "tapcell");

  return EXIT_SUCCESS;
}

}  // namespace tcl
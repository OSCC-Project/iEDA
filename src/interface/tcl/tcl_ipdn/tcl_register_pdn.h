#pragma once
/**
 * @File Name: tcl_register.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include "UserShell.hh"
#include "tcl_ipdn.h"

using namespace ieda;

namespace tcl {

int registerCmdPDN()
{
  /// pdn
  registerTclCmd(TclPdnAddIO, "add_pdn_io");
  registerTclCmd(TclPdnGlobalConnect, "global_net_connect");
  registerTclCmd(TclPdnPlacePort, "place_pdn_port");
  registerTclCmd(TclPdnCreateGrid, "create_grid");
  registerTclCmd(TclPdnCreateStripe, "create_stripe");
  registerTclCmd(TclPdnConnectLayer, "connect_two_layer");
  registerTclCmd(TclPdnConnectMacro, "connect_macro_pdn");
  registerTclCmd(TclPdnConnectIOPin, "connect_io_pin_to_pdn");
  registerTclCmd(TclPdnConnectStripe, "connect_pdn_stripe");
  registerTclCmd(TclPdnAddSegmentStripe, "add_segment_stripe");
  registerTclCmd(TclPdnAddSegmentVia, "add_segment_via");

  return EXIT_SUCCESS;
}

}  // namespace tcl
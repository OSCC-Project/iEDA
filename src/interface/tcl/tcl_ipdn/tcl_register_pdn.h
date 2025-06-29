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
  registerTclCmd(TclRunPNP, "run_pnp");
  ///如要增加，直接拷贝即可

  return EXIT_SUCCESS;
}

}  // namespace tcl
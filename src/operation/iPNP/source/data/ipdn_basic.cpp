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
#include "ipdn_basic.h"

#include "IdbSpecialNet.h"
#include "IdbSpecialWire.h"

namespace ipdn {

RouteInfo::RouteInfo()
{
  idb::IdbSpecialWire* power = new idb::IdbSpecialWire();
  idb::IdbSpecialWire* ground = new idb::IdbSpecialWire();
  _power_route.insert(std::make_pair("VDD", power));
  _power_route.insert(std::make_pair("VSS", ground));
};

void RouteInfo::add_special_wire_segment(std::string special_net_name, idb::IdbSpecialWireSegment* special_wire_segment)
{
  _power_route[special_net_name]->add_segment(special_wire_segment);
}

}  // namespace ipdn

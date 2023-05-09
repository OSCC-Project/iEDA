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
/**
 * @project		iDB
 * @file		IdbHalo.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe halo & routed halo information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbHalo.h"

namespace idb {

IdbHalo::IdbHalo()
{
  _extend_left = -1;
  _extend_right = -1;
  _extend_top = -1;
  _extend_bottom = -1;
  _is_soft = false;
}

bool IdbHalo::set_bounding_box(IdbRect* instance_bounding_box)
{
  int32_t ll_x = instance_bounding_box->get_low_x() - _extend_left;
  int32_t ll_y = instance_bounding_box->get_low_y() - _extend_bottom;
  int32_t ur_x = instance_bounding_box->get_high_x() + _extend_right;
  int32_t ur_y = instance_bounding_box->get_high_y() + _extend_top;

  return IdbObject::set_bounding_box(ll_x, ll_y, ur_x, ur_y);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbRouteHalo::IdbRouteHalo()
{
  _route_distance = -1;
  _layer_bottom = nullptr;
  _layer_top = nullptr;
}

}  // namespace idb

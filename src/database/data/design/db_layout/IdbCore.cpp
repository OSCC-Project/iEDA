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
 * @file		IdbCore.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Core information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbCore.h"

namespace idb {

IdbCoreSide IdbCore::get_side(IdbCoordinate<int32_t>* coordinate)
{
  IdbRect* bounding_box = get_bounding_box();
  if (bounding_box->containPoint(coordinate)) {
    return IdbCoreSide::kNone;
  }

  if (coordinate->get_x() < bounding_box->get_low_x()) {
    return IdbCoreSide::kLeft;
  } else if (coordinate->get_x() > bounding_box->get_high_x()) {
    return IdbCoreSide::kRight;
  } else if (coordinate->get_y() < bounding_box->get_low_y()) {
    return IdbCoreSide::kBottom;
  } else if (coordinate->get_y() > bounding_box->get_high_y()) {
    return IdbCoreSide::kTop;
  } else {
    return IdbCoreSide::kNone;
  }
}

}  // namespace idb

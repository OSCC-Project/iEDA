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
/*
 * @Author: S.J Chen
 * @Date: 2022-04-01 12:07:29
 * @LastEditTime: 2022-04-01 12:50:51
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/operator/pre_placer/center_place/CenterPlace.cc
 * Contact : https://github.com/sjchanson
 */

#include "CenterPlace.hh"

namespace ipl {

void CenterPlace::runCenterPlace()
{
  Point<int32_t> core_center = _placer_db->get_layout()->get_core_shape().get_center();

  auto inst_list = _placer_db->get_design()->get_instance_list();
  for (auto* inst : inst_list) {
    if (inst->isUnPlaced()) {
      inst->update_center_coordi(core_center);
    }
  }
}

}  // namespace ipl
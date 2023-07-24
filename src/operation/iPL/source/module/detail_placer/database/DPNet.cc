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
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-03-01 17:51:38
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-10 16:05:03
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/database/DPNet.cc
 * @Description:
 *
 *
 */
#include "DPNet.hh"

namespace ipl {

DPNet::DPNet(std::string name) : _dp_net_id(-1), _name(name), _netweight(1.0f), _driver_pin(nullptr)
{
}

DPNet::~DPNet()
{
}

int64_t DPNet::calCurrentHPWL()
{
  if (_pins.empty()) {
    return 0;
  }

  int32_t lower_x = INT32_MAX;
  int32_t lower_y = INT32_MAX;
  int32_t upper_x = INT32_MIN;
  int32_t upper_y = INT32_MIN;

  for (auto* pin : _pins) {
    pin->get_x_coordi() < lower_x ? lower_x = pin->get_x_coordi() : lower_x;
    pin->get_y_coordi() < lower_y ? lower_y = pin->get_y_coordi() : lower_y;
    pin->get_x_coordi() > upper_x ? upper_x = pin->get_x_coordi() : upper_x;
    pin->get_y_coordi() > upper_y ? upper_y = pin->get_y_coordi() : upper_y;
  }

  return (upper_x - lower_x) + (upper_y - lower_y);
}

Rectangle<int32_t> DPNet::obtainBoundingBox()
{
  if (_pins.empty()) {
    return Rectangle<int32_t>(0, 0, 0, 0);
  }

  int32_t lower_x = INT32_MAX;
  int32_t lower_y = INT32_MAX;
  int32_t upper_x = INT32_MIN;
  int32_t upper_y = INT32_MIN;

  for (auto* pin : _pins) {
    pin->get_x_coordi() < lower_x ? lower_x = pin->get_x_coordi() : lower_x;
    pin->get_y_coordi() < lower_y ? lower_y = pin->get_y_coordi() : lower_y;
    pin->get_x_coordi() > upper_x ? upper_x = pin->get_x_coordi() : upper_x;
    pin->get_y_coordi() > upper_y ? upper_y = pin->get_y_coordi() : upper_y;
  }

  return Rectangle<int32_t>(lower_x, lower_y, upper_x, upper_y);
}

}  // namespace ipl
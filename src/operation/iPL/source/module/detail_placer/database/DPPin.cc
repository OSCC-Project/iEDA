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
#include "DPPin.hh"

#include <cstdint>

namespace ipl {

DPPin::DPPin(std::string name)
    : _dp_pin_id(-1),
      _name(name),
      _x_coordi(INT32_MIN),
      _y_coordi(INT32_MIN),
      _offset_x(INT32_MIN),
      _offset_y(INT32_MIN),
      _internal_id(INT32_MIN),
      _net(nullptr),
      _instance(nullptr)
{
}

DPPin::~DPPin()
{
}

}  // namespace ipl
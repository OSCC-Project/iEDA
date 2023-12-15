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
 * @file CtsInstance.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "CtsInstance.hh"

namespace icts {

void CtsInstance::addPin(CtsPin* pin)
{
  pin->set_instance(this);
  _pin_list.push_back(pin);
}

CtsPin* CtsInstance::get_in_pin() const
{
  return find_pin(CtsPinType::kIn);
}

CtsPin* CtsInstance::get_out_pin() const
{
  CtsPin* found_pin = nullptr;
  for (auto* pin : _pin_list) {
    if (pin->is_io()) {
      if (pin->get_pin_type() == CtsPinType::kIn) {
        found_pin = pin;
        break;
      }
    } else {
      if (pin->get_pin_type() == CtsPinType::kOut) {
        found_pin = pin;
        break;
      }
    }
  }
  return found_pin;
  // return find_pin(CtsPinType::kOut);
}

CtsPin* CtsInstance::get_clk_pin() const
{
  return find_pin(CtsPinType::kClock);
}

CtsPin* CtsInstance::find_pin(CtsPinType type) const
{
  CtsPin* found_pin = nullptr;
  for (auto* pin : _pin_list) {
    if (pin->get_pin_type() == type) {
      found_pin = pin;
      break;
    }
  }
  return found_pin;
}

CtsPin* CtsInstance::get_load_pin() const
{
  CtsPin* pin = get_clk_pin();
  return pin ? pin : get_in_pin();
}

}  // namespace icts
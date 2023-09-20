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
 * @file CtsPin.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "CtsPin.hh"

namespace icts {
CtsPin::CtsPin() : _pin_name(""), _instance(nullptr), _net(nullptr)
{
}

CtsPin::CtsPin(const std::string& pin_name) : _pin_name(pin_name), _instance(nullptr), _net(nullptr)
{
}

CtsPin::CtsPin(const std::string& pin_name, CtsPinType pin_type, CtsInstance* instance, CtsNet* net)
    : _pin_name(pin_name), _pin_type(pin_type), _instance(instance), _net(net)
{
}

std::string CtsPin::get_full_name() const
{
  return _instance ? _instance->get_name() + "/" + _pin_name : _pin_name;
}
}  // namespace icts
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

#include "PlanarCoord.hpp"

namespace irt {

enum class PACheckType
{
  kNone = 0,
  kPrefOrien = 1,
  kViaOrien = 2
};

class PANode : public AccessPoint
{
 public:
  PANode() = default;
  explicit PANode(const AccessPoint& access_point) : AccessPoint(access_point) {}
  ~PANode() = default;
  // getter
  irt_int get_net_idx() { return _net_idx; }
  PAPin* get_pin_ptr() { return _pin_ptr; }
  // setter
  void set_net_idx(const irt_int net_idx) { _net_idx = net_idx; }
  void set_pin_ptr(PAPin* pin_ptr) { _pin_ptr = pin_ptr; }

 private:
  irt_int _net_idx = -1;
  PAPin* _pin_ptr = nullptr;
};

}  // namespace irt

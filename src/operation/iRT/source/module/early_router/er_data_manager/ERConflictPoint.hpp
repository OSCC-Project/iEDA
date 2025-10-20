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

#include "ERPin.hpp"

namespace irt {

class ERConflictPoint : public LayerCoord
{
 public:
  ERConflictPoint() = default;
  ~ERConflictPoint() = default;
  // getter
  ERPin* get_er_pin() { return _er_pin; }
  AccessPoint* get_access_point() { return _access_point; }
  // setter
  void set_er_pin(ERPin* er_pin) { _er_pin = er_pin; }
  void set_access_point(AccessPoint* access_point) { _access_point = access_point; }
  // function
 private:
  ERPin* _er_pin = nullptr;
  AccessPoint* _access_point = nullptr;
};

}  // namespace irt

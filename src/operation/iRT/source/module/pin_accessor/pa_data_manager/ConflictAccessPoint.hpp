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

#include "PAPin.hpp"

namespace irt {

class ConflictAccessPoint
{
 public:
  ConflictAccessPoint() = default;
  ~ConflictAccessPoint() = default;
  // getter
  PAPin* get_pa_pin() { return _pa_pin; }
  int32_t get_access_point_idx() const { return _access_point_idx; }
  PlanarCoord& get_real_coord() { return _real_coord; }
  // setter
  void set_pa_pin(PAPin* pa_pin) { _pa_pin = pa_pin; }
  void set_access_point_idx(const int32_t access_point_idx) { _access_point_idx = access_point_idx; }
  void set_real_coord(const PlanarCoord& real_coord) { _real_coord = real_coord; }
  // function
 private:
  PAPin* _pa_pin = nullptr;
  int32_t _access_point_idx = -1;
  PlanarCoord _real_coord;
};

}  // namespace irt

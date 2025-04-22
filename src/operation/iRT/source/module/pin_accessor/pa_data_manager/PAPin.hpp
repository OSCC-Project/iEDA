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

#include "Pin.hpp"

namespace irt {

class PAPin : public Pin
{
 public:
  PAPin() = default;
  explicit PAPin(const Pin& pin) : Pin(pin) {}
  ~PAPin() = default;
  // getter
  std::vector<AccessPoint>& get_access_point_list() { return _access_point_list; }
  PlanarCoord& get_key_grid_coord() { return _key_grid_coord; }
  AccessPoint& get_best_access_point() { return _best_access_point; }
  // setter
  void set_access_point_list(const std::vector<AccessPoint>& access_point_list) { _access_point_list = access_point_list; }
  void set_key_grid_coord(const PlanarCoord& key_grid_coord) { _key_grid_coord = key_grid_coord; }
  void set_best_access_point(const AccessPoint& best_access_point) { _best_access_point = best_access_point; }
  // function
 private:
  std::vector<AccessPoint> _access_point_list;
  PlanarCoord _key_grid_coord;
  AccessPoint _best_access_point;
};

}  // namespace irt

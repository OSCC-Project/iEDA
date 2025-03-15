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

#include "LayerRect.hpp"
#include "PlanarRect.hpp"
#include "Segment.hpp"
#include "Violation.hpp"
#include "DRCShape.hpp"

namespace idrc {

class DRCBox
{
 public:
  DRCBox() = default;
  ~DRCBox() = default;
  // getter
  PlanarRect& get_box_rect() { return _box_rect; }
  std::vector<DRCShape*>& get_drc_shape_list() { return _drc_shape_list; }
  std::vector<Violation>& get_violation_list() { return _violation_list; }
  // setter
  void set_box_rect(const PlanarRect& box_rect) { _box_rect = box_rect; }
  void set_drc_shape_list(const std::vector<DRCShape*>& drc_shape_list) { _drc_shape_list = drc_shape_list; }
  void set_violation_list(const std::vector<Violation>& violation_list) { _violation_list = violation_list; }
  // function

 private:
  PlanarRect _box_rect;
  std::vector<DRCShape*> _drc_shape_list;
  std::vector<Violation> _violation_list;
};

}  // namespace idrc

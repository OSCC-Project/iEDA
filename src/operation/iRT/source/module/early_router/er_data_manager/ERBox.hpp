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

#include "ERBoxId.hpp"
#include "EXTPlanarRect.hpp"

namespace irt {

class ERBox
{
 public:
  ERBox() = default;
  ~ERBox() = default;
  // getter
  EXTPlanarRect& get_box_rect() { return _box_rect; }
  ERBoxId& get_er_box_id() { return _er_box_id; }

  // setter
  void set_box_rect(const EXTPlanarRect& box_rect) { _box_rect = box_rect; }
  void set_er_box_id(const ERBoxId& er_box_id) { _er_box_id = er_box_id; }
  // function

 private:
  EXTPlanarRect _box_rect;
  ERBoxId _er_box_id;
};

}  // namespace irt

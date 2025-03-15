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

#include "DRCShape.hpp"
#include "DRCBox.hpp"

namespace idrc {

class DRCModel
{
 public:
  DRCModel() = default;
  ~DRCModel() = default;
  // getter
  std::vector<DRCShape>& get_drc_shape_list() { return _drc_shape_list; }
  std::vector<DRCBox>& get_drc_box_list() { return _drc_box_list; }

  // setter
  void set_drc_shape_list(const std::vector<DRCShape>& drc_shape_list) { _drc_shape_list = drc_shape_list; }
  void set_drc_box_list(const std::vector<DRCBox>& drc_box_list) { _drc_box_list = drc_box_list; }

 private:
  std::vector<DRCShape> _drc_shape_list;
  std::vector<DRCBox> _drc_box_list;
};

}  // namespace idrc

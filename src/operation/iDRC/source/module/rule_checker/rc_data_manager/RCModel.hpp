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
#include "RCBox.hpp"

namespace idrc {

class RCModel
{
 public:
  RCModel() = default;
  ~RCModel() = default;
  // getter
  std::vector<DRCShape>& get_drc_env_shape_list() { return _drc_env_shape_list; }
  std::vector<DRCShape>& get_drc_result_shape_list() { return _drc_result_shape_list; }
  std::vector<RCBox>& get_rc_box_list() { return _rc_box_list; }

  // setter
  void set_drc_env_shape_list(const std::vector<DRCShape>& drc_env_shape_list) { _drc_env_shape_list = drc_env_shape_list; }
  void set_drc_result_shape_list(const std::vector<DRCShape>& drc_result_shape_list) { _drc_result_shape_list = drc_result_shape_list; }
  void set_rc_box_list(const std::vector<RCBox>& rc_box_list) { _rc_box_list = rc_box_list; }

 private:
  std::vector<DRCShape> _drc_env_shape_list;
  std::vector<DRCShape> _drc_result_shape_list;
  std::vector<RCBox> _rc_box_list;
};

}  // namespace idrc

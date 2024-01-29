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

#include "BoundingBox.hpp"
#include "ConnectType.hpp"
#include "GridMap.hpp"
#include "Guide.hpp"
#include "MTree.hpp"
#include "PhysicalNode.hpp"
#include "Pin.hpp"

namespace irt {

class Net
{
 public:
  Net() = default;
  ~Net() = default;
  // getter
  irt_int get_net_idx() const { return _net_idx; }
  std::string& get_net_name() { return _net_name; }
  ConnectType get_connect_type() const { return _connect_type; }
  // PinAccessor
  std::vector<Pin>& get_pin_list() { return _pin_list; }
  BoundingBox& get_bounding_box() { return _bounding_box; }
  // GlobalRouter
  MTree<Guide>& get_gr_result_tree() { return _gr_result_tree; }
  // TrackAssigner
  std::vector<Segment<LayerCoord>>& get_ta_result_list() { return _ta_result_list; }

  // setter
  void set_net_idx(const irt_int net_idx) { _net_idx = net_idx; }
  void set_net_name(const std::string& net_name) { _net_name = net_name; }
  void set_connect_type(const ConnectType& connect_type) { _connect_type = connect_type; }
  // PinAccessor
  void set_pin_list(const std::vector<Pin>& pin_list) { _pin_list = pin_list; }
  void set_bounding_box(const BoundingBox& bounding_box) { _bounding_box = bounding_box; }
  // GlobalRouter
  void set_gr_result_tree(const MTree<Guide>& gr_result_tree) { _gr_result_tree = gr_result_tree; }
  // TrackAssigner
  void set_ta_result_list(const std::vector<Segment<LayerCoord>>& ta_result_list) { _ta_result_list = ta_result_list; }

 private:
  irt_int _net_idx = -1;
  std::string _net_name;
  ConnectType _connect_type = ConnectType::kNone;
  // PinAccessor
  std::vector<Pin> _pin_list;
  BoundingBox _bounding_box;
  // GlobalRouter
  MTree<Guide> _gr_result_tree;
  // TrackAssigner
  std::vector<Segment<LayerCoord>> _ta_result_list;
  // DetailedRouter
  
};

}  // namespace irt

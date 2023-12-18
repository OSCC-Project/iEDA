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
#include "GuideSegNode.hpp"
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
  Pin& get_driving_pin() { return _driving_pin; }
  BoundingBox& get_bounding_box() { return _bounding_box; }
  // ResourceAllocator
  GridMap<double>& get_ra_cost_map() { return _ra_cost_map; }
  // GlobalRouter
  MTree<Guide>& get_gr_result_tree() { return _gr_result_tree; }
  // TrackAssigner
  std::vector<Segment<LayerCoord>>& get_ta_result_list() { return _ta_result_list; }
  // DetailedRouter
  MTree<PhysicalNode>& get_dr_result_tree() { return _dr_result_tree; }
  // ViolationRepairer
  MTree<PhysicalNode>& get_vr_result_tree() { return _vr_result_tree; }

  // setter
  void set_net_idx(const irt_int net_idx) { _net_idx = net_idx; }
  void set_net_name(const std::string& net_name) { _net_name = net_name; }
  void set_connect_type(const ConnectType& connect_type) { _connect_type = connect_type; }
  // PinAccessor
  void set_pin_list(const std::vector<Pin>& pin_list) { _pin_list = pin_list; }
  void set_driving_pin(const Pin& driving_pin) { _driving_pin = driving_pin; }
  void set_bounding_box(const BoundingBox& bounding_box) { _bounding_box = bounding_box; }
  // ResourceAllocator
  void set_ra_cost_map(const GridMap<double>& ra_cost_map) { _ra_cost_map = ra_cost_map; }
  // GlobalRouter
  void set_gr_result_tree(const MTree<Guide>& gr_result_tree) { _gr_result_tree = gr_result_tree; }
  // TrackAssigner
  void set_ta_result_list(const std::vector<Segment<LayerCoord>>& ta_result_list) { _ta_result_list = ta_result_list; }
  // DetailedRouter
  void set_dr_result_tree(const MTree<PhysicalNode>& dr_result_tree) { _dr_result_tree = dr_result_tree; }
  // ViolationRepairer
  void set_vr_result_tree(const MTree<PhysicalNode>& vr_result_tree) { _vr_result_tree = vr_result_tree; }

 private:
  irt_int _net_idx = -1;
  std::string _net_name;
  ConnectType _connect_type = ConnectType::kNone;
  // PinAccessor
  std::vector<Pin> _pin_list;
  Pin _driving_pin;
  BoundingBox _bounding_box;
  // ResourceAllocator
  GridMap<double> _ra_cost_map;
  // GlobalRouter
  MTree<Guide> _gr_result_tree;
  // TrackAssigner
  std::vector<Segment<LayerCoord>> _ta_result_list;
  // DetailedRouter
  MTree<PhysicalNode> _dr_result_tree;
  // ViolationRepairer
  MTree<PhysicalNode> _vr_result_tree;
};

}  // namespace irt

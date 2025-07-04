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
#include "RoutingLayer.hpp"
#include "SortStatus.hpp"
#include "ViaMasterIdx.hpp"

namespace irt {

class ViaMaster
{
 public:
  ViaMaster() = default;
  ~ViaMaster() = default;
  // getter
  ViaMasterIdx& get_via_master_idx() { return _via_master_idx; }
  std::string& get_via_name() { return _via_name; }
  LayerRect& get_above_enclosure() { return _above_enclosure; }
  Direction get_above_direction() const { return _above_direction; }
  LayerRect& get_below_enclosure() { return _below_enclosure; }
  Direction get_below_direction() const { return _below_direction; }
  std::vector<PlanarRect>& get_cut_shape_list() { return _cut_shape_list; }
  int32_t get_cut_layer_idx() const { return _cut_layer_idx; }
  // const getter
  const LayerRect& get_above_enclosure() const { return _above_enclosure; }
  const LayerRect& get_below_enclosure() const { return _below_enclosure; }
  // setter
  void set_via_master_idx(const ViaMasterIdx& via_master_idx) { _via_master_idx = via_master_idx; }
  void set_via_master_idx(const int32_t below_layer_idx, const int32_t via_idx)
  {
    _via_master_idx.set_below_layer_idx(below_layer_idx);
    _via_master_idx.set_via_idx(via_idx);
  }
  void set_via_name(const std::string& via_name) { _via_name = via_name; }
  void set_above_enclosure(const LayerRect& above_enclosure) { _above_enclosure = above_enclosure; }
  void set_above_direction(const Direction& above_direction) { _above_direction = above_direction; }
  void set_below_enclosure(const LayerRect& below_enclosure) { _below_enclosure = below_enclosure; }
  void set_below_direction(const Direction& below_direction) { _below_direction = below_direction; }
  void set_cut_shape_list(const std::vector<PlanarRect>& cut_shape_list) { _cut_shape_list = cut_shape_list; }
  void set_cut_layer_idx(const int32_t cut_layer_idx) { _cut_layer_idx = cut_layer_idx; }
  // function

 private:
  ViaMasterIdx _via_master_idx;
  std::string _via_name;
  LayerRect _above_enclosure;
  Direction _above_direction = Direction::kNone;
  LayerRect _below_enclosure;
  Direction _below_direction = Direction::kNone;
  std::vector<PlanarRect> _cut_shape_list;
  int32_t _cut_layer_idx;
};

struct CmpViaMaster
{
  bool operator()(const ViaMaster& a, const ViaMaster& b, std::vector<Direction>& direction_list) const
  {
    SortStatus sort_status = SortStatus::kEqual;
    // 层方向优先
    if (sort_status == SortStatus::kEqual) {
      Direction above_layer_direction = direction_list[a.get_above_enclosure().get_layer_idx()];
      Direction below_layer_direction = direction_list[a.get_below_enclosure().get_layer_idx()];
      Direction a_above_direction = (a.get_above_direction() == Direction::kNone ? above_layer_direction : a.get_above_direction());
      Direction a_below_direction = (a.get_below_direction() == Direction::kNone ? below_layer_direction : a.get_below_direction());
      Direction b_above_direction = (b.get_above_direction() == Direction::kNone ? above_layer_direction : b.get_above_direction());
      Direction b_below_direction = (b.get_below_direction() == Direction::kNone ? below_layer_direction : b.get_below_direction());
      if (a_above_direction == above_layer_direction && b_above_direction != above_layer_direction) {
        sort_status = SortStatus::kTrue;
      } else if (a_above_direction != above_layer_direction && b_above_direction == above_layer_direction) {
        sort_status = SortStatus::kFalse;
      } else {
        if (a_below_direction == below_layer_direction && b_below_direction != below_layer_direction) {
          sort_status = SortStatus::kTrue;
        } else if (a_below_direction != below_layer_direction && b_below_direction == below_layer_direction) {
          sort_status = SortStatus::kFalse;
        } else {
          sort_status = SortStatus::kEqual;
        }
      }
    }
    // 宽度升序
    if (sort_status == SortStatus::kEqual) {
      int32_t a_above_enclosure_width = a.get_above_enclosure().getWidth();
      int32_t a_below_enclosure_width = a.get_below_enclosure().getWidth();
      int32_t b_above_enclosure_width = b.get_above_enclosure().getWidth();
      int32_t b_below_enclosure_width = b.get_below_enclosure().getWidth();
      if (a_above_enclosure_width < b_above_enclosure_width) {
        sort_status = SortStatus::kTrue;
      } else if (a_above_enclosure_width > b_above_enclosure_width) {
        sort_status = SortStatus::kFalse;
      } else {
        if (a_below_enclosure_width < b_below_enclosure_width) {
          sort_status = SortStatus::kTrue;
        } else if (a_below_enclosure_width > b_below_enclosure_width) {
          sort_status = SortStatus::kFalse;
        } else {
          sort_status = SortStatus::kEqual;
        }
      }
    }
    // 长度升序
    if (sort_status == SortStatus::kEqual) {
      int32_t a_above_enclosure_length = a.get_above_enclosure().getLength();
      int32_t a_below_enclosure_length = a.get_below_enclosure().getLength();
      int32_t b_above_enclosure_length = b.get_above_enclosure().getLength();
      int32_t b_below_enclosure_length = b.get_below_enclosure().getLength();
      if (a_above_enclosure_length < b_above_enclosure_length) {
        sort_status = SortStatus::kTrue;
      } else if (a_above_enclosure_length > b_above_enclosure_length) {
        sort_status = SortStatus::kFalse;
      } else {
        if (a_below_enclosure_length < b_below_enclosure_length) {
          sort_status = SortStatus::kTrue;
        } else if (a_below_enclosure_length > b_below_enclosure_length) {
          sort_status = SortStatus::kFalse;
        } else {
          sort_status = SortStatus::kEqual;
        }
      }
    }
    // 对称升序(对称值越小越对称)
    if (sort_status == SortStatus::kEqual) {
      // via_master的ll为负数,ur为正数
      int32_t a_above_enclosure_symmetry = std::abs(a.get_above_enclosure().get_ll_x() + a.get_above_enclosure().get_ur_x());
      int32_t b_above_enclosure_symmetry = std::abs(b.get_above_enclosure().get_ll_x() + b.get_above_enclosure().get_ur_x());
      int32_t a_below_enclosure_symmetry = std::abs(a.get_below_enclosure().get_ll_x() + a.get_below_enclosure().get_ur_x());
      int32_t b_below_enclosure_symmetry = std::abs(b.get_below_enclosure().get_ll_x() + b.get_below_enclosure().get_ur_x());
      if (a_above_enclosure_symmetry < b_above_enclosure_symmetry) {
        sort_status = SortStatus::kTrue;
      } else if (a_above_enclosure_symmetry > b_above_enclosure_symmetry) {
        sort_status = SortStatus::kFalse;
      } else {
        if (a_below_enclosure_symmetry < b_below_enclosure_symmetry) {
          sort_status = SortStatus::kTrue;
        } else if (a_below_enclosure_symmetry > b_below_enclosure_symmetry) {
          sort_status = SortStatus::kFalse;
        } else {
          sort_status = SortStatus::kEqual;
        }
      }
    }
    if (sort_status == SortStatus::kTrue) {
      return true;
    } else if (sort_status == SortStatus::kFalse) {
      return false;
    }
    return false;
  }
};

}  // namespace irt

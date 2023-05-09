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
#ifndef IDRC_SRC_DB_DRCRECT_H_
#define IDRC_SRC_DB_DRCRECT_H_

#include <algorithm>

#include "DrcEnum.h"
#include "DrcRectangle.h"

namespace idrc {
class DrcPolygon;
class DrcRect
{
 public:
  DrcRect()
      : _layer_id(-1),
        _net_id(-1),
        _is_fixed(false),
        _owner_type(RectOwnerType::kNone),
        _op_object(nullptr),
        _scope_type(ScopeType::kNone),
        _owner_polygon(nullptr)
  {
  }

  DrcRect(const int net_id, const int lb_x, const int lb_y, const int rt_x, const int rt_y)
      : _net_id(net_id), _owner_type(RectOwnerType::kNone), _scope_type(ScopeType::kNone)
  {
    _rectangle.set_lb(lb_x, lb_y);
    _rectangle.set_rt(rt_x, rt_y);
  }
  DrcRect(const int net_id, const int lb_x, const int lb_y, const int rt_x, const int rt_y, const int layer_id)
      : _layer_id(layer_id), _net_id(net_id), _owner_type(RectOwnerType::kNone), _scope_type(ScopeType::kNone)
  {
    _rectangle.set_lb(lb_x, lb_y);
    _rectangle.set_rt(rt_x, rt_y);
  }

  DrcRect(int layerId, const DrcRectangle<int>& rect)
      : _layer_id(layerId),
        _net_id(-1),
        _is_fixed(false),
        _owner_type(RectOwnerType::kNone),
        _op_object(nullptr),
        _scope_type(ScopeType::kNone),
        _owner_polygon(nullptr),
        _rectangle(rect)
  {
  }
  DrcRect(const DrcRect& other)
  {
    _layer_id = other._layer_id;
    _net_id = other._net_id;
    _is_fixed = other._is_fixed;
    _owner_type = other._owner_type;
    _rectangle = other._rectangle;
    _scope_type = other._scope_type;
  }
  DrcRect(DrcRect&& other)
  {
    _layer_id = std::move(other._layer_id);
    _net_id = std::move(other._net_id);
    _is_fixed = std::move(other._is_fixed);
    _owner_type = std::move(other._owner_type);
    _rectangle = std::move(other._rectangle);
    _scope_type = std::move(other._scope_type);
  }
  ~DrcRect() {}
  DrcRect& operator=(const DrcRect& other)
  {
    _layer_id = other._layer_id;
    _net_id = other._net_id;
    _is_fixed = other._is_fixed;
    _owner_type = other._owner_type;
    _rectangle = other._rectangle;
    _scope_type = other._scope_type;
    return *this;
  }
  DrcRect& operator=(DrcRect&& other)
  {
    _layer_id = std::move(other._layer_id);
    _net_id = std::move(other._net_id);
    _is_fixed = std::move(other._is_fixed);
    _owner_type = std::move(other._owner_type);
    _rectangle = std::move(other._rectangle);
    _scope_type = std::move(other._scope_type);
    return *this;
  }

  // setter
  // void set_cut_class(std::vector<std::shared_ptr<idb::cutlayer::Lef58Cutclass>>& cut_class_list)
  // {
  //   for (auto& cut_class : cut_class_list) {
  //     int width = cut_class->get_via_width();
  //     int length = cut_class->get_via_length();
  //     if (getWidth() == width && getLength() == length) {
  //     }
  //   }
  // }
  void set_layer_id(int layer_id) { _layer_id = layer_id; }
  void set_net_id(int net_id) { _net_id = net_id; }
  void set_is_fixed(bool isFixed) { _is_fixed = isFixed; }
  void set_owner_type(RectOwnerType owner_type) { _owner_type = owner_type; }
  void set_rectangle(const DrcRectangle<int>& rect) { _rectangle = rect; }
  void set_lb(const int lb_x, const int lb_y) { _rectangle.set_lb(lb_x, lb_y); }
  void set_rt(const int rt_x, const int rt_y) { _rectangle.set_rt(rt_x, rt_y); }
  void set_coordinate(const int lb_x, const int lb_y, const int rt_x, const int rt_y)
  {
    _rectangle.set_lb(lb_x, lb_y);
    _rectangle.set_rt(rt_x, rt_y);
  }
  // void set_via_idx(int idx) { _via_idx = idx; }
  void set_owner_polygon(DrcPolygon* polygon) { _owner_polygon = polygon; }
  void set_op_object(void* irt_object) { _op_object = irt_object; }
  void set_scope_owner(void* in) { _scope_owner = in; }
  void set_min_scope(DrcRect* in) { _min_scope = in; }
  void set_max_scope(DrcRect* in) { _max_scope = in; }
  void set_is_max_scope(bool in) { _is_max_scope = in; }
  void setScopeType(ScopeType in) { _scope_type = in; }

  // getter
  int get_layer_id() const { return _layer_id; }
  int get_net_id() const { return _net_id; }
  // int get_via_idx() const { return _via_idx; }
  bool is_fixed() const { return _is_fixed; }
  RectOwnerType get_owner_type() const { return _owner_type; }
  DrcRectangle<int> get_rectangle() const { return _rectangle; }
  DrcPolygon* get_owner_polygon() { return _owner_polygon; }
  void* get_op_object() { return _op_object; }
  void* get_scope_owner() { return _scope_owner; }
  DrcRect* get_min_scope() { return _min_scope; }
  DrcRect* get_max_scope() { return _max_scope; }
  ScopeType getScopeType() { return _scope_type; }
  bool is_scope_max() { return _is_max_scope; }

  bool isSegmentRect() const { return _owner_type == RectOwnerType::kSegment; }
  bool isPinRect() const { return _owner_type == RectOwnerType::kPin; }
  bool isViaMetalRect() const { return _owner_type == RectOwnerType::kViaMetal; }
  bool isBlockRect() const { return _owner_type == RectOwnerType::kBlockage; }
  bool isSpotMark() const { return _owner_type == RectOwnerType::kSpotMark; }

  // function
  int get_left() const { return std::min(_rectangle.get_lb_x(), _rectangle.get_rt_x()); }
  int get_right() const { return std::max(_rectangle.get_lb_x(), _rectangle.get_rt_x()); }
  int get_bottom() const { return std::min(_rectangle.get_lb_y(), _rectangle.get_rt_y()); }
  int get_top() const { return std::max(_rectangle.get_lb_y(), _rectangle.get_rt_y()); }
  int getXSpan() const { return get_right() - get_left(); }
  int getYSpan() const { return get_top() - get_bottom(); }
  bool isHorizontal() const { return getXSpan() >= getYSpan(); }
  bool isVertical() const { return getXSpan() <= getYSpan(); }
  int getWidth() const { return std::min(getXSpan(), getYSpan()); }
  int getLength() const { return std::max(getXSpan(), getYSpan()); }

 private:
  int _layer_id = -1;
  int _net_id = -1;
  bool _is_fixed = false;
  bool _is_cut = false;
  bool _is_max_scope = false;
  std::string _cut_class_name;
  void* _scope_owner = nullptr;
  DrcRect* _min_scope = nullptr;
  DrcRect* _max_scope = nullptr;
  RectOwnerType _owner_type;
  void* _op_object = nullptr;

  ScopeType _scope_type;
  DrcPolygon* _owner_polygon = nullptr;
  DrcRectangle<int> _rectangle;
};
}  // namespace idrc

#endif
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

#ifndef IMP_INSTANCE_H
#define IMP_INSTANCE_H
#include "Cell.hh"
#include "Object.hh"
namespace imp {
enum class INSTANCE_TYPE
{
  kNone,
  kNormal,
  kCross,
  kOutside,
  kPseudo
};

enum class INSTANCE_STATE
{
  kNone,
  kUnPlaced,
  kPlaced,
  kFixed
};

class Instance final : public Object
{
 public:
  Instance(std::string name, std::shared_ptr<Cell> cell = nullptr, std::shared_ptr<Object> parent = nullptr);
  ~Instance() = default;

  virtual OBJ_TYPE object_type() const override { return OBJ_TYPE::kInstance; }
  virtual geo::box<int32_t> boundingbox() const override { return Object::transform(_cell->get_shape()); }

  // setter
  void set_cell_master(std::shared_ptr<Cell> cell) { _cell = cell; }
  void set_type(INSTANCE_TYPE type) { _type = type; }
  void set_state(INSTANCE_STATE state) { _state = state; }
  void set_extend_left(int32_t value) { _extend_left = value; }
  void set_extend_right(int32_t value) { _extend_right = value; }
  void set_extend_top(int32_t value) { _extend_top = value; }
  void set_extend_bottom(int32_t value) { _extend_bottom = value; }
  void set_extend(int32_t, int32_t, int32_t, int32_t);
  void set_halo_min_corner(int32_t halo_lx, int32_t halo_ly) { set_min_corner(halo_lx + _extend_left, halo_ly + _extend_bottom); }
  void set_halo_min_corner(const geo::point<int32_t>& point) { set_min_corner(point.x() + _extend_left, point.y() + _extend_bottom); }

  int32_t get_halo_width() const { return _cell->get_width() + _extend_left + _extend_right; }
  int32_t get_halo_height() const { return _cell->get_height() + _extend_bottom + _extend_top; }
  geo::point<int32_t> get_halo_min_corner() const
  {
    auto min_corner = get_min_corner();
    return geo::point<int32_t>(min_corner.x() - _extend_left, min_corner.y() - _extend_bottom);
  }

  double get_area() const { return double(geo::width(_cell->get_shape())) * geo::height(_cell->get_shape()); }
  const Cell& get_cell_master() const { return *_cell; }
  const int32_t get_extend_left() const { return _extend_left; }
  const int32_t get_extend_right() const { return _extend_right; }
  const int32_t get_extend_top() const { return _extend_top; }
  const int32_t get_extend_bottom() const { return _extend_bottom; }

  bool isNormal() const { return _type == INSTANCE_TYPE::kNormal; }
  bool isOutside() const { return _type == INSTANCE_TYPE::kOutside; }
  bool isPseudo() const { return _type == INSTANCE_TYPE::kPseudo; }
  bool isUnPlaced() const { return _state == INSTANCE_STATE::kUnPlaced; }
  bool isPlaced() const { return _state == INSTANCE_STATE::kPlaced; }
  bool isFixed() const { return _state == INSTANCE_STATE::kFixed; }

 private:
  int32_t _extend_left = 0;
  int32_t _extend_right = 0;
  int32_t _extend_top = 0;
  int32_t _extend_bottom = 0;
  INSTANCE_TYPE _type;
  INSTANCE_STATE _state;
  std::shared_ptr<Cell> _cell;
};

inline Instance::Instance(std::string name, std::shared_ptr<Cell> cell, std::shared_ptr<Object> parent)
    : Object::Object(name, parent), _cell(cell)
{
}
inline void Instance::set_extend(int32_t left, int32_t right, int32_t bottom, int32_t top)
{
  set_extend_left(left);
  set_extend_right(right);
  set_extend_top(bottom);
  set_extend_bottom(top);
}
}  // namespace imp

#endif
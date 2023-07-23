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
/*
 * @Author: S.J Chen
 * @Date: 2022-01-20 18:34:47
 * @LastEditTime: 2022-03-04 15:51:14
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/database/Instance.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_INSTANCE_H
#define IPL_INSTANCE_H

#include <string>
#include <vector>

#include "Cell.hh"
#include "Orient.hh"
#include "Pin.hh"
#include "Rectangle.hh"
#include "module/logger/Log.hh"

namespace ipl {

class Region;

enum class INSTANCE_TYPE
{
  kNone,
  kNormal,
  kCross,
  kOutside,
  kFakeInstance
};

enum class INSTANCE_STATE
{
  kNone,
  KUnPlaced,
  kPlaced,
  kFixed
};

class Instance
{
 public:
  Instance() = delete;
  explicit Instance(std::string inst_name)
      : _inst_id(-1),
        _instance_name(std::move(inst_name)),
        _cell_master(nullptr),
        _belong_region(nullptr),
        _instance_type(INSTANCE_TYPE::kNone),
        _instance_state(INSTANCE_STATE::kNone)
  {
  }
  Instance(const Instance& other) = default;
  Instance(Instance&& other) noexcept
  {
    _instance_name = std::move(other._instance_name);
    _cell_master = other._cell_master;
    _shape = std::move(other._shape);
    _pin_list = std::move(other._pin_list);
    _belong_region = other._belong_region;
    _orient = other._orient;
    _instance_type = other._instance_type;
    _instance_state = other._instance_state;

    other._instance_name = "";
    other._cell_master = nullptr;
    other._pin_list.clear();
    other._belong_region = nullptr;
    other._orient = Orient::kNone;
    other._instance_type = INSTANCE_TYPE::kNone;
    other._instance_state = INSTANCE_STATE::kNone;
  }
  ~Instance() = default;

  Instance& operator=(const Instance&) = default;
  Instance& operator=(Instance&&) = delete;

  // getter.
  int32_t get_inst_id() const { return _inst_id; }
  std::string get_name() const { return _instance_name; }
  Cell* get_cell_master() const { return _cell_master; }
  Region* get_belong_region() const { return _belong_region; }
  Orient get_orient() const { return _orient; }
  INSTANCE_STATE get_instance_state() const { return _instance_state; }

  Rectangle<int32_t> get_shape() const { return _shape; }
  int32_t get_shape_width() { return _shape.get_width(); }
  int32_t get_shape_height() { return _shape.get_height(); }

  Point<int32_t> get_coordi() { return _shape.get_lower_left(); }
  Point<int32_t> get_center_coordi() { return _shape.get_center(); }

  const std::vector<Pin*>& get_pins() const { return _pin_list; }
  std::vector<Pin*> get_inpins() const;
  std::vector<Pin*> get_outpins() const;

  bool isNormalInstance() { return _instance_type == INSTANCE_TYPE::kNormal; }
  bool isOutsideInstance() { return _instance_type == INSTANCE_TYPE::kOutside; }
  bool isFakeInstance() { return _instance_type == INSTANCE_TYPE::kFakeInstance; }

  bool isUnPlaced() { return _instance_state == INSTANCE_STATE::KUnPlaced; }
  bool isPlaced() { return _instance_state == INSTANCE_STATE::kPlaced; }
  bool isFixed() { return _instance_state == INSTANCE_STATE::kFixed; }

  // setter.
  void set_inst_id(int32_t id) { _inst_id = id; }
  void set_cell_master(Cell* cell_master) { _cell_master = cell_master; }
  void set_shape(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y) { _shape.set_rectangle(ll_x, ll_y, ur_x, ur_y); }
  void set_instance_type(INSTANCE_TYPE type) { _instance_type = type; }
  void set_instance_state(INSTANCE_STATE state) { _instance_state = state; }
  void add_pin(Pin* pin) { _pin_list.push_back(pin); }
  void set_belong_region(Region* region) { _belong_region = region; }
  void set_orient(Orient orient);

  // function.
  void update_coordi(int32_t lower_x, int32_t lower_y);
  void update_coordi(const Point<int32_t> lower_point) { this->update_coordi(lower_point.get_x(), lower_point.get_y()); }
  void update_center_coordi(int32_t center_x, int32_t center_y);
  void update_center_coordi(const Point<int32_t> center_coordi)
  {
    this->update_center_coordi(center_coordi.get_x(), center_coordi.get_y());
  }

 private:
  int32_t _inst_id;
  std::string _instance_name;
  Cell* _cell_master;

  Rectangle<int32_t> _shape;
  std::vector<Pin*> _pin_list;
  Region* _belong_region;
  Orient _orient;

  INSTANCE_TYPE _instance_type;
  INSTANCE_STATE _instance_state;

  void update_pins_coordi();
};

inline std::vector<Pin*> Instance::get_inpins() const
{
  std::vector<Pin*> inpins;
  for (auto* pin : _pin_list) {
    if (pin->isInstanceInput() || pin->isInstanceInputOutput()) {
      inpins.push_back(pin);
    }
  }
  return inpins;
}

inline std::vector<Pin*> Instance::get_outpins() const
{
  std::vector<Pin*> outpins;
  for (auto* pin : _pin_list) {
    if (pin->isInstanceOutput() || pin->isInstanceInputOutput()) {
      outpins.push_back(pin);
    }
  }
  return outpins;
}

inline void Instance::set_orient(Orient orient)
{
  LOG_ERROR_IF(!_cell_master) << _instance_name + " is not setting the cell master!";
  _orient = orient;

  // set the inst shape.
  int32_t ll_x = _shape.get_ll_x();
  int32_t ll_y = _shape.get_ll_y();
  int32_t width = _cell_master->get_width();
  int32_t height = _cell_master->get_height();
  switch (_orient) {
    case Orient::kNone:
    case Orient::kN_R0:
    case Orient::kS_R180:
    case Orient::kFN_MY:
    case Orient::kFS_MX: {
      this->set_shape(ll_x, ll_y, ll_x + width, ll_y + height);
      break;
    }
    case Orient::kW_R90:
    case Orient::kE_R270:
    case Orient::kFW_MX90:
    case Orient::kFE_MY90: {
      this->set_shape(ll_x, ll_y, ll_x + height, ll_y + width);
      break;
    }
    default:
      break;
  }

  // update pins coordi.
  update_pins_coordi();
}

inline void Instance::update_pins_coordi()
{
  int32_t inst_center_x = this->get_center_coordi().get_x();
  int32_t inst_center_y = this->get_center_coordi().get_y();
  for (auto* pin : _pin_list) {
    int32_t origin_offset_x = pin->get_offset_coordi().get_x();
    int32_t origin_offset_y = pin->get_offset_coordi().get_y();

    int32_t modify_offset_x = INT32_MAX;
    int32_t modify_offset_y = INT32_MAX;

    if (_orient == Orient::kN_R0) {
      modify_offset_x = origin_offset_x;
      modify_offset_y = origin_offset_y;
    } else if (_orient == Orient::kW_R90) {
      modify_offset_x = (-1) * origin_offset_y;
      modify_offset_y = origin_offset_x;
    } else if (_orient == Orient::kS_R180) {
      modify_offset_x = (-1) * origin_offset_x;
      modify_offset_y = (-1) * origin_offset_y;
    } else if (_orient == Orient::kFW_MX90) {
      modify_offset_x = origin_offset_y;
      modify_offset_y = origin_offset_x;
    } else if (_orient == Orient::kFN_MY) {
      modify_offset_x = (-1) * origin_offset_x;
      modify_offset_y = origin_offset_y;
    } else if (_orient == Orient::kFE_MY90) {
      modify_offset_x = (-1) * origin_offset_y;
      modify_offset_y = (-1) * origin_offset_x;
    } else if (_orient == Orient::kFS_MX) {
      modify_offset_x = origin_offset_x;
      modify_offset_y = (-1) * origin_offset_y;
    } else if (_orient == Orient::kE_R270) {
      modify_offset_x = origin_offset_y;
      modify_offset_y = (-1) * origin_offset_x;
    } else {
      LOG_WARNING << this->get_name() + " has not the orient!";
    }

    int32_t pin_cx = inst_center_x + modify_offset_x;
    int32_t pin_cy = inst_center_y + modify_offset_y;
    pin->set_center_coordi(pin_cx, pin_cy);
  }
}

inline void Instance::update_coordi(int32_t lower_x, int32_t lower_y)
{
  _shape.set_rectangle(lower_x, lower_y, lower_x + this->get_shape_width(), lower_y + this->get_shape_height());

  // update pins coordi.
  update_pins_coordi();
}

inline void Instance::update_center_coordi(int32_t center_x, int32_t center_y)
{
  _shape.set_center(center_x, center_y);

  // update pins coordi.
  update_pins_coordi();
}

}  // namespace ipl

#endif
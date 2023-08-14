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
/**
 * @file Inst.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
namespace icts {
class Pin;
}
#include <string>

#include "CTSAPI.hpp"
#include "CtsInstance.h"
#include "Enum.hh"
#include "Pin.hh"
namespace icts {

class Inst
{
 public:
  // basic
  Inst(CtsInstance* cts_inst, const InstType& type = InstType::kSink) : _cts_inst(cts_inst), _type(type)
  {
    _name = _cts_inst->get_name();
    _location = _cts_inst->get_location();
    init();
  }
  // node to buffer
  Inst(CtsInstance* cts_inst, Node* driver_node) : _cts_inst(cts_inst)
  {
    _name = _cts_inst->get_name();
    _location = _cts_inst->get_location();
    _type = InstType::kBuffer;
    init(driver_node);
  }
  ~Inst() { release(); }
  // get
  const std::string& get_name() const { return _name; }
  const Point& get_location() const { return _location; }
  CtsInstance* get_cts_inst() const { return _cts_inst; }
  const InstType& get_type() const { return _type; }
  const std::string& get_cell_master() const { return _cell_master; }
  const double& get_insert_delay() const { return _insert_delay; }
  Pin* get_driver_pin() const { return _driver_pin; }
  Pin* get_load_pin() const { return _load_pin; }

  const double& getCapLoad() const;
  const double& getCapOut() const;
  // set
  void set_name(const std::string& name) { _name = name; }
  void set_location(const Point& location)
  {
    _location = location;
    updatePinLocation(location);
    _cts_inst->set_location(location);
  }
  void set_cts_inst(CtsInstance* cts_inst) { _cts_inst = cts_inst; }
  void set_type(const InstType& type) { _type = type; }
  void set_cell_master(const std::string& cell_master)
  {
    _cell_master = cell_master;
    _cts_inst->set_cell_master(cell_master);
  }
  void set_insert_delay(const double& insert_delay) { _insert_delay = insert_delay; }
  void set_driver_pin(Pin* driver_pin) { _driver_pin = driver_pin; }
  void set_load_pin(Pin* load_pin) { _load_pin = load_pin; }

  // bool
  bool isSink() const { return _type == InstType::kSink; }
  bool isBuffer() const { return _type == InstType::kBuffer; }

 private:
  void init(Node* driver_node = nullptr);
  void release();
  void updatePinLocation(const Point& location);
  std::string _name = "";
  Point _location = Point(-1, -1);
  CtsInstance* _cts_inst = nullptr;
  InstType _type = InstType::kSink;
  std::string _cell_master = "";
  double _insert_delay = 0;
  Pin* _driver_pin = nullptr;
  Pin* _load_pin = nullptr;
};
}  // namespace icts
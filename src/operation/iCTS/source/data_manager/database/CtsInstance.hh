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
 * @file CtsInstance.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include <cassert>
#include <string>
#include <vector>

#include "CtsPin.hh"
#include "CtsPoint.hh"
#include "DesignObject.hh"
namespace icts {
class CtsPin;
enum class CtsPinType;

enum class CtsInstanceType
{
  kBuffer,     // up level buffer
  kSink,       // flip flop
  kClockGate,  // clock gate
  kBufferSink,
  kLogic,
  kMux
};

class CtsInstance : public DesignObject
{
 public:
  CtsInstance() = default;
  CtsInstance(const std::string& name) : _name(name) {}
  CtsInstance(const std::string& name, const std::string& cell_master, CtsInstanceType type, Point location)
      : _name(name), _cell_master(cell_master), _type(type), _location(location)
  {
  }

  // getter
  const std::string& get_name() { return _name; }
  std::string get_cell_master() const { return _cell_master; }
  CtsInstanceType get_type() const { return _type; }
  Point get_location() const { return _location; }
  std::vector<CtsPin*>& get_pin_list() { return _pin_list; }
  CtsPin* get_in_pin() const;
  CtsPin* get_out_pin() const;
  CtsPin* get_clk_pin() const;
  CtsPin* get_load_pin() const;
  int get_level() const { return _level; }
  bool is_virtual() { return _b_virtual; }

  // setter
  void set_name(const std::string& name) { _name = name; }
  void set_cell_master(const std::string& cell_master) { _cell_master = cell_master; }
  void set_type(CtsInstanceType type) { _type = type; }
  void set_location(const Point& location) { _location = location; }
  void set_location(int x, int y)
  {
    _location.x(x);
    _location.y(y);
  }
  void set_virtual(bool b_virtual) { _b_virtual = b_virtual; }
  void set_level(const int& level) { _level = level; }
  void addPin(CtsPin* pin);

  // bool
  bool isSink() const { return _type == CtsInstanceType::kSink; }
  bool isBuffer() const { return _type == CtsInstanceType::kBuffer; }
  bool isMux() const { return _type == CtsInstanceType::kMux; }

 private:
  CtsPin* find_pin(CtsPinType type) const;

 private:
  std::string _name;
  std::string _cell_master;
  std::vector<CtsPin*> _pin_list;
  CtsInstanceType _type = CtsInstanceType::kSink;
  Point _location;
  int _level = 0;
  bool _b_virtual = false;
};

}  // namespace icts

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
 * @file CtsPin.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include <string>

#include "CtsInstance.hh"
#include "CtsNet.hh"
#include "CtsPoint.hh"
#include "DesignObject.hh"

namespace icts {

class CtsInstance;
class CtsNet;

enum class CtsPinType
{
  kClock,
  kIn,
  kOut,
  kInOut,
  kOther
};

class CtsPin : public DesignObject
{
 public:
  CtsPin();
  explicit CtsPin(const std::string& pin_name);
  CtsPin(const std::string& pin_name, CtsPinType pin_type, CtsInstance* instance, CtsNet* net);
  CtsPin(const CtsPin& pin) = default;
  ~CtsPin() = default;

  // getter
  const std::string& get_pin_name() const { return _pin_name; }
  std::string get_full_name() const;
  CtsPinType get_pin_type() const { return _pin_type; }
  Point get_location() const { return _location; }
  CtsInstance* get_instance() const { return _instance; }
  CtsNet* get_net() const { return _net; }
  bool is_io() { return _b_io; }

  // setter
  void set_pin_name(const std::string& pin_name) { _pin_name = pin_name; }
  void set_pin_type(CtsPinType pin_type) { _pin_type = pin_type; }
  void set_location(const Point& location) { _location = location; }
  void set_instance(CtsInstance* instance) { _instance = instance; }
  void set_net(CtsNet* net) { _net = net; }
  void set_io(bool b_io = false) { _b_io = b_io; }

 private:
  std::string _pin_name;
  CtsPinType _pin_type = CtsPinType::kOther;
  Point _location;
  CtsInstance* _instance = nullptr;
  CtsNet* _net = nullptr;
  bool _b_io = false;
};

}  // namespace icts

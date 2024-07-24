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
#ifndef SRC_EVALUATOR_SOURCE_CONGESTION_DATABASE_CONGNET_HPP_
#define SRC_EVALUATOR_SOURCE_CONGESTION_DATABASE_CONGNET_HPP_

#include "CongInst.hpp"
#include "CongPin.hpp"

namespace eval {

class CongNet
{
 public:
  CongNet() = default;
  ~CongNet() = default;

  // getter
  std::string get_name() const { return _name; }
  std::vector<CongPin*> get_pin_list() const { return _pin_list; }
  std::vector<CongInst*> get_inst_list() const { return _inst_list; }
  int64_t get_lx();
  int64_t get_ly();
  int64_t get_ux();
  int64_t get_uy();
  int64_t get_width() { return _ux - _lx; }
  int64_t get_height() { return _uy - _ly; }
  NET_CONNECT_TYPE get_connect_type() const { return _connect_type; }

  // setter
  void set_name(const std::string& name) { _name = name; }
  void set_pin_list(const std::vector<CongPin*>& pin_list) { _pin_list = pin_list; }
  void set_connect_type(const NET_CONNECT_TYPE& type) { _connect_type = type; }

  // adder
  void add_pin(const int64_t& x, const int64_t& y, const std::string& name);
  void add_pin(CongPin* pin) { _pin_list.push_back(pin); }
  void add_inst(CongInst* inst) { _inst_list.push_back(inst); }

 private:
  std::string _name;
  NET_CONNECT_TYPE _connect_type;
  std::vector<CongPin*> _pin_list;
  std::vector<CongInst*> _inst_list;

  int64_t _lx;
  int64_t _ly;
  int64_t _ux;
  int64_t _uy;
};

inline int64_t CongNet::get_lx()
{
  _lx = INT_MAX;
  for (auto& pin : _pin_list) {
    if (pin->get_x() < _lx) {
      _lx = pin->get_x();
    }
  }
  if (_lx < 0) {
    _lx = 0;
  }
  return _lx;
}

inline int64_t CongNet::get_ly()
{
  _ly = INT_MAX;
  for (auto& pin : _pin_list) {
    if (pin->get_y() < _ly) {
      _ly = pin->get_y();
    }
  }
  if (_ly < 0) {
    _ly = 0;
  }
  return _ly;
}

inline int64_t CongNet::get_ux()
{
  _ux = 0;
  for (auto& pin : _pin_list) {
    if (pin->get_x() > _ux) {
      _ux = pin->get_x();
    }
  }
  return _ux;
}

inline int64_t CongNet::get_uy()
{
  _uy = 0;
  for (auto& pin : _pin_list) {
    if (pin->get_y() > _uy) {
      _uy = pin->get_y();
    }
  }
  return _uy;
}

inline void CongNet::add_pin(const int64_t& x, const int64_t& y, const std::string& name)
{
  CongPin* pin = new CongPin();
  pin->set_x(x);
  pin->set_y(y);
  pin->set_name(name);
  _pin_list.push_back(pin);
}

}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_CONGESTION_DATABASE_CONGNET_HPP_

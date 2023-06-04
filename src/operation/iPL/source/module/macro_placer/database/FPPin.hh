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

#include "FPInst.hh"
namespace ipl::imp {
class FPNet;
class FPPin
{
 public:
  FPPin();
  ~FPPin();

  // setter
  void set_name(std::string name) { _name = name; }
  void set_x(int32_t x) { _coordinate->set_x(x); }
  void set_y(int32_t y) { _coordinate->set_y(y); }
  void set_instance(FPInst* instance) { _instance = instance; }
  void set_net(FPNet* net) { _net = net; }
  void set_io_pin() { _is_io_pin = true; }
  void set_weight(float weight) { _weight = weight; }

  // getter
  std::string get_name() const { return _name; }
  FPInst* get_instance() const { return _instance; }
  FPNet* get_net() const { return _net; }
  int32_t get_x() const;
  int32_t get_y() const;
  bool is_io_pin() const { return _is_io_pin; }
  float get_weight() const { return _weight; }
  int32_t get_offset_x() const;
  int32_t get_offset_y() const;
  int32_t get_orig_xoff() const { return _coordinate->get_x(); }
  int32_t get_orig_yoff() const { return _coordinate->get_y(); }

 private:
  std::string _name;
  bool _is_io_pin;
  Coordinate* _coordinate;
  FPInst* _instance;  // index of the macro in which the Pin is located
  FPNet* _net;        // index of net to which Pin is attached
  float _weight;      // instance's area is the weight
};

}  // namespace ipl::imp
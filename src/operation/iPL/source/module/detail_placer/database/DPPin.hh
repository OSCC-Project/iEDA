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
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-03-01 17:57:09
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-10 10:40:33
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/database/DPPin.hh
 * @Description: Pin of detail placement.
 *
 *
 */
#ifndef IPL_DPPIN_H
#define IPL_DPPIN_H

#include <string>

namespace ipl {

class DPInstance;
class DPNet;

class DPPin
{
 public:
  DPPin() = delete;
  explicit DPPin(std::string name);
  DPPin(const DPPin&) = delete;
  DPPin(DPPin&&) = delete;
  ~DPPin();

  DPPin& operator=(const DPPin&) = delete;
  DPPin& operator=(DPPin&&) = delete;

  // getter
  int32_t get_pin_id() const { return _dp_pin_id; }
  std::string get_name() const { return _name; }
  int32_t get_x_coordi() const { return _x_coordi; }
  int32_t get_y_coordi() const { return _y_coordi; }
  int32_t get_offset_x() const { return _offset_x; }
  int32_t get_offset_y() const { return _offset_y; }
  int32_t get_internal_id() const { return _internal_id; }
  DPNet* get_net() const { return _net; }
  DPInstance* get_instance() const { return _instance; }

  // setter
  void set_pin_id(int32_t id) { _dp_pin_id = id; }
  void set_x_coordi(int32_t x_coordi) { _x_coordi = x_coordi; }
  void set_y_coordi(int32_t y_coordi) { _y_coordi = y_coordi; }
  void set_offset_x(int32_t offset_x) { _offset_x = offset_x; }
  void set_offset_y(int32_t offset_y) { _offset_y = offset_y; }
  void set_internal_id(int32_t id) { _internal_id = id; }
  void set_net(DPNet* net) { _net = net; }
  void set_instance(DPInstance* instance) { _instance = instance; }

  // function

 private:
  int32_t _dp_pin_id;
  std::string _name;
  int32_t _x_coordi;
  int32_t _y_coordi;
  int32_t _offset_x;
  int32_t _offset_y;
  int32_t _internal_id;
  DPNet* _net;
  DPInstance* _instance;
};
}  // namespace ipl
#endif
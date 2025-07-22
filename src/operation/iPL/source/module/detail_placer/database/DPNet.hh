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
 * @Date: 2023-03-01 17:51:38
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-10 11:16:32
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/database/DPNet.hh
 * @Description: Net of detail placement
 *
 *
 */
#ifndef IPL_DPNET_H
#define IPL_DPNET_H

#include <cstdint>
#include <string>
#include <vector>

#include "DPPin.hh"
#include "data/Rectangle.hh"

namespace ipl {

enum class DPNET_TYPE
{
  kNone,
  kClock,
  kSignal
};

enum class DPNET_STATE
{
  kNone,
  kNormal,
  kDontCare
};

class DPNet
{
 public:
  DPNet() = delete;
  explicit DPNet(std::string name);
  DPNet(const DPNet&) = delete;
  DPNet(DPNet&&) = delete;
  ~DPNet();

  DPNet& operator=(const DPNet&) = delete;
  DPNet& operator=(DPNet&&) = delete;

  // getter
  int32_t get_net_id() const { return _dp_net_id; }
  std::string get_name() const { return _name; }
  DPNET_TYPE get_net_type() const { return _type; }
  DPNET_STATE get_net_state() const { return _state; }
  float get_netwight() const { return _netweight; }
  DPPin* get_driver_pin() const { return _driver_pin; }
  const std::vector<DPPin*>& get_pins() const { return _pins; }

  // setter
  void set_net_id(int32_t id) { _dp_net_id = id; }
  void set_driver_pin(DPPin* pin) { _driver_pin = pin; }
  void add_pin(DPPin* pin) { _pins.push_back(pin); }
  void set_netweight(float weight) { _netweight = weight; }
  void set_net_type(DPNET_TYPE net_type) { _type = net_type; }
  void set_net_state(DPNET_STATE net_state) { _state = net_state; }

  // function
  int64_t calCurrentHPWL();
  Rectangle<int32_t> obtainBoundingBox();

 private:
  int32_t _dp_net_id;
  std::string _name;
  DPNET_TYPE _type;
  DPNET_STATE _state;
  float _netweight;
  DPPin* _driver_pin;
  std::vector<DPPin*> _pins;
};
}  // namespace ipl
#endif
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
 * @Date: 2022-01-20 21:57:19
 * @LastEditTime: 2022-12-05 11:24:18
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/imp/src/database/Net.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IMP_NET_H
#define IMP_NET_H

#include <string>
#include <vector>

namespace imp {

enum class NET_TYPE
{
  kNone,
  kSignal,
  kClock,
  kReset,
  kFakeNet
};

enum class NET_STATE
{
  kNone,
  kNormal,
  kDontCare
};

class Net
{
 public:
  Net() = delete;
  explicit Net(std::string name);
  Net(const Net&) = delete;
  Net(Net&&) = delete;
  ~Net() = default;

  Net& operator=(const Net&) = delete;
  Net& operator=(Net&&) = delete;

  // getter.
  std::string get_name() const { return _name; }
  float get_net_weight() const { return _netweight; }
  NET_TYPE get_net_type() const { return _net_type; }
  NET_STATE get_net_state() const { return _net_state; }

  bool isSignalNet() const { return _net_type == NET_TYPE::kSignal; }
  bool isClockNet() const { return _net_type == NET_TYPE::kClock; }
  bool isResetNet() const { return _net_type == NET_TYPE::kReset; }
  bool isFakeNet() const { return _net_type == NET_TYPE::kFakeNet; }

  bool isNormalStateNet() const { return _net_state == NET_STATE::kNormal; }
  bool isDontCareNet() const { return _net_state == NET_STATE::kDontCare; }

  // setter.
  void set_netweight(float weight) { _netweight = weight; }
  void set_net_type(NET_TYPE type) { _net_type = type; }
  void set_net_state(NET_STATE state) { _net_state = state; }

 private:
  std::string _name;
  float _netweight = 1.;

  NET_TYPE _net_type;
  NET_STATE _net_state;
};

inline Net::Net(std::string name) : _name(std::move(name)), _netweight(1.0), _net_type(NET_TYPE::kNone), _net_state(NET_STATE::kNone)
{
}

}  // namespace imp

#endif
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
 * @Date: 2022-03-09 21:31:05
 * @LastEditTime: 2022-10-17 16:55:28
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/operator/global_placer/nesterov_place/database/NesNet.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_OPERATOR_NESTEROV_PLACE_DATABASE_NESNET_H
#define IPL_OPERATOR_NESTEROV_PLACE_DATABASE_NESNET_H

#include <string>
#include <vector>

#include "NesPin.hh"

namespace ipl {

class NesNet
{
 public:
  NesNet() = delete;
  explicit NesNet(std::string name);
  NesNet(const NesNet&) = delete;
  NesNet(NesNet&&) = delete;
  ~NesNet() = default;

  NesNet& operator=(const NesNet&) = delete;
  NesNet& operator=(NesNet&&) = delete;

  // getter.
  int32_t get_net_id() const { return _n_net_id; }
  std::string get_name() const { return _name; }
  float get_weight() const { return _weight; }
  float get_delta_weight() const { return _delta_weight; }
  bool isDontCare() const { return _is_dont_care == 1; }

  NesPin* get_driver() const { return _driver; }
  std::vector<NesPin*> get_loader_list() const { return _loader_list; }
  std::vector<NesPin*> get_nPin_list() const;

  // setter.
  void set_net_id(int32_t id) { _n_net_id = id; }
  void set_weight(float weight) { _weight = weight; }
  void set_delta_weight(float delta_weight) { _delta_weight = delta_weight; }
  void set_dont_care() { _is_dont_care = 1; }
  void set_driver(NesPin* nPin) { _driver = nPin; }
  void add_loader(NesPin* nPin) { _loader_list.push_back(nPin); }

 private:
  int32_t _n_net_id;
  std::string _name;
  float _weight;
  float _delta_weight;
  unsigned char _is_dont_care : 1;

  NesPin* _driver;
  std::vector<NesPin*> _loader_list;
};
inline NesNet::NesNet(std::string name) : _n_net_id(-1), _name(name), _weight(1.0F), _delta_weight(0.0F), _is_dont_care(0), _driver(nullptr)
{
}

inline std::vector<NesPin*> NesNet::get_nPin_list() const
{
  std::vector<NesPin*> nPin_list;
  if (_driver) {
    nPin_list.push_back(_driver);
  }
  nPin_list.insert(nPin_list.end(), _loader_list.begin(), _loader_list.end());
  return nPin_list;
}

}  // namespace ipl

#endif
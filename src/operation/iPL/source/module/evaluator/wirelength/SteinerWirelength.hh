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
 * @Date: 2022-04-11 12:04:50
 * @LastEditTime: 2022-12-06 22:13:14
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/evaluator/wirelength/SteinerWirelength.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_EVALUATOR_STEINERWL_H
#define IPL_EVALUATOR_STEINERWL_H

#include <unordered_map>

#include "Log.hh"
#include "Wirelength.hh"
#include "flute3/flute.h"
#include "utility/MultiTree.hh"
#include "utility/Utility.hh"

namespace ipl {

class SteinerWirelength : public Wirelength
{
 public:
  SteinerWirelength() = delete;
  explicit SteinerWirelength(TopologyManager* topology_manager);
  SteinerWirelength(const SteinerWirelength&) = delete;
  SteinerWirelength(SteinerWirelength&&) = delete;
  ~SteinerWirelength() override = default;

  SteinerWirelength& operator=(const SteinerWirelength&) = delete;
  SteinerWirelength& operator=(SteinerWirelength&&) = delete;

  void updateAllNetWorkPointPair();
  void updateNetWorkPointPair(NetWork* network);
  void updatePartOfNetWorkPointPair(const std::vector<NetWork*>& network_list);

  const std::vector<std::pair<Point<int32_t>, Point<int32_t>>>& obtainPointPairList(NetWork* network);
  MultiTree* obtainPointMultiTree(Point<int32_t> root_point, std::vector<std::pair<Point<int32_t>, Point<int32_t>>>& point_pair_list);
  MultiTree* obtainMultiTree(NetWork* network);

  int64_t obtainTotalWirelength();
  int64_t obtainNetWirelength(int32_t net_id);
  int64_t obtainPartOfNetWirelength(int32_t net_id, int32_t sink_pin_id);

 private:
  std::unordered_map<NetWork*, std::vector<std::pair<Point<int32_t>, Point<int32_t>>>> _point_pair_map;

  void initAllNetWorkPointPair();
  void obtainNetWorkPointPair(NetWork* network, std::vector<std::pair<Point<int32_t>, Point<int32_t>>>& point_pair);
  void obtainFlutePointPair(std::vector<Point<int32_t>>& point_vec, std::vector<std::pair<Point<int32_t>, Point<int32_t>>>& point_pair);
};
inline SteinerWirelength::SteinerWirelength(TopologyManager* topology_manager) : Wirelength(topology_manager)
{
  Flute::readLUT();
}

}  // namespace ipl

#endif
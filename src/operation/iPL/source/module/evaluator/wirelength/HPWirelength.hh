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
 * @Date: 2022-03-09 15:08:52
 * @LastEditTime: 2022-11-23 12:11:40
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/evaluator/wirelength/HPWirelength.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_EVALUATOR_HPWL_H
#define IPL_EVALUATOR_HPWL_H

#include "PlacerDB.hh"
#include "Wirelength.hh"
namespace ipl {

class HPWirelength : public Wirelength
{
 public:
  HPWirelength() = delete;
  explicit HPWirelength(TopologyManager* topology_manager);
  HPWirelength(const HPWirelength&) = delete;
  HPWirelength(HPWirelength&&) = delete;
  ~HPWirelength() override = default;

  HPWirelength& operator=(const HPWirelength&) = delete;
  HPWirelength& operator=(HPWirelength&&) = delete;

  int64_t obtainTotalWirelength();
  int64_t obtainNetWirelength(int32_t net_id);
  int64_t obtainPartOfNetWirelength(int32_t net_id, int32_t sink_pin_id);
  std::vector<std::vector<std::pair<int32_t, int32_t>>> constructPointSets();
};
inline HPWirelength::HPWirelength(TopologyManager* topology_manager) : Wirelength(topology_manager)
{
}

}  // namespace ipl

#endif
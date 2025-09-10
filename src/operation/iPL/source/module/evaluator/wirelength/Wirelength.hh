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
 * @Date: 2022-03-07 12:10:03
 * @LastEditTime: 2022-11-23 12:12:17
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/evaluator/wirelength/Wirelength.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_EVALUATOR_WIRELENGTH_H
#define IPL_EVALUATOR_WIRELENGTH_H

namespace ipl {
class TopologyManager;

class Wirelength
{
 public:
  Wirelength() = delete;
  explicit Wirelength(TopologyManager* topology_manager);
  Wirelength(const Wirelength&) = delete;
  Wirelength(Wirelength&&) = delete;
  virtual ~Wirelength() = default;

  Wirelength& operator=(const Wirelength&) = delete;
  Wirelength& operator=(Wirelength&&) = delete;

  virtual int64_t obtainTotalWirelength() = 0;
  virtual int64_t obtainNetWirelength(int32_t net_id) = 0;
  virtual int64_t obtainPartOfNetWirelength(int32_t net_id, int32_t sink_pin_id) = 0;

 protected:
  TopologyManager* _topology_manager;
};
inline Wirelength::Wirelength(TopologyManager* topology_manager) : _topology_manager(topology_manager)
{
}

}  // namespace ipl

#endif
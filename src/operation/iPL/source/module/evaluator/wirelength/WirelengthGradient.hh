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
 * @Date: 2022-03-08 22:36:24
 * @LastEditTime: 2022-11-23 11:57:32
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/evaluator/wirelength/WirelengthGradient.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_EVALUATOR_WIRELENGTH_GRADIENT_H
#define IPL_EVALUATOR_WIRELENGTH_GRADIENT_H

#include "TopologyManager.hh"
#include "data/Point.hh"
#include "GridManager.hh"

namespace ipl {

class WirelengthGradient
{
 public:
  WirelengthGradient() = delete;
  explicit WirelengthGradient(TopologyManager* topology_manager);
  WirelengthGradient(const WirelengthGradient&) = delete;
  WirelengthGradient(WirelengthGradient&&) = delete;
  virtual ~WirelengthGradient() = default;

  WirelengthGradient& operator=(const WirelengthGradient&) = delete;
  WirelengthGradient& operator=(WirelengthGradient&&) = delete;

  virtual void updateWirelengthForce(float coeff_x, float coeff_y, float min_force_bar, int32_t thread_num) = 0;
  virtual Point<float> obtainWirelengthGradient(int32_t inst_id, float coeff_x, float coeff_y) = 0;
  virtual void updateWirelengthForceDirect(float coeff_x, float coeff_y, float min_force_bar, int32_t thread_num, GridManager* grid_manager) = 0;

  // Debug
  virtual void waWLAnalyzeForDebug(float coeff_x, float coeff_y) = 0;

 protected:
  TopologyManager* _topology_manager;
};
inline WirelengthGradient::WirelengthGradient(TopologyManager* topology_manager) : _topology_manager(topology_manager)
{
}

}  // namespace ipl

#endif
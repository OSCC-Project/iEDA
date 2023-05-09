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
/**
 * @file PwrPropagateConst.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Propagate const vertexes from VDD or GND.
 * @version 0.1
 * @date 2023-03-27
 */

#pragma once

#include "core/PwrFunc.hh"
#include "core/PwrGraph.hh"
#include "core/PwrSeqGraph.hh"

namespace ipower {
/**
 * @brief Propagate const vertexes from VDD or GND.
 *
 */
class PwrPropagateConst : public PwrFunc {
 public:
  unsigned operator()(PwrVertex* the_vertex) override;
  unsigned operator()(PwrGraph* the_graph) override;
  void setTieCellFanout(PwrSeqGraph* the_seq_graph);

 private:
  void setConstStaToPwrVertex(StaVertex* const_sta_vertex,
                              PwrVertex* const_pwr_vertex);
  void setConstPwrToSeqVertex(PwrVertex* const_pwr_vertex);

};
}  // namespace ipower

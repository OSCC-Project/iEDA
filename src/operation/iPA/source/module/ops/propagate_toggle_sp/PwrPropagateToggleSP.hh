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
 * @file PwrPropagateToggleSP.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Propagate toggle and sp.
 * @version 0.1
 * @date 2023-04-06
 */

#pragma once

#include "core/PwrFunc.hh"
#include "core/PwrGraph.hh"
#include "core/PwrSeqGraph.hh"

/**
 * @brief struct for toggle and sp value in power data.
 *
 */
struct PwrToggleSPData {
  double _toggle_value;
  double _sp_value;
};

namespace ipower {
/**
 * @brief Propagate toggle and sp.
 *
 */
class PwrPropagateToggleSP : public PwrFunc {
 public:
  unsigned operator()(PwrGraph* the_graph) override;
  unsigned operator()(PwrVertex* the_vertex) override;

 private:
  PwrToggleSPData getToggleSPData(PwrVertex* the_vertex);
  PwrToggleSPData calcSeqDataOutToggleSP(
      PwrVertex* data_in_vertex, PwrVertex* data_out_vertex,
      const PwrToggleSPData& seq_data_in_toggle_sp);
};
}  // namespace ipower
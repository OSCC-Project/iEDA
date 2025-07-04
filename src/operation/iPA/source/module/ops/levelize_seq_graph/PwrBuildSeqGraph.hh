// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file PwrBuildSeqGraph.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Build the sequential graph.
 * @version 0.1
 * @date 2023-03-03
 */

#pragma once

#include "core/PwrFunc.hh"
#include "core/PwrGraph.hh"
#include "core/PwrSeqGraph.hh"

namespace ipower {
/**
 * @brief Build the sequential graph.
 *
 */
class PwrBuildSeqGraph : public PwrFunc {
 public:
  unsigned operator()(PwrGraph* the_graph) override;
  unsigned operator()(PwrVertex* the_vertex) override;

  auto& takePwrSeqGraph() { return _seq_graph; }

 private:
  unsigned buildSeqVertexes(PwrGraph* the_graph);
  unsigned buildPortVertexes(PwrGraph* the_graph);

  unsigned buildSeqArcs(PwrGraph* the_graph);

  std::set<std::pair<PwrSeqVertex*, PwrSeqVertex*>>
      _macro_arcs;  // for filter out repeat macro connection, because macro has
                    // multiple input and output pin, maybe built repeatedly
                    // macro connection.

  PwrSeqGraph _seq_graph;  //!< the sequential graph
};
}  // namespace ipower
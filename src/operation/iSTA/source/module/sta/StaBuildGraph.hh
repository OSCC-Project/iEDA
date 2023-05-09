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
 * @file StaBuildGraph.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The functor of build sta graph from the design netlist.
 * @version 0.1
 * @date 2021-08-10
 */

#pragma once

#include "StaFunc.hh"
#include "StaGraph.hh"

namespace ista {

/**
 * @brief The functor of build graph.
 *
 */
class StaBuildGraph : public StaFunc {
 public:
  unsigned buildPort(StaGraph* the_graph, Port* port);
  unsigned buildInst(StaGraph* the_graph, Instance* inst);
  unsigned buildNet(StaGraph* the_graph, Net* net);
  unsigned buildConst(StaGraph* the_graph, Instance* inst);

  unsigned operator()(StaGraph* the_graph) override;
};

}  // namespace ista

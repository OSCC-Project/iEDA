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
 * @file PwrBuildGraph.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief Build power graph, consider the power graph is the same with timing
 * graph, so we just need to update the power information to the timing graph.
 * @version 0.1
 * @date 2023-01-01
 */
#pragma once

#include "core/PwrGraph.hh"
#include "sta/StaFunc.hh"
#include "sta/StaGraph.hh"

namespace ipower {

/**
 * @brief The class for build power graph, we can build power graph and annotate
 * power info based on timing graph.
 *
 */
class PwrBuildGraph : public StaFunc {
 public:
  explicit PwrBuildGraph(PwrGraph& power_graph) : _power_graph(power_graph) {}
  ~PwrBuildGraph() override = default;
  unsigned operator()(StaGraph* the_graph) override;

  auto& takePowerGraph() { return _power_graph; }

 private:
  unsigned annotateInternalPower(PwrInstArc* inst_power_arc, Instance* inst);

  PwrGraph& _power_graph;  //!< The power graph to be build.
};

}  // namespace ipower

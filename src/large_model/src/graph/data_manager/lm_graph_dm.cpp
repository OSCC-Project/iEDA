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

#include "lm_graph_dm.h"

#include "Log.hh"
#include "lm_graph_check.hh"
#include "lm_net_graph_gen.hh"
#include "omp.h"
#include "usage.hh"

namespace ilm {

bool LmGraphDataManager::buildGraphData()
{
  LmNetGraphGenerator gen;
  auto wire_graphs = gen.buildGraphs();

  auto& layout_graph = _layout->get_graph();
  for (auto wire : wire_graphs) {
    auto net_id = 0;
    LmNetWire lm_wire;
    layout_graph.add_net_wire(net_id, lm_wire);
  }

  return true;
}

}  // namespace ilm
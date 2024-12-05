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
#include "lm_graph_init.h"
#include "omp.h"
#include "usage.hh"

namespace ilm {

bool LmGraphDataManager::buildGraphData(const std::string path)
{
  LmNetGraphGenerator gen;
  auto wire_graphs = gen.buildGraphs();
}

}  // namespace ilm
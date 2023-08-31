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
 * @file PwrPropagateClock.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief Propagate clock vertexes.
 * @version 0.1
 * @date 2023-04-15
 */

#include "PwrPropagateClock.hh"

#include "include/PwrConfig.hh"

namespace ipower {
using ieda::Stats;

/**
 * @brief Propagate clock vertexes.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned PwrPropagateClock::operator()(PwrVertex* the_vertex) {
  VERBOSE_LOG(1) << "propagate clock vertex: " << the_vertex->getName();

  // CP pin is register, not set clock network.
  the_vertex->set_is_clock_network();
  auto* propagated_clock = get_propagated_sta_clock();
  // set power data.
  the_vertex->addData(c_default_clock_toggle / propagated_clock->getPeriodNs(),
                      c_default_clock_sp, PwrDataSource::kClockPropagation);

  if (the_vertex->get_sta_vertex()->is_clock()) {
    return 1;
  }

  FOREACH_SRC_PWR_ARC(the_vertex, the_arc) {
    auto* the_snk_vertex = the_arc->get_snk();
    the_snk_vertex->exec(*this);
  }

  return 1;
}

/**
 * @brief Propagate clock vertexes.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned PwrPropagateClock::operator()(PwrGraph* the_graph) {
  Stats stats;
  LOG_INFO << "propagate clock start";
  set_the_pwr_graph(the_graph);

  for (auto* sta_clock : the_graph->get_sta_clocks()) {
    set_propagated_sta_clock(sta_clock);
    auto& clock_vertexes = sta_clock->get_clock_vertexes();
    for (auto* clock_vertex : clock_vertexes) {
      auto* pwr_clock_vertex = the_graph->staToPwrVertex(clock_vertex);
      pwr_clock_vertex->exec(*this);
    }
  }

  LOG_INFO << "propagate clock end";
  double memory_delta = stats.memoryDelta();
  LOG_INFO << "propagate clock memory usage " << memory_delta << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "propagate clock time elapsed " << time_delta << "s";
  return 1;
}
}  // namespace ipower
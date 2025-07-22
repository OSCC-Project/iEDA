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
 * @file StaClockSlewDelayPropagation.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The clock slew and propagation together using BFS method.
 * @version 0.1
 * @date 2024-12-26
 */

#include "StaClockSlewDelayPropagation.hh"

#include "StaDelayPropagation.hh"
#include "StaSlewPropagation.hh"
#include "ThreadPool/ThreadPool.h"

namespace ista {

/**
 * @brief propagate the arc to calc slew and delay of the snk vertex.
 *
 * @param the_arc
 * @return unsigned
 */
unsigned StaClockSlewDelayPropagation::operator()(StaArc* the_arc) {
  std::lock_guard<std::mutex> lk(the_arc->get_snk()->get_fwd_mutex());
  StaSlewPropagation slew_propagation;
  StaDelayPropagation delay_propagation;

  slew_propagation(the_arc);
  delay_propagation(the_arc);
  return 1;
}

/**
 * @brief propagate the vertex, and get the next bfs vertexes.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaClockSlewDelayPropagation::operator()(StaVertex* the_vertex) {
  if (the_vertex->is_const()) {
    return 1;
  }

  auto* vertex_own_cell = the_vertex->getOwnCell();

  // clock propagation end at the clock vertex.
  if (the_vertex->is_clock() && vertex_own_cell && !vertex_own_cell->isICG()) {
    the_vertex->set_is_slew_prop();
    the_vertex->set_is_delay_prop();
    return 1;
  }

  unsigned is_ok = 1;
  FOREACH_SRC_ARC(the_vertex, src_arc) {
    if (!src_arc->isDelayArc()) {
      continue;
    }

    if (src_arc->is_loop_disable()) {
      continue;
    }

    if (src_arc->isInstArc() &&
        !src_arc->get_snk()->get_design_obj()->get_net()) {
      // skip the instance output not connected to the net.
      continue;
    }

    // get the next bfs vertex and add it to the queue.
    auto* snk_vertex = src_arc->get_snk();
    if (!isIdealClock()) {
      is_ok = src_arc->exec(*this);
      if (!is_ok) {
        LOG_FATAL << "slew propgation error";
        break;
      }
    } else {
      snk_vertex->initSlewData();
    }

    addNextBFSQueue(snk_vertex);
  }

  the_vertex->set_is_slew_prop();
  the_vertex->set_is_delay_prop();

  return 1;
}

/**
 * @brief propagate from the clock source vertex.
 *
 * @return unsigned
 */
unsigned StaClockSlewDelayPropagation::operator()(StaGraph*) {
  ieda::Stats stats;
  LOG_INFO << "clock slew delay propagation start";
  unsigned is_ok = 1;

  Sta* ista = getSta();
  auto& clocks = ista->get_clocks();

  for (auto& clock : clocks) {
    _propagate_clock = clock.get();

    auto& vertexes = clock->get_clock_vertexes();
    for (auto* vertex : vertexes) {
      vertex->initSlewData();
      _bfs_queue.emplace_back(vertex);
    }
  }

  // lambda for propagate the current queue.
  auto propagate_current_queue = [this](auto& current_queue) {
    LOG_INFO << "propagating current clock queue vertexes number is "
             << current_queue.size();
#if 0
    // create thread pool
    unsigned num_threads = getNumThreads();
    ThreadPool pool(num_threads);

    for (auto* the_vertex : current_queue) {
      // bfs start from the root vertex, traverse to the clock pin vertex.
      if (the_vertex->get_src_arcs().empty()) {
        continue;
      }

      pool.enqueue([this](StaVertex* the_vertex) { return the_vertex->exec(*this); },
                   the_vertex);
    }
#else
    for (auto* the_vertex : current_queue) {
      the_vertex->exec(*this);
    }
#endif
  };

  // do the bfs traverse for calc the clock slew/delay.
  do {
    propagate_current_queue(_bfs_queue);
    _bfs_queue.clear();

    // swap to the next bfs queue.
    std::swap(_bfs_queue, _next_bfs_queue);

  } while (!_bfs_queue.empty());

  LOG_INFO << "clock slew delay propagation end";

  double memory_delta = stats.memoryDelta();
  LOG_INFO << "clock slew delay propagation memory usage " << memory_delta
           << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "clock slew delay propagation time elapsed " << time_delta << "s";

  return is_ok;
}

}  // namespace ista
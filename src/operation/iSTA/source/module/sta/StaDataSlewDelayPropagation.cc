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
 * @file StaDataSlewDelayPropagation.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The Data slew delay propagation using BFS method.
 * @version 0.1
 * @date 2024-12-26
 */
#include "StaDataSlewDelayPropagation.hh"
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
unsigned StaDataSlewDelayPropagation::operator()(StaArc* the_arc) {
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
unsigned StaDataSlewDelayPropagation::operator()(StaVertex* the_vertex) {
  if (the_vertex->is_const()) {
    return 1;
  }

  // data propagation end at the clock vertex.
  if (the_vertex->is_end()) {
    return 1;
  }

  unsigned is_ok = 1;
  FOREACH_SRC_ARC(the_vertex, src_arc) {
    if (!src_arc->isDelayArc()) {
      // calculate the check arc constrain value.
      if (src_arc->isCheckArc()) {
        StaDelayPropagation delay_propagation;
        src_arc->exec(delay_propagation);
      }
      continue;
    }

    if (src_arc->is_loop_disable()) {
      continue;
    }

    is_ok = src_arc->exec(*this);
    if (!is_ok) {
      LOG_FATAL << "slew propgation error";
      break;
    }

    // get the next level bfs vertex and add it to the queue.
    auto* snk_vertex = src_arc->get_snk();
    if (snk_vertex->get_level() == the_vertex->get_level() + 1) {
      _next_bfs_queue.push_back(snk_vertex);
    }
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
unsigned StaDataSlewDelayPropagation::operator()(StaGraph* the_graph) {
  unsigned is_ok = 1;

  StaVertex* start_vertex;
  FOREACH_START_VERTEX(the_graph, start_vertex) {
    // start from the vertex which is level one.
    if (start_vertex->get_level() == 1) {
      _bfs_queue.emplace_back(start_vertex);
    }
  }

  // lambda for propagate the current queue.
  auto propagate_current_queue = [this](auto& current_queue) {
    LOG_INFO << "Propagating current queue vertexes number is "
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

      pool.enqueue([](StaFunc& func,
                      StaVertex* the_vertex) { return the_vertex->exec(func); },
                   *this, the_vertex);
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

  return is_ok;
}

}  // namespace ista
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
 * @file StaFindStartOrEnd.cc
 * @author longshy (longshy@pcl.ac.cn)
 * @brief The implemention of the class for finding start or end points of the
 * timing path.
 * @version 0.1
 * @date 2023-07-21
 */
#include "StaFindStartOrEnd.hh"

#include "ThreadPool/ThreadPool.h"

namespace ista {

/**
 * @brief Find the end pins from the vertex.
 *
 * @param the_vertex
 * @return unsigned 1 if success, 0 else fail.
 */
unsigned StaFindEnd::operator()(StaVertex* the_vertex) {
  std::lock_guard<std::mutex> lk(the_vertex->get_fwd_mutex());

  if (the_vertex->is_foward_find()) {
    return 1;
  }

  if (the_vertex->is_end()) {
    the_vertex->addFanoutEndVertex(the_vertex);
    the_vertex->set_is_foward_find();
    return 1;
  }

  auto& src_arcs = the_vertex->get_src_arcs();
  for (auto& src_arc : src_arcs) {
    if (!src_arc->isDelayArc()) {
      continue;
    }

    // if the snk vertex is clock pin, don't consider the clock path.
    StaVertex* snk_vertex = src_arc->get_snk();
    if (snk_vertex->is_clock()) {
      continue;
    }

    if ((*this)(snk_vertex)) {
      auto& end_vertexs = snk_vertex->get_fanout_end_vertexes();

      if (!end_vertexs.empty()) {
        the_vertex->addFanoutEndVertex(end_vertexs);
      }
    }
  }
  the_vertex->set_is_foward_find();
  return 1;
}

/**
 * @brief Find the end pins from the graph start_vertexes.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned StaFindEnd::operator()(StaGraph* the_graph) {
  LOG_INFO << "start finding end";
  unsigned is_ok = 1;
#if 1
  unsigned num_threads = getNumThreads();
  ThreadPool pool(num_threads);
  StaVertex* start_vertex;
  FOREACH_START_VERTEX(the_graph, start_vertex) {
    if (start_vertex->is_clock()) {
      pool.enqueue(
          [](StaFunc& func, StaVertex* start_vertex) {
            return start_vertex->exec(func);
          },
          *this, start_vertex);
    }
  }

#else
  StaVertex* start_vertex;
  FOREACH_START_VERTEX(the_graph, start_vertex) {
    if (start_vertex->is_clock()) {
      start_vertex->exec(*this);
    }
  }
#endif
  LOG_INFO << "end finding end";
  return is_ok;
}

/**
 * @brief Find the start pins from the vertex.
 *
 * @param the_vertex
 * @return unsigned 1 if success, 0 else fail.
 */
unsigned StaFindStart::operator()(StaVertex* the_vertex) {
  std::lock_guard<std::mutex> lk(the_vertex->get_bwd_mutex());

  if (the_vertex->is_backward_find()) {
    return 1;
  }

  if (the_vertex->is_start()) {
    the_vertex->addFaninStartVertex(the_vertex);
    the_vertex->set_is_backward_find();
    return 1;
  }

  auto& snk_arcs = the_vertex->get_snk_arcs();
  for (auto& snk_arc : snk_arcs) {
    if (!snk_arc->isDelayArc()) {
      continue;
    }

    StaVertex* src_vertex = snk_arc->get_src();

    if ((*this)(src_vertex)) {
      auto& start_vertexs = src_vertex->get_fanin_start_vertexes();

      if (!start_vertexs.empty()) {
        the_vertex->addFaninStartVertex(start_vertexs);
      }
    }
  }
  the_vertex->set_is_backward_find();
  return 1;
}

/**
 * @brief Find the start pins from the graph end_vertexes.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned StaFindStart::operator()(StaGraph* the_graph) {
  LOG_INFO << "start finding start";
  unsigned is_ok = 1;
#if 1
  unsigned num_threads = getNumThreads();
  ThreadPool pool(num_threads);
  StaVertex* end_vertex;
  FOREACH_END_VERTEX(the_graph, end_vertex) {
    if (end_vertex->is_end()) {
      pool.enqueue([](StaFunc& func,
                      StaVertex* end_vertex) { return end_vertex->exec(func); },
                   *this, end_vertex);
    }
  }
#else
  StaVertex* end_vertex;
  FOREACH_END_VERTEX(the_graph, end_vertex) {
    if (end_vertex->is_end()) {
      end_vertex->exec(*this);
    }
  }

#endif
  LOG_INFO << "end finding start";
  return is_ok;
}
}  // namespace ista
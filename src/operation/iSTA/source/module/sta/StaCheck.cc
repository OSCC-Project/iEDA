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
 * @file StaCheck.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The timing check implemention.
 * @version 0.1
 * @date 2021-03-01
 */

#include "StaCheck.hh"
#include "log/Log.hh"
#include "sta/StaVertex.hh"

namespace ista {

/**
 * @brief print loop record.
 *
 */
void StaCombLoopCheck::printAndBreakLoop(bool is_fwd) {
  LOG_INFO << "found loop : ";
  const char* direction = is_fwd ? " <- " : " -> ";
  std::string loop_name;
  auto* loop_point = _loop_record.front();
  StaVertex* last_vertex = nullptr;
  while (!_loop_record.empty()) {
    auto* the_vertex = _loop_record.front();
    loop_name += the_vertex->getName() + direction;
    _loop_record.pop();

    // found loop point and break loop.
    if (_loop_record.front() == loop_point) {
      loop_name += _loop_record.front()->getName();

      StaArc* loop_arc = nullptr;
      if (is_fwd) {
        loop_arc = last_vertex->getSrcArc(the_vertex).front();
      } else {
        loop_arc = last_vertex->getSnkArc(the_vertex).front();
      }
      loop_arc->set_is_loop_disable(true);

      break;
    }

    last_vertex = the_vertex;
  }

  // clear queue.
  std::queue<StaVertex*> empty_vertex;
  _loop_record.swap(empty_vertex);

  LOG_INFO << loop_name;
}

/**
 * @brief The combination loop check implemention.
 *
 * @param the_graph
 * @return unsigned return 1 if found loop, else return 0.
 */
unsigned StaCombLoopCheck::operator()(StaGraph* the_graph) {
  std::function<unsigned(StaVertex*, bool)> traverse_data_path =
      [&traverse_data_path, this](StaVertex* the_vertex,
                                  bool is_fwd) -> unsigned {
    /*The traverse end at the end or start node.*/
    if (is_fwd && the_vertex->is_end()) {
      return 0;
    }

    if (!is_fwd && the_vertex->is_start()) {
      return 0;
    }

    the_vertex->setGray();
    auto& next_arcs =
        is_fwd ? the_vertex->get_src_arcs() : the_vertex->get_snk_arcs();

    for (auto* arc : next_arcs) {
      if (!arc->isDelayArc()) {
        continue;
      }

      auto* next_vertex = is_fwd ? arc->get_snk() : arc->get_src();

      if (next_vertex->isBlack()) {
        continue;
      }
      if (next_vertex->isWhite()) {
        // continue traverse the data path.
        if (traverse_data_path(next_vertex, is_fwd)) {
          /* found loop */
          _loop_record.push(next_vertex);
          // for loop found, we do not propagate the vertex, set is black.
          the_vertex->setBlack();
          return 1;
        }
      } else {
        LOG_FATAL_IF(!next_vertex->isGray()) << "the vertex should be gray.";
        /* found loop */
        _loop_record.push(next_vertex);
        // for loop found, we do not propagate the vertex, set is black.
        the_vertex->setBlack();
        return 1;
      }
    }

    // if all arcs is traversed, then set the vertex is black.
    the_vertex->setBlack();

    return 0;
  };

  LOG_INFO << "found loop fwd start";
  // reset the vertex color.
  the_graph->resetVertexColor();

  StaVertex* start_vertex;
  FOREACH_START_VERTEX(the_graph, start_vertex) {
    auto src_arcs = start_vertex->get_src_arcs();
    for (auto* arc : src_arcs) {
      // traverse from the data path start point.
      if (arc->isDelayArc()) {
        StaVertex* data_start_point = arc->get_snk();
        if (traverse_data_path(data_start_point, true)) {
          _loop_record.push(data_start_point);
          printAndBreakLoop(true);
        }
      }
    }
  }

  LOG_INFO << "found loop fwd end";

  LOG_INFO << "found loop bwd start";

  // reset the vertex color.
  the_graph->resetVertexColor();

  StaVertex* end_vertex;
  FOREACH_END_VERTEX(the_graph, end_vertex) {
    auto snk_arcs = end_vertex->get_snk_arcs();
    for (auto* arc : snk_arcs) {
      if (arc->isDelayArc()) {
        StaVertex* data_end_point = arc->get_src();
        // traverse from the data path end point.
        if (traverse_data_path(data_end_point, false)) {
          _loop_record.push(data_end_point);
          printAndBreakLoop(false);
        }
      }
    }
  }

  the_graph->resetVertexColor();

  LOG_INFO << "found loop bwd end";

  return 1;
}

}  // namespace ista

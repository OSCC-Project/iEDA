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
 * @author shy long (longshy@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2023-07-21
 */
#include "StaFindStartOrEnd.hh"

namespace ista {

/**
 * @brief Find the end pins from the vertex.
 *
 * @param the_vertex
 * @return unsigned 1 if success, 0 else fail.
 */
unsigned StaFindEnd::operator()(StaVertex* the_vertex) {
  if (the_vertex->is_foward_find()) {
    return 1;
  }

  if (the_vertex->is_end()) {
    the_vertex->addEndVertex(the_vertex);
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
      auto& end_vertexs = snk_vertex->get_end_vertexes();

      if (!end_vertexs.empty()) {
        the_vertex->addEndVertex(end_vertexs);
      }
    }
  }
  the_vertex->set_is_foward_find();
  return 1;
}

unsigned StaFindEnd::operator()(StaGraph* the_graph) {}

/**
 * @brief Find the start pins from the vertex.
 *
 * @param the_vertex
 * @return unsigned 1 if success, 0 else fail.
 */
unsigned StaFindStart::operator()(StaVertex* the_vertex) {
  if (the_vertex->is_backward_find()) {
    return 1;
  }

  if (the_vertex->is_start()) {
    if (the_vertex->is_clock()) {
      LOG_INFO << "Debug";
    }
    the_vertex->addStartVertex(the_vertex);
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
      auto& start_vertexs = src_vertex->get_start_vertexes();

      if (!start_vertexs.empty()) {
        the_vertex->addStartVertex(start_vertexs);
      }
    }
  }
  the_vertex->set_is_backward_find();
  return 1;
}

unsigned StaFindStart::operator()(StaGraph* the_graph) {}
}  // namespace ista
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
 * @file StaLevelization.cc
 * @author longshy (longshy@pcl.ac.cn)
 * @brief The levelization implemention from end vertex of the graph.
 * @version 0.1
 * @date 2021-09-16
 */

#include "StaLevelization.hh"

namespace ista {

/**
 * @brief Levelization from the vertex.
 *
 * @param the_vertex
 * @return unsigned  1 if success, 0 else fail.
 */
unsigned StaLevelization::operator()(StaVertex* the_vertex) {  
  if (the_vertex->is_start()) {
    VLOG(1) << "set start vertex " << the_vertex->getName() << " level 1";
    the_vertex->set_level(1);
    return 1;
  }

  // only levelization data path.
  if (the_vertex->is_clock()) {
    VLOG(1) << "set clock vertex " << the_vertex->getName() << " level 1";
    the_vertex->set_level(1);
  }

  if (the_vertex->isSetLevel()) {
    return 1;
  }

  FOREACH_SNK_ARC(the_vertex, snk_arc) {
    if (snk_arc->isDelayArc()) {
      snk_arc->exec(*this);
    }
  }

  return 1;
}

/**
 * @brief Levelization from the arc.
 *
 * @param the_arc
 * @return unsigned  1 if success, 0 else fail.
 */
unsigned StaLevelization::operator()(StaArc* the_arc) {
  if (the_arc->is_loop_disable()) {
    return 1;
  }

  auto* src_vertex = the_arc->get_src();
  auto* snk_vertex = the_arc->get_snk();
  src_vertex->exec(*this);

  unsigned src_level = src_vertex->get_level();
  snk_vertex->set_level(src_level + 1);

  return 1;
}

/**
 * @brief Levelization from the graph port vertex.
 *
 * @param the_graph
 * @return unsigned  1 if success, 0 else fail.
 */
unsigned StaLevelization::operator()(StaGraph* the_graph) {
  StaVertex* end_vertex;

  FOREACH_END_VERTEX(the_graph, end_vertex) { end_vertex->exec(*this); }

  return 1;
}

}  // namespace ista
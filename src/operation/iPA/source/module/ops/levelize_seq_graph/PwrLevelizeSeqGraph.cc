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
 * @file PwrLevelizeSeq.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief Ranking of sequential logic units.
 * @version 0.1
 * @date 2023-02-27
 */

#include "PwrLevelizeSeqGraph.hh"

#include <fstream>

#include "ops/dump/PwrDumpSeqGraph.hh"

namespace ipower {

using ieda::Stats;

/**
 * @brief Set level to seq vertexes.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned PwrLevelizeSeq::operator()(PwrSeqVertex* the_vertex) {
  // Find snk arcs of the seq vertex.
  auto& snk_arcs = the_vertex->get_snk_arcs();
  unsigned the_level = 0;

  /*get level from src vertexes.*/
  for (auto* snk_arc : snk_arcs) {
    // Get the src vertex of the snk arc.
    auto* the_src_vertex = snk_arc->get_src();

    // The seq arc may be self loop or loop-marked arc.
    if (snk_arc->isSelfLoop() || snk_arc->isPipelineLoop()) {
      continue;
    }

    // The src vertex may be input port, the level is 0.
    if (the_src_vertex->isInputPort()) {
      if (!the_src_vertex->isLevelSet()) {
        get_the_pwr_seq_graph()->addLevelSeqVertex(0, the_src_vertex);
        the_src_vertex->set_is_level_set();
      }
      continue;
    }
    // The src vertex's level is not set.
    if (!the_src_vertex->isLevelSet()) {
      the_src_vertex->exec(*this);
    }
    // The level value prepare for the next seq vertex.
    unsigned the_src_level = the_src_vertex->get_level();
    if (the_src_level > the_level) {
      // The current path has a higher level.
      the_level = the_src_level;
    }
  }

  /*Set level for the vertex.*/
  get_the_pwr_seq_graph()->addLevelSeqVertex(the_level + 1, the_vertex);
  the_vertex->set_level(the_level + 1);
  the_vertex->set_is_level_set();
  return 1;
}

/**
 * @brief Traversing the seq graph to levelize seq vertexes.
 *
 * @param the_seq_graph
 * @return unsigned
 */
unsigned PwrLevelizeSeq::operator()(PwrSeqGraph* the_seq_graph) {
  Stats stats;
  set_the_pwr_seq_graph(the_seq_graph);

  LOG_INFO << "seq levelization start";
  // Start path from output port seq vertex.
  auto& output_port_vertexes = the_seq_graph->get_output_port_vertexes();
  for (auto* output_port_vertex : output_port_vertexes) {
    output_port_vertex->exec(*this);
    // LOG_INFO << "output port " << output_port_vertex->get_obj_name()
    //          << " level " << output_port_vertex->get_level();
  }
  LOG_INFO << "seq levelization end";

  LOG_INFO << "the seq graph's level depth "
           << the_seq_graph->get_level_depth();

  bool is_debug = false;
  if (is_debug) {
    std::string dump_file_name = "level.txt";
    std::ofstream level_file(dump_file_name, std::ios::trunc);
    the_seq_graph->printSeqLevelInfo(level_file);
    level_file.close();

    // LOG_FATAL << "levelization done";
  }

  double memory_delta = stats.memoryDelta();
  LOG_INFO << "levelization memory usage " << memory_delta << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "levelization time elapsed " << time_delta << "s";
  return 1;
}

}  // namespace ipower

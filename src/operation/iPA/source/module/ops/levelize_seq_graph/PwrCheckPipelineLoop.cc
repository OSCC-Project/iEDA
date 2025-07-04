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
 * @file PwrCheckPipelineLoop.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief Check the pipeline loop for the seq graph.
 * @version 0.1
 * @date 2023-03-14
 */
#include "PwrCheckPipelineLoop.hh"

#include <fstream>
#include <iostream>
#include <ranges>

namespace ipower {

using ieda::Stats;

/**
 * @brief Mark seq vertexes using the tricolor mark method.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned PwrCheckPipelineLoop::operator()(PwrSeqVertex* the_vertex) {
  pushStack(the_vertex);
  // Set the seq vertex's color as grey.
  the_vertex->setGrey();
  // Find snk arcs of the seq vertex.
  auto& snk_arcs = the_vertex->get_snk_arcs();
  for (auto* snk_arc : snk_arcs) {
    // The seq arc may be self loop.
    if (snk_arc->isSelfLoop()) {
      continue;
    }
    // Get the src vertex of the src arc.
    auto* the_src_vertex = snk_arc->get_src();
    TricolorMark the_src_tricolor_mark = the_src_vertex->get_tricolor_mark();
    // If the src vertex is a white vertex, the src vertex is not visited.
    if (the_src_tricolor_mark == TricolorMark::kWhite) {
      // If it is a output port vertex, the src vertex turn black.
      if (the_src_vertex->isInputPort()) {
        the_src_vertex->setBlack();
        continue;
      }
      // The src vertex is a seq vertex.
      the_src_vertex->exec(*this);
    }
    // If the src vertex is a black vertex, continue.
    if (the_src_tricolor_mark == TricolorMark::kBlack) {
      continue;
    }

    // If the src vertex is a grey vertex, find a pipeline loop.
    if (the_src_tricolor_mark == TricolorMark::kGrey) {
      // record loop.
      if (_record_num > 0) {
        auto seq_loop = getLoopFromStack(the_src_vertex);
        // seq_loop.print(std::cout);
        addLoop(std::move(seq_loop));
        --_record_num;
      }

      ++_loop_num;
      // Set the snk src as a pipeline loop arc.
      snk_arc->set_is_pipeline_loop();
      continue;
    }
  }

  // Visited all src vertexes of this vertex, the vertex turn black.
  the_vertex->setBlack();
  popStack();
  return 1;
}

/**
 * @brief from dfs stack get the loop vertex begin point.
 *
 * @param loop_vertex
 * @return PwrSeqLoop
 */
PwrSeqLoop PwrCheckPipelineLoop::getLoopFromStack(PwrSeqVertex* loop_vertex) {
  PwrSeqLoop seq_loop;
  seq_loop.insertVertex(loop_vertex);
  for (auto* the_vertex : _dfs_stack | std::ranges::views::reverse) {
    // found loop point.
    if (the_vertex == loop_vertex) {
      seq_loop.insertVertex(the_vertex);
      break;
    }
    seq_loop.insertVertex(the_vertex);
  }

  return seq_loop;
}

/**
 * @brief Check pipeline loop in the power seq graph.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned PwrCheckPipelineLoop::operator()(PwrSeqGraph* the_graph) {
  Stats stats;

  LOG_INFO << "check pipeline loop start";

  // Start path from output port seq vertex.
  auto& output_port_vertexes = the_graph->get_output_port_vertexes();
  for (auto* output_port_vertex : output_port_vertexes) {
    output_port_vertex->exec(*this);
  }
  LOG_INFO << "check pipeline loop end";

  std::ofstream loop_file("/home/shaozheqing/iSO/src/iPower/test/loop.txt",
                          std::ios::trunc);
  printPipelineLoop(loop_file);
  loop_file.close();

  double memory_delta = stats.memoryDelta();
  LOG_INFO << "check pipeline loop memory usage " << memory_delta << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "check pipeline loop time elapsed " << time_delta << "s";

  return 1;
}

}  // namespace ipower
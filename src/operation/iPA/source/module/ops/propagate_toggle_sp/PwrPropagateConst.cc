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
 * @file PwrPropagateConst.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief Propagate const vertexes from VDD or GND.
 * @version 0.1
 * @date 2023-03-27
 */

#include "PwrPropagateConst.hh"

#include <queue>
#include <ranges>

#include "log/Log.hh"
#include "ops/calc_toggle_sp/PwrCalcToggleSP.hh"

namespace ipower {
using ieda::Stats;

/**
 * @brief set const power vertex by const sta vertex.
 *
 * @param const_sta_vertex
 * @param const_pwr_vertex
 */
void PwrPropagateConst::setConstStaToPwrVertex(StaVertex* const_sta_vertex,
                                               PwrVertex* const_pwr_vertex) {
  const_pwr_vertex->set_is_const();

  // if the vertex is const 1 or const 0.
  if (const_sta_vertex->is_const_vdd()) {
    const_pwr_vertex->set_is_const_vdd();
  } else {
    const_pwr_vertex->set_is_const_gnd();
  }
}

/**
 * @brief set const seq vertex by const power vertex.
 *
 * @param const_sta_vertex
 * @param const_seq_vertex
 */
void PwrPropagateConst::setConstPwrToSeqVertex(PwrVertex* const_pwr_vertex) {
  auto* const_seq_vertex = const_pwr_vertex->get_own_seq_vertex();

  if (const_pwr_vertex->isSeqDataIn()) {
    const_seq_vertex->set_is_const();
    // if the vertex is const 1 or const 0.
    if (const_pwr_vertex->is_const_vdd()) {
      const_seq_vertex->set_is_const_vdd();
    } else {
      const_seq_vertex->set_is_const_gnd();
    }
  }
}

/**
 * @brief set fanout seq vertex for tie cells.
 *
 * @param the_seq_graph
 */
void PwrPropagateConst::setTieCellFanout(PwrSeqGraph* the_seq_graph) {
  PwrSeqVertex* seq_vertex;

  std::function<void(PwrSeqVertex * seq_vertex, PwrVertex * pwr_vertex)>
      set_fanout_vertex = [&set_fanout_vertex](PwrSeqVertex* seq_vertex,
                                               PwrVertex* pwr_vertex) {
        pwr_vertex->addFanoutSeqVertex(seq_vertex, 1);
        FOREACH_SNK_PWR_ARC(pwr_vertex, snk_arc) {
          auto* the_src_vertex = snk_arc->get_src();
          if (!the_src_vertex->get_fanout_seq_vertexes().contains(seq_vertex)) {
            set_fanout_vertex(seq_vertex, the_src_vertex);
          }
          
        }
      };

  FOREACH_SEQ_VERTEX(the_seq_graph, seq_vertex) {
    if (seq_vertex->isMacro()) {
      continue;
    }

    auto& data_in_vertexes = seq_vertex->get_seq_in_vertexes();
    for (auto* data_in_vertex : data_in_vertexes) {
      FOREACH_SNK_PWR_ARC(data_in_vertex, the_arc) {
        auto* the_src_vertex = the_arc->get_src();
        if (the_src_vertex->isSeqClockPin()) {
          continue;
        }
        /*Find paths that have not been marked with fanout vertex.*/
        if (the_src_vertex->get_fanout_seq_vertexes().empty()) {
          LOG_INFO << "set tie cell fanout from " << the_src_vertex->getName()
                   << " data in vertex " << data_in_vertex->getName();
          set_fanout_vertex(seq_vertex, the_src_vertex);
        }
      }
    }
  }
}

/**
 * @brief propagate by dfs until the start vertex.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned PwrPropagateConst::operator()(PwrVertex* the_vertex) {
  if (the_vertex->is_const_propagated()) {
    return 1;
  }

  auto* the_graph = get_the_pwr_graph();

  FOREACH_SNK_PWR_ARC(the_vertex, the_arc) {
    auto* the_src_pwr_vertex = the_arc->get_src();

    if (the_src_pwr_vertex->isSeqClockPin()) {
      continue;
    }

    // dfs until the previous level seq vertex data out vertex or port vertex.
    if (the_src_pwr_vertex->get_own_seq_vertex() ||
        the_src_pwr_vertex->is_input_port()) {
      continue;
    }

    // dfs src vertex.
    if (!the_src_pwr_vertex->exec(*this)) {
      LOG_FATAL << "propagte const error.";
      return 0;
    }
  }

  // if the pin is output pin or inout pin(need to be assistant node means
  // output pin), calc toggle and SP of cell output.
  auto* the_obj = the_vertex->get_sta_vertex()->get_design_obj();
  if (the_vertex->isHaveConstSrcVertex()) {
    if (the_obj->isPin() && (the_obj->isOutput() &&
                             (!the_obj->isInout() ||
                              the_vertex->get_sta_vertex()->is_assistant()))) {
      PwrCalcToggleSP calc_toggle_sp_func;
      calc_toggle_sp_func.set_the_pwr_graph(the_graph);
      calc_toggle_sp_func(the_vertex);
    } else {
      if (!the_vertex->get_snk_arcs().empty()) {
        the_vertex->set_is_const();
        the_vertex->get_snk_arcs().front()->get_src()->is_const_vdd()
            ? the_vertex->set_is_const_vdd()
            : the_vertex->set_is_const_gnd();
      }
    }
  }

  the_vertex->set_is_const_propagated();

  return 1;
}

/**
 * @brief Propagate const vertexes from VDD or GND.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned PwrPropagateConst::operator()(PwrGraph* the_graph) {
  Stats stats;
  LOG_INFO << "propagate const start";

  set_the_pwr_graph(the_graph);
  set_the_pwr_seq_graph(the_graph->get_pwr_seq_graph());

  // lambda function to compare power vertex min fanout seq level.
  auto seq_level_cmp = [](PwrVertex* lhs, PwrVertex* rhs) {
    auto lhs_min_seq_vertex = lhs->getFanoutMinSeqLevel();
    auto rhs_min_seq_vertex = rhs->getFanoutMinSeqLevel();
    if (!lhs_min_seq_vertex) {
      return false;
    }

    if (!rhs_min_seq_vertex) {
      return true;
    }

    return (*lhs_min_seq_vertex)->get_level() >
           (*rhs_min_seq_vertex)->get_level();
  };

  auto* the_sta_graph = the_graph->get_sta_graph();
  StaVertex* const_sta_vertex;

  std::priority_queue<PwrVertex*, std::vector<PwrVertex*>,
                      decltype(seq_level_cmp)>
      const_pwr_vertex_queue;

  /*Get the initial const vertex from sta graph, mark to pwr graph and seq
   * graph.*/
  FOREACH_CONST_VERTEX(the_sta_graph, const_sta_vertex) {
    auto* const_pwr_vertex = the_graph->staToPwrVertex(const_sta_vertex);

    /*Mark const vertex to power graph.*/
    setConstStaToPwrVertex(const_sta_vertex, const_pwr_vertex);

    // add const power vertex to queue.
    const_pwr_vertex_queue.push(const_pwr_vertex);

    /*Mark const vertex to seq graph.*/
    if (auto* const_seq_vertex = const_pwr_vertex->get_own_seq_vertex();
        const_seq_vertex) {
      setConstPwrToSeqVertex(const_pwr_vertex);
    }
  }

  // lambda function propagate the fanout end vertex from seq data in power
  // vertex.
  auto propagate_const_by_seq = [&const_pwr_vertex_queue,
                                 this](PwrSeqVertex* fanout_seq_vertex) {
    auto& data_in_pwr_vertexes = fanout_seq_vertex->get_seq_in_vertexes();

    for (auto* data_in_pwr_vertex : data_in_pwr_vertexes) {
      data_in_pwr_vertex->exec(*this);

      // set seq vertex const.
      if (data_in_pwr_vertex->is_const() && data_in_pwr_vertex->isSeqDataIn()) {
        if (!fanout_seq_vertex->isMacro()) {
          setConstPwrToSeqVertex(data_in_pwr_vertex);

          // set seq data out as const if seq vertex is const.
          auto& data_out_vertexes = fanout_seq_vertex->get_seq_out_vertexes();
          for (auto* data_out_vertex : data_out_vertexes) {
            data_out_vertex->set_is_const();
            const_pwr_vertex_queue.push(data_out_vertex);
          }
        } else {
          // TODO macro have more data in port, judge const is not the same.
        }
      }
    }
  };

  // for tie cell, may not found fanout seq vertex, need first set.
  setTieCellFanout(get_the_pwr_seq_graph());

  std::set<PwrSeqVertex*> visited_const_seq_vertexes;
  // traverse const power vertex level by level, the min level first propgate
  // to fanout seq vertex by dfs, then set seq vertex output power vertex is
  // const, continue propagate to next level seq vertex until never found
  // const vertex.
  unsigned const_vertex_num = 0;
  while (!const_pwr_vertex_queue.empty()) {
    auto* const_pwr_vertex = const_pwr_vertex_queue.top();
    ++const_vertex_num;
    const_pwr_vertex_queue.pop();

    VERBOSE_LOG(2)
        << "propagate const vertex " << const_pwr_vertex->getName()
        << " liberty cell "
        << const_pwr_vertex->getOwnInstance()->get_inst_cell()->get_cell_name();

    auto& fanout_seq_vertexes = const_pwr_vertex->get_fanout_seq_vertexes();
    for (auto* fanout_seq_vertex :
         fanout_seq_vertexes | std::views::filter([&visited_const_seq_vertexes](
                                                      auto* the_seq_vertex) {
           return !visited_const_seq_vertexes.contains(the_seq_vertex);
         })) {
      visited_const_seq_vertexes.insert(fanout_seq_vertex);
      propagate_const_by_seq(fanout_seq_vertex);
    }
  }
  VERBOSE_LOG(1) << "const vertex num: " << const_vertex_num;

  LOG_INFO << "propagate const end";
  double memory_delta = stats.memoryDelta();
  LOG_INFO << "propagate const memory usage " << memory_delta << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "propagate const time elapsed " << time_delta << "s";

  return 1;
}
}  // namespace ipower
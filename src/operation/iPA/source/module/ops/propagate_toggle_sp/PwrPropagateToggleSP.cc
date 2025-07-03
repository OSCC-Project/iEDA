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
 * @file PwrPropagateToggleSP.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief Propagate toggle and sp.
 * @version 0.1
 * @date 2023-04-06
 */

#include "PwrPropagateToggleSP.hh"

#include <future>
#include <mutex>

#include "ThreadPool/ThreadPool.h"
#include "include/PwrConfig.hh"
#include "ops/calc_toggle_sp/PwrCalcToggleSP.hh"
#include "ops/dump/PwrDumpGraph.hh"

#define MULTI_THREAD 1

namespace ipower {
using ieda::Stats;

/**
 * @brief Get toggle and SP of a power vertex.
 *
 * @param the_vertex
 * @return PwrToggleSPValue
 */
PwrToggleSPData PwrPropagateToggleSP::getToggleSPData(PwrVertex* the_vertex) {
  double toggle_value =
      the_vertex->getToggleData(PwrDataSource::kDataPropagation);
  double sp_value = the_vertex->getSPData(PwrDataSource::kDataPropagation);

  return {._toggle_value = toggle_value, ._sp_value = sp_value};
}

/**
 * @brief calc the seq data out according seq data in toggle and sp data.
 *
 * @param seq_data_in_toggle_sp
 * @return PwrToggleSPData
 */
PwrToggleSPData PwrPropagateToggleSP::calcSeqDataOutToggleSP(
    PwrVertex* data_in_vertex, PwrVertex* data_out_vertex,
    const PwrToggleSPData& seq_data_in_toggle_sp) {
  auto* the_graph = get_the_pwr_graph();

  auto launch_clock_domain = data_in_vertex->getOwnFastestClockDomain();
  LOG_INFO_IF(!launch_clock_domain)
      << "power vertex " << data_in_vertex->getName()
      << " not found launch clock domain.";
  if (data_out_vertex->getName() == "swerv_exu__30129_:QN") {
    LOG_INFO << "Debug";
  }
  auto capcure_clock_domain = data_out_vertex->getOwnFastestClockDomain();
  LOG_FATAL_IF(!capcure_clock_domain) << " not found capture clock domain.";
  // calc toggle and sp for data out vertex.

  double launch_period =
      launch_clock_domain
          ? (*launch_clock_domain)->getPeriodNs()
          : the_graph->get_fastest_clock().get_clock_period_ns();
  double capture_period = (*capcure_clock_domain)->getPeriodNs();

  // In one period clock toggle is 2, change to 1 ns unit, toggle is 2/period
  double clock_toggle = 2 / capture_period;

  double data_out_toggle = seq_data_in_toggle_sp._toggle_value;

  // if data out clock domain is slower, toggle data should be toggle/period
  // times.
  if (capture_period > launch_period) {
    data_out_toggle = seq_data_in_toggle_sp._toggle_value / capture_period;
  }

  // data out toggle is not more than half of clock toggle, if less than half
  // of clock toggle, use the data in toggle.
  if (data_out_toggle > (clock_toggle / 2)) {
    data_out_toggle = clock_toggle / 2;
  }

  return {data_out_toggle, seq_data_in_toggle_sp._sp_value};
}

/**
 * @brief propagare by dfs.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned PwrPropagateToggleSP::operator()(PwrVertex* the_vertex) {
  if (the_vertex->is_toggle_sp_propagated()) {
    if (isTrace()) {
      addTraceVertex(the_vertex);
    }
    return 1;
  }

  // dfs until the previous level seq vertex data out vertex or port vertex
  // or const vertex.
  if (the_vertex->is_const() ||
      (the_vertex->get_own_seq_vertex() &&
       !the_vertex->get_sta_vertex()->is_end()) ||
      the_vertex->isSeqClockPin() || the_vertex->is_input_port()) {
    if (isTrace()) {
      addTraceVertex(the_vertex);
    }
    return 1;
  }

  VERBOSE_LOG(2) << "before get mutex propagate toggle sp to vertex "
                 << the_vertex->getName();

  std::lock_guard lk(the_vertex->get_mutex());

  VERBOSE_LOG(2) << "after get mutex propagate toggle sp to vertex "
                 << the_vertex->getName();

  if (the_vertex->is_toggle_sp_propagated()) {
    return 1;
  }

  auto* the_graph = get_the_pwr_graph();

  FOREACH_SNK_PWR_ARC(the_vertex, the_arc) {
    auto* the_src_pwr_vertex = the_arc->get_src();
    if (the_vertex == the_src_pwr_vertex) {
      continue;
    }

    if (isTrace()) {
      addTraceArc(the_arc);
    }

    // dfs src vertex.
    if (!the_src_pwr_vertex->exec(*this)) {
      LOG_FATAL << "propagte toggle error.";
      return 0;
    }
  }

  // if the pin is output pin or inout pin(need to be assistant node means
  // output pin), calc toggle and SP of cell output.
  auto* the_obj = the_vertex->get_sta_vertex()->get_design_obj();
  if (the_obj->isPin() &&
      (the_obj->isOutput() &&
       (!the_obj->isInout() || the_vertex->get_sta_vertex()->is_assistant()))) {
    PwrCalcToggleSP calc_toggle_sp_func;
    calc_toggle_sp_func.set_the_pwr_graph(get_the_pwr_graph());
    calc_toggle_sp_func(the_vertex);
  } else {
    if (!the_vertex->get_snk_arcs().empty()) {
      // copy src vertex (data out) data to the vertex (data in) data.
      auto* driver_pwr_vertex = the_vertex->get_snk_arcs().front()->get_src();
      // get driver vertex toggle sp data.
      auto toggle_sp_data = getToggleSPData(driver_pwr_vertex);
      auto& the_fastest_clock = the_graph->get_fastest_clock();
      the_vertex->addData(toggle_sp_data._toggle_value,
                          toggle_sp_data._sp_value,
                          PwrDataSource::kDataPropagation, &the_fastest_clock);
    }
  }
  if (isTrace()) {
    addTraceVertex(the_vertex);
  }

  the_vertex->set_is_toggle_sp_propagated();

  return 1;
}

/**
 * @brief Propagate toggle and SP.
 *
 * @param the_seq_graph
 * @return unsigned
 */
unsigned PwrPropagateToggleSP::operator()(PwrGraph* the_graph) {
  Stats stats;
  set_the_pwr_graph(the_graph);
  set_the_pwr_seq_graph(the_graph->get_pwr_seq_graph());

  auto* the_seq_graph = get_the_pwr_seq_graph();

  LOG_INFO << "propagate toggle sp start";
  {
    /*Lambda function of propagate toggle and SP from a seq vertex.*/
    auto propagate_from_seq = [this](PwrSeqVertex* seq_vertex) {
      /*Start from data in vertex and work forward.*/
      auto data_in_vertexes = seq_vertex->getDataInVertexes();
      unsigned is_ok = 1;

      for (auto* data_in_vertex : data_in_vertexes) {
        VERBOSE_LOG(2) << "propagate toggle sp from data in vertex "
                       << data_in_vertex->getName() << " start.";
        is_ok &= data_in_vertex->exec(*this);

        VERBOSE_LOG(2) << "propagate toggle sp from data in vertex "
                       << data_in_vertex->getName() << " end.";
      }

      /*Save the results to dataout vertex.*/
      if (!seq_vertex->isOutputPort()) {
        // TODO Disregard macro, (data in /data out) have only one vertex.
        auto* data_in_vertex = (*data_in_vertexes.begin());
        auto& data_out_vertexes = seq_vertex->get_seq_out_vertexes();
        for (auto* data_out_vertex : data_out_vertexes) {
          auto* seq_data_out_net = data_out_vertex->getDesignObj()->get_net();
          // If the net is not loaded, skip this data out vertex.
          if (seq_data_out_net && seq_data_out_net->getLoads().empty()) {
            continue;
          }
          
          // If the dataout vertex is not in data path, skip this data out.
          auto capcure_clock_domain = data_out_vertex->getOwnFastestClockDomain();
          if (!capcure_clock_domain) {
            continue;
          }

          // Get the data in vertex's data.
          auto seq_toggle_sp_data = getToggleSPData(data_in_vertex);

          // Convert the seq data in toggle sp data to data out toggle sp data.
          auto seq_data_out_toggle_sp = calcSeqDataOutToggleSP(
              data_in_vertex, data_out_vertex, seq_toggle_sp_data);
          // get the fastest clock
          auto* the_pwr_graph = get_the_pwr_graph();
          auto& the_fastest_clock = the_pwr_graph->get_fastest_clock();
          data_out_vertex->addData(seq_data_out_toggle_sp._toggle_value,
                                   seq_data_out_toggle_sp._sp_value,
                                   PwrDataSource::kDataPropagation,
                                   &the_fastest_clock);
        }
      }

      return is_ok;
    };

/*Calculate the toggle and SP of the current layer starting from 1th
 * level.*/
#if MULTI_THREAD
    ThreadPool the_same_level_thread_pool(get_num_threads());
#endif
    auto& level_to_seq_vertex = the_seq_graph->get_level_to_seq_vertex();
    for (auto& [level, the_level_seq_vertexes] : level_to_seq_vertex) {
      if (level == 0) {
        continue;
      }

      VERBOSE_LOG(2) << "level " << level << " seq vertex is propagated start.";

#if MULTI_THREAD
      std::vector<std::pair<PwrSeqVertex*, std::future<unsigned>>> futures;
#endif

      for (auto* seq_vertex : the_level_seq_vertexes) {
        // If the seq_vertex is const, no need to calc.
        if (seq_vertex->isConst()) {
          continue;
        }

/*Select whether to use multithreading for propagate toggle and SP from a seq
 * vertex.*/
#if MULTI_THREAD
        futures.emplace_back(seq_vertex, the_same_level_thread_pool.enqueue(
                                             propagate_from_seq, seq_vertex));
#else
        propagate_from_seq(seq_vertex);
#endif

        if (isTrace()) {
          set_is_trace(false);
          PwrDumpGraphYaml dump_graph_yaml;
          printVertexTraceStack("prop_vertex.yaml", dump_graph_yaml);

          PwrDumpGraphViz dump_graphviz;
          printArcTraceStack("prop_arc.dot", dump_graphviz);
        }
      }

#if MULTI_THREAD
      // The same level needs to be fully calculated before moving to the next
      // level.
      for (auto& [seq_vertex, future] : futures) {
        VERBOSE_LOG(2) << "seq vertex " << seq_vertex->get_obj_name()
                       << " is wait propagated.";

        unsigned result = future.get();

        VERBOSE_LOG(2) << "seq vertex " << seq_vertex->get_obj_name()
                       << " is propagated finish "
                       << (result ? "success" : "failed");
      }
#endif

      VERBOSE_LOG(2) << "level " << level << " seq vertex is propagated end.";
    }
  }
  LOG_INFO << "propagate toggle sp end";

  double memory_delta = stats.memoryDelta();
  LOG_INFO << "propagate toggle sp memory usage " << memory_delta << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "propagate toggle sp time elapsed " << time_delta << "s";

  return 1;
}

}  // namespace ipower
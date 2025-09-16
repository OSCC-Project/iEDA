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
 * @file PwrBuildGraph.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief update the power information to sta graph.
 * @version 0.1
 * @date 2023-01-01
 */
#include "PwrBuildGraph.hh"

#include "liberty/Lib.hh"
#include "netlist/Instance.hh"
#include "string/Str.hh"

namespace ipower {

/**
 * @brief annotate the instance internal power arc.
 *
 * @param inst_power_arc the inst power arc to be annotated.
 * @param inst
 * @return unsigned
 */
unsigned PwrBuildGraph::annotateInternalPower(PwrInstArc* inst_power_arc,
                                              Instance* inst) {
  auto* src_power_vertex = inst_power_arc->get_src();
  auto* src_design_obj = src_power_vertex->getDesignObj();
  auto* snk_power_vertex = inst_power_arc->get_snk();
  auto* snk_design_obj = snk_power_vertex->getDesignObj();

  auto [src_name, src_index] =
      ieda::Str::matchBusName(src_design_obj->get_name());
  auto [snk_name, snk_index] =
      ieda::Str::matchBusName(snk_design_obj->get_name());

  // build inst power arc
  auto* lib_cell = inst->get_inst_cell();
  LibPowerArcSet* power_arc_set;
  FOREACH_POWER_ARC_SET(lib_cell, power_arc_set) {
    auto* power_arc = power_arc_set->front();
    auto [src_port_name, src_port_index] =
        ieda::Str::matchBusName(power_arc->get_src_port());
    auto [snk_port_name, snk_port_index] =
        ieda::Str::matchBusName(power_arc->get_snk_port());

    if ((src_name == src_port_name) and (snk_name == snk_port_name)) {
      // found the internal power arc set.
      inst_power_arc->set_power_arc_set(power_arc_set);
      break;
    }
  }

  return 1;
}

/**
 * @brief build power graph based on sta graph, annotate the interal power
 * information.
 *
 * @param sta_graph
 * @return unsigned
 */
unsigned PwrBuildGraph::operator()(StaGraph* sta_graph) {
  ieda::Stats stats;
  LOG_INFO << "build power graph start";

  _power_graph.set_sta_graph(sta_graph);
  // build power vertex based on sta vertex.
  StaVertex* sta_vertex;
  FOREACH_VERTEX(sta_graph, sta_vertex) {
    // build the power vertex.
    auto power_vertex = std::make_unique<PwrVertex>(sta_vertex);
    _power_graph.addStaAndPwrCrossRef(sta_vertex, power_vertex.get());
    _power_graph.addPowerVertex(std::move(power_vertex));
    if (sta_vertex->is_bidirection()) {
      auto* assistant_vertex = sta_graph->getAssistant(sta_vertex);
      LOG_FATAL_IF(!assistant_vertex) << "assistant is null.";

      auto power_assistant_vertex =
          std::make_unique<PwrVertex>(assistant_vertex);
      _power_graph.addStaAndPwrCrossRef(assistant_vertex,
                                        power_assistant_vertex.get());
      _power_graph.addPowerVertex(std::move(power_assistant_vertex));
    }
  }

  // build power arc based on sta arc.
  StaArc* sta_arc;
  FOREACH_ARC(sta_graph, sta_arc) {
    auto* sta_src_vertex = sta_arc->get_src();
    auto* sta_snk_vertex = sta_arc->get_snk();
    auto* pwr_src_vertex = _power_graph.staToPwrVertex(sta_src_vertex);
    auto* pwr_snk_vertex = _power_graph.staToPwrVertex(sta_snk_vertex);

    if (sta_arc->is_loop_disable()) {
      continue;
    }

    std::unique_ptr<PwrArc> power_arc;
    // power arc only exist for delay arc.
    if (sta_arc->isInstArc()) {
      if (sta_arc->isDelayArc()) {
        power_arc =
            std::make_unique<PwrInstArc>(pwr_src_vertex, pwr_snk_vertex);
        // annoate the internal power to inst arc.
        annotateInternalPower(dynamic_cast<PwrInstArc*>(power_arc.get()),
                              dynamic_cast<StaInstArc*>(sta_arc)->get_inst());
      }
    } else {
      // net arc
      power_arc = std::make_unique<PwrNetArc>(pwr_src_vertex, pwr_snk_vertex);
      dynamic_cast<PwrNetArc*>(power_arc.get())
          ->set_net(dynamic_cast<StaNetArc*>(sta_arc)->get_net());
    }

    if (power_arc) {
      pwr_src_vertex->addSrcArc(power_arc.get());
      pwr_snk_vertex->addSnkArc(power_arc.get());

      _power_graph.addStaAndPwrArc(sta_arc, power_arc.get());

      _power_graph.addPowerArc(std::move(power_arc));
    }
  }

  // build power cell.
  auto* nl = sta_graph->get_nl();
  Instance* design_inst;
  FOREACH_INSTANCE(nl, design_inst) {
    auto power_cell = std::make_unique<PwrCell>(design_inst);
    _power_graph.addPowerCell(std::move(power_cell));
  }

  // set the port vertexes.
  StaVertex* vertex;
  FOREACH_PORT_VERTEX(sta_graph, vertex) {
    DesignObject* obj = vertex->get_design_obj();
    if (obj->isInout()) {
      auto* input_port_vertex = _power_graph.staToPwrVertex(vertex);
      input_port_vertex->set_is_input_port();
      _power_graph.addInputPortVertex(input_port_vertex);
      auto* output_sta_vertex = sta_graph->getAssistant(vertex);
      auto* output_port_vertex = _power_graph.staToPwrVertex(output_sta_vertex);
      output_port_vertex->set_is_output_port();
      _power_graph.addOutputPortVertex(output_port_vertex);

    } else if (obj->isInput()) {
      auto* input_port_vertex = _power_graph.staToPwrVertex(vertex);
      input_port_vertex->set_is_input_port();
      _power_graph.addInputPortVertex(input_port_vertex);
    } else if (obj->isOutput()) {
      auto* output_port_vertex = _power_graph.staToPwrVertex(vertex);
      output_port_vertex->set_is_output_port();
      _power_graph.addOutputPortVertex(output_port_vertex);
    }
  }

  LOG_INFO << "build power graph end";
  double memory_delta = stats.memoryDelta();
  LOG_INFO << "build power graph memory usage " << memory_delta << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "build power graph time elapsed " << time_delta << "s";

  return 1;
}

}  // namespace ipower

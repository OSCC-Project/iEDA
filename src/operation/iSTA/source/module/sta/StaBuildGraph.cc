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
 * @file StaBuildGraph.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The functor of build sta graph from the design netlist.
 * @version 0.1
 * @date 2021-08-10
 */
#include "StaBuildGraph.hh"

#include "netlist/Netlist.hh"

namespace ista {

/**
 * @brief Build the port into graph port vertex.
 *
 * @param the_graph
 * @param port
 * @return unsigned
 */
unsigned StaBuildGraph::buildPort(StaGraph* the_graph, Port* port) {
  auto the_vertex = std::make_unique<StaVertex>(port);
  the_vertex->set_is_port();

  if (port->isInout()) {
    // for inout port, we set input as main, output as assistant.
    the_vertex->set_is_start();
    the_vertex->set_is_bidirection();
    the_graph->addStartVertex(the_vertex.get());

    auto assistant_vertex = std::make_unique<StaVertex>(port);
    assistant_vertex->set_is_port();
    assistant_vertex->set_is_end();
    assistant_vertex->set_is_bidirection();
    assistant_vertex->set_is_assistant();

    the_graph->addEndVertex(assistant_vertex.get());
    the_graph->addMainAssistantCrossReference(the_vertex.get(),
                                              std::move(assistant_vertex));

  } else if (port->isInput()) {
    the_vertex->set_is_start();
    the_graph->addStartVertex(the_vertex.get());

  } else if (port->isOutput()) {
    the_vertex->set_is_end();  // may be the port is inout, fixme.
    the_graph->addEndVertex(the_vertex.get());
  }

  the_graph->addPortVertex(port, std::move(the_vertex));

  return 1;
}

/**
 * @brief Build the inst into graph vertex and cell arc.
 *
 * @param the_graph
 * @param inst
 * @return unsigned
 */
unsigned StaBuildGraph::buildInst(StaGraph* the_graph, Instance* inst) {
  // build pin vertex
  Pin* pin;
  FOREACH_INSTANCE_PIN(inst, pin) {
    auto the_vertex = std::make_unique<StaVertex>(pin);

    if (pin->isInout()) {
      // for inout pin, we set input as main, output as assistant.
      auto assistant_vertex = std::make_unique<StaVertex>(pin);
      the_vertex->set_is_bidirection();
      assistant_vertex->set_is_bidirection();
      assistant_vertex->set_is_assistant();
      the_graph->addMainAssistantCrossReference(the_vertex.get(),
                                                std::move(assistant_vertex));
    }

    the_graph->addPinVertex(pin, std::move(the_vertex));
  }

  // lambda function, build one inst arc.
  auto build_inst_arc = [the_graph, inst](Pin* src_pin, Pin* snk_pin,
                                          LibArc* cell_arc) {
    auto src_vertex = the_graph->findVertex(src_pin);
    LOG_FATAL_IF(!src_vertex);

    auto snk_vertex = the_graph->findVertex(snk_pin);
    LOG_FATAL_IF(!snk_vertex);

    // for inout pin, we need find the assistant node, main node is input,
    // assistant node is output.
    if (snk_pin->isInout()) {
      snk_vertex = the_graph->getAssistant(*snk_vertex);
    }

    auto inst_arc =
        std::make_unique<StaInstArc>(*src_vertex, *snk_vertex, cell_arc, inst);
    (*src_vertex)->addSrcArc(inst_arc.get());
    (*snk_vertex)->addSnkArc(inst_arc.get());

    if (cell_arc->isDisableArc()) {
      inst_arc->set_is_disable_arc(true);
    }

    the_graph->addArc(std::move(inst_arc));

    if (cell_arc->isCheckArc()) {
      the_graph->addEndVertex(*snk_vertex);
      (*src_vertex)->set_is_clock();
    }

    // add clock gating check arc
    if (cell_arc->isClockGateCheckArc()) {
      (*snk_vertex)->set_is_clock_gate_end();
      (*src_vertex)->set_is_clock_gate_clock();
    }
  };

  // build inst timing arc
  LibCell* lib_cell = inst->get_inst_cell();
  for (auto& cell_arc_set : lib_cell->get_cell_arcs()) {
    auto* cell_arc = cell_arc_set->front();
    const char* src_port_name = cell_arc->get_src_port();
    const char* snk_port_name = cell_arc->get_snk_port();

    auto* src_port = lib_cell->get_cell_port_or_port_bus(src_port_name);
    LOG_FATAL_IF(!src_port) << "src port " << src_port_name << " is not found.";
    auto* snk_port = lib_cell->get_cell_port_or_port_bus(snk_port_name);
    LOG_FATAL_IF(!snk_port) << "snk port " << snk_port_name << " is not found.";

    std::vector<std::string> src_ports;
    std::vector<std::string> snk_ports;

    if (src_port->isLibertyPortBus()) {
      auto src_port_bus_size =
          dynamic_cast<LibPortBus*>(src_port)->getBusSize();
      for (unsigned src_index = 0; src_index < src_port_bus_size; ++src_index) {
        std::string one_src_port =
            Str::printf("%s[%d]", src_port_name, src_index);
        src_ports.emplace_back(std::move(one_src_port));
      }
    } else {
      src_ports.emplace_back(std::string(src_port_name));
    }

    if (snk_port->isLibertyPortBus()) {
      auto snk_port_bus_size =
          dynamic_cast<LibPortBus*>(snk_port)->getBusSize();

      for (unsigned snk_index = 0; snk_index < snk_port_bus_size; ++snk_index) {
        std::string one_snk_port =
            Str::printf("%s[%d]", snk_port_name, snk_index);
        snk_ports.emplace_back(std::move(one_snk_port));
      }
    } else {
      snk_ports.emplace_back(std::string(snk_port_name));
    }

    for (auto& one_src_port_name : src_ports) {
      for (auto& one_snk_port_name : snk_ports) {
        auto src_pin = inst->getPin(one_src_port_name.c_str());
        auto snk_pin = inst->getPin(one_snk_port_name.c_str());
        if (!src_pin || !snk_pin) {
          continue;
        }

        build_inst_arc(*src_pin, *snk_pin, cell_arc);
      }
    }
  }

  return 1;
}

/**
 * @brief Build the net into graph net arc.
 *
 * @param the_graph
 * @param net
 * @return unsigned
 */
unsigned StaBuildGraph::buildNet(StaGraph* the_graph, Net* net) {
  DesignObject* pin_port;
  DesignObject* driver = net->getDriver();
  auto driver_vertex = the_graph->findVertex(driver);

  std::pair<StaVertex*, std::vector<StaVertex*>> inout_pair{nullptr, {}};
  // for inout pin, we need find the assistant node, main node is input,
  // assistant node is output.
  if (driver && driver->isInout()) {
    if (driver->isPin()) {
      inout_pair.first = *driver_vertex;
      driver_vertex = the_graph->getAssistant(*driver_vertex);
    } else {
      inout_pair.first = the_graph->getAssistant(*driver_vertex);
    }
  }

  if (!driver_vertex) {
    DLOG_INFO << "net " << net->get_name() << " has no driver.";
    return 1;
  }

  if (driver_vertex && driver->isPin() && driver->isConst()) {
    the_graph->addConstVertex(*driver_vertex);
  } else if (!driver_vertex) {
    auto const_vertex = std::make_unique<StaVertex>(nullptr);
    const_vertex->set_is_const();
    (*driver_vertex) = const_vertex.get();
    the_graph->addConstVertex(*driver_vertex);
    the_graph->addVertex(std::move(const_vertex));
  }

  // lambda function to create net arc.
  auto create_net_arc = [](auto* src_vertex, auto* snk_vertex, auto* net) {
    auto net_arc = std::make_unique<StaNetArc>(src_vertex, snk_vertex, net);
    src_vertex->addSrcArc(net_arc.get());
    snk_vertex->addSnkArc(net_arc.get());
    return net_arc;
  };

  FOREACH_NET_PIN(net, pin_port) {
    if (driver == pin_port) {
      continue;
    }

    // build net timing arc
    auto load_vertex = the_graph->findVertex(pin_port);

    // for inout port, we need find the assistant node, main node is input,
    // assistant node is output.
    if (pin_port->isInout()) {
      if (pin_port->isPort()) {
        inout_pair.second.emplace_back(*load_vertex);
        load_vertex = the_graph->getAssistant(*load_vertex);
      } else {
        inout_pair.second.emplace_back(the_graph->getAssistant(*load_vertex));
      }
    }

    auto net_arc = create_net_arc(*driver_vertex, *load_vertex, net);

    // FIXME disable the power gate arc.
    if (auto* vertex_own_cell = (*driver_vertex)->getOwnCell();
        vertex_own_cell && vertex_own_cell->isSequentialCell() &&
        !vertex_own_cell->isICG() && !vertex_own_cell->isMacroCell()) {
      if ((*load_vertex)->is_clock()) {
        net_arc->set_is_disable_arc(true);
      }
    }

    the_graph->addArc(std::move(net_arc));
  }

  // for inout vertex, we need build another director net arc.
  if (auto* another_load_vertex = inout_pair.first; another_load_vertex) {
    for (auto* another_driver_vertex : inout_pair.second) {
      auto net_arc =
          create_net_arc(another_driver_vertex, another_load_vertex, net);
      the_graph->addArc(std::move(net_arc));
    }
  }

  return 1;
}

/**
 * @brief Build the const pin for const vertex.
 *
 * @param the_graph
 * @param inst
 * @return unsigned
 */
unsigned StaBuildGraph::buildConst(StaGraph* the_graph, Instance* inst) {
  Pin* pin;
  FOREACH_INSTANCE_PIN(inst, pin) {
    auto the_vertex = the_graph->findVertex(pin);
    LOG_FATAL_IF(!the_vertex);
    auto port_func = pin->get_cell_port()->get_func_expr();
    if (!pin->get_net() || (port_func && (rust_expr_func_is_one(port_func) ||
                                          rust_expr_func_is_zero(port_func)))) {
      (*the_vertex)->set_is_const();
      the_graph->addConstVertex(*the_vertex);

      if (port_func) {
        if (rust_expr_func_is_one(port_func)) {
          (*the_vertex)->set_is_const_vdd();
        } else if (rust_expr_func_is_zero(port_func)) {
          (*the_vertex)->set_is_const_gnd();
        }
      }
    }
  }

  return 1;
}

unsigned StaBuildGraph::operator()(StaGraph* the_graph) {
  LOG_INFO << "build graph start";

  Netlist* nl = the_graph->get_nl();
  Port* port;

  // build port vertex
  FOREACH_PORT(nl, port) { buildPort(the_graph, port); }

  Instance* inst;
  FOREACH_INSTANCE(nl, inst) { buildInst(the_graph, inst); }

  // build net arc
  Net* net;
  FOREACH_NET(nl, net) { buildNet(the_graph, net); }

  FOREACH_INSTANCE(nl, inst) { buildConst(the_graph, inst); }

  LOG_INFO << "build graph end";

  return 1;
}

}  // namespace ista

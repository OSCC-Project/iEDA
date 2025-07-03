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
 * @file PwrGraph.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The power graph class of power analysis.
 * @version 0.1
 * @date 2023-01-17
 */
#pragma once

#include <memory>
#include <vector>

#include "BTreeSet.hh"
#include "PwrArc.hh"
#include "PwrCell.hh"
#include "PwrClock.hh"
#include "PwrVertex.hh"
#include "netlist/Instance.hh"
#include "sta/StaClock.hh"
#include "sta/StaGraph.hh"

namespace ipower {

/**
 * @brief power vertex sort compare by vertex name.
 *
 */
struct PwrVertexComp {
  bool operator()(const PwrVertex* const& lhs,
                  const PwrVertex* const& rhs) const {
    return lhs->getName() > rhs->getName();
  }
};

/**
 * @brief The power analysis DAG graph.
 *
 */
class PwrGraph {
 public:
  explicit PwrGraph(StaGraph* sta_graph) : _sta_graph(sta_graph) {}
  PwrGraph() = default;
  ~PwrGraph() = default;

  PwrGraph(PwrGraph&& other) noexcept = default;
  PwrGraph& operator=(PwrGraph&& other) noexcept = default;

  using PwrVertexSet = BTreeSet<PwrVertex*, PwrVertexComp>;

  void set_sta_graph(StaGraph* sta_graph) { _sta_graph = sta_graph; }
  auto* get_sta_graph() { return _sta_graph; }

  void set_pwr_seq_graph(PwrSeqGraph* pwr_seq_graph) {
    _pwr_seq_graph = pwr_seq_graph;
  }
  auto* get_pwr_seq_graph() { return _pwr_seq_graph; }

  void addPowerVertex(std::unique_ptr<PwrVertex> vertex) {
    _vertexes.emplace_back(std::move(vertex));
  }
  auto& get_vertexes() { return _vertexes; }
  auto numVertex() { return _vertexes.size(); }

  PwrVertex* getPowerVertex(DesignObject* obj);

  void addStaAndPwrCrossRef(StaVertex* sta_vertex, PwrVertex* pwr_vertex) {
    _vertex_sta_to_pwr[sta_vertex] = pwr_vertex;
    _vertex_pwr_to_sta[pwr_vertex] = sta_vertex;
  }
  PwrVertex* staToPwrVertex(StaVertex* sta_vertex) {
    return _vertex_sta_to_pwr[sta_vertex];
  }
  StaVertex* pwrToStaVertex(PwrVertex* pwr_vertex) {
    return _vertex_pwr_to_sta[pwr_vertex];
  }

  void addPowerArc(std::unique_ptr<PwrArc> arc) {
    _arcs.emplace_back(std::move(arc));
  }
  auto& get_arcs() { return _arcs; }
  auto numArc() { return _arcs.size(); }

  void addPowerCell(std::unique_ptr<PwrCell> cell) {
    _inst_name_to_pwr_cell[cell->get_design_inst()->get_name()] = cell.get();
    _cells.emplace_back(std::move(cell));
  }
  auto& get_cells() { return _cells; }
  auto* getCell(std::string_view inst_name) {
    LOG_FATAL_IF(!_inst_name_to_pwr_cell.contains(inst_name))
        << inst_name << " is not found power cell.";
    return _inst_name_to_pwr_cell[inst_name];
  }

  PwrVertex* getDriverVertex(const std::string& net_name);

  void addInputPortVertex(PwrVertex* input_port_vertex) {
    _input_port_vertexes.insert(input_port_vertex);
  }
  void addOutputPortVertex(PwrVertex* output_port_vertex) {
    _output_port_vertexes.insert(output_port_vertex);
  }
  auto& get_input_port_vertexes() { return _input_port_vertexes; }
  auto& get_output_port_vertexes() { return _output_port_vertexes; }

  void setFastestClock(const char* clock_name, double clock_period_ns) {
    _fastest_clock.set_clock_name(clock_name);
    _fastest_clock.set_clock_period_ns(clock_period_ns);
  }
  void set_fastest_clock(PwrClock&& fastest_clock) {
    _fastest_clock = std::move(fastest_clock);
  }
  auto& get_fastest_clock() { return _fastest_clock; }

  void set_sta_clocks(Vector<StaClock*>&& sta_clocks) {
    _sta_clocks = std::move(sta_clocks);
  }
  auto& get_sta_clocks() { return _sta_clocks; }

  unsigned exec(std::function<unsigned(PwrGraph*)> the_power_func) {
    return the_power_func(this);
  }

 private:
  StaGraph* _sta_graph =
      nullptr;  //!< The sta graph for get timing information.
  PwrSeqGraph* _pwr_seq_graph = nullptr;  //!< The power sequence graph.

  std::vector<std::unique_ptr<PwrVertex>>
      _vertexes;  //!< The power graph vertex, mapped to pin/port, to calculate
                  //!< the pin internal power.
  std::vector<std::unique_ptr<PwrArc>>
      _arcs;  //!< The power graph arc, mapped to internal power arc, to
              //!< calculate the arc internal power and swtich power.
  std::vector<std::unique_ptr<PwrCell>>
      _cells;  //!< The power graph cell, mapped to design instance, to
               //!< calculate the cell leakage power.

  PwrClock _fastest_clock;        //!< The fastest clock domain used for default
                                  //!< toggle calc.
  Vector<StaClock*> _sta_clocks;  //!< The sta all clocks.

  std::map<std::string_view, PwrCell*>
      _inst_name_to_pwr_cell;  //!< The map of inst name and power cell.

  std::unordered_map<StaVertex*, PwrVertex*>
      _vertex_sta_to_pwr;  //!< The sta and pwr vertex crossref.
  std::unordered_map<PwrVertex*, StaVertex*> _vertex_pwr_to_sta;
  PwrVertexSet _input_port_vertexes;   //<! The input port vertexes of the
                                       // propagation path.
  PwrVertexSet _output_port_vertexes;  //<! The output port vertexes of the
                                       // propagation path.
};

/**
 * @brief The macro of foreach vertex, usage:
 * PwrGraph* graph;
 * PwrVertex* vertex;
 * FOREACH_PWR_VERTEX(graph, vertex)
 * {
 *    do_something_for_vertex();
 * }
 */
#define FOREACH_PWR_VERTEX(graph, vertex)                          \
  if (auto& vertexes = (graph)->get_vertexes(); !vertexes.empty()) \
    for (auto p = vertexes.begin();                                \
         p != vertexes.end() ? vertex = p->get(), true : false; ++p)

/**
 * @brief The macro of foreach arc, usage:
 * PwrGraph* graph;
 * PwrArc* arc;
 * FOREACH_PWR_ARC(graph, arc)
 * {
 *    do_something_for_arc();
 * }
 */
#define FOREACH_PWR_ARC(graph, arc)                  \
  if (auto& arcs = graph->get_arcs(); !arcs.empty()) \
    for (auto p = arcs.begin();                      \
         p != arcs.end() ? arc = p->get(), true : false; ++p)

/**
 * @brief The macro of foreach cell, usage:
 * PwrGraph* graph;
 * PwrCell* cell;
 * FOREACH_PWR_CELL(graph, cell)
 * {
 *    do_something_for_cell();
 * }
 */
#define FOREACH_PWR_CELL(graph, cell)                   \
  if (auto& cells = graph->get_cells(); !cells.empty()) \
    for (auto p = cells.begin();                        \
         p != cells.end() ? cell = p->get(), true : false; ++p)

}  // namespace ipower

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
 * @file StaGraph.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The graph for sta.
 * @version 0.1
 * @date 2021-02-17
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "BTreeMap.hh"
#include "BTreeSet.hh"
#include "StaArc.hh"
#include "StaVertex.hh"
#include "Vector.hh"
#include "netlist/Netlist.hh"

namespace ista {

class StaFunc;

/**
 * @brief The static timing analysis DAG graph.
 *
 */
class StaGraph {
 public:
  explicit StaGraph(Netlist* nl);
  ~StaGraph() = default;

  StaGraph(StaGraph&& other) noexcept = default;
  StaGraph& operator=(StaGraph&& other) noexcept = default;

  Netlist* get_nl() { return _nl; }
  void set_nl(Netlist* nl) { _nl = nl; }

  void addStartVertex(StaVertex* start_vertex);
  void removeStartVertex(StaVertex* start_vertex);
  void addEndVertex(StaVertex* end_vertex);
  void addConstVertex(StaVertex* const_vertex);

  void addVertex(std::unique_ptr<StaVertex>&& vertex) {
    _vertexes.emplace_back(std::move(vertex));
  }

  void addCrossReference(DesignObject* obj, StaVertex* the_vertex) {
    _obj2vertex[obj] = the_vertex;
    _vertex2obj[the_vertex] = obj;
  }

  void removeCrossReference(DesignObject* obj, StaVertex* the_vertex) {
    _obj2vertex.erase(obj);
    _vertex2obj.erase(the_vertex);
  }

  void addPortVertex(Port* port, std::unique_ptr<StaVertex>&& port_vertex) {
    addCrossReference(port, port_vertex.get());
    _port_vertexes.insert(port_vertex.get());

    addVertex(std::move(port_vertex));
  }

  void addPinVertex(Pin* pin, std::unique_ptr<StaVertex>&& pin_vertex) {
    addCrossReference(pin, pin_vertex.get());
    addVertex(std::move(pin_vertex));
  }

  void removePinVertex(Pin* pin, StaVertex* pin_vertex) {
    removeCrossReference(pin, pin_vertex);
    auto it = std::find_if(
        _vertexes.begin(), _vertexes.end(),
        [pin_vertex](auto& vertex) { return pin_vertex == vertex.get(); });

    LOG_FATAL_IF(it == _vertexes.end());
    _vertexes.erase(it);
  }

  void addMainAssistantCrossReference(
      StaVertex* main_vertex, std::unique_ptr<StaVertex> assistant_vertex) {
    _assistant2main[assistant_vertex.get()] = main_vertex;
    _main2assistant[main_vertex] = std::move(assistant_vertex);
  }

  StaVertex* getAssistant(StaVertex* main_vertex) {
    return _main2assistant[main_vertex].get();
  }

  std::vector<StaVertex*> getAssistants() {
    std::vector<StaVertex*> assistants;
    for (auto& [main_vertex, assistant_vertex] : _main2assistant) {
      assistants.push_back(assistant_vertex.get());
    }
    return assistants;
  }

  auto& get_main2assistant() { return _main2assistant; }

  StaVertex* getMain(StaVertex* assistant_vertex) {
    return _assistant2main[assistant_vertex];
  }

  void addArc(std::unique_ptr<StaArc>&& arc) {
    _arcs.emplace_back(std::move(arc));
  }

  void removeArc(StaArc* the_arc) {
    LOG_FATAL_IF(!std::erase_if(_arcs, [the_arc](std::unique_ptr<StaArc>& arc) {
      return arc.get() == the_arc;
    }));
  }

  BTreeSet<StaVertex*>& get_start_vertexes() { return _start_vertexes; }
  BTreeSet<StaVertex*>& get_end_vertexes() { return _end_vertexes; }
  BTreeSet<StaVertex*>& get_const_vertexes() { return _const_vertexes; }
  BTreeSet<StaVertex*>& get_port_vertexes() { return _port_vertexes; }

  std::vector<std::unique_ptr<StaVertex>>& get_vertexes() { return _vertexes; }
  void sortVertexByLevel() {
    std::stable_sort(_vertexes.begin(), _vertexes.end(),
              [](auto& lhs, auto& rhs) { return lhs->get_level() < rhs->get_level(); });

  }
  std::vector<std::unique_ptr<StaArc>>& get_arcs() { return _arcs; }

  std::size_t numVertex() const { return _vertexes.size(); }
  std::size_t numArc() const { return _arcs.size(); }

  std::optional<StaVertex*> findVertex(DesignObject* obj);
  std::optional<DesignObject*> findObj(StaVertex* vertex);

  void initGraph();
  void reset();
  void resetVertexColor();
  void resetVertexData();
  void resetArcData();

  unsigned exec(std::function<unsigned(StaGraph*)>);

 private:
  Netlist* _nl;
  BTreeSet<StaVertex*> _port_vertexes;
  BTreeSet<StaVertex*>
      _start_vertexes;  //<! The start vertexes of the timing path.
  BTreeSet<StaVertex*>
      _end_vertexes;  //<! The endpoint vertexes of the timing path.
  BTreeSet<StaVertex*> _const_vertexes;               //<! The const vertexes.
  std::vector<std::unique_ptr<StaVertex>> _vertexes;  //!< all vertexes.
  std::vector<std::unique_ptr<StaArc>> _arcs;         //!< all arcs.
  ieda::BTreeMap<DesignObject*, StaVertex*>
      _obj2vertex;  //!< design obj to vertex.
  ieda::BTreeMap<StaVertex*, DesignObject*>
      _vertex2obj;  //!< vertex to design obj.
  ieda::BTreeMap<StaVertex*, std::unique_ptr<StaVertex>>
      _main2assistant;  //!< for inout node, set one node main, another
                        //!< assistant. for inout port, we set input as main,
                        //!< output as assistant
  ieda::BTreeMap<StaVertex*, StaVertex*>
      _assistant2main;  //!< assistant to main map.
};

/**
 * @brief The macro of foreach start vertex, usage:
 * StaGraph* graph;
 * StaVertex* vertex;
 * FOREACH_START_VERTEX(graph, vertex)
 * {
 *    do_something_for_vertex();
 * }
 */
#define FOREACH_START_VERTEX(graph, vertex)               \
  if (auto& start_vertexes = graph->get_start_vertexes(); \
      !start_vertexes.empty())                            \
    for (auto p = start_vertexes.begin();                 \
         p != start_vertexes.end() ? vertex = *p, true : false; ++p)

/**
 * @brief The macro of foreach end vertex, usage:
 * StaGraph* graph;
 * StaVertex* vertex;
 * FOREACH_END_VERTEX(graph, vertex)
 * {
 *    do_something_for_vertex();
 * }
 */
#define FOREACH_END_VERTEX(graph, vertex)                                    \
  if (auto& end_vertexes = graph->get_end_vertexes(); !end_vertexes.empty()) \
    for (auto p = end_vertexes.begin();                                      \
         p != end_vertexes.end() ? vertex = *p, true : false; ++p)

/**
 * @brief The macro of foreach end vertex, usage:
 * StaGraph* graph;
 * StaVertex* vertex;
 * FOREACH_END_VERTEX(graph, vertex)
 * {
 *    do_something_for_vertex();
 * }
 */
#define FOREACH_CONST_VERTEX(graph, vertex)               \
  if (auto& const_vertexes = graph->get_const_vertexes(); \
      !const_vertexes.empty())                            \
    for (auto p = const_vertexes.begin();                 \
         p != const_vertexes.end() ? vertex = *p, true : false; ++p)

/**
 * @brief The macro of foreach vertex, usage:
 * StaGraph* graph;
 * StaVertex* vertex;
 * FOREACH_VERTEX(graph, vertex)
 * {
 *    do_something_for_vertex();
 * }
 */
#define FOREACH_VERTEX(graph, vertex)                              \
  if (auto& vertexes = (graph)->get_vertexes(); !vertexes.empty()) \
    for (auto p = vertexes.begin();                                \
         p != vertexes.end() ? vertex = p->get(), true : false; ++p)

/**
 * @brief The macro of foreach vertex, usage:
 * StaGraph* graph;
 * FOREACH_ASSISTANT_VERTEX(graph, assistant)
 * {
 *    do_something_for_assistant();
 * }
 */
#define FOREACH_ASSISTANT_VERTEX(graph, assistant)          \
  if (auto& main2assistant = (graph)->get_main2assistant(); \
      !main2assistant.empty())                              \
    for (auto& [main, assistant] : main2assistant)

/**
 * @brief The macro of foreach vertex, usage:
 * StaGraph* graph;
 * StaVertex* vertex;
 * FOREACH_PORT_VERTEX(graph, vertex)
 * {
 *    do_something_for_vertex();
 * }
 */
#define FOREACH_PORT_VERTEX(graph, vertex)                            \
  if (auto& vertexes = graph->get_port_vertexes(); !vertexes.empty()) \
    for (auto p = vertexes.begin();                                   \
         p != vertexes.end() ? vertex = *p, true : false; ++p)

/**
 * @brief The macro of foreach arc, usage:
 * StaGraph* graph;
 * StaArc* arc;
 * FOREACH_ARC(graph, arc)
 * {
 *    do_something_for_arc();
 * }
 */
#define FOREACH_ARC(graph, arc)                      \
  if (auto& arcs = graph->get_arcs(); !arcs.empty()) \
    for (auto p = arcs.begin();                      \
         p != arcs.end() ? arc = p->get(), true : false; ++p)

}  // namespace ista
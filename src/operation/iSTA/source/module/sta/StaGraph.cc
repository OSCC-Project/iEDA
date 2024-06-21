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
 * @file StaGraph.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of StaGraph.
 * @version 0.1
 * @date 2021-02-17
 */

#include "StaGraph.hh"

#include <utility>

#include "StaFunc.hh"
#include "liberty/Lib.hh"
#include "log/Log.hh"

namespace ista {

StaGraph::StaGraph(Netlist* nl) : _nl(nl) {}

/**
 * @brief Add start vertex.
 *
 * @param start_vertex
 */
void StaGraph::addStartVertex(StaVertex* start_vertex) {
  start_vertex->set_is_start();
  _start_vertexes.insert(start_vertex);
}

/**
 * @brief remove start vertex
 *
 * @param start_vertex
 */
void StaGraph::removeStartVertex(StaVertex* start_vertex) {
  auto it =
      std::find(_start_vertexes.begin(), _start_vertexes.end(), start_vertex);
  if (it != _start_vertexes.end()) {
    start_vertex->reset_is_start();
    _start_vertexes.erase(start_vertex);
  }
}

/**
 * @brief Add end vertex.
 *
 * @param end_vertex
 */
void StaGraph::addEndVertex(StaVertex* end_vertex) {
  end_vertex->set_is_end();
  _end_vertexes.insert(end_vertex);
}

/**
 * @brief  Add const vertex
 *
 * @param end_vertex
 */
void StaGraph::addConstVertex(StaVertex* const_vertex) {
  const_vertex->set_is_const();
  _const_vertexes.insert(const_vertex);
}

/**
 * @brief Init the all vertex state in the graph.
 *
 */
void StaGraph::initGraph() {
  StaVertex* vertex;
  FOREACH_VERTEX(this, vertex) {
    vertex->resetColor();
    vertex->resetLevel();
    vertex->reset_is_slew_prop();
    vertex->reset_is_delay_prop();
    vertex->reset_is_fwd();
    vertex->reset_is_bwd();
  }
}

/**
 * @brief clear graph content.
 *
 */
void StaGraph::reset() {
  _port_vertexes.clear();
  _start_vertexes.clear();
  _end_vertexes.clear();
  _const_vertexes.clear();
  _vertexes.clear();
  _arcs.clear();
  _obj2vertex.clear();
  _vertex2obj.clear();
  _main2assistant.clear();
  _assistant2main.clear();
}

/**
 * @brief Reset the all vertex color in the graph.
 *
 */
void StaGraph::resetVertexColor() {
  StaVertex* vertex;
  FOREACH_VERTEX(this, vertex) { vertex->resetColor(); }
}

/**
 * @brief Reset the sta vertex data of the all vertex in the graph.
 *
 */
void StaGraph::resetVertexData() {
  StaVertex* vertex;
  FOREACH_VERTEX(this, vertex) {
    vertex->resetSlewBucket();
    vertex->resetClockBucket();
    vertex->resetPathDelayBucket();
  }

  FOREACH_ASSISTANT_VERTEX(this, assistant) {
    assistant->resetSlewBucket();
    assistant->resetClockBucket();
    assistant->resetPathDelayBucket();
  }
}

/**
 * @brief Reset the sta arc data of the all vertex in the graph.
 *
 */
void StaGraph::resetArcData() {
  StaArc* arc;
  FOREACH_ARC(this, arc) { arc->resetArcDelayBucket(); }
}

/**
 * @brief Find the graph vertex.
 *
 * @param obj
 * @return std::optional<StaVertex*>
 */
std::optional<StaVertex*> StaGraph::findVertex(DesignObject* obj) {
  auto p = _obj2vertex.find(obj);
  if (p != _obj2vertex.end()) {
    return p->second;
  }
  return std::nullopt;
}

/**
 * @brief Find the obj in the design netlist.
 *
 * @param vertex
 * @return std::optional<StaVertex*>
 */
std::optional<DesignObject*> StaGraph::findObj(StaVertex* vertex) {
  auto p = _vertex2obj.find(vertex);
  if (p != _vertex2obj.end()) {
    return p->second;
  }
  return std::nullopt;
}

/**
 * @brief The graph execuate the sta functor.
 *
 */
unsigned StaGraph::exec(std::function<unsigned(StaGraph*)> func) {
  return func(this);
}

}  // namespace ista

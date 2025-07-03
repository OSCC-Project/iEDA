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
 * @file PwrSeqGraph.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief the sequential graph for labeling propagation levels.
 * @version 0.1
 * @date 2023-02-27
 */

#include <fstream>

#include "PwrSeqGraph.hh"
#include "sta/Sta.hh"

namespace ipower {

// static std::shared_mutex rw_mutex;  //! For synchronization read write seq data.
/**
 * @brief add power vertex of the graph.
 *
 * @param vertex
 */
void PwrSeqGraph::addPwrSeqVertex(PwrSeqVertex* vertex) {
  static std::mutex mt;
  std::lock_guard lk(mt);
  _vertexes.emplace_back(vertex);
}

/**
 * @brief add power arc of the graph.
 *
 * @param arc
 */
void PwrSeqGraph::addPwrSeqArc(PwrSeqArc* arc) {
  static std::mutex mt;
  std::lock_guard lk(mt);
  // std::unique_lock<std::shared_mutex> lock(rw_mutex);
  _arcs.emplace_back(arc);
}

/**
 * @brief find seq arc.
 * 
 * @param src_vertex 
 * @param snk_vertex 
 * @return PwrSeqArc* 
 */
PwrSeqArc* PwrSeqGraph::findSeqArc(PwrSeqVertex* src_vertex,
                                   PwrSeqVertex* snk_vertex) {
  // std::shared_lock<std::shared_mutex> lock(rw_mutex);
  auto it = std::find_if(
      _arcs.begin(), _arcs.end(), [src_vertex, snk_vertex](auto& arc) {
        return arc->get_src() == src_vertex && arc->get_snk() == snk_vertex;
      });
  return it != _arcs.end() ? it->get() : nullptr;
}

/**
 * @brief insert seq instance to and seq vertex mapping.
 *
 * @param seq_inst
 * @param seq_vertex
 */
void PwrSeqGraph::insertInstToVertex(Instance* seq_inst,
                                     PwrSeqVertex* seq_vertex) {
  static std::mutex mt;
  std::lock_guard lk(mt);
  _obj_to_vertex[seq_inst] = seq_vertex;
}

/**
 * @brief get seq vertex max fanout and max fanin.
 *
 * @return std::pair<std::size_t, std::size_t>
 */
std::pair<std::size_t, std::size_t>
PwrSeqGraph::getSeqVertexMaxFanoutAndMaxFain() {
  PwrSeqVertex* max_fanout_vertex;
  std::size_t max_fanout = 1;
  std::size_t max_fanin = 1;
  std::ranges::for_each(_vertexes, [&max_fanout, &max_fanin,
                                    &max_fanout_vertex](auto& the_vertex) {
    if (auto fanout = the_vertex->get_src_arcs().size(); fanout > max_fanout) {
      max_fanout = fanout;
    }
    if (auto fanin = the_vertex->get_snk_arcs().size(); fanin > max_fanin) {
      max_fanin = fanin;
    }
  });

  return std::make_pair(max_fanout, max_fanin);
}

/**
 * @brief print seq level information for debug.
 *
 * @param out
 */
void PwrSeqGraph::printSeqLevelInfo(std::ostream& out) {
  unsigned total_node_num = 0;
  for (auto& [level, level_seq_vertexes] : _level_to_seq_vertex) {
    out << "level: " << level << " number: " << level_seq_vertexes.size()
        << std::endl;
    total_node_num += level_seq_vertexes.size();
  }

  out << "total level node : " << total_node_num << std::endl;
}

/**
 * @brief seq arc sort by src_vertex name first, then snk vertex name.
 *
 * @param lhs
 * @param rhs
 * @return true
 * @return false
 */
bool PwrSeqArcComp::operator()(const PwrSeqArc* const& lhs,
                               const PwrSeqArc* const& rhs) const {
  auto* lhs_src_vertex = lhs->get_src();
  auto* rhs_src_vertex = rhs->get_src();
  if (lhs_src_vertex != rhs_src_vertex) {
    return (lhs_src_vertex->get_obj_name() > rhs_src_vertex->get_obj_name());
  }

  auto* lhs_snk_vertex = lhs->get_snk();
  auto* rhs_snk_vertex = rhs->get_snk();

  return (lhs_snk_vertex->get_obj_name() > rhs_snk_vertex->get_obj_name());
}

/**
 * @brief add src arc from the vertex.
 *
 * @param src_arc
 */
void PwrSeqVertex::addSrcArc(PwrSeqArc* src_arc) {
  static std::mutex mt;
  std::lock_guard lk(mt);
  _src_arcs.insert(src_arc);
}

/**
 * @brief add snk arc to the vertex.
 *
 * @param snk_arc
 */
void PwrSeqVertex::addSnkArc(PwrSeqArc* snk_arc) {
  static std::mutex mt;
  std::lock_guard lk(mt);
  _snk_arcs.insert(snk_arc);
}

/**
 * @brief get seq vertex data in vertex worst slack and depth for dataflow
 * created.
 *
 * @return std::pair<double, unsigned>
 */
std::pair<std::optional<double>, unsigned>
PwrSeqVertex::getDataInVertexWorstSlackAndDepth() {
  auto data_in_vertexes = getDataInVertexes();

  std::optional<double> worst_slack = 0;
  unsigned depth = 0;
  for (auto* data_in_vertex : data_in_vertexes) {
    auto* data_in_sta_vertex = data_in_vertex->get_sta_vertex();
    auto data_in_vertex_worst_slack =
        data_in_sta_vertex->getWorstSlackNs(AnalysisMode::kMax);
    auto data_in_vertex_depth =
        data_in_sta_vertex->GetWorstPathDepth(AnalysisMode::kMax);
    if (!worst_slack || (data_in_vertex_worst_slack &&
                         (*data_in_vertex_worst_slack < *worst_slack))) {
      worst_slack = data_in_vertex_worst_slack;
      depth = data_in_vertex_depth;
    }
  }

  return std::make_pair(*worst_slack, depth);
}

}  // namespace ipower
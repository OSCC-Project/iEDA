/**
 * @file PwrSeqGraph.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief the sequential graph for labeling propagation levels.
 * @version 0.1
 * @date 2023-02-27
 */

#include "PwrSeqGraph.hh"

#include <fstream>

namespace ipower {

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
  _arcs.emplace_back(arc);
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
  _inst_to_vertex[seq_inst] = seq_vertex;
}

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

}  // namespace ipower
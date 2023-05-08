/**
 * @file StaConstPropagation.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of const propagation.
 * @version 0.1
 * @date 2021-03-04
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "StaConstPropagation.hh"

#include <queue>

namespace ista {

/**
 * @brief Propagation the const node, const node do not need timing analysis.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned StaConstPropagation::operator()(StaGraph* the_graph) {
  auto prop_vertex = [the_graph](StaVertex* the_vertex,
                                 std::queue<StaVertex*>& bfs_queue) -> void {
    for (auto* arc : the_vertex->get_src_arcs()) {
      // propgate the const to the other vertex.
      StaVertex* snk_vertx = arc->get_snk();
      // only propgate the net arc and buf/inv arc, other arc fix me.
      if (arc->isNetArc() || arc->isBufInvArc()) {
        snk_vertx->set_is_const();
        if (the_vertex->is_const_vdd()) {
          snk_vertx->set_is_const_vdd();
        } else if (the_vertex->is_const_gnd()) {
          snk_vertx->set_is_const_gnd();
        }

        the_graph->addConstVertex(snk_vertx);

        bfs_queue.push(snk_vertx);
      }
    }
  };

  auto const_prop = [&prop_vertex](std::queue<StaVertex*>& bfs_queue) -> void {
    while (!bfs_queue.empty()) {
      StaVertex* the_vertex = bfs_queue.front();
      prop_vertex(the_vertex, bfs_queue);
      bfs_queue.pop();
    }
  };

  StaVertex* the_vertex;
  std::queue<StaVertex*> bfs_queue;
  FOREACH_CONST_VERTEX(the_graph, the_vertex) { bfs_queue.push(the_vertex); }

  const_prop(bfs_queue);

  return 1;
}

}  // namespace ista

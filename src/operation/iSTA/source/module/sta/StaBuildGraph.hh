/**
 * @file StaBuildGraph.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The functor of build sta graph from the design netlist.
 * @version 0.1
 * @date 2021-08-10
 */

#pragma once

#include "StaFunc.hh"
#include "StaGraph.hh"

namespace ista {

/**
 * @brief The functor of build graph.
 *
 */
class StaBuildGraph : public StaFunc {
 public:
  unsigned buildPort(StaGraph* the_graph, Port* port);
  unsigned buildInst(StaGraph* the_graph, Instance* inst);
  unsigned buildNet(StaGraph* the_graph, Net* net);
  unsigned buildConst(StaGraph* the_graph, Instance* inst);

  unsigned operator()(StaGraph* the_graph) override;
};

}  // namespace ista

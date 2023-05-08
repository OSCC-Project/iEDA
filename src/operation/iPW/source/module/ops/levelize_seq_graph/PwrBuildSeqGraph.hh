/**
 * @file PwrBuildSeqGraph.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Build the sequential graph.
 * @version 0.1
 * @date 2023-03-03
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include "core/PwrFunc.hh"
#include "core/PwrGraph.hh"
#include "core/PwrSeqGraph.hh"

namespace ipower {
/**
 * @brief Build the sequential graph.
 *
 */
class PwrBuildSeqGraph : public PwrFunc {
 public:
  unsigned operator()(PwrGraph* the_graph) override;
  unsigned operator()(PwrVertex* the_vertex) override;

  auto& takePwrSeqGraph() { return _seq_graph; }

 private:
  unsigned buildSeqVertexes(PwrGraph* the_graph);
  unsigned buildPortVertexes(PwrGraph* the_graph);
  
  unsigned buildSeqArcs(PwrGraph* the_graph);

  PwrSeqGraph _seq_graph;  //!< the sequential graph
};
}  // namespace ipower
/**
 * @file PwrBuildGraph.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief Build power graph, consider the power graph is the same with timing
 * graph, so we just need to update the power information to the timing graph.
 * @version 0.1
 * @date 2023-01-01
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once

#include "core/PwrGraph.hh"
#include "sta/StaFunc.hh"
#include "sta/StaGraph.hh"

namespace ipower {

/**
 * @brief The class for build power graph, we can build power graph and annotate
 * power info based on timing graph.
 *
 */
class PwrBuildGraph : public StaFunc {
 public:
  unsigned operator()(StaGraph* the_graph) override;

  auto& takePowerGraph() { return _power_graph; }

 private:
  unsigned annotateInternalPower(PwrInstArc* inst_power_arc, Instance* inst);

  PwrGraph _power_graph;  //!< The power graph to be build.
};

}  // namespace ipower

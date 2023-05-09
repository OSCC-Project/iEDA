/**
 * @file StaCrossTalkPropagation.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of crosstalk delay propagation.
 * @version 0.1
 * @date 2022-10-31
 */

#pragma once

#include "StaFunc.hh"

namespace ista {

class StaArc;
class StaVertex;
class StaGraph;

class StaCrossTalkPropagation : public StaFunc {
 public:
  unsigned operator()(StaArc* the_arc) override;
  unsigned operator()(StaVertex* the_vertex) override;
  unsigned operator()(StaGraph* the_graph) override;
};

}  // namespace ista

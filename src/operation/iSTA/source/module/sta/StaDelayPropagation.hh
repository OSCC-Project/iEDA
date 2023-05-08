/**
 * @file StaDelayPropagation.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of delay propagation.
 * @version 0.1
 * @date 2021-04-10
 */
#pragma once

#include "StaFunc.hh"

namespace ista {

class StaArc;
class StaVertex;
class StaGraph;

class StaDelayPropagation : public StaFunc {
 public:
  unsigned operator()(StaArc* the_arc);
  unsigned operator()(StaVertex* the_vertex);
  unsigned operator()(StaGraph* the_graph);

  AnalysisMode get_analysis_mode() override { return AnalysisMode::kMaxMin; }
};

}  // namespace ista

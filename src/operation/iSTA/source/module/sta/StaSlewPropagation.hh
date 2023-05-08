/**
 * @file StaSlewPropagation.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The slew propagation from input port.
 * @version 0.1
 * @date 2021-04-08
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

#include "StaFunc.hh"

namespace ista {

/**
 * @brief The slew propagation for calculating the gate delay.
 *
 */
class StaSlewPropagation : public StaFunc {
 public:
  unsigned operator()(StaArc* the_arc) override;
  unsigned operator()(StaVertex* the_vertex) override;
  unsigned operator()(StaGraph* the_graph) override;

  AnalysisMode get_analysis_mode() override { return AnalysisMode::kMaxMin; }
};

}  // namespace ista

/**
 * @file StaAnalyze.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of static timing analysis.
 * @version 0.1
 * @date 2021-03-14
 */

#pragma once

#include <vector>

#include "StaFunc.hh"
#include "StaPathData.hh"
#include "StaVertex.hh"
#include "Type.hh"

namespace ista {

/**
 * @brief The timing analysis top class.
 *
 */
class StaAnalyze : public StaFunc {
 public:
  unsigned operator()(StaGraph* the_graph) override;

 private:
  StaClockPair analyzeClockRelation(StaClockData* launch_clock_data,
                                    StaClockData* capture_clock_data);
  unsigned analyzeSetupHold(StaVertex* end_vertex, StaArc* check_arc,
                            AnalysisMode analysis_mode);
  unsigned analyzePortSetupHold(StaVertex* port_vertex,
                                AnalysisMode analysis_mode);
  unsigned analyzeClockGateCheck(StaVertex* end_vertex, StaArc* check_arc,
                                 AnalysisMode analysis_mode);
};

}  // namespace ista

// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
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

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

#pragma once

#include <set>
#include <string>
#include <vector>

#include "BufferedOption.h"
#include "RoutingTree.h"
#include "define.h"

namespace ito {
using ito::approximatelyLess;
using ito::approximatelyLessEqual;
using ito::BufferedOptionSeq;
class SetupOptimizer
{
 public:
  SetupOptimizer() {}
  ~SetupOptimizer() {}

  // open functions
  void optimizeSetup();

 private:
  TOLibertyCellSeq _available_lib_cell_sizes;

  /// init
  void init();
  void checkAndFindVioaltion(TOVertexSeq& end_pts_setup_violation);
  StaSeqPathData* worstRequiredPath();
  void initBufferCell();
  TOVertexSet getEndPoints();
  void findEndpointsWithSetupViolation(TOVertexSet end_points, TOVertexSeq &setup_violations);

  /// process
  void optimizeViolation(TOVertexSeq& end_pts_setup_violation);
  void optimizeSetup(StaVertex *vertex, bool perform_gs, bool perform_buf);
  int getFanoutNumber(Pin* pin);
  bool netConnectToOutputPort(Net* net);
  bool netConnectToPort(Net* net);
  void incrUpdateRCAndTiming();
  // std::optional<TOSlack> getNodeWorstSlack(StaVertex* node);
  bool checkSlackDecrease(TOSlack& current_slack, TOSlack& last_slack, int& number_of_decreasing_slack_iter);

  // gate sizing function
  bool performGateSizing(float cap_load, float driver_res, Pin *in_pin, Pin *driver_pin);
  bool heuristicGateSizing(float cap_load, float driver_res, Pin *in_pin, Pin *out_pin);

  // heuristic buffer insertion function
  bool performSplitBufferingIfNecessary(StaVertex *driver_vertex, int fanout);
  void insertBufferSeparateLoads(StaVertex *driver_vertex);

  // VG style buffer insertion function
  bool              performBufferingIfNecessary(Pin *driver_pin, int fanout);
  void              performVGBuffering(Pin *pin, int &num_insert_buf);
  BufferedOptionSeq findBufferSolution(RoutingTree *tree, int curr_id, int prev_id);
  BufferedOptionSeq mergeBranch(BufferedOptionSeq buf_opt_left,
                                BufferedOptionSeq buf_opt_right, Point curr_loc);
  BufferedOptionSeq addWire(BufferedOptionSeq buf_opt_seq, Point curr_loc,
                            Point prev_loc);
  BufferedOptionSeq addBuffer(BufferedOptionSeq buf_opt_seq, Point prev_loc);
  void              implementVGSolution(BufferedOption *buf_opt, Net *net);

  /// report
  void report(int begin_buffer_num, int begin_resize_num);
  void reportWNSAndTNS();
};

}  // namespace ito
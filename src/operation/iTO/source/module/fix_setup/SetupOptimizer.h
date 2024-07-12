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
#include "define.h"
#include "tree_build/TreeBuild.h"

namespace ito {
using ito::approximatelyLess;
using ito::approximatelyLessEqual;
using ito::BufferedOptionSeq;

#define toOptSetup SetupOptimizer::getInstance()

class SetupOptimizer
{
 public:
  static SetupOptimizer* getInstance()
  {
    if (nullptr == _instance) {
      _instance = new SetupOptimizer();
    }
    return _instance;
  }

  // open functions
  void optimizeSetup();

  void performBuffering(const char* net_name);

 private:
  static SetupOptimizer* _instance;
  TOLibertyCellSeq _available_lib_cell_sizes;

  bool _has_estimate_all_net = false;

  /// init
  void init();
  void checkAndFindVioaltion(TOVertexSeq& end_pts_setup_violation);
  StaSeqPathData* worstRequiredPath();
  void initBufferCell();
  TOVertexSet getEndPoints();
  void findEndpointsWithSetupViolation(TOVertexSet end_points, TOVertexSeq &setup_violations);

  /// process
  void optimizeViolationProcess(TOVertexSeq& end_pts_setup_violation);
  void optimizeSetupViolation(StaVertex *vertex, bool perform_gs, bool perform_buf); // main optimzation function
  int getFanoutNumber(Pin* pin);
  bool netConnectToOutputPort(Net* net);
  bool netConnectToPort(Net* net);
  void incrUpdateRCAndTiming();
  bool checkSlackDecrease(TOSlack& current_slack, TOSlack& last_slack, int& number_of_decreasing_slack_iter);

  // gate sizing function
  bool performGateSizing(float cap_load, float driver_res, Pin *in_pin, Pin *driver_pin);
  bool heuristicGateSizing(float cap_load, float driver_res, Pin *in_pin, Pin *out_pin);

  // heuristic buffer insertion function
  bool performSplitBufferingIfNecessary(StaVertex *driver_vertex, int fanout);
  void insertBufferDivideFanout(StaVertex *driver_vertex);

  // buffer insertion function
  bool performBufferingIfNecessary(Pin* driver_pin, int fanout);
  // for VG style buffer insertion
  void performVGBuffering(Pin* pin, int& num_insert_buf);
  void implementVGSolution(BufferedOption* buf_opt, Net* net);

  /// report
  void report(int begin_buffer_num, int begin_resize_num);
  void reportWNSAndTNS();

  /// constuctor
  SetupOptimizer() {}
  ~SetupOptimizer() {}
};

}  // namespace ito
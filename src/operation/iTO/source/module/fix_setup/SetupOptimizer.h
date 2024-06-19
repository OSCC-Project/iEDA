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
#include "ViolationOptimizer.h"

#include "ids.hpp"

namespace ito {
using ito::BufferedOptionSeq;
using ito::approximatelyLess;
using ito::approximatelyLessEqual;
class SetupOptimizer {
 public:
  SetupOptimizer(DbInterface *dbinterface);

  ~SetupOptimizer() {
    delete _parasitics_estimator;
    delete _violation_fixer;
  }
  SetupOptimizer(const SetupOptimizer &other) = delete;
  SetupOptimizer(SetupOptimizer &&other) = delete;

  // open functions
  void optimizeSetup();

 private:
  void initBufferCell();

  void optimizeSetup(StaSeqPathData *worst_path, TOSlack path_slack);
  void optimizeSetup(StaVertex *vertex, TOSlack path_slack);

  void performBuffering(Pin *pin);

  void insertBufferSeparateLoads(StaVertex *drvr_vertex, TOSlack drvr_slack);

  LibertyCell *repowerCell(LibertyPort *in_port, LibertyPort *drvr_port, float load_cap,
                          float prev_drive);

  BufferedOptionSeq bottomUpBuffering(RoutingTree *tree, int curr_id, int prev_id,
                                      int level);

  BufferedOptionSeq mergeBranch(BufferedOptionSeq buf_opt_left,
                                BufferedOptionSeq buf_opt_right, Point curr_loc);

  BufferedOptionSeq addWireAndBuffer(BufferedOptionSeq buf_opt_seq,
                                     Point curr_loc, Point prev_loc, int level);

  BufferedOptionSeq addBuffer(BufferedOptionSeq buf_opt_seq, Point prev_loc);

  void topDownImplementBuffering(BufferedOption *buf_opt, Net *net, int level);

  float calcDelayOfBuffer(LibertyCell *buffer_cell, float load_cap);
  float calcDelayOfBuffer(LibertyCell *buffer_cell, float load_cap, TransType rf);

  float calcDelayOfGate(LibertyPort *drvr_port, float load_cap, TransType rf);

  float calcDelayOfGate(LibertyPort *drvr_port, float load_cap);

  int getFanoutNumber(Pin *pin);

  bool netConnectToOutputPort(Net *net);

  void setLocation(Instance *inst, int x, int y);

  StaSeqPathData *worstRequiredPath();

  bool netConnectToPort(Net *net);

  TOSlack getWorstSlack(StaVertex *vertex, AnalysisMode mode);
  TOVertexSet getEndPoints();
  void      findEndpointsWithSetupViolation(TOVertexSet end_points, TOSlack slack_margin,
                                            TOVertexSeq &setup_violations);

  // data
  DbInterface     *_db_interface;
  TimingEngine    *_timing_engine;
  TimingDBAdapter *_db_adapter;

  EstimateParasitics *_parasitics_estimator;
  ViolationOptimizer *_violation_fixer;

  int _resize_instance_count;
  int _inserted_buffer_count;

  // to name the instance
  int _insert_instance_index;
  // to name the net
  int _make_net_index;

  int _dbu;

  static int _rise;
  static int _fall;

  TOLibertyCellSeq _available_buffer_cells;

  friend class HoldOptimizer;
};

} // namespace ito
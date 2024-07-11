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

#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "define.h"

namespace ito {

const int _mode_max = (int) AnalysisMode::kMax - 1;
const int _mode_min = (int) AnalysisMode::kMin - 1;

#define toOptHold HoldOptimizer::getInstance()

class HoldOptimizer
{
 public:
  static HoldOptimizer* getInstance()
  {
    if (nullptr == _instance) {
      _instance = new HoldOptimizer();
    }
    return _instance;
  }

  // open functions
  void optimizeHold();

 private:
  float _target_slack = 0.0;

  /// init
  void init();


  /// buffer operation
  void initBufferCell();
  void insertHoldDelay(string insert_buf_name, string pin_name, int insert_number = 1);
  void calcBufferCap();
  void insertBufferOptHold(StaVertex* driver_vertex, int insert_number, TODesignObjSeq& pins_loaded);
  void insertLoadBuffer(LibCell* load_buffer, StaVertex* driver_vtx, int insert_num);
  void insertLoadBuffer(TOVertexSeq fanins);
  LibCell *ensureInsertBufferSize();

  /// process
  void process();
  int checkAndOptimizeHold();
  bool checkAndFindVioaltion();
  int performOptimizationProcess();
  void optimizeHoldViolation();// main optimization function
  int optHoldViolationEnd(TOVertexSeq fanins);
  bool findEndpointsWithHoldViolation(TOVertexSet end_points, TOSlack& worst_slack, TOVertexSet& hold_violations);
  void calcStaVertexSlacks(StaVertex* vertex, TOSlacks slacks);
  TOSlack calcSlackGap(StaVertex* vertex);
  float calcHoldDelayOfBuffer(LibCell* buffer);
  TOVertexSet getEndPoints();
  TOVertexSeq getFanins(TOVertexSet end_points);
  bool isFindEndpointsWithHoldViolation(TOSlack &worst_slack);

  /// report
  void report(int begin_buffer_num);
  void reportWNSAndTNS();

  /// constuctor
  HoldOptimizer();
  ~HoldOptimizer() {}

  static HoldOptimizer* _instance;

  bool _has_estimate_all_net = false;
  TOLibertyCellSeq _available_buffer_cells;
  LibCell *_hold_insert_buf_cell;
  TODelay _hold_insert_buf_cell_delay = 0.0;

  TOVertexSet _all_end_points;
  TOVertexSet _end_pts_hold_violation;

  bool _allow_setup_violation = true;
  int _max_numb_insert_buf = 0;

  std::vector<std::pair<double, LibCell*>> _buffer_cap_pair;
};

}  // namespace ito
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

/////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2019, The Regents of the University of California
// All rights reserved.
//
// BSD 3-Clause License
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////////


#pragma once

#include "ViolationOptimizer.h"
#include "ids.hpp"

namespace ito {

class HoldOptimizer {
 public:
  HoldOptimizer(DbInterface *dbinterface);

  ~HoldOptimizer() {
    delete _parasitics_estimator;
    delete _violation_fixer;
  }
  HoldOptimizer(const HoldOptimizer &other) = delete;
  HoldOptimizer(HoldOptimizer &&other) = delete;

  // open functions
  void optimizeHold();

  void insertHoldDelay(string insert_buf_name, string pin_name, int insert_number = 1);

 private:
  int checkAndOptimizeHold(TOVertexSet end_points, LibertyCell *insert_buf_cell);

  void initBufferCell();

  void calcBufferCap();

  LibertyCell *findBufferWithMaxDelay();

  bool findEndpointsWithHoldViolation(TOVertexSet end_points,
                                      TOSlack &worst_slack, TOVertexSet &hold_violations);

  int  fixHoldVioPath(TOVertexSeq fanins, LibertyCell *insert_buffer_cell);

  void insertBufferDelay(StaVertex *drvr_vertex, int insert_number,
                         TODesignObjSeq &load_pins, LibertyCell *insert_buffer_cell);

  void  calcStaVertexSlacks(StaVertex *vertex,
                     TOSlacks slacks);
  TOSlack calcSlackGap(StaVertex *vertex);

  void setLocation(Instance *inst, int x, int y);

  float calcHoldDelayOfBuffer(LibertyCell *buffer);

  TOVertexSet getEndPoints();

  TOVertexSet getFanins(TOVertexSet end_points);

  TOVertexSeq sortFanins(TOVertexSet fanins);

  TOSlack getWorstSlack(AnalysisMode mode);
  TOSlack getWorstSlack(StaVertex *vertex, AnalysisMode mode);

  void insertLoadBuffer(LibertyCell *load_buffer, StaVertex *drvr_vtx, int insert_num);
  void insertLoadBuffer(TOVertexSeq fanins);

  void reportWNSAndTNS();

  // data
  DbInterface     *_db_interface;
  TimingEngine    *_timing_engine;
  TimingDBAdapter *_db_adapter;

  EstimateParasitics *_parasitics_estimator;
  ViolationOptimizer *_violation_fixer;

  TOLibertyCellSeq _available_buffer_cells;

  int _dbu;

  float _target_slack = 0.0;
  bool  _allow_setup_violation = true;
  int   _max_numb_insert_buf = 0;

  int _number_insert_buffer = 0;
  int _inserted_load_buffer_count = 0;

  // to name the instance
  int _insert_instance_index = 1;
  int _insert_load_instance_index = 1;
  // to name the net
  int _make_net_index = 1;
  int _make_load_net_index = 1;

  static int _mode_max;
  static int _mode_min;
  static int _rise;
  static int _fall;

  vector<pair<double, LibertyCell *>> _buffer_cap_pair;
};

} // namespace ito
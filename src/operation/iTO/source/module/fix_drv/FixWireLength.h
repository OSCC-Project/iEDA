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

#include "DbInterface.h"
#include "RoutingTree.h"
// #include "DesignCalculator.h"
// #include "EstimateParasitics.h"

#include "ids.hpp"

namespace ito {
class FixWireLength {
 public:
  FixWireLength(ista::TimingEngine *timing);
  ~FixWireLength() = default;

  void fixMaxLength(int max_length);

  void set_insert_buffer(LibertyCell *insert_buf);

  void fixMaxLength(Net *net, int max_length, bool fix = false);
 private:

  void fixMaxLength(RoutingTree *tree, int curr_pt, int prev_pt, Net *net, int max_length,
                    int level,
                    // Return values.
                    int &wire_length, TODesignObjSeq &load_pins);

  template <class T1, class T2>
  void determineFixSide(T1 max_numb, T2 left, T2 middle, T2 right, bool &fix_left,
                        bool &fix_middle, bool &fix_right);

  void insertBuffer(int x, int y, Net *net, LibertyCell *insert_buf_cell, int level,
                    int &wire_length, TODesignObjSeq &load_pins);

  void setLocation(Instance *inst, int x, int y);

  TimingEngine    *_timing_engine;
  TimingDBAdapter *_db_adapter;

  LibertyCell *_insert_buffer_cell;
  // to name the instance
  int _insert_instance_index = 1;
  // to name the net
  int _make_net_index = 1;
};
} // namespace ito

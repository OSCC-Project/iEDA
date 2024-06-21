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

#include "DbInterface.h"
#include "DesignCalculator.h"
#include "EstimateParasitics.h"
#include "GDSwriter.h"

#include "ids.hpp"

namespace ito {

class ViolationOptimizer {
 public:
  ViolationOptimizer(DbInterface *dbinterface);
  ~ViolationOptimizer() { delete _parasitics_estimator; }
  ViolationOptimizer(const ViolationOptimizer &other) = delete;
  ViolationOptimizer(ViolationOptimizer &&other) = delete;

  void fixViolations();
  void fixViolations(const char *net_name);

 private:
  void initBuffer();
  void repairViolations(Net *net, StaVertex *drvr, bool check_slew, bool check_cap,
                        int &repair_count, int &slew_violations, int &cap_violations,
                        int &length_violations);

  void checkViolations();

  void fixViolations(RoutingTree *tree, int curr_pt, int prev_pt, Net *net, float max_cap,
                     int level, int &wire_length, float &pin_cap,
                     TODesignObjSeq &load_pins);

  void fixLargeNet(Net *net, int max_fanout, LibertyCell *insert_buf_cell);

  RoutingTree *makeClusterTree(Net *net);

  template <class T1, class T2>
  void determineFixSide(T1 max_numb, T2 left, T2 middle, T2 right, bool &fix_left,
                        bool &fix_middle, bool &fix_right);

  void insertBuffer(int x, int y, Net *net, LibertyCell *insert_buf_cell, int level,
                    int &wire_length, float &cap, TODesignObjSeq &load_pins);

  void checkFanoutViolation(DesignObject *drvr_pin, double &max_fanout,
                            int &fanout_violations, bool &repair_fanout,
                            vector<int> &fanouts);

  void checkCapacitanceViolation(DesignObject *drvr_pin, double &max_drvr_cap,
                                 int &cap_violations, bool &repair_cap);

  void   checkSlewViolation(DesignObject *drvr_pin, double &max_drvr_cap,
                            int &slew_violations, bool &repair_slew);
  double calcLoadCap(LibertyPort *drvr_port, double slew);
  double calcSlewDiff(LibertyPort *drvr_port, double target_slew, double load_cap);
  void   calcGateRiseFallDelays(LibertyPort *drvr_port, float load_cap, TODelay delays[],
                                TOSlew slews[]);
  void gateRiseFallDelay(TransType rf, LibertyArc *arc, float load_cap, TODelay delays[],
                         TOSlew slews[]);

  void setLocation(Instance *inst, int x, int y);

  void increDesignArea(float delta) { _db_interface->increDesignArea(delta); }

  bool repowerInstance(Pin *drvr_pin);
  bool repowerInstance(Instance *inst, LibertyCell *replace);

  double calcDelayOfBuffer(LibertyCell *buffer_cell, float load);

  int portFanoutLoadNum(LibertyPort *port);

  bool netConnectToPort(Net *net);

  DbInterface     *_db_interface;
  TimingEngine    *_timing_engine;
  TimingDBAdapter *_db_adapter;

  bool _check_cap = true;
  bool _check_slew = true;

  vector<float>        _slew_record;
  vector<const char *> _fanout_vio_net;

  int _number_resized_instance;
  int _number_insert_buffer;

  // to name the instance
  int _insert_instance_index = 1;
  // to name the net
  int _make_net_index = 1;

  int _dbu;

  static int _rise;
  static int _fall;

  // If there are still a violation nets, the secondary fix is performed.
  vector<Net *> _still_violation_net;

  // Instances with multiple output ports that have been resized.
  std::set<Instance *> _resized_multi_output_insts;

  EstimateParasitics *_parasitics_estimator;

  LibertyCell *_insert_buffer_cell;

  friend class SetupOptimizer;
  friend class HoldOptimizer;
};

} // namespace ito
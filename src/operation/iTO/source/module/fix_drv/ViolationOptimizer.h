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

#include "EstimateParasitics.h"
#include "GDSwriter.h"
#include "data_manager.h"
#include "define.h"
#include "tree_build/TreeBuild.h"

namespace ito {

#define toOptDrv ViolationOptimizer::getInstance()

class ViolationOptimizer
{
 public:
  static ViolationOptimizer* getInstance()
  {
    if (nullptr == _instance) {
      _instance = new ViolationOptimizer();
    }
    return _instance;
  }

  void fixViolations();
  void fixSpecialNet(const char* net_name);

 private:
  static ViolationOptimizer* _instance;

  bool _has_estimate_all_net = false;

  ista::LibCell* _insert_buffer_cell;
  // If there are still a violation nets, the secondary fix is performed.
  std::map<ista::Net*, double> _violation_nets_map;
  int _number_slew_violation_net = 0;
  int _number_cap_violation_net = 0;
  double _max_buf_load_cap = 0.0;
  double _slew_2_cap_factor = 1.0;

  /// init
  bool init();

  /// process
  void checkAndRepair();
  void iterCheckAndRepair();

  /// check
  void checkViolations();
  bool isNeedRepair(ista::Net* net, double& cap_load_allowed_max);
  bool checkCapacitanceViolation(double &max_driver_cap, DesignObject *driver_pin);
  bool checkSlewViolation(double &max_driver_cap, DesignObject *driver_pin);

  double calcLoadCap(ista::LibPort* driver_port, double slew);
  double calcSlew(ista::LibPort* driver_port, double cap_load);
  bool netConnectToPort(ista::Net* net);
  int portFanoutLoadNum(ista::LibPort* port);
  void dereaseSlewCapFactor(int prev_violation_num, int last_violation_num);

  /// repair
  void optimizeViolationNet(ista::Net* net, double cap_load_allowed_max);
  void repairViolationNetByDP(TreeBuild* tree, int curr_pt, int father_pt, ista::Net* net, float cap_load_allowed_max, int& wire_length,
                              float& pin_cap, TODesignObjSeq& pins_loaded);
  template <class T1, class T2>
  void determineFixSide(T1 max_numb, T2 left, T2 middle, T2 right, bool& fix_left, bool& fix_middle, bool& fix_right);

  /// buffer operation
  bool initBuffer();
  void insertBuffer(int x, int y, ista::Net* net, ista::LibCell* insert_buf_cell, int& wire_length, float& cap,
                    TODesignObjSeq& pins_loaded);

  /// constuctor
  ViolationOptimizer() {};
  ~ViolationOptimizer() {};
};

}  // namespace ito
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
 * @file GOCA.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include <string>
#include <vector>

#include "CTSAPI.hpp"
#include "ClockTopo.h"
#include "CtsCellLib.h"
#include "CtsConfig.h"
#include "CtsInstance.h"
#include "Node.hh"
#include "log/Log.hh"

namespace icts {
class GOCA
{
 public:
  GOCA() = delete;
  GOCA(const std::string& net_name, const std::vector<CtsInstance*>& instances) : _net_name(net_name), _instances(instances)
  {
    auto* config = CTSAPIInst.get_config();
    // unit
    _unit_res = CTSAPIInst.getClockUnitRes() / 1000;
    _unit_cap = CTSAPIInst.getClockUnitCap();
    _db_unit = CTSAPIInst.getDbUnit();
    // constraint
    _skew_bound = config->get_skew_bound();
    _max_cap = config->get_max_cap();
    _max_buf_tran = config->get_max_buf_tran();
    _max_sink_tran = config->get_max_sink_tran();
    _max_fanout = config->get_max_fanout();
    _max_length = config->get_max_length();
    // lib
    _delay_libs = CTSAPIInst.getAllBufferLibs();
  }

  ~GOCA() = default;
  // run
  void run();

 private:
  // flow
  void downscale();
  void globalAssign();
  void clustering(const std::vector<Node*>& nodes);

  // get
  std::vector<ClockTopo> get_clk_topos() const { return _clock_topos; }
  // report
  void reportTiming() const;
  // member
  std::string _net_name;
  std::vector<CtsInstance*> _instances;
  std::vector<ClockTopo> _clock_topos;

  // design info
  // unit
  double _unit_res = 0.0;
  double _unit_cap = 0.0;
  size_t _db_unit = 0;
  // constraint
  double _skew_bound = 0.0;
  double _max_cap = 0.0;
  double _max_buf_tran = 0.0;
  double _max_sink_tran = 0.0;
  size_t _max_fanout = 0;
  double _max_length = 0;
  // lib
  std::vector<CtsCellLib*> _delay_libs;
};
}  // namespace icts
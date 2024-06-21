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
#include "DbInterface.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "builder.h"

namespace ino {
DbInterface *DbInterface::_db_interface = nullptr;

DbInterface *DbInterface::get_db_interface(NoConfig *config, IdbBuilder *idb,
                                           TimingEngine *timing) {
  static std::mutex mt;
  if (_db_interface == nullptr) {
    std::lock_guard<std::mutex> lock(mt);
    if (_db_interface == nullptr) {
      _db_interface = new DbInterface(config);
      _db_interface->_timing_engine = timing;
      _db_interface->_idb = idb;
      _db_interface->initData();
    }
  }
  return _db_interface;
}

void DbInterface::destroyDbInterface() {
  if (_db_interface != nullptr) {
    delete _db_interface;
    _db_interface = nullptr;
  }
}

void DbInterface::initData() {
  // log report
  string report_path = _config->get_report_file();
  _reporter = new Reporter(report_path);
  _reporter->reportTime(true);
}

void DbInterface::set_eval_data() {
  if (!_eval_data.empty()) {
    _eval_data.clear();
  }
  auto clk_list = _timing_engine->getClockList();
  for (auto clk : clk_list) {
    auto clk_name = clk->get_clock_name();
    auto setup_wns = _timing_engine->getWNS(clk_name, ista::AnalysisMode::kMax);
    auto setup_tns = _timing_engine->getTNS(clk_name, ista::AnalysisMode::kMax);
    auto hold_wns = _timing_engine->getWNS(clk_name, ista::AnalysisMode::kMin);
    auto hold_tns = _timing_engine->getTNS(clk_name, ista::AnalysisMode::kMin);
    auto freq = 1000.0 / (clk->getPeriodNs() - setup_wns);
    _eval_data.push_back({clk_name, setup_wns, setup_tns, hold_wns, hold_tns, freq});
  }
}

} // namespace ino

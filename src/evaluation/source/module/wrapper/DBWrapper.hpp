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
#ifndef SRC_PLATFORM_EVALUATOR_SOURCE_WRAPPER_DBWRAPPER_HPP_
#define SRC_PLATFORM_EVALUATOR_SOURCE_WRAPPER_DBWRAPPER_HPP_

#include "Config.hpp"
#include "EvalDB.hpp"
#include "EvalLog.hpp"

namespace eval {

using namespace idb;

class DBWrapper
{
 public:
  DBWrapper() = delete;
  explicit DBWrapper(Config* config);
  DBWrapper(const DBWrapper&) = delete;
  DBWrapper(DBWrapper&&) = delete;
  ~DBWrapper();

  DBWrapper& operator=(const DBWrapper&) = delete;
  DBWrapper& operator=(DBWrapper&&) = delete;

  const Layout* get_layout() const { return _eval_db->_layout; }
  Design* get_design() const { return _eval_db->_design; }
  EvalDB* get_eval_db() const { return _eval_db; }

 private:
  DBConfig _db_config;
  CongConfig _cong_config;
  DRCConfig _drc_config;
  GDSWrapperConfig _gds_wrapper_config;
  PowerConfig _power_config;
  TimingConfig _timing_config;
  WLConfig _wl_config;

  EvalDB* _eval_db;

  void initIDB();
  void wrapIDBData();
  void wrapLayout(IdbLayout* idb_layout);
  void wrapDesign(IdbDesign* idb_design);
  // wrap different netlists
  void wrapWLNetlists(IdbDesign* idb_design);
  void wrapCongNetlists(IdbDesign* idb_design);
  void wrapGDSNetlist(IdbBuilder* idb_builder);

  // wrap different pins
  WLPin* wrapWLPin(IdbPin* idb_pin);
  CongPin* wrapCongPin(IdbPin* idb_pin);
  GDSPin* wrapGDSPin(IdbPin* idb_pin);
  // wrap instances
  void wrapInstances(IdbDesign* idb_design);

  bool isCoreOverlap(IdbInstance* idb_inst);
  std::string fixSlash(std::string raw_str);
};

}  // namespace eval

#endif  // SRC_PLATFORM_EVALUATOR_SOURCE_WRAPPER_DBWRAPPER_HPP_

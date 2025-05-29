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

#include "JsonParser.h"
#include "NoConfig.h"
#include "Reporter.h"
#include "api/TimingEngine.hh"
#include "builder.h"
#include "ids.hpp"

namespace ino {
using idb::IdbBuilder;
using ista::TimingEngine;

using ino::NoConfig;

const float kInf = 1E+30F;

struct EvalData {
  std::string name;
  double      setup_wns;
  double      setup_tns;
  double      hold_wns;
  double      hold_tns;
  double      freq;
};

class DbInterface {
 public:
  static DbInterface *get_db_interface(NoConfig *config, IdbBuilder *idb,
                                       TimingEngine *timing = nullptr);
  static void         destroyDbInterface();

  TimingEngine *get_timing_engine() { return _timing_engine; }

  IdbBuilder *get_idb() { return _idb; }

  string get_output_def_file() { return _config->get_output_def_file(); }
  string get_insert_buffer() { return _config->get_insert_buffer(); }

  int get_max_fanout() { return _config->get_max_fanout(); }

  Reporter *report() { return _reporter; }

  void                  set_eval_data();
  std::vector<EvalData> eval_data() { return _eval_data; }

 private:
  DbInterface(NoConfig *config) : _config(config) {};
  ~DbInterface() {};

  void initData();

  static DbInterface *_db_interface;

  NoConfig     *_config = nullptr;
  IdbBuilder   *_idb = nullptr;
  TimingEngine *_timing_engine = nullptr;

  Reporter *_reporter = nullptr;

  std::vector<EvalData> _eval_data;
};
} // namespace ino

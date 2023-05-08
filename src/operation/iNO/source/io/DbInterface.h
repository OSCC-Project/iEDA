#pragma once

#include "JsonParser.h"
#include "NoConfig.h"
#include "Reporter.h"
#include "ids.hpp"

namespace ino {
using idb::IdbBuilder;
using ista::TimingEngine;

using ino::NoConfig;

const float kInf = 1E+30F;

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

 private:
  DbInterface(NoConfig *config) : _config(config){};
  ~DbInterface(){};

  void initData();

  static DbInterface *_db_interface;

  NoConfig     *_config = nullptr;
  IdbBuilder   *_idb = nullptr;
  TimingEngine *_timing_engine = nullptr;

  Reporter *_reporter = nullptr;
};
} // namespace ino

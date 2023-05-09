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
#include "NoApi.hpp"
#include "NoConfig.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "builder.h"
#include "iNO.h"
#include "idm.h"

namespace ino {

NoApi *NoApi::_no_api_instance = nullptr;

NoApi &NoApi::getInst() {
  if (_no_api_instance == nullptr) {
    _no_api_instance = new NoApi();
  }
  return *_no_api_instance;
}

void NoApi::destroyInst() {
  if (_no_api_instance != nullptr) {
    delete _no_api_instance;
    _no_api_instance = nullptr;
  }
}

void NoApi::initNO(const std::string &ITO_CONFIG_PATH) {
  _ino = new ino::iNO(ITO_CONFIG_PATH);
}

void NoApi::iNODataInit(idb::IdbBuilder *idb, ista::TimingEngine *timing) {
  if (nullptr == idb) {
    // init idb
    idb = initIDB();
  }

  if (nullptr == timing) {
    // init timing
    timing = initISTA(idb);
  }

  _idb = idb;
  _timing_engine = timing;

  _ino->initialization(idb, timing);
}

idb::IdbBuilder *NoApi::initIDB() {
  // if (dmInst->get_idb_builder()) {
  //   return dmInst->get_idb_builder();
  // }
  auto idb_builder = new IdbBuilder();

  NoConfig      *no_config = _ino->get_config();
  string         def_file = no_config->get_def_file();
  vector<string> lef_files = no_config->get_lef_files();

  idb_builder->buildLef(lef_files);
  idb_builder->buildDef(def_file);
  return idb_builder;
}

ista::TimingEngine *NoApi::initISTA(idb::IdbBuilder *idb) {
  auto timing_engine = ista::TimingEngine::getOrCreateTimingEngine();

  NoConfig            *no_config = _ino->get_config();
  const char          *design_work_space = no_config->get_design_work_space().c_str();
  vector<const char *> lib_files;
  for (auto &lib : no_config->get_lib_files()) {
    lib_files.push_back(lib.c_str());
  }

  timing_engine->set_num_threads(50);
  timing_engine->set_design_work_space(design_work_space);
  timing_engine->readLiberty(lib_files);

  auto idb_adapter = std::make_unique<TimingIDBAdapter>(timing_engine->get_ista());
  idb_adapter->set_idb(idb);
  idb_adapter->convertDBToTimingNetlist();
  timing_engine->set_db_adapter(std::move(idb_adapter));

  const char *sdc_file = no_config->get_sdc_file().c_str();
  if (sdc_file != nullptr) {
    timing_engine->readSdc(sdc_file);
  }

  timing_engine->buildGraph();
  timing_engine->updateTiming();
  return timing_engine;
}

void NoApi::fixFanout() { _ino->fixFanout(); }

void NoApi::saveDef(string saved_def_path) {
  if (saved_def_path.empty()) {
    saved_def_path = _ino->get_config()->get_output_def_file();
  }
  _idb->saveDef(saved_def_path);
}

NoConfig *NoApi::get_no_config() { return _ino->get_config(); }

void NoApi::reportTiming() { _timing_engine->reportTiming(); }
} // namespace ino

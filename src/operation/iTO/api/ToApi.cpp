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
#include "ToApi.hpp"
#include "CTSViolationFixer.h"
#include "ToConfig.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "builder.h"
#include "iTO.h"
#include "idm.h"

namespace ito {

ToApi *ToApi::_to_api_instance = nullptr;

Tree *ToApi::get_tree(const int &size) { return new Tree(size); }

void ToApi::addTopoEdge(Tree *topo, const int &first_id, const int &second_id,
                        const int &x1, const int &y1, const int &x2, const int &y2) {
  topo->idToLocation(first_id, Point(x1, y1));
  topo->idToLocation(second_id, Point(x2, y2));
  topo->add_edge(ito::Edge(first_id, second_id));
}
void ToApi::topoIdToDesignObject(ito::Tree *topo, const int &id,
                                 ista::DesignObject *sta_pin) {
  topo->idToDesignObject(id, sta_pin);
}
void   ToApi::topoSetDriverId(ito::Tree *topo, const int &id) { topo->set_drvr_id(id); }
ToApi &ToApi::getInst() {
  if (_to_api_instance == nullptr) {
    _to_api_instance = new ToApi();
  }
  return *_to_api_instance;
}

void ToApi::destroyInst() {
  if (_to_api_instance != nullptr) {
    delete _to_api_instance;
    _to_api_instance = nullptr;
  }
}

void ToApi::initTO(const std::string &ITO_CONFIG_PATH) {
  if (_ito == nullptr) {
    _ito = new ito::iTO(ITO_CONFIG_PATH);
  }
}

void ToApi::iTODataInit(idb::IdbBuilder *idb, ista::TimingEngine *timing) {
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

  _ito->initialization(idb, timing);
}

void ToApi::resetiTOData(idb::IdbBuilder *idb, ista::TimingEngine *timing) {
  LOG_ERROR_IF(!idb) << "[ERROR] Function loss parameter idb::IdbBuilder.";
  if (nullptr == timing) {
    timing = initISTA(idb);
  }

  _idb = idb;
  _timing_engine = timing;

  _ito->resetInitialization(idb, timing);
}

idb::IdbBuilder *ToApi::initIDB() {
  // if (dmInst->get_idb_builder()) {
  //   return dmInst->get_idb_builder();
  // }
  auto idb_builder = new IdbBuilder();

  ToConfig      *to_config = _ito->get_config();
  string         def_file = to_config->get_def_file();
  vector<string> lef_files = to_config->get_lef_files();

  idb_builder->buildLef(lef_files);
  idb_builder->buildDef(def_file);
  return idb_builder;
}

ista::TimingEngine *ToApi::initISTA(idb::IdbBuilder *idb) {
  auto timing_engine = ista::TimingEngine::getOrCreateTimingEngine();

  ToConfig            *to_config = _ito->get_config();
  const char          *design_work_space = to_config->get_design_work_space().c_str();
  vector<const char *> lib_files;
  for (auto &lib : to_config->get_lib_files()) {
    lib_files.push_back(lib.c_str());
  }

  timing_engine->set_num_threads(50);
  timing_engine->set_design_work_space(design_work_space);
  timing_engine->readLiberty(lib_files);

  auto idb_adapter = std::make_unique<TimingIDBAdapter>(timing_engine->get_ista());
  idb_adapter->set_idb(idb);
  idb_adapter->convertDBToTimingNetlist();
  timing_engine->set_db_adapter(std::move(idb_adapter));

  const char *sdc_file = to_config->get_sdc_file().c_str();
  if (sdc_file != nullptr) {
    timing_engine->readSdc(sdc_file);
  }

  timing_engine->buildGraph();
  timing_engine->updateTiming();
  return timing_engine;
}

void ToApi::runTO() { _ito->runTO(); }

void ToApi::optimizeDesignViolation() { _ito->optimizeDesignViolation(); }

void ToApi::optimizeSetup() { _ito->optimizeSetup(); }

void ToApi::optimizeHold() { _ito->optimizeHold(); }

void ToApi::initCTSDesignViolation(idb::IdbBuilder *idb, ista::TimingEngine *timing) {
  CTSViolationFixer::get_cts_violation_fixer(idb, timing);
}

std::vector<idb::IdbNet *> ToApi::optimizeCTSDesignViolation(idb::IdbNet *idb_net,
                                                             Tree        *topo) {
  CTSViolationFixer *cts_drv_opt = CTSViolationFixer::get_cts_violation_fixer();
  return cts_drv_opt->fixTiming(idb_net, topo);
}

void ToApi::saveDef(string saved_def_path) {
  if (saved_def_path.empty()) {
    saved_def_path = _ito->get_config()->get_output_def_file();
  }
  _idb->saveDef(saved_def_path);
}

ToConfig *ToApi::get_to_config() { return _ito->get_config(); }

void ToApi::resetConfigLibs(std::vector<std::string> &paths) {
  ToConfig *config = _ito->get_config();
  if (config != nullptr) {
    config->set_lib_files(paths);
  }
}

void ToApi::resetConfigSdc(std::string &path) {
  ToConfig *config = _ito->get_config();
  if (config != nullptr) {
    config->set_sdc_file(path);
  }
}

void ToApi::reportTiming() { _timing_engine->reportTiming(); }
} // namespace ito

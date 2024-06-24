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
#include "ToConfig.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "builder.h"
#include "feature_ito.h"
#include "iTO.h"
#include "idm.h"

namespace ito {

ToApi *ToApi::_to_api_instance = nullptr;

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

  ToConfig *     to_config = _ito->get_config();
  string         def_file = to_config->get_def_file();
  vector<string> lef_files = to_config->get_lef_files();

  idb_builder->buildLef(lef_files);
  idb_builder->buildDef(def_file);
  return idb_builder;
}

ista::TimingEngine *ToApi::initISTA(idb::IdbBuilder *idb) {
  ista::TimingEngine::destroyTimingEngine();

  auto timing_engine = ista::TimingEngine::getOrCreateTimingEngine();

  ToConfig *           to_config = _ito->get_config();
  const char *         design_work_space = to_config->get_design_work_space().c_str();
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

// void ToApi::initCTSDesignViolation(idb::IdbBuilder *idb, ista::TimingEngine *timing) {
//   CTSViolationFixer::get_cts_violation_fixer(idb, timing);
// }

// std::vector<idb::IdbNet *> ToApi::optimizeCTSDesignViolation(idb::IdbNet *idb_net,
//                                                              Tree        *topo) {
//   CTSViolationFixer *cts_drv_opt = CTSViolationFixer::get_cts_violation_fixer();
//   return cts_drv_opt->fixTiming(idb_net, topo);
// }

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

ieda_feature::TimingOptSummary ToApi::outputSummary() {
  ieda_feature::TimingOptSummary to_summary;

  std::map<std::string, ieda_feature::TONetTimingCmp> summary_map;

  // origin data，tns，wns，freq
  auto to_eval_data = getEvalData();
  for (auto eval_data : to_eval_data) {
    ieda_feature::TONetTiming net_timing;
    std::string               net_name = eval_data.name;
    net_timing.tns = eval_data.initial_tns;
    net_timing.wns = eval_data.initial_wns;
    net_timing.suggest_freq = eval_data.initial_freq;

    ieda_feature::TONetTimingCmp net_cmp;
    memset(&net_cmp, 0, sizeof(ieda_feature::TONetTimingCmp));
    net_cmp.origin = net_timing;
    summary_map[net_name] = net_cmp;
  }

  // after optimize timing
  auto clk_list = _timing_engine->getClockList();

  std::ranges::for_each(clk_list, [&](ista::StaClock *clk) {
    auto clk_name = clk->get_clock_name();
    auto drv_tns = _timing_engine->getTNS(clk_name, AnalysisMode::kMax);
    auto drv_wns = _timing_engine->getWNS(clk_name, AnalysisMode::kMax);
    auto suggest_freq = 1000.0 / (clk->getPeriodNs() - drv_wns);

    ieda_feature::TONetTiming net_timing;
    std::string               net_name = clk_name;
    net_timing.tns = drv_tns;
    net_timing.wns = drv_wns;
    net_timing.suggest_freq = suggest_freq;

    summary_map[net_name].opt = net_timing;
  });

  for (auto [net_name, net_timings] : summary_map) {
    net_timings.net_name = net_name;

    net_timings.delta.tns = net_timings.opt.tns - net_timings.origin.tns;
    net_timings.delta.wns = net_timings.opt.wns - net_timings.origin.wns;
    net_timings.delta.suggest_freq =
        net_timings.opt.suggest_freq - net_timings.origin.suggest_freq;

    to_summary.net_timings.push_back(net_timings);
  }

  return to_summary;
}

} // namespace ito

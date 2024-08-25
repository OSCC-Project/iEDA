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
 * @file init_sta.cpp
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-25
 */

#include "init_sta.h"

#include "TimingEngine.hh"
#include "idm.h"

namespace ieval {
#define STA_INST ista::TimingEngine::getOrCreateTimingEngine()
InitSTA::InitSTA(const std::string& work_dir)
{
  // init iSTA interface
  auto* timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  auto sta_db_adapter = std::make_unique<ista::TimingIDBAdapter>(STA_INST->get_ista());
  sta_db_adapter->set_idb(dmInst->get_idb_builder());
  STA_INST->set_db_adapter(std::move(sta_db_adapter));

  // read lib files
  STA_INST->set_design_work_space(work_dir.c_str());
  STA_INST->set_num_threads(80);
  std::vector<const char*> lib_paths;
  auto db_config = dmInst->get_config();
  std::ranges::for_each(db_config.get_lib_paths(), [&](const std::string& lib_path) { lib_paths.push_back(lib_path.c_str()); });
  STA_INST->readLiberty(lib_paths);

  // report setting
  STA_INST->get_ista()->set_n_worst_path_per_clock(10);

  // convert db to timing netlist
  STA_INST->resetNetlist();
  STA_INST->resetGraph();
  STA_INST->get_db_adapter()->convertDBToTimingNetlist();
  const char* sdc_path = db_config.get_sdc_path().c_str();
  STA_INST->readSdc(sdc_path);
  STA_INST->buildGraph();

  // rc tree init
  STA_INST->initRcTree();
}

InitSTA::~InitSTA()
{
  STA_INST->destroyTimingEngine();
}

void InitSTA::runSTA()
{
  // TODO: build rc tree for each net

  // report timing
  STA_INST->updateTiming();
  STA_INST->reportTiming({}, true, true);
}

}  // namespace ieval
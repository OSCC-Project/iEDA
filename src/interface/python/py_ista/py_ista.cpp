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
#include "py_ista.h"

#include <tool_manager.h>

#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "sta/Sta.hh"

namespace python_interface {
bool staRun(const std::string& output)
{
  bool run_ok = iplf::tmInst->autoRunSTA(output);
  return run_ok;
}

bool staInit(const std::string& output)
{
  bool run_ok = iplf::tmInst->initSTA(output);
  return run_ok;
}

bool staReport(const std::string& output)
{
  bool run_ok = iplf::tmInst->runSTA(output);
  return run_ok;
}

bool setDesignWorkSpace(const std::string& design_workspace)
{
  auto* ista = ista::Sta::getOrCreateSta();
  ista->set_design_work_space(design_workspace.c_str());
  return true;
}

bool read_lef_def(std::vector<std::string>& lef_files, const std::string& def_file)
{
  auto* timing_engine = ista::TimingEngine::getOrCreateTimingEngine();

  auto* db_builder = new idb::IdbBuilder();
  db_builder->buildLef(lef_files);

  db_builder->buildDef(def_file);

  auto db_adapter = std::make_unique<ista::TimingIDBAdapter>(timing_engine->get_ista());
  db_adapter->set_idb(db_builder);
  unsigned is_ok = db_adapter->convertDBToTimingNetlist();

  return is_ok;
}

bool readVerilog(const std::string& file_name)
{
  auto* ista = ista::Sta::getOrCreateSta();

  ista->readVerilogWithRustParser(file_name.c_str());
  return true;
}

bool readLiberty(std::vector<std::string>& lib_files)
{
  auto* ista = ista::Sta::getOrCreateSta();
  ista->readLiberty(lib_files);
  return true;
}

bool linkDesign(const std::string& cell_name)
{
  auto* ista = ista::Sta::getOrCreateSta();
  ista->set_top_module_name(cell_name.c_str());
  ista->linkDesignWithRustParser(cell_name.c_str());
  return true;
}

bool readSpef(const std::string& file_name)
{
  auto* ista = ista::Sta::getOrCreateSta();
  ista->readSpef(file_name.c_str());
  return true;
}

bool readSdc(const std::string& file_name)
{
  auto* ista = ista::Sta::getOrCreateSta();
  return ista->readSdc(file_name.c_str());
}

bool reportTiming(int digits, const std::string& delay_type, std::set<std::string> exclude_cell_names, bool derate)
{
  auto* ista = ista::Sta::getOrCreateSta();
  ista->buildGraph();
  ista->updateTiming();
  ista->reportTiming(std::move(exclude_cell_names), derate);
  return true;
}

std::vector<std::string> get_used_libs()
{
  auto* ista = ista::Sta::getOrCreateSta();
  auto used_libs = ista->getUsedLibs();

  std::vector<std::string> ret;
  for (auto& lib : used_libs) {
    ret.push_back(lib->get_file_name());
  }

  return ret;
}

}  // namespace python_interface
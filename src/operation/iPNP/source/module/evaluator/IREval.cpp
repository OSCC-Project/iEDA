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
 * @file IREval.cpp
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#include "IREval.hh"
#include "iPNPIdbWrapper.hh"
#include "log/Log.hh"

namespace ipnp {

void IREval::runIREval(idb::IdbBuilder* idb_builder)
{
  _power_engine = ipower::PowerEngine::getOrCreatePowerEngine();

  std::string power_net_name = "VDD";

  _power_engine->resetIRAnalysisData();
  _power_engine->buildPGNetWireTopo();
  _power_engine->runIRAnalysis(power_net_name);

  _power_engine->reportIRAnalysis();

  _coord_ir_map = _power_engine->displayIRDropMap();
}

void IREval::initIREval(idb::IdbBuilder* idb_builder, PNPConfig* pnp_config)
{
  LOG_INFO << "Start initialize IREval";

  _timing_engine = TimingEngine::getOrCreateTimingEngine();
  _timing_engine->set_num_threads(48);
  
  // Set working directory
  std::string design_work_space;
  if (pnp_config && !pnp_config->get_timing_design_workspace().empty()) {
    design_work_space = pnp_config->get_timing_design_workspace();
  }
  else {
    design_work_space = "/home/sujianrong/iEDA/src/operation/iPNP/data/ir/ir_temp_directory";
  }
  _timing_engine->set_design_work_space(design_work_space.c_str());

  //  
  std::vector<const char*> lib_files;
  if (pnp_config && !pnp_config->get_liberty_files().empty()) {
    const auto& liberty_files = pnp_config->get_liberty_files();
    lib_files.reserve(liberty_files.size());
    for (const auto& lib : liberty_files) {
      lib_files.push_back(lib.c_str());
    }
  } else {
    lib_files = {
      "/home/sujianrong/T28/lib/tcbn28hpcplusbwp40p140ssg0p81v125c.lib",
      "/home/sujianrong/T28/lib/tcbn28hpcplusbwp40p140hvtssg0p81v125c.lib",
      "/home/sujianrong/T28/lib/tcbn28hpcplusbwp35p140ssg0p81v125c.lib",
      "/home/sujianrong/T28/lib/tcbn28hpcplusbwp35p140lvtssg0p81v125c.lib",
      "/home/sujianrong/T28/lib/tcbn28hpcplusbwp40p140lvtssg0p81v125c.lib",
      "/home/sujianrong/T28/lib/tcbn28hpcplusbwp30p140lvtssg0p81v125c.lib",
      "/home/sujianrong/T28/lib/tcbn28hpcplusbwp30p140ssg0p81v125c.lib"
    };
  }
  _timing_engine->readLiberty(lib_files);

  _timing_engine->get_ista()->set_analysis_mode(ista::AnalysisMode::kMaxMin);
  _timing_engine->get_ista()->set_n_worst_path_per_clock(1);

  // Read LEF and DEF files
  std::vector<std::string> lef_files;
  if (pnp_config && !pnp_config->get_lef_files().empty()) {
    lef_files = pnp_config->get_lef_files();
  } else {
    lef_files = {
    "/home/sujianrong/T28/tlef/tsmcn28_9lm6X2ZUTRDL.tlef",
    "/home/sujianrong/T28/lef/PLLTS28HPMLAINT.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp30p140opplvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp35p140lvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp35p140uhvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140hvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140oppuhvt.lef",
    "/home/sujianrong/T28/lef/ts5n28hpcplvta256x32m4fw_130a.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp30p140cg.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp30p140oppuhvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp35p140mbhvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp35p140ulvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140uhvt.lef",
    "/home/sujianrong/T28/lef/ts5n28hpcplvta64x100m2fw_130a.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp30p140hvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp30p140oppulvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp35p140mb.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140cgcwhvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140lvt.lef",
    "/home/sujianrong/T28/lef/tpbn28v_9lm.lef",
    "/home/sujianrong/T28/lef/ts5n28hpcplvta64x128m2f_130a.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp30p140.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp30p140uhvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp35p140mblvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140cgcw.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140mbhvt.lef",
    "/home/sujianrong/T28/lef/tpbn28v.lef",
    "/home/sujianrong/T28/lef/ts5n28hpcplvta64x128m2fw_130a.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp30p140lvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp30p140ulvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp35p140opphvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140cgehvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140mb.lef",
    "/home/sujianrong/T28/lef/tphn28hpcpgv18_9lm.lef",
    "/home/sujianrong/T28/lef/ts5n28hpcplvta64x88m2fw_130a.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp30p140mb.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp35p140cghvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp35p140opp.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140cghvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140oppehvt.lef",
    "/home/sujianrong/T28/lef/ts1n28hpcplvtb2048x48m8sw_180a.lef",
    "/home/sujianrong/T28/lef/ts5n28hpcplvta64x92m2fw_130a.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp30p140mblvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp35p140cg.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp35p140opplvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140cg.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140opphvt.lef",
    "/home/sujianrong/T28/lef/ts1n28hpcplvtb512x128m4sw_180a.lef",
    "/home/sujianrong/T28/lef/ts5n28hpcplvta64x96m2fw_130a.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp30p140opphvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp35p140hvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp35p140oppuhvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140cguhvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140opp.lef",
    "/home/sujianrong/T28/lef/ts1n28hpcplvtb512x64m4sw_180a.lef",
    "/home/sujianrong/T28/lef/ts6n28hpcplvta2048x32m8sw_130a.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp30p140opp.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp35p140.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp35p140oppulvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140ehvt.lef",
    "/home/sujianrong/T28/lef/tcbn28hpcplusbwp40p140opplvt.lef",
    "/home/sujianrong/T28/lef/ts1n28hpcplvtb8192x64m8sw_180a.lef"
    };
  }

  std::string def_file;
  if (pnp_config && !pnp_config->get_output_def_path().empty()) {
    def_file = pnp_config->get_output_def_path();
  }else{
    def_file = "/home/sujianrong/iEDA/src/operation/iPNP/data/test/output.def";
  }

  // Read DEF design
  _timing_engine->readDefDesign(def_file, lef_files);

  // Read SDC file
  std::string sdc_file;
  if (pnp_config && !pnp_config->get_sdc_file().empty()) {
    sdc_file = pnp_config->get_sdc_file();
  }
  else {
    sdc_file = "/home/sujianrong/iEDA/src/operation/iPNP/data/test/output.sdc";
  }
  _timing_engine->readSdc(sdc_file.c_str());

  _timing_engine->buildGraph();
  _timing_engine->get_ista()->updateTiming();
  _timing_engine->reportTiming();

  _ista = Sta::getOrCreateSta();
  _ipower = ipower::Power::getOrCreatePower(&(_ista->get_graph()));

  _ipower->runCompleteFlow();

  LOG_INFO << "End initialize IREval";

}

double IREval::getMaxIRDrop() const
{
  if (_coord_ir_map.empty()) {
    return 0.0;
  }
  double max_ir_drop = -std::numeric_limits<double>::max();
  for (auto it = _coord_ir_map.begin(); it != _coord_ir_map.end(); ++it) {
    if (it->second > max_ir_drop) {
      max_ir_drop = it->second;
    }
  }
  return max_ir_drop;
}

double IREval::getMinIRDrop() const
{
  if (_coord_ir_map.empty()) {
    return 0.0;
  }
  double min_ir_drop = std::numeric_limits<double>::max();
  for (auto it = _coord_ir_map.begin(); it != _coord_ir_map.end(); ++it) {
    if (it->second < min_ir_drop) {
      min_ir_drop = it->second;
    }
  }
  return min_ir_drop;
}

double IREval::getAvgIRDrop() const
{
  if (_coord_ir_map.empty()) {
    return 0.0;
  }
  double sum_ir_drop = 0.0;
  int count = 0;
  for (auto it = _coord_ir_map.begin(); it != _coord_ir_map.end(); ++it) {
    sum_ir_drop += it->second;
    count++;
  }
  return sum_ir_drop / count;
}

}  // namespace ipnp
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
 * @file PNP.cpp
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#include "PNP.hh"

#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <fstream>
#include <iomanip>

#include "CongestionEval.hh"
#include "FastPlacer.hh"
#include "IREval.hh"
#include "NetworkSynthesis.hh"
#include "PNPConfig.hh"
#include "PNPIdbWrapper.hh"
#include "PdnOptimizer.hh"
#include "idm.h"
#include "log/Log.hh"

namespace ipnp {

PNP::PNP()
{
}

PNP::PNP(const std::string& config_file)
{
  if (!pnpConfig->loadConfigFromJson(config_file)) {
    LOG_WARNING << "Initializing PNP with default configuration" << std::endl;
  }
}

void PNP::runSynthesis()
{
  NetworkSynthesis network_synthesizer(SysnType::kDefault, _input_network);
  network_synthesizer.synthesizeNetwork();
  _initialized_network = network_synthesizer.get_network();
  _current_opt_network = _initialized_network;
}

void PNP::runOptimize()
{
  saveToIdb();
  PdnOptimizer pdn_optimizer;

  pdn_optimizer.optimizeGlobal(_initialized_network);
  _current_opt_network = pdn_optimizer.get_out_put_grid();
}

void PNP::runFastPlacer()
{
  FastPlacer fast_placer;
  fast_placer.runFastPlacer();
}

void PNP::init()
{
  // Initialize the input network
  _input_network = PNPGridManager();

  _input_network.set_power_layers(pnpConfig->get_power_layers());
  _input_network.set_ho_region_num(pnpConfig->get_ho_region_num());
  _input_network.set_ver_region_num(pnpConfig->get_ver_region_num());

  _input_network.set_layer_count(_input_network.get_power_layers().size());
  _input_network.set_die_width(dmInst->get_idb_layout()->get_die()->get_width());
  _input_network.set_die_height(dmInst->get_idb_layout()->get_die()->get_height());

  // Initialize with configuration-based templates if available
  _input_network.init_PNPGridManager_data();
}

void PNP::initIRAnalysis()
{
  static bool is_init = false;
  if (!is_init) {
    // Initialize IREval
    _ir_eval.initIREval();
    is_init = true;
  }
}

void PNP::runAnalysis()
{
  saveToIdb();

  CongestionEval cong_eval;

  cong_eval.evalEGR();

  _ir_eval.runIREval();

  // Get analysis results
  double max_ir_drop = _ir_eval.getMaxIRDrop();
  double min_ir_drop = _ir_eval.getMinIRDrop();
  double avg_ir_drop = _ir_eval.getAvgIRDrop();
  int32_t overflow = cong_eval.get_total_overflow_union();

  // Build final_report path
  std::string final_report_path = "final_report.txt";
  if (!pnpConfig->get_report_path().empty()) {
    std::string dir_path = pnpConfig->get_report_path();
    if (dir_path.back() != '/') {
      dir_path += '/';
    }
    final_report_path = dir_path + "final_report.txt";
    LOG_INFO << "Using directory from config for final report: " << dir_path << std::endl;
  }

  LOG_INFO << "Final report will be written to: " << final_report_path << std::endl;

  std::string directory = final_report_path.substr(0, final_report_path.find_last_of('/'));
  if (!directory.empty()) {
    auto create_directories = [](const std::string& path) -> bool {
      size_t pos = 0;
      std::string dir;
      int ret = 0;
      if (path[0] == '/') {
        pos = 1;
      }

      while ((pos = path.find('/', pos)) != std::string::npos) {
        dir = path.substr(0, pos++);
        if (dir.empty())
          continue;

        ret = mkdir(dir.c_str(), 0755);
        if (ret != 0 && errno != EEXIST) {
          return false;
        }
      }
      return true;
    };

    if (!create_directories(directory)) {
      LOG_ERROR << "Failed to create directory: " << directory << ", error: " << strerror(errno) << std::endl;
    } else {
      LOG_INFO << "Directory ensured: " << directory << std::endl;
    }
  }

  // Open file for writing
  std::ofstream report_file(final_report_path.c_str(), std::ios::out | std::ios::trunc);
  if (!report_file.is_open()) {
    LOG_ERROR << "Failed to open final report file for writing: " << final_report_path << std::endl;
  } else {
    LOG_INFO << "Successfully opened final report file for writing" << std::endl;

    report_file << std::fixed << std::setprecision(6);

    report_file << "======================================================================" << std::endl;
    report_file << "                     FINAL POWER NETWORK ANALYSIS REPORT               " << std::endl;
    report_file << "======================================================================" << std::endl;
    report_file << std::endl;

    report_file << "IR DROP ANALYSIS:" << std::endl;
    report_file << "  Maximum IR Drop  : " << std::setw(12) << max_ir_drop << std::endl;
    report_file << "  Minimum IR Drop  : " << std::setw(12) << min_ir_drop << std::endl;
    report_file << "  Average IR Drop  : " << std::setw(12) << avg_ir_drop << std::endl;
    report_file << std::endl;

    report_file << "CONGESTION ANALYSIS:" << std::endl;
    report_file << "  Total Overflow   : " << std::setw(12) << overflow << std::endl;
    report_file << std::endl;

    report_file << "----------------------------------------------------------------------" << std::endl;

    report_file.close();
    LOG_INFO << "Final analysis report written successfully" << std::endl;
  }
}

void PNP::connect_M2_M1()
{
  _idb_wrapper.connect_M2_M1_Layer();
}

void PNP::run()
{
  // initializing
  init();
  initIRAnalysis();

  // running
  runSynthesis();
  runFastPlacer();
  runOptimize();
  runAnalysis();
}

void PNP::writeIdbToDef(std::string def_path)
{
  dmInst->saveDef(def_path);
}

}  // namespace ipnp

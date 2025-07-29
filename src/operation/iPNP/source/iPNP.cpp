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
 * @file iPNP.cpp
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#include "iPNP.hh"

#include "FastPlacer.hh"
#include "NetworkSynthesis.hh"
#include "PdnOptimizer.hh"
#include "iPNPIdbWrapper.hh"
#include "CongestionEval.hh"
#include "IREval.hh"
#include "PNPConfig.hh"
#include "log/Log.hh"
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <string.h>

namespace ipnp {

  iPNP::iPNP() {
    _pnp_config = new PNPConfig();
  }

  iPNP::iPNP(const std::string& config_file) {
    _pnp_config = new PNPConfig();
    if (!loadConfigFromJson(config_file, _pnp_config)) {
      LOG_WARNING << "Initializing iPNP with default configuration" << std::endl;
    }

    // If LEF and DEF file paths are specified in the configuration, read the DEF file
    if (!_pnp_config->get_lef_files().empty() && !_pnp_config->get_def_path().empty()) {
      readLefDef(_pnp_config->get_lef_files(), _pnp_config->get_def_path());
      LOG_INFO << "DEF file read: " << _pnp_config->get_def_path() << std::endl;
    }
    else {
      LOG_WARNING << "LEF files or DEF file path not specified in configuration" << std::endl;
    }

    // Set output DEF file path
    if (!_pnp_config->get_output_def_path().empty()) {
      _output_def_path = _pnp_config->get_output_def_path();
    }
    else {
      _output_def_path = "./output.def";
    }
  }

  iPNP::~iPNP() {
    if (_pnp_config) {
      delete _pnp_config;
      _pnp_config = nullptr;
    }
  }

  void iPNP::runSynthesis()
  {
    NetworkSynthesis network_synthesizer(SysnType::kDefault, _input_network);
    network_synthesizer.synthesizeNetwork();
    _initialized_network = network_synthesizer.get_network();
    _current_opt_network = _initialized_network;
  }

  void iPNP::runOptimize()
  {
    saveToIdb();
    PdnOptimizer pdn_optimizer;
    pdn_optimizer.set_config(_pnp_config);
    
    pdn_optimizer.optimizeGlobal(_initialized_network, _idb_wrapper.get_idb_builder());
    _current_opt_network = pdn_optimizer.get_out_put_grid();
  }

  void iPNP::runFastPlacer()
  {
    FastPlacer fast_placer;
    fast_placer.set_config(_pnp_config);
    fast_placer.runFastPlacer(_idb_wrapper.get_idb_builder());
  }

  void iPNP::readLefDef(std::vector<std::string> lef_files, std::string def_path)
  {
    auto* db_builder = new idb::IdbBuilder();
    db_builder->buildLef(lef_files);
    db_builder->buildDef(def_path);

    auto* idb_design = db_builder->get_def_service()->get_design();

    _idb_wrapper.set_idb_design(idb_design);
    _idb_wrapper.set_idb_builder(db_builder);
  }

  void iPNP::init()
  {
    // Initialize the input network
    _input_network = GridManager();
    if (_pnp_config) {
      _input_network.set_power_layers(_pnp_config->get_power_layers());
      _input_network.set_ho_region_num(_pnp_config->get_ho_region_num());
      _input_network.set_ver_region_num(_pnp_config->get_ver_region_num());
    }
    else {
      _input_network.set_power_layers({ 9,8,7,6,5,4,3 });
      _input_network.set_ho_region_num(2);
      _input_network.set_ver_region_num(2);
    }
    _input_network.set_layer_count(_input_network.get_power_layers().size());
    _input_network.set_die_width(_idb_wrapper.get_input_die_width());
    _input_network.set_die_height(_idb_wrapper.get_input_die_height());

    // Initialize with configuration-based templates if available
    if (_pnp_config) {
      _input_network.init_GridManager_data(_pnp_config);
    }
    else {
      _input_network.init_GridManager_data();
    }

  }

  void iPNP::initIRAnalysis() {
    static bool is_init = false;
    if (!is_init) {
      // Initialize IREval
      _ir_eval.initIREval(_idb_wrapper.get_idb_builder(), _pnp_config);
      is_init = true;
    }
  }

  void iPNP::runAnalysis()
  {
    saveToIdb();

    _cong_eval.set_config(_pnp_config);
    _cong_eval.evalEGR(_idb_wrapper.get_idb_builder());

    _ir_eval.runIREval(_idb_wrapper.get_idb_builder());
    
    // Get analysis results
    double max_ir_drop = _ir_eval.getMaxIRDrop();
    double min_ir_drop = _ir_eval.getMinIRDrop();
    double avg_ir_drop = _ir_eval.getAvgIRDrop();
    int32_t overflow = _cong_eval.get_total_overflow_union();
    
    // Build final_report path
    std::string final_report_path = "final_report.txt";
    if (_pnp_config && !_pnp_config->get_report_path().empty()) {
      std::string dir_path = _pnp_config->get_report_path();
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
          if (dir.empty()) continue;
          
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

  void iPNP::outputDef()
  {
    readLefDef(_pnp_config->get_lef_files(), _pnp_config->get_def_path());
    saveToIdb();
    writeIdbToDef(_output_def_path);
  }

  void iPNP::connect_M2_M1()
  {
    _idb_wrapper.connect_M2_M1_Layer();
  }

  void iPNP::run()
  {
    if (_idb_wrapper.get_idb_design()) {

      init();
      initIRAnalysis();
      runSynthesis();
      runFastPlacer();
      initIRAnalysis();
      
      runOptimize();
      runAnalysis();
      outputDef();


      LOG_INFO << "Output written to DEF file: " << _output_def_path << std::endl;

    }
    else {
      LOG_ERROR << "Warning: idb design is empty!" << std::endl;
    }
  }

}  // namespace ipnp

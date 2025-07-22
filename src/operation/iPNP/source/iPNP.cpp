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
    pdn_optimizer.optimizeGlobal(_initialized_network, _idb_wrapper.get_idb_builder());
    _current_opt_network = pdn_optimizer.get_out_put_grid();
  }

  void iPNP::runFastPlacer()
  {
    FastPlacer fast_placer;
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
    // Initialize IREval
    _ir_eval.initIREval(_idb_wrapper.get_idb_builder(), _pnp_config);

  }

  void iPNP::runAnalysis()
  {
    saveToIdb();

    _cong_eval.set_config(_pnp_config);
    _cong_eval.evalEGR(_idb_wrapper.get_idb_builder());

    _ir_eval.runIREval(_idb_wrapper.get_idb_builder());
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

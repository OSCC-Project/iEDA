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
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

#include "iPNP.hh"

#include "FastPlacer.hh"
#include "CongestionEval.hh"
#include "NetworkSynthesis.hh"
#include "PdnOptimizer.hh"
#include "iPNPIdbWrapper.hh"
#include "IREval.hh"
#include "utility/DefConverter.hh"

namespace ipnp {

iPNP::iPNP(){}

iPNP::iPNP(const std::string& config_file)
{
  /**
   * @todo add config
   * @brief need json module
   */
  
  // _pnp_config = new PNPConfig;
  // JsonParser* json = JsonParser::get_json_parser();
  // json->parse(config_file, _pnp_config);

}

/**
 * @brief Generate initial solution. Decide which region to place, and place templates on regions randomly.
 * @attention Version_1.0 only consider rectangular grid region.
 */
void iPNP::runSynthesis()
{
  NetworkSynthesis network_synthesizer(SysnType::kDefault, _input_network);

  network_synthesizer.synthesizeNetwork();

  _initialized_network = network_synthesizer.get_network();

}

void iPNP::runOptimize()
{
  PdnOptimizer pdn_optimizer;
  pdn_optimizer.optimizeGlobal(_initialized_network, _idb_wrapper.get_idb_builder());
  _current_opt_network = pdn_optimizer.get_out_put_grid();
}

void iPNP::readDef(std::vector<std::string> lef_files, std::string def_path)
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
  _input_network.set_power_layers({ 9,8,7,6,5,4,3 });
  _input_network.set_layer_count(_input_network.get_power_layers().size());
  _input_network.set_ho_region_num(3);
  _input_network.set_ver_region_num(3);
  _input_network.set_core_width(_idb_wrapper.get_input_core_width());
  _input_network.set_core_height(_idb_wrapper.get_input_core_height());
  _input_network.set_die_width(_idb_wrapper.get_input_die_width());
  _input_network.set_die_height(_idb_wrapper.get_input_die_height());
  _input_network.set_core_llx(_idb_wrapper.get_input_core_lx());
  _input_network.set_core_lly(_idb_wrapper.get_input_core_ly());
  _input_network.set_core_urx(_idb_wrapper.get_input_core_hx());
  _input_network.set_core_ury(_idb_wrapper.get_input_core_hy());
  _input_network.init_GridManager_data();
  
}

void iPNP::run()
{
  if (_idb_wrapper.get_idb_design()) {

    init();
    
    runSynthesis();

    // FastPlacer fast_placer;
    // fast_placer.runFastPlacer(_idb_wrapper.get_idb_builder());

    _current_opt_network = _initialized_network;

    idb::IdbBuilder* idb_builder = _idb_wrapper.get_idb_builder();

    saveToIdb();

    runOptimize();

    writeIdbToDef(_output_def_path);

    // CongestionEval cong_eval;
    // cong_eval.evalEGR(_idb_wrapper.get_idb_builder());
    
    IREval ir_eval;
    ir_eval.initIREval(_idb_wrapper.get_idb_builder());
    ir_eval.runIREval(_idb_wrapper.get_idb_builder());
    
    // DefConverter def_converter;
    // def_converter.runDefConverter("/home/sujianrong/iEDA/src/operation/iPNP/data/test/output.def",
    //   "/home/sujianrong/iEDA/src/operation/iPNP/data/test/aes_no_pwr_with_PL.def",
    //   "COMPONENTS 15826 ;",
    //   "END COMPONENTS");

    
  }
  else {
    std::cout << "Warning: idb design is empty!" << std::endl;
  }
}

}  // namespace ipnp

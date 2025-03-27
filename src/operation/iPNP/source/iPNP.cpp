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
#include "NetworkSynthesis.hh"
#include "PdnOptimizer.hh"
#include "iPNPIdbWrapper.hh"

namespace ipnp {

class PndOptimizer;

iPNP::iPNP()
{
  // Initialize the input network
  _input_network = GridManager();
  _input_network.set_power_layers({ 9,8,7,6 });
  _input_network.set_layer_count(_input_network.get_power_layers().size());
  _input_network.set_ho_region_num(3);
  _input_network.set_ver_region_num(3);
  _input_network.set_core_width(_input_core_width);
  _input_network.set_core_height(_input_core_height);
  _input_network.update_GridManager_data();
}

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
void iPNP::initSynthesize()
{
  /**
   * @todo add template_lib infomation to _initialized_network
   */

  // Create network synthesizer
  NetworkSynthesis network_synthesizer(SysnType::kDefault, _input_network);

  // Generate the initial network
  network_synthesizer.synthesizeNetwork();

  // Get the initial network
  _initialized_network = network_synthesizer.get_network();

  /**
   * @todo version_2.0: Add result of fastplacer
   * @brief consider regions with irregular shapes
   */
  // FastPlacer fast_placer;
  // fast_placer.fastPlace(_input_netlist);
  // idb::IdbLayer* fast_place_result = fast_placer.getPlaceResult();
}

void iPNP::optimize()
{
  PdnOptimizer pdn_optimizer;
  pdn_optimizer.optimize(_initialized_network);
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

void iPNP::getIdbDesignInfo()
{
  _input_core_width = _idb_wrapper.get_input_core_width();
  _input_core_height = _idb_wrapper.get_input_core_height();

  // for (int i = 0; i < _idb_wrapper.get_input_macro_nums(); i++) {
  //   std::pair<int32_t, int32_t> l_coordinate(_idb_wrapper.get_input_macro_lx(), _idb_wrapper.get_input_macro_ly());
  //   std::pair<int32_t, int32_t> h_coordinate(_idb_wrapper.get_input_macro_hx(), _idb_wrapper.get_input_macro_hy());
  //   std::pair<std::pair<int32_t, int32_t>, std::pair<int32_t, int32_t>> macro_coordinate(l_coordinate, h_coordinate);
  //   _input_macro_coordinate.push_back(macro_coordinate);
  // }
}

void iPNP::run()
{
  
  if (_idb_wrapper.get_idb_design()) {
    
    // Get the idb design information
    getIdbDesignInfo();
    
    // Initialize the network
    initSynthesize();

    // Optimize the network
    optimize();

    // Save the network to idb
    saveToIdb();

    // Write the network to def
    writeIdbToDef(_output_def_path);
    
  }
  else {
    std::cout << "Warning: idb design is empty!" << std::endl;
  }
}

}  // namespace ipnp

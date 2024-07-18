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
   * @todo add template_lib infomation to _input_network
   */

  NetworkSynthesis network_synthesizer(SysnType::Default, _input_network);
  network_synthesizer.synthesizeNetwork();
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

void iPNP::readFromIdb(std::string input_def)
{
  iPNPIdbWrapper ipnp_idb_wrapper;
  ipnp_idb_wrapper.readFromIdb(input_def);
  _input_network = ipnp_idb_wrapper.get_input_db_pdn();
}

void iPNP::writeToIdb()
{
  iPNPIdbWrapper ipnp_idb_wrapper;
  ipnp_idb_wrapper.writeToIdb(_current_opt_network);
}

void iPNP::run()
{
  readFromIdb("<input def path>");
  initSynthesize();
  optimize();
  writeToIdb();
}

}  // namespace ipnp

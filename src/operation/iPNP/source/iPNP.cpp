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
#include "iPNP.hh"

#include "FastPlacer.hh"
#include "NetworkSynthesis.hh"
#include "PdnOptimizer.hh"

namespace ipnp {

class PndOptimizer;

iPNP::iPNP(){
}

iPNP::iPNP(const std::string& config_file)
{
  //TODO: add config
  //need json module

  /*
  _pnp_config = new PNPConfig;
  JsonParser *json = JsonParser::get_json_parser();
  json->parse(config_file, _pnp_config);
  */
}

iPNP::~iPNP()
{
}

void iPNP::initSynthesize()
{
  //TODO: 使用fastplacer后决定region摆放，但哪个region上放哪个Template用随机
  //v1.0先不考虑不规则region(fastplacer)
  //should include writeDef().
  
  /*
  FastPlacer fast_placer;
  fast_placer.fastPlace(_input_netlist);
  idb::IdbLayer* fast_place_result = fast_placer.getPlaceResult();
  */
  
  GridManager empty_grid;
  NetworkSynthesis network_synthesizer("default", empty_grid);
  network_synthesizer.synthesizeNetwork();
  _initialized_network = network_synthesizer.getNetwork();
}

void iPNP::optimize()
{
  PdnOptimizer pdn_optimizer;
  _current_opt_network = pdn_optimizer.optimize(_initialized_network);


  optimizer -> middle data -> syn(optimizer) , set_syn(middle data)

  // TODO:
  void evaluate();  // 整个评估，包括IR、congestion，后续可拓展EM，也可整个被ML Model代替

}

void iPNP::synthesizeNetwork(){
  NetworkSynthesis network("optimizer", _current_opt_network);
  network.writeDef();
}

void iPNP::run()
{
  initSynthesize();
  optimize();
  synthesizeNetwork();
}

}  // namespace ipnp

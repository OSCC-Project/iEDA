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
/*
 * @Author: Li Jiangkao
 * @Date: 2021-08-23 22:17:47
 * @LastEditTime: 2022-02-13 12:47:14
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /LJK-iEDA/iEDA/src/iFP/MacroPlacer/MacroPlacer.cpp
 */
/**
 * @file Inst.h
 * @author xingquan li (lixq01@pcl.ac.cn)
 * @brief Tool Data;
 * Store instance
 * @version 0.1
 * @date 2021-4-1
 **/
#include "MacroPlacer.hh"

namespace ipl::imp {

MacroPlacer::MacroPlacer(MPDB* mdb, ipl::Config* config) : _mdb(mdb)
{
  _set = new Setting();
  setConfig(config);
  addHalo();
  updateDensity();
}

void MacroPlacer::runMacroPlacer()
{
  time = clock();

  // parition
  MPPartition* partition = new MPPartition(_mdb, _set);
  partition->runPartition();
  buildNewNetList();
  _mdb->updatePlaceMacroList();

  // simulate anneal
  if (_set->get_solver_type() == SolverType::kSimulate_anneal) {
    SolutionFactory factory = SolutionFactory();
    MPSolution* mp_solution = factory.createSolution(_mdb->get_place_macro_list(), _set);
    MPEvaluation* mp_evaluation = new MPEvaluation(_mdb, _set, mp_solution);
    SimulateAnneal* anneal = new SimulateAnneal(_set, mp_evaluation);
    anneal->runAnneal();
  } else if (_set->get_solver_type() == SolverType::kAnalytical) {
  }

  _mdb->writeDB();
  std::string output_path = _set->get_output_path();
  plotGDS();
  _mdb->writeResult(output_path);
  writeSummary();
}

void MacroPlacer::setConfig(Config* config)
{
  MacroPlacerConfig mp_config = config->get_mp_config();

  // set
  _set->set_new_macro_density(mp_config.get_new_macro_density());
  _set->set_output_path(mp_config.get_output_path());
  _set->set_macro_halo_x(mp_config.get_halo_x());
  _set->set_macro_halo_y(mp_config.get_halo_y());
  _set->set_parts(mp_config.get_parts());  // the number of cluster
  _set->set_ncon(5);                       // The number of balancing constraints
  _set->set_ufactor(mp_config.get_ufactor());
  // simulate anneal
  _set->set_perturb_per_step(mp_config.get_perturb_per_step());
  _set->set_cool_rate(mp_config.get_cool_rate());
  // solution type
  LOG_INFO << "solution type: " << mp_config.get_solution_tpye();
  if (mp_config.get_solution_tpye() == "BStarTree") {
    _set->set_solution_type(SolutionType::kBStar_tree);
  } else if ("SequencePair" == mp_config.get_solution_tpye()) {
    _set->set_solution_type(SolutionType::kSequence_pair);
  } else {
    LOG_ERROR << "error illegal type: " << mp_config.get_solution_tpye();
  }

  // set fixed macro
  for (FPInst* macro : _mdb->get_total_macro_list()) {
    macro->set_fixed(false);
  }
  std::vector<std::string> fixed_macro = mp_config.get_fixed_macro();
  std::vector<int32_t> fixed_macro_coord = mp_config.get_fixed_macro_coordinate();
  for (size_t i = 0; i < fixed_macro.size(); ++i) {
    _mdb->setMacroFixed(fixed_macro[i], fixed_macro_coord[2 * i], fixed_macro_coord[2 * i + 1]);
  }

  // set blockage
  std::vector<int32_t> blockage = mp_config.get_blockage();
  for (size_t i = 0; i < blockage.size(); i += 4) {
    FPRect* rect = new FPRect();
    rect->set_x(blockage[i]);
    rect->set_y(blockage[i + 1]);
    rect->set_width(blockage[i + 2] - blockage[i]);
    rect->set_height(blockage[i + 3] - blockage[i + 1]);
    _mdb->add_blockage(rect);
  }

  // set guidance
  std::vector<std::string> guidance_macro_list = mp_config.get_guidance_macro();
  std::vector<int32_t> guidance = mp_config.get_guidance();
  for (size_t i = 0; i < guidance_macro_list.size(); ++i) {
    FPRect* rect = new FPRect();
    rect->set_x(guidance[i]);
    rect->set_y(guidance[i + 1]);
    rect->set_width(guidance[i + 2] - guidance[i]);
    rect->set_height(guidance[i + 3] - guidance[i + 1]);
    _mdb->add_guidance_to_macro_name(rect, guidance_macro_list[i]);
  }
}

void MacroPlacer::updateDensity()
{
  // update density
  float total_macro_area = 0;
  float total_std_cell_area = 0;
  float core_area = 0;
  float density = 1;
  for (FPInst* macro : _mdb->get_design()->get_macro_list()) {
    total_macro_area += macro->get_area();
  }
  for (FPInst* std_cell : _mdb->get_design()->get_std_cell_list()) {
    total_std_cell_area += std_cell->get_area();
  }
  core_area = _mdb->get_layout()->get_core_shape()->get_area();
  density = total_std_cell_area / (core_area - total_macro_area) + 0.2;
  density = std::min(density, float(1));
  density = std::max(density, _set->get_new_macro_density());
  _set->set_new_macro_density(density);
  LOG_INFO << "new_macro_density: " << _set->get_new_macro_density();
}

void MacroPlacer::addHalo()
{
  // set halo
  uint32_t halo_x = _set->get_macro_halo_x();
  uint32_t halo_y = _set->get_macro_halo_y();
  for (FPInst* macro : _mdb->get_design()->get_macro_list()) {
    macro->set_halo_x(halo_x);
    macro->set_halo_y(halo_y);
    macro->addHalo();
  }

  LOG_INFO << "halo_x: " << _set->get_macro_halo_x();
  LOG_INFO << "halo_y: " << _set->get_macro_halo_y();
}

void MacroPlacer::deleteHalo()
{
  for (FPInst* macro : _mdb->get_design()->get_macro_list()) {
    macro->deleteHalo();
  }
}

void MacroPlacer::writeSummary()
{
  double time_consume = double(clock() - time) / CLOCKS_PER_SEC;
  LOG_INFO << "time consume: " << time_consume << "s";

  std::ofstream config;
  time_t now = std::time(0);
  char* dt = ctime(&now);
  config.open(_set->get_output_path() + "/config_set.txt");
  config << "new_macro_density: " << _set->get_new_macro_density() << std::endl;
  config << "macro_halo_x: " << _set->get_macro_halo_x() << std::endl;
  config << "macro_halo_y: " << _set->get_macro_halo_y() << std::endl;
  config << "parts: " << _set->get_parts() << std::endl;
  config << "ufactor: " << _set->get_ufactor() << std::endl;
  config << "perturb_per_step: " << _set->get_perturb_per_step() << std::endl;
  config << "cool_rate: " << _set->get_cool_rate() << std::endl;
  config << "time consume: " << time_consume << "s" << std::endl;
  config << "date: " << dt << std::endl;
  config.close();
}

void MacroPlacer::darwPartition()
{
  int part = _set->get_parts();
  map<FPInst*, int> partition_result = partitionInst(part);
  plotPartitionGDS(partition_result);
}

map<FPInst*, int> MacroPlacer::partitionInst(int part)
{
  map<int, FPInst*> index_to_inst_map;
  map<FPInst*, int> inst_to_index_map;
  int index = 0;
  int inst_num = 0;
  inst_num += _mdb->get_design()->get_macro_list().size();
  inst_num += _mdb->get_design()->get_std_cell_list().size();
  for (FPInst* macro : _mdb->get_design()->get_macro_list()) {
    index_to_inst_map.insert(std::pair<int, FPInst*>(index, macro));
    inst_to_index_map.insert(std::pair<FPInst*, int>(macro, index));
    index++;
  }

  for (FPInst* inst : _mdb->get_design()->get_std_cell_list()) {
    index_to_inst_map.insert(std::pair<int, FPInst*>(index, inst));
    inst_to_index_map.insert(std::pair<FPInst*, int>(inst, index));
    index++;
  }

  std::vector<std::vector<int>> hyper_edge_list;
  for (FPNet* net : _mdb->get_design()->get_net_list()) {
    std::vector<int> hyper_edge;
    for (FPPin* pin : net->get_pin_list()) {
      FPInst* inst = pin->get_instance();
      if (inst == nullptr) {
        continue;
      }
      // map<FPInst, int>::iterator index_ite;
      auto index_ite = inst_to_index_map.find(inst);
      if (index_ite == inst_to_index_map.end()) {
        continue;
      } else {
        hyper_edge.emplace_back((*index_ite).second);
      }
    }
    if (hyper_edge.size() > 1) {
      hyper_edge_list.emplace_back(hyper_edge);
    }
  }

  Hmetis* hmetis = new Hmetis();
  hmetis->set_ufactor(20);
  hmetis->set_nparts(part);
  hmetis->partition(inst_num, hyper_edge_list);
  std::vector<int> partition_result = hmetis->get_result();
  delete hmetis;

  map<FPInst*, int> result;
  for (size_t i = 0; i < partition_result.size(); ++i) {
    FPInst* inst;
    auto inst_ite = index_to_inst_map.find(i);
    if (inst_ite == index_to_inst_map.end()) {
      continue;
    } else {
      inst = (*inst_ite).second;
    }
    result.insert(std::pair<FPInst*, int>(inst, partition_result[i]));
  }

  return result;
}

void MacroPlacer::plotGDS()
{
  std::string gds_path = _set->get_output_path() + "/plane_result_macro.gds";
  GDSPlotter* plotter = new GDSPlotter(gds_path);
  plotter->plotDie(_mdb->get_layout()->get_die_shape());
  plotter->plotCore(_mdb->get_layout()->get_core_shape());

  for (FPInst* macro : _mdb->get_total_macro_list()) {
    if (macro->isMacro()) {
      macro->addHalo();
      plotter->plotInst(macro, 2);
      macro->deleteHalo();
      plotter->plotInst(macro, 3);
    } else if (macro->isNewMacro()) {
      plotter->plotInst(macro, 4);
    }
  }

  plotter->plotNetList(_mdb->get_new_net_list(), 5);
  delete plotter;
}

void MacroPlacer::plotPartitionGDS(map<FPInst*, int> partition_result)
{
  std::string gds_path = _set->get_output_path() + "/partition_result.gds";
  GDSPlotter* plotter = new GDSPlotter(gds_path);
  plotter->plotDie(_mdb->get_layout()->get_die_shape());
  plotter->plotCore(_mdb->get_layout()->get_core_shape());

  for (auto iter = partition_result.begin(); iter != partition_result.end(); ++iter) {
    plotter->plotInst(iter->first, iter->second + 2);
  }
  delete plotter;
}

void MacroPlacer::buildNewNetList()
{
  _mdb->clearNewNetList();
  std::vector<FPNet*> old_net_list = _mdb->get_net_list();
  for (FPNet* old_net : old_net_list) {
    std::vector<FPPin*> pin_list = old_net->get_pin_list();
    if (pin_list.size() == 0) {
      continue;
    }

    if (pin_list.size() > 50) {
      // LOG_INFO << "degree of net " << old_net->get_name() << " : " << pin_list.size();
      continue;
    }

    std::set<FPInst*> net_macro_set;

    // create new net
    FPNet* new_net = new FPNet();
    new_net->set_name(old_net->get_name());
    // read instance pin
    for (FPPin* old_pin : pin_list) {
      if (old_pin->is_io_pin()) {
        new_net->add_pin(old_pin);
        continue;
      }
      FPInst* old_inst = old_pin->get_instance();
      if (nullptr == old_inst) {
        continue;
      }
      if (old_inst->isMacro()) {
        new_net->add_pin(old_pin);
        // net_macro_set.insert(old_inst);
      } else {
        FPInst* new_macro = _mdb->findNewMacro(old_inst);
        if (nullptr == new_macro) {
          continue;
        }
        net_macro_set.insert(new_macro);
      }
    }

    // create new pin
    if (net_macro_set.size() < 1 || ((net_macro_set.size() == 1) && (new_net->get_pin_list().size() == 0))) {
      delete new_net;
      continue;
    }
    for (std::set<FPInst*>::iterator it = net_macro_set.begin(); it != net_macro_set.end(); ++it) {
      FPPin* new_pin = new FPPin();
      new_pin->set_instance(*it);
      (*it)->add_pin(new_pin);
      new_pin->set_x(0);
      new_pin->set_y(0);
      new_pin->set_net(new_net);
      new_net->add_pin(new_pin);
    }
    _mdb->add_new_net(new_net);
  }

  LOG_INFO << "_mdb's netlist have build, the num of net: " << _mdb->get_new_net_list().size();
  _mdb->showNewNetMessage();
}

}  // namespace ipl::imp
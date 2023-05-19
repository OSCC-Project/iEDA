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

#include <time.h>

#include <algorithm>
#include <ctime>
#include <string>
#include <vector>

using namespace std;
namespace ipl::imp {

MacroPlacer::MacroPlacer(MPDB* mdb, ipl::Config* config) : _mdb(mdb)
{
  _mp_config = config->get_mp_config();
  init();
}

void MacroPlacer::runMacroPlacer()
{
  clock_t start = clock();

  // parition
  MPPartition* partition = new MPPartition(_mdb, _set);
  partition->runPartition();
  _mdb->buildNetList();
  LOG_INFO << "_mdb's netlist have build, the num of net: " << _mdb->get_new_net_list().size();
  _mdb->updatePlaceMacroList();
  LOG_INFO << "halo_x: " << _set->get_macro_halo_x();
  LOG_INFO << "halo_y: " << _set->get_macro_halo_y();
  LOG_INFO << "new_macro_density: " << _set->get_new_macro_density();

  // simulate anneal
  SolutionFactory factory = SolutionFactory();
  MPSolution* mp_solution = factory.createSolution(_mdb->get_place_macro_list(), _set);
  MPEvaluation* mp_evaluation = new MPEvaluation(_mdb, _set, mp_solution);
  SimulateAnneal* anneal = new SimulateAnneal(_set, mp_evaluation);
  anneal->runAnneal();

  for (FPInst* macro : _mdb->get_place_macro_list()) {
    LOG_INFO << macro->get_name() << " " << macro->get_x() << " " << macro->get_y() << " " << macro->get_width() << " "
             << macro->get_height();
  }

  _mdb->writeDB();
  string output_path = _set->get_output_path();
  plotGDS();
  _mdb->writeResult(output_path);
  double time = double(clock() - start) / CLOCKS_PER_SEC;
  LOG_INFO << "time consume: " << time << "s";
  writeSummary(time);
  for (FPInst* macro : _mdb->get_total_macro_list()) {
    if (macro->isMacro()) {
      LOG_INFO << macro->get_x() << "," << macro->get_y() << ",";
    }
  }
}

void MacroPlacer::init()
{
  // set
  _set = new Setting();
  _set->set_new_macro_density(_mp_config.get_new_macro_density());
  _set->set_output_path(_mp_config.get_output_path());
  _set->set_macro_halo_x(_mp_config.get_halo_x());
  _set->set_macro_halo_y(_mp_config.get_halo_y());
  _set->set_partition_type(PartitionType::Hmetis);
  _set->set_parts(_mp_config.get_parts());  // the number of cluster
  _set->set_ncon(5);                        // The number of balancing constraints
  _set->set_ufactor(_mp_config.get_ufactor());
  // simulate anneal
  _set->set_max_num_step(200);
  _set->set_perturb_per_step(_mp_config.get_perturb_per_step());
  _set->set_cool_rate(_mp_config.get_cool_rate());
  _set->set_init_temperature(1000);
  // cost weight
  _set->set_weight_area(1);      // 1
  _set->set_weight_wl(12);       // 12
  _set->set_weight_e_area(96);   // 96
  _set->set_weight_guidance(0);  // 3

  _set->set_weight_boundary(22);  // 22

  // solution type
  LOG_INFO << "solution type: " << _mp_config.get_solution_tpye();
  if (_mp_config.get_solution_tpye() == "BStarTree") {
    _set->set_solution_type(SolutionTYPE::BST);
  } else if ("SequencePair" == _mp_config.get_solution_tpye()) {
    _set->set_solution_type(SolutionTYPE::SP);
  } else {
    LOG_ERROR << "error illegal type: " << _mp_config.get_solution_tpye();
  }

  // B* tree
  _set->set_swap_pro(0.5);  // the probability of swap
  _set->set_move_pro(0.5);  // the probability of move

  setFixedMacro();
  addHalo();
  addBlockage();
  addGuidance();
  updateDensity();
  // set guidance
  initLocation();
}

void MacroPlacer::updateDensity()
{
  // update density
  float total_inst_area = 0;
  float core_area = 0;
  float density = 1;
  for (FPInst* macro : _mdb->get_design()->get_macro_list()) {
    total_inst_area += macro->get_area();
  }
  for (FPInst* std_cell : _mdb->get_design()->get_std_cell_list()) {
    total_inst_area += std_cell->get_area();
  }
  core_area = _mdb->get_layout()->get_core_shape()->get_area();
  density = total_inst_area / core_area;
  density = std::min(density, float(1));
  density = std::max(density, _set->get_new_macro_density());
  _set->set_new_macro_density(density);
}

void MacroPlacer::setFixedMacro()
{
  // set fixed macro
  for (FPInst* macro : _mdb->get_total_macro_list()) {
    macro->set_fixed(false);
  }
  std::vector<std::string> fixed_macro = _mp_config.get_fixed_macro();
  std::vector<int32_t> fixed_macro_coord = _mp_config.get_fixed_macro_coordinate();
  for (size_t i = 0; i < fixed_macro.size(); ++i) {
    _mdb->setMacroFixed(fixed_macro[i], fixed_macro_coord[2 * i], fixed_macro_coord[2 * i + 1]);
  }
}

void MacroPlacer::addHalo()
{
  // set halo
  uint32_t halo_x = _set->get_macro_halo_x();
  uint32_t halo_y = _set->get_macro_halo_y();
  for (FPInst* macro : _mdb->get_design()->get_macro_list()) {
    // uint32_t self_adaption_halo_x = macro->get_width() / 20;
    // uint32_t self_adaption_halo_y = macro->get_height() / 20;
    // halo_x = max(halo_x, self_adaption_halo_x);
    // halo_y = max(halo_y, self_adaption_halo_y);
    macro->set_halo_x(halo_x);
    macro->set_halo_y(halo_y);
    macro->addHalo();
  }
}

void MacroPlacer::deleteHalo()
{
  for (FPInst* macro : _mdb->get_design()->get_macro_list()) {
    macro->deleteHalo();
  }
}

void MacroPlacer::addBlockage()
{
  // set blockage
  std::vector<int32_t> blockage = _mp_config.get_blockage();
  for (size_t i = 0; i < blockage.size(); i += 4) {
    FPRect* rect = new FPRect();
    rect->set_x(blockage[i]);
    rect->set_y(blockage[i + 1]);
    rect->set_width(blockage[i + 2] - blockage[i]);
    rect->set_height(blockage[i + 3] - blockage[i + 1]);
    _mdb->add_blockage(rect);
  }
}

void MacroPlacer::addGuidance()
{
  // set guidance
  std::vector<std::string> guidance_macro_list = _mp_config.get_guidance_macro();
  std::vector<int32_t> guidance = _mp_config.get_guidance();
  for (size_t i = 0; i < guidance_macro_list.size(); ++i) {
    FPRect* rect = new FPRect();
    rect->set_x(guidance[i]);
    rect->set_y(guidance[i + 1]);
    rect->set_width(guidance[i + 2] - guidance[i]);
    rect->set_height(guidance[i + 3] - guidance[i + 1]);
    _mdb->add_guidance_to_macro_name(rect, guidance_macro_list[i]);
  }
}

void MacroPlacer::initLocation()
{
  for (FPInst* macro : _mdb->get_design()->get_macro_list()) {
    LOG_INFO << macro->get_name() << ": " << macro->get_x() << " " << macro->get_y() << " " << macro->get_width() << " "
             << macro->get_height();
    FPRect* guidance = new FPRect();
    guidance->set_x(macro->get_x());
    guidance->set_y(macro->get_y());
    guidance->set_width(macro->get_width());
    guidance->set_height(macro->get_height());
    _mdb->add_guidance_to_macro_name(guidance, macro);
  }
}

void MacroPlacer::writeSummary(double time)
{
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
  config << "time consume: " << time << "s" << std::endl;
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
    index_to_inst_map.insert(pair<int, FPInst*>(index, macro));
    inst_to_index_map.insert(pair<FPInst*, int>(macro, index));
    index++;
  }

  for (FPInst* inst : _mdb->get_design()->get_std_cell_list()) {
    index_to_inst_map.insert(pair<int, FPInst*>(index, inst));
    inst_to_index_map.insert(pair<FPInst*, int>(inst, index));
    index++;
  }

  vector<vector<int>> hyper_edge_list;
  for (FPNet* net : _mdb->get_design()->get_net_list()) {
    vector<int> hyper_edge;
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
  vector<int> partition_result = hmetis->get_result();
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
    result.insert(pair<FPInst*, int>(inst, partition_result[i]));
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

}  // namespace ipl::imp
/*
 * @Author: your name
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
 * @copyright Copyright (c) 2021 PCNL EDA.
 **/
#include "MacroPlacer.hh"

#include <time.h>

#include <algorithm>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
namespace ipl::imp {

void MacroPlacer::runMacroPlacer()
{
  // SequencePair* sq = new SequencePair(_mdb->get_design()->get_macro_list(), _set, true);
  // string output_path = _set->get_output_path();
  // _mdb->writeResult(output_path);
  // string file = output_path + "/plane_result_macro.gds";
  // _mdb->writeGDS(file);

  clock_t start = clock();

  // parition
  MPPartition* partition = new MPPartition(_mdb, _set);
  partition->runPartition();
  _mdb->buildNetList();
  cout << "_mdb's netlist have build, the num of net: " << _mdb->get_new_net_list().size() << endl;
  _mdb->updatePlaceMacroList();

  // simulate anneal
  SolutionFactory factory = SolutionFactory();
  MPSolution* mp_solution = factory.createSolution(_mdb->get_place_macro_list(), _set);
  MPEvaluation* mp_evaluation = new MPEvaluation(_mdb, _set, mp_solution);
  SimulateAnneal* anneal = new SimulateAnneal(_set, mp_evaluation);
  anneal->runAnneal();

  for (FPInst* macro : _mdb->get_place_macro_list()) {
    std::cout << macro->get_name() << " " << macro->get_x() << " " << macro->get_y() << " " << macro->get_width() << " "
              << macro->get_height() << std::endl;
  }

  _mdb->writeDB();
  string output_path = _set->get_output_path();
  string file = output_path + "/plane_result_macro.gds";
  _mdb->writeGDS(file);
  _mdb->writeResult(output_path);
  double time = double(clock() - start) / CLOCKS_PER_SEC;
  std::cout << "time consume: " << time << "s" << std::endl;
  writeSummary(time);
  for (FPInst* macro : _mdb->get_total_macro_list()) {
    if (macro->isMacro()) {
      std::cout << macro->get_x() << "," << macro->get_y() << ",";
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
  _set->set_partition_type(PartitionType::Metis);
  _set->set_parts(_mp_config.get_parts());  // the number of cluster
  _set->set_ncon(5);                        // The number of balancing constraints
  _set->set_ufactor(_mp_config.get_ufactor());
  // simulate anneal
  _set->set_max_num_step(500);
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
  std::cout << "solution type: " << _mp_config.get_solution_tpye() << std::endl;
  if (_mp_config.get_solution_tpye() == "BStarTree") {
    _set->set_solution_type(SolutionTYPE::BST);
  } else if ("SequencePair" == _mp_config.get_solution_tpye()) {
    _set->set_solution_type(SolutionTYPE::SP);
  } else {
    std::cout << "error illegal type: " << _mp_config.get_solution_tpye() << std::endl;
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
    float halo_area = 0;
    halo_area += float(macro->get_halo_x()) * float(macro->get_height()) * 2;
    halo_area += float(macro->get_halo_y()) * float(macro->get_width()) * 2;
    halo_area += float(macro->get_halo_x()) * float(macro->get_halo_y()) * 4;
    total_inst_area += halo_area;
  }
  for (FPInst* std_cell : _mdb->get_design()->get_std_cell_list()) {
    total_inst_area += std_cell->get_area();
  }
  core_area = float(_mdb->get_layout()->get_core_shape()->get_width()) * float(_mdb->get_layout()->get_core_shape()->get_height());
  density = total_inst_area / core_area + 0.05;
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
    uint32_t self_adaption_halo_x = macro->get_width() / 20;
    uint32_t self_adaption_halo_y = macro->get_height() / 20;
    halo_x = max(halo_x, self_adaption_halo_x);
    halo_y = max(halo_y, self_adaption_halo_y);
    halo_x = min(halo_x, halo_y);
    macro->set_halo_x(halo_x);
    macro->set_halo_y(halo_x);
    macro->addHalo();
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
    std::cout << macro->get_name() << ": " << macro->get_x() << " " << macro->get_y() << " " << macro->get_width() << " "
              << macro->get_height() << std::endl;
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

}  // namespace ipl::imp
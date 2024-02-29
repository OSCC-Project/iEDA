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
 * @Author: S.J Chen
 * @Date: 2022-03-06 21:26:05
 * @LastEditTime: 2022-10-27 19:41:20
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/operator/global_placer/nesterov_place/database/NesterovDatabase.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_OPERATOR_GP_NESTEROV_DATABASE_H
#define IPL_OPERATOR_GP_NESTEROV_DATABASE_H

#include <map>
#include <string>
#include <vector>

#include "BinGrid.hh"
#include "Density.hh"
#include "ElectricFieldGradient.hh"
#include "GridManager.hh"
#include "HPWirelength.hh"
#include "NesInstance.hh"
#include "NesNet.hh"
#include "NesPin.hh"
#include "PlacerDB.hh"
#include "TopologyManager.hh"
#include "WAWirelengthGradient.hh"
#include "nesterov/Nesterov.hh"
#include "timing/TimingAnnotation.hh"

namespace ipl {

class NesterovDatabase
{
 public:
  NesterovDatabase();
  NesterovDatabase(const NesterovDatabase&) = delete;
  NesterovDatabase(NesterovDatabase&&) = delete;
  ~NesterovDatabase();

  NesterovDatabase& operator=(const NesterovDatabase&) = delete;
  NesterovDatabase& operator=(NesterovDatabase&&) = delete;

  void reset();

 private:
  PlacerDB* _placer_db;
  std::vector<NesInstance*> _nInstance_list;
  std::vector<NesNet*> _nNet_list;
  std::vector<NesPin*> _nPin_list;

  int32_t _nInstances_range;
  int32_t _nNets_range;
  int32_t _nPins_range;

  std::map<Instance*, NesInstance*> _nInstance_map;
  std::map<Net*, NesNet*> _nNet_map;
  std::map<Pin*, NesPin*> _nPin_map;

  std::map<NesInstance*, Instance*> _instance_map;
  std::map<NesNet*, Net*> _net_map;
  std::map<NesPin*, Pin*> _pin_map;

  // wirelength.
  TopologyManager* _topology_manager;
  TimingAnnotation* _timing_annotation;
  Wirelength* _wirelength;
  WirelengthGradient* _wirelength_gradient;
  float _wirelength_coef;
  float _base_wirelength_coef;
  float _wirelength_grad_sum;

  // density.
  GridManager* _grid_manager;
  BinGrid* _bin_grid;
  Density* _density;
  DensityGradient* _density_gradient;
  float _density_penalty;
  float _density_grad_sum;

  // nesterov solver.
  bool _is_diverged;
  Nesterov* _nesterov_solver;

  friend class NesterovPlace;
};
inline NesterovDatabase::NesterovDatabase()
    : _placer_db(nullptr),
      _nInstances_range(0),
      _nNets_range(0),
      _nPins_range(0),
      _topology_manager(nullptr),
      _timing_annotation(nullptr),
      _wirelength(nullptr),
      _wirelength_gradient(nullptr),
      _wirelength_coef(0.0F),
      _base_wirelength_coef(0.0F),
      _wirelength_grad_sum(0.0F),
      _grid_manager(nullptr),
      _bin_grid(nullptr),
      _density(nullptr),
      _density_gradient(nullptr),
      _density_penalty(0.0F),
      _density_grad_sum(0.0F),
      _is_diverged(false),
      _nesterov_solver(new Nesterov())
{
}
inline NesterovDatabase::~NesterovDatabase()
{
  delete _wirelength_gradient;
  delete _wirelength;
  delete _topology_manager;
  delete _timing_annotation;

  delete _density_gradient;
  delete _density;
  delete _bin_grid;
  delete _grid_manager;

  delete _nesterov_solver;

  for (auto* nInst : _nInstance_list) {
    delete nInst;
  }
  for (auto* nNet : _nNet_list) {
    delete nNet;
  }
  for (auto* nPin : _nPin_list) {
    delete nPin;
  }

  _nInstances_range = 0;
  _nNets_range = 0;
  _nPins_range = 0;
}

inline void NesterovDatabase::reset()
{
  //
}

}  // namespace ipl

#endif

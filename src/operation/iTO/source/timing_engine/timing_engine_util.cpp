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

#include "timing_engine_util.h"

#include "EstimateParasitics.h"
#include "ToConfig.h"
#include "data_manager.h"
#include "timing_engine.h"

namespace ito {

TOLibRepowerInstance::TOLibRepowerInstance(Pin* driver_pin)
{
  _driver_pin = driver_pin;
}

bool TOLibRepowerInstance::repowerInstance()
{
  if (false == is_repower()) {
    return false;
  }

  if (false == find_best_cell()) {
    repowerInstance(_repower_size_best, _inst);
  }

  return true;
}

float TOLibRepowerInstance::calLoad(ista::LibCell* cell)
{
  return timingEngine->get_target_map()->get_load(cell);
}

float TOLibRepowerInstance::calDelay(ista::LibCell* cell)
{
  return cell->isBuffer() || cell->isInverter() ? timingEngine->calcDelayOfBuffer(_load, cell) : 0.0;
}

float TOLibRepowerInstance::calDist(float target_cell_load)
{
  return abs(_load - target_cell_load);
}

bool TOLibRepowerInstance::is_repower()
{
  _inst = _driver_pin->get_own_instance();
  if (!_inst) {
    return false;
  }

  ista::Net* net = _driver_pin->get_net();
  if (net) {
    toEvalInst->estimateInvalidNetParasitics(net, _driver_pin);
  }

  _load = _driver_pin->get_net()->getLoad(AnalysisMode::kMax, TransType::kRise);
  if (_load <= 0.0) {
    return false;
  }

  /// init original cell
  _repower_size_best = _inst->get_inst_cell();
  if (!_repower_size_best) {
    return false;
  }

  _cell_target_load_best = timingEngine->get_target_map()->get_load(_repower_size_best);
  _load_margin_best = abs(_load - _cell_target_load_best);

  _b_buffer = _repower_size_best->isBuffer() || _repower_size_best->isInverter();
  _cell_delay_best = _b_buffer ? timingEngine->calcDelayOfBuffer(_load, _repower_size_best) : 0.0;

  return true;
};

/// @brief  find the best cell for pin
/// @return true : no need change cell, false : need to change cell
bool TOLibRepowerInstance::find_best_cell()
{
  auto origin_cell = _repower_size_best;
  for (auto* target_lib_cell : *timingEngine->get_sta_engine()->classifyCells(_repower_size_best)) {
    if (is_best(target_lib_cell)) {
      set_best(target_lib_cell);
    }
  }

  return origin_cell == _repower_size_best;
}

bool TOLibRepowerInstance::is_best(ista::LibCell* cell)
{
  const char* buf_name = cell->get_cell_name();
  if (strstr(buf_name, "CLK") != NULL || !timingEngine->canFindLibertyCell(cell)) {
    return false;
  }

  float target_cell_load = calLoad(cell);
  float delay = calDelay(cell);
  float dist = calDist(target_cell_load);

  return _b_buffer ? check_buf(dist, delay) : check_load(dist, target_cell_load);
}

void TOLibRepowerInstance::set_best(ista::LibCell* cell)
{
  _repower_size_best = cell;

  _cell_target_load_best = calLoad(cell);
  _cell_delay_best = calDelay(cell);
  _load_margin_best = calDist(_cell_target_load_best);
}

bool TOLibRepowerInstance::repowerInstance(ista::LibCell* repower_size, ista::Instance* repowered_inst)
{
  const char* replaced_lib_cell_name = repower_size->get_cell_name();

  TimingIDBAdapter* idb_adapter = timingEngine->get_sta_adapter();
  idb::IdbLayout* layout = idb_adapter->get_idb()->get_def_service()->get_layout();
  idb::IdbCellMaster* replaced_cell_master = layout->get_cell_master_list()->find_cell_master(replaced_lib_cell_name);
  if (replaced_cell_master) {
    idb::IdbInstance* dinst = idb_adapter->staToDb(repowered_inst);
    idb::IdbCellMaster* idb_master = dinst->get_cell_master();
    Master* master = new Master(idb_master);
    float area_master = toDmInst->calcMasterArea(master, toDmInst->get_dbu());

    if (replaced_cell_master->get_name() == idb_master->get_name()) {
      return false;
    }

    toDmInst->increDesignArea(-area_master);

    idb::IdbCellMaster* replace_master_idb = idb_adapter->staToDb(repower_size);

    Master* replace_master = new Master(replace_master_idb);
    float area_replace_master = toDmInst->calcMasterArea(replace_master, toDmInst->get_dbu());

    idb_adapter->substituteCell(repowered_inst, repower_size);
    timingEngine->get_sta_engine()->repowerInstance(repowered_inst->get_name(), repower_size->get_cell_name());
    toDmInst->increDesignArea(area_replace_master);

    return true;
  }
  return false;
}

}  // namespace ito
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
#include "DataManager.hpp"

#include "Monitor.hpp"
#include "Utility.hpp"

namespace idrc {

// public

void DataManager::initInst()
{
  if (_dm_instance == nullptr) {
    _dm_instance = new DataManager();
  }
}

DataManager& DataManager::getInst()
{
  if (_dm_instance == nullptr) {
    DRCLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_dm_instance;
}

void DataManager::destroyInst()
{
  if (_dm_instance != nullptr) {
    delete _dm_instance;
    _dm_instance = nullptr;
  }
}

// function

void DataManager::input(std::map<std::string, std::any>& config_map)
{
  Monitor monitor;
  DRCLOG.info(Loc::current(), "Starting...");
  DRCI.input(config_map);
  buildConfig();
  buildDatabase();
  printConfig();
  printDatabase();
  DRCLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void DataManager::output()
{
  Monitor monitor;
  DRCLOG.info(Loc::current(), "Starting...");
  DRCI.output();
  DRCLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

#if 1  // 获得唯一的pitch

int32_t DataManager::getOnlyPitch()
{
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();

  std::vector<int32_t> pitch_list;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    pitch_list.push_back(routing_layer.get_pitch());
  }
  for (int32_t pitch : pitch_list) {
    if (pitch_list.front() != pitch) {
      DRCLOG.error(Loc::current(), "The pitch is not equal!");
    }
  }
  return pitch_list.front();
}

#endif

// private

DataManager* DataManager::_dm_instance = nullptr;

#if 1  // build

void DataManager::buildConfig()
{
  //////////////////////////////////////////////
  // **********        DRC         ********** //
  _config.temp_directory_path = std::filesystem::absolute(_config.temp_directory_path);
  _config.temp_directory_path += "/";
  if (_config.golden_directory_path != "null") {
    _config.golden_directory_path = std::filesystem::absolute(_config.golden_directory_path);
    _config.golden_directory_path += "/";
  }
  _config.log_file_path = _config.temp_directory_path + "drc.log";
  // **********     RuleValidator  ********** //
  _config.rv_temp_directory_path = _config.temp_directory_path + "rule_validator/";
  // **********     GDSPlotter     ********** //
  _config.gp_temp_directory_path = _config.temp_directory_path + "gds_plotter/";
  /////////////////////////////////////////////
  // **********        DRC         ********** //
  DRCUTIL.removeDir(_config.temp_directory_path);
  DRCUTIL.createDir(_config.temp_directory_path);
  DRCUTIL.createDirByFile(_config.log_file_path);
  // **********   RuleValidator    ********** //
  DRCUTIL.createDir(_config.rv_temp_directory_path);
  // **********     GDSPlotter     ********** //
  DRCUTIL.createDir(_config.gp_temp_directory_path);
  //////////////////////////////////////////////
  DRCLOG.openLogFileStream(_config.log_file_path);
}

void DataManager::buildDatabase()
{
  buildDie();
  buildDesignRule();
  buildLayerList();
  buildLayerInfo();
}

void DataManager::buildDie()
{
  makeDie();
  checkDie();
}

void DataManager::makeDie()
{
}

void DataManager::checkDie()
{
  Die& die = _database.get_die();

  if (die.get_ll_x() < 0 || die.get_ll_y() < 0 || die.get_ur_x() < 0 || die.get_ur_y() < 0) {
    DRCLOG.error(Loc::current(), "The die '(", die.get_ll_x(), " , ", die.get_ll_y(), ") - (", die.get_ur_x(), " , ", die.get_ur_y(), ")' is wrong!");
  }
  if ((die.get_ur_x() <= die.get_ll_x()) || (die.get_ur_y() <= die.get_ll_y())) {
    DRCLOG.error(Loc::current(), "The die '(", die.get_ll_x(), " , ", die.get_ll_y(), ") - (", die.get_ur_x(), " , ", die.get_ur_y(), ")' is wrong!");
  }
}

void DataManager::buildDesignRule()
{
  std::map<int32_t, int32_t>& routing_idb_layer_id_to_idx_map = _database.get_routing_idb_layer_id_to_idx_map();
  MaxViaStackRule& max_via_stack_rule = _database.get_max_via_stack_rule();
  max_via_stack_rule.bottom_routing_layer_idx = routing_idb_layer_id_to_idx_map[max_via_stack_rule.bottom_routing_layer_idx];
  max_via_stack_rule.top_routing_layer_idx = routing_idb_layer_id_to_idx_map[max_via_stack_rule.top_routing_layer_idx];
}

void DataManager::buildLayerList()
{
  transLayerList();
  makeLayerList();
  checkLayerList();
}

void DataManager::transLayerList()
{
  std::map<int32_t, int32_t>& routing_idb_layer_id_to_idx_map = _database.get_routing_idb_layer_id_to_idx_map();
  std::map<int32_t, int32_t>& cut_idb_layer_id_to_idx_map = _database.get_cut_idb_layer_id_to_idx_map();

  for (RoutingLayer& routing_layer : _database.get_routing_layer_list()) {
    routing_layer.set_layer_idx(routing_idb_layer_id_to_idx_map[routing_layer.get_layer_idx()]);
  }
  for (CutLayer& cut_layer_list : _database.get_cut_layer_list()) {
    cut_layer_list.set_layer_idx(cut_idb_layer_id_to_idx_map[cut_layer_list.get_layer_idx()]);
  }
}

void DataManager::makeLayerList()
{
  makeRoutingLayerList();
  makeCutLayerList();
}

void DataManager::makeRoutingLayerList()
{
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();

  auto getFrequentNum = [](const std::vector<int32_t>& num_list) {
    if (num_list.empty()) {
      DRCLOG.error(Loc::current(), "The num_list is empty!");
    }
    std::map<int32_t, int32_t> num_count_map;
    for (int32_t num : num_list) {
      num_count_map[num]++;
    }
    std::map<int32_t, std::vector<int32_t>, std::greater<int32_t>> count_num_list_map;
    for (auto& [num, count] : num_count_map) {
      count_num_list_map[count].push_back(num);
    }
    int32_t frequent_num = INT32_MAX;
    for (int32_t num : count_num_list_map.begin()->second) {
      frequent_num = std::min(frequent_num, num);
    }
    return frequent_num;
  };
  int32_t step_length;
  {
    std::vector<int32_t> pitch_list;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      pitch_list.push_back(routing_layer.get_pitch());
    }
    step_length = getFrequentNum(pitch_list);
  }
  for (RoutingLayer& routing_layer : routing_layer_list) {
    routing_layer.set_pitch(step_length);
  }
}

void DataManager::makeCutLayerList()
{
  std::vector<CutLayer>& cut_layer_list = _database.get_cut_layer_list();

  for (size_t i = 1; i < cut_layer_list.size(); i++) {
    DifferentLayerCutSpacingRule& pre_different_layer_cut_spacing_rule = cut_layer_list[i - 1].get_different_layer_cut_spacing_rule();
    DifferentLayerCutSpacingRule& curr_different_layer_cut_spacing_rule = cut_layer_list[i].get_different_layer_cut_spacing_rule();
    pre_different_layer_cut_spacing_rule.above_spacing = curr_different_layer_cut_spacing_rule.below_spacing;
    pre_different_layer_cut_spacing_rule.above_prl = curr_different_layer_cut_spacing_rule.below_prl;
    pre_different_layer_cut_spacing_rule.above_prl_spacing = curr_different_layer_cut_spacing_rule.below_prl_spacing;
  }
  cut_layer_list.back().get_different_layer_cut_spacing_rule().above_spacing = 0;
  cut_layer_list.back().get_different_layer_cut_spacing_rule().above_prl = 0;
  cut_layer_list.back().get_different_layer_cut_spacing_rule().above_prl_spacing = 0;
}

void DataManager::checkLayerList()
{
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = _database.get_cut_layer_list();

  if (routing_layer_list.empty()) {
    DRCLOG.error(Loc::current(), "The routing_layer_list is empty!");
  }
  if (cut_layer_list.empty()) {
    DRCLOG.error(Loc::current(), "The cut_layer_list is empty!");
  }
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::string& layer_name = routing_layer.get_layer_name();
    if (routing_layer.get_prefer_direction() == Direction::kNone) {
      DRCLOG.error(Loc::current(), "The layer '", layer_name, "' prefer_direction is none!");
    }
    if (routing_layer.get_pitch() <= 0) {
      DRCLOG.error(Loc::current(), "The layer '", layer_name, "' pitch '", routing_layer.get_pitch(), "' is wrong!");
    }
  }
}

void DataManager::buildLayerInfo()
{
  std::map<int32_t, std::vector<int32_t>>& routing_to_adjacent_cut_map = _database.get_routing_to_adjacent_cut_map();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = _database.get_cut_to_adjacent_routing_map();

  std::vector<std::tuple<int32_t, bool, int32_t>> order_routing_layer_list;
  for (RoutingLayer& routing_layer : _database.get_routing_layer_list()) {
    order_routing_layer_list.emplace_back(routing_layer.get_layer_order(), true, routing_layer.get_layer_idx());
  }
  for (CutLayer& cut_layer : _database.get_cut_layer_list()) {
    order_routing_layer_list.emplace_back(cut_layer.get_layer_order(), false, cut_layer.get_layer_idx());
  }
  std::sort(order_routing_layer_list.begin(), order_routing_layer_list.end(),
            [](std::tuple<int32_t, bool, int32_t>& a, std::tuple<int32_t, bool, int32_t>& b) { return std::get<0>(a) < std::get<0>(b); });
  for (int32_t i = 0; i < static_cast<int32_t>(order_routing_layer_list.size()); i++) {
    if (std::get<1>(order_routing_layer_list[i]) == true) {
      if (i - 1 >= 0) {
        routing_to_adjacent_cut_map[std::get<2>(order_routing_layer_list[i])].push_back(std::get<2>(order_routing_layer_list[i - 1]));
      }
      if (i + 1 < static_cast<int32_t>(order_routing_layer_list.size())) {
        routing_to_adjacent_cut_map[std::get<2>(order_routing_layer_list[i])].push_back(std::get<2>(order_routing_layer_list[i + 1]));
      }
    } else {
      if (i - 1 >= 0) {
        cut_to_adjacent_routing_map[std::get<2>(order_routing_layer_list[i])].push_back(std::get<2>(order_routing_layer_list[i - 1]));
      }
      if (i + 1 < static_cast<int32_t>(order_routing_layer_list.size())) {
        cut_to_adjacent_routing_map[std::get<2>(order_routing_layer_list[i])].push_back(std::get<2>(order_routing_layer_list[i + 1]));
      }
    }
  }
}

void DataManager::printConfig()
{
  /////////////////////////////////////////////
  // **********        DRC         ********** //
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(0), "DRC_CONFIG_INPUT");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(1), "temp_directory_path");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(2), _config.temp_directory_path);
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(1), "thread_number");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(2), _config.thread_number);
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(1), "golden_directory_path");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(2), _config.golden_directory_path);
  // **********        DRC         ********** //
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(0), "DRC_CONFIG_BUILD");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(1), "log_file_path");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(2), _config.log_file_path);
  // **********     DRCEngine     ********** //
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(1), "RuleValidator");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(2), "rv_temp_directory_path");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(3), _config.rv_temp_directory_path);
  // **********     GDSPlotter     ********** //
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(1), "GDSPlotter");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(2), "gp_temp_directory_path");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(3), _config.gp_temp_directory_path);
  /////////////////////////////////////////////
}

void DataManager::printDatabase()
{
  ////////////////////////////////////////////////
  // ********** DRC ********** //
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(0), "DRC_DATABASE");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(1), "design_name");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(2), _database.get_design_name());
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(1), "lef_file_path_list");
  for (std::string& lef_file_path : _database.get_lef_file_path_list()) {
    DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(2), lef_file_path);
  }
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(1), "def_file_path");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(2), _database.get_def_file_path());
  // **********     MicronDBU     ********** //
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(1), "micron_dbu");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(2), _database.get_micron_dbu());
  // **********  ManufactureGrid  ********** //
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(1), "manufacture_grid");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(2), _database.get_manufacture_grid());
  // **********        Die        ********** //
  Die& die = _database.get_die();
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(1), "die");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(2), "(", die.get_ll_x(), ",", die.get_ll_y(), ")-(", die.get_ur_x(), ",", die.get_ur_y(), ")");
  // ********** RoutingLayer ********** //
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(1), "routing_layer_num");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(2), routing_layer_list.size());
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(1), "routing_layer");
  for (RoutingLayer& routing_layer : routing_layer_list) {
    DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(2), "idx:", routing_layer.get_layer_idx(), " order:", routing_layer.get_layer_order(),
                " name:", routing_layer.get_layer_name(), " prefer_direction:", GetDirectionName()(routing_layer.get_prefer_direction()),
                " pitch:", routing_layer.get_pitch());
  }
  // ********** CutLayer ********** //
  std::vector<CutLayer>& cut_layer_list = _database.get_cut_layer_list();
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(1), "cut_layer_num");
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(2), cut_layer_list.size());
  DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(1), "cut_layer");
  for (CutLayer& cut_layer : cut_layer_list) {
    DRCLOG.info(Loc::current(), DRCUTIL.getSpaceByTabNum(2), "idx:", cut_layer.get_layer_idx(), " order:", cut_layer.get_layer_order(),
                " name:", cut_layer.get_layer_name());
  }
  ////////////////////////////////////////////////
}

#endif

}  // namespace idrc

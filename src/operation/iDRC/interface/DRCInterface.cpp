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

#include "DRCInterface.hpp"

#include "DataManager.hpp"
#include "GDSPlotter.hpp"
#include "Monitor.hpp"
#include "RuleValidator.hpp"
#include "feature_manager.h"
#include "idm.h"

namespace idrc {

// public

DRCInterface& DRCInterface::getInst()
{
  if (_drc_interface_instance == nullptr) {
    _drc_interface_instance = new DRCInterface();
  }
  return *_drc_interface_instance;
}

void DRCInterface::destroyInst()
{
  if (_drc_interface_instance != nullptr) {
    delete _drc_interface_instance;
    _drc_interface_instance = nullptr;
  }
}

#if 1  // 外部调用DRC的API

#if 1  // iDRC

void DRCInterface::initDRC(std::map<std::string, std::any> config_map, bool enable_quiet)
{
  Logger::initInst();
  if (enable_quiet) {
    DRCLOG.enableQuiet();
  }
  // clang-format off
  DRCLOG.info(Loc::current(), ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
  DRCLOG.info(Loc::current(), "______________________________   _____________________________________   ");
  DRCLOG.info(Loc::current(), "___(_)__  __ \\__  __ \\_  ____/   __  ___/__  __/__    |__  __ \\__  __/");
  DRCLOG.info(Loc::current(), "__  /__  / / /_  /_/ /  /        _____ \\__  /  __  /| |_  /_/ /_  /     ");
  DRCLOG.info(Loc::current(), "_  / _  /_/ /_  _, _// /___      ____/ /_  /   _  ___ |  _, _/_  /       ");
  DRCLOG.info(Loc::current(), "/_/  /_____/ /_/ |_| \\____/      /____/ /_/    /_/  |_/_/ |_| /_/       ");
  DRCLOG.info(Loc::current(), ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
  // clang-format on
  DRCLOG.printLogFilePath();
  //////////////////////////////////////////////////////
  //////////////////////////////////////////////////////
  //////////////////////////////////////////////////////
  Monitor monitor;
  DRCLOG.info(Loc::current(), "Starting...");

  DataManager::initInst();
  DRCDM.input(config_map);
  GDSPlotter::initInst();
  DRCGP.init();
  RuleValidator::initInst();

  DRCLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void DRCInterface::checkDef()
{
  std::map<std::string, std::vector<ids::Violation>> type_violation_map;
  for (ids::Violation& ids_violation : getViolationList(buildEnvShapeList(), buildResultShapeList())) {
    type_violation_map[ids_violation.violation_type].push_back(ids_violation);
  }
  printSummary(type_violation_map);
  outputViolationJson(type_violation_map);
  outputSummary(type_violation_map);
}

void DRCInterface::destroyDRC()
{
  Monitor monitor;
  DRCLOG.info(Loc::current(), "Starting...");

  RuleValidator::destroyInst();
  DRCGP.destroy();
  GDSPlotter::destroyInst();
  DRCDM.output();
  DataManager::destroyInst();

  DRCLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());

  DRCLOG.printLogFilePath();
  // clang-format off
  DRCLOG.info(Loc::current(), ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
  DRCLOG.info(Loc::current(), "______________________________   _____________________   _____________________  __  ");
  DRCLOG.info(Loc::current(), "___(_)__  __ \\__  __ \\_  ____/   ___  ____/___  _/__  | / /___  _/_  ___/__  / / /");
  DRCLOG.info(Loc::current(), "__  /__  / / /_  /_/ /  /        __  /_    __  / __   |/ / __  / _____ \\__  /_/ /  ");
  DRCLOG.info(Loc::current(), "_  / _  /_/ /_  _, _// /___      _  __/   __/ /  _  /|  / __/ /  ____/ /_  __  /    ");
  DRCLOG.info(Loc::current(), "/_/  /_____/ /_/ |_| \\____/      /_/      /___/  /_/ |_/  /___/  /____/ /_/ /_/    ");
  DRCLOG.info(Loc::current(), ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
  // clang-format on
  Logger::destroyInst();
}

std::vector<ids::Violation> DRCInterface::getViolationList(const std::vector<ids::Shape>& ids_env_shape_list,
                                                           const std::vector<ids::Shape>& ids_result_shape_list)
{
  std::vector<DRCShape> drc_env_shape_list;
  drc_env_shape_list.reserve(ids_env_shape_list.size());
  for (const ids::Shape& ids_env_shape : ids_env_shape_list) {
    drc_env_shape_list.push_back(convertToDRCShape(ids_env_shape));
  }
  std::vector<DRCShape> drc_result_shape_list;
  drc_result_shape_list.reserve(ids_result_shape_list.size());
  for (const ids::Shape& ids_result_shape : ids_result_shape_list) {
    drc_result_shape_list.push_back(convertToDRCShape(ids_result_shape));
  }
  std::vector<ids::Violation> ids_violation_list;
  for (Violation& violation : DRCRV.verify(drc_env_shape_list, drc_result_shape_list)) {
    ids::Violation ids_violation;
    ids_violation.violation_type = GetViolationTypeName()(violation.get_violation_type());
    ids_violation.ll_x = violation.get_ll_x();
    ids_violation.ll_y = violation.get_ll_y();
    ids_violation.ur_x = violation.get_ur_x();
    ids_violation.ur_y = violation.get_ur_y();
    ids_violation.layer_idx = violation.get_layer_idx();
    ids_violation.is_routing = violation.get_is_routing();
    ids_violation.violation_net_set = violation.get_violation_net_set();
    ids_violation.required_size = violation.get_required_size();
    ids_violation_list.push_back(ids_violation);
  }
  return ids_violation_list;
}

#endif

#endif

#if 1  // DRC调用外部的API

#if 1  // TopData

#if 1  // input

void DRCInterface::input(std::map<std::string, std::any>& config_map)
{
  wrapConfig(config_map);
  wrapDatabase();
}

void DRCInterface::wrapConfig(std::map<std::string, std::any>& config_map)
{
  /////////////////////////////////////////////
  DRCDM.getConfig().temp_directory_path = DRCUTIL.getConfigValue<std::string>(config_map, "-temp_directory_path", "./drc_temp_directory");
  DRCDM.getConfig().thread_number = DRCUTIL.getConfigValue<int32_t>(config_map, "-thread_number", 128);
  DRCDM.getConfig().golden_directory_path = DRCUTIL.getConfigValue<std::string>(config_map, "-golden_directory_path", "null");
  omp_set_num_threads(std::max(DRCDM.getConfig().thread_number, 1));
  /////////////////////////////////////////////
}

void DRCInterface::wrapDatabase()
{
  wrapMicronDBU();
  wrapDie();
  wrapDesignRule();
  wrapLayerList();
  wrapLayerInfo();
}

void DRCInterface::wrapMicronDBU()
{
  DRCDM.getDatabase().set_micron_dbu(dmInst->get_idb_def_service()->get_design()->get_units()->get_micron_dbu());
}

void DRCInterface::wrapDie()
{
  idb::IdbDie* idb_die = dmInst->get_idb_lef_service()->get_layout()->get_die();

  Die& die = DRCDM.getDatabase().get_die();
  die.set_ll(idb_die->get_llx(), idb_die->get_lly());
  die.set_ur(idb_die->get_urx(), idb_die->get_ury());
}

void DRCInterface::wrapDesignRule()
{
  std::set<ViolationType>& exist_rule_set = DRCDM.getDatabase().get_exist_rule_set();

  // default
  {
    exist_rule_set.insert(ViolationType::kOutOfDie);
  }
  // MaxViaStackRule
  {
    MaxViaStackRule& max_via_stack_rule = DRCDM.getDatabase().get_max_via_stack_rule();
    idb::IdbLayers* idb_layer_list = dmInst->get_idb_def_service()->get_layout()->get_layers();
    idb::IdbMaxViaStack* idb_max_via_stack = dmInst->get_idb_lef_service()->get_layout()->get_max_via_stack();
    if (idb_max_via_stack != nullptr) {
      max_via_stack_rule.max_via_stack_num = idb_max_via_stack->get_stacked_via_num();
      max_via_stack_rule.bottom_routing_layer_idx = idb_layer_list->find_layer(idb_max_via_stack->get_layer_bottom())->get_id();
      max_via_stack_rule.top_routing_layer_idx = idb_layer_list->find_layer(idb_max_via_stack->get_layer_top())->get_id();
      exist_rule_set.insert(ViolationType::kMaxViaStack);
    }
  }
  // OffGridOrWrongWayRule
  {
    OffGridOrWrongWayRule& off_grid_or_wrong_way_rule = DRCDM.getDatabase().get_off_grid_or_wrong_way_rule();
    off_grid_or_wrong_way_rule.manufacture_grid = dmInst->get_idb_lef_service()->get_layout()->get_munufacture_grid();
    exist_rule_set.insert(ViolationType::kOffGridOrWrongWay);
  }
}

void DRCInterface::wrapLayerList()
{
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();

  std::vector<idb::IdbLayer*>& idb_layers = dmInst->get_idb_lef_service()->get_layout()->get_layers()->get_layers();
  for (idb::IdbLayer* idb_layer : idb_layers) {
    if (idb_layer->is_routing()) {
      idb::IdbLayerRouting* idb_routing_layer = dynamic_cast<idb::IdbLayerRouting*>(idb_layer);
      RoutingLayer routing_layer;
      routing_layer.set_layer_idx(idb_routing_layer->get_id());
      routing_layer.set_layer_order(idb_routing_layer->get_order());
      routing_layer.set_layer_name(idb_routing_layer->get_name());
      routing_layer.set_prefer_direction(getDRCDirectionByDB(idb_routing_layer->get_direction()));
      wrapTrackAxis(routing_layer, idb_routing_layer);
      wrapRoutingDesignRule(routing_layer, idb_routing_layer);
      routing_layer_list.push_back(std::move(routing_layer));
    } else if (idb_layer->is_cut()) {
      idb::IdbLayerCut* idb_cut_layer = dynamic_cast<idb::IdbLayerCut*>(idb_layer);
      CutLayer cut_layer;
      cut_layer.set_layer_idx(idb_cut_layer->get_id());
      cut_layer.set_layer_order(idb_cut_layer->get_order());
      cut_layer.set_layer_name(idb_cut_layer->get_name());
      wrapCutDesignRule(cut_layer, idb_cut_layer);
      cut_layer_list.push_back(std::move(cut_layer));
    }
  }
}

void DRCInterface::wrapTrackAxis(RoutingLayer& routing_layer, idb::IdbLayerRouting* idb_layer)
{
  if (idb_layer->get_prefer_track_grid() != nullptr) {
    routing_layer.set_pitch(idb_layer->get_prefer_track_grid()->get_track()->get_pitch());
  }
}

void DRCInterface::wrapRoutingDesignRule(RoutingLayer& routing_layer, idb::IdbLayerRouting* idb_layer)
{
  std::set<ViolationType>& exist_rule_set = DRCDM.getDatabase().get_exist_rule_set();

  // default
  {
    exist_rule_set.insert(ViolationType::kMetalShort);
  }
  // CornerFillSpacingRule
  {
    CornerFillSpacingRule& corner_fill_spacing_rule = routing_layer.get_corner_fill_spacing_rule();
    idb::routinglayer::Lef58CornerFillSpacing* idb_corner_fill = idb_layer->get_lef58_corner_fill_spacing().get();
    if (idb_corner_fill != nullptr) {
      corner_fill_spacing_rule.has_corner_fill = true;
      corner_fill_spacing_rule.corner_fill_spacing = idb_corner_fill->get_spacing();
      corner_fill_spacing_rule.edge_length_1 = idb_corner_fill->get_edge_length1();
      corner_fill_spacing_rule.edge_length_2 = idb_corner_fill->get_edge_length2();
      corner_fill_spacing_rule.adjacent_eol = idb_corner_fill->get_eol_width();
      exist_rule_set.insert(ViolationType::kCornerFillSpacing);
    }
  }
  // EndOfLineSpacingRule
  {
    std::vector<EndOfLineSpacingRule>& end_of_line_spacing_rule_list = routing_layer.get_end_of_line_spacing_rule_list();
    if (!idb_layer->get_lef58_spacing_eol_list().empty()) {
      for (std::shared_ptr<idb::routinglayer::Lef58SpacingEol> idb_spacing_eol : idb_layer->get_lef58_spacing_eol_list()) {
        EndOfLineSpacingRule end_of_line_spacing_rule;

        end_of_line_spacing_rule.eol_spacing = idb_spacing_eol.get()->get_eol_space();
        end_of_line_spacing_rule.eol_width = idb_spacing_eol.get()->get_eol_width();
        end_of_line_spacing_rule.eol_within = idb_spacing_eol.get()->get_eol_within().value();

        end_of_line_spacing_rule.has_ete = idb_spacing_eol.get()->get_end_to_end().has_value();
        if (idb_spacing_eol.get()->get_end_to_end().has_value()) {
          end_of_line_spacing_rule.ete_spacing = idb_spacing_eol.get()->get_end_to_end().value().get_end_to_end_space();
        }

        end_of_line_spacing_rule.has_par = idb_spacing_eol.get()->get_parallel_edge().has_value();
        if (idb_spacing_eol.get()->get_parallel_edge().has_value()) {
          end_of_line_spacing_rule.has_subtrace_eol_width = idb_spacing_eol.get()->get_parallel_edge().value().is_subtract_eol_width();
          end_of_line_spacing_rule.par_spacing = idb_spacing_eol.get()->get_parallel_edge().value().get_par_space();
          end_of_line_spacing_rule.par_within = idb_spacing_eol.get()->get_parallel_edge().value().get_par_within();
          end_of_line_spacing_rule.has_two_edges = idb_spacing_eol.get()->get_parallel_edge().value().is_two_edges();
          end_of_line_spacing_rule.has_min_length = idb_spacing_eol.get()->get_parallel_edge().value().get_min_length().has_value();
          if (idb_spacing_eol.get()->get_parallel_edge().value().get_min_length().has_value()) {
            end_of_line_spacing_rule.min_length = idb_spacing_eol.get()->get_parallel_edge().value().get_min_length().value();
          }
          end_of_line_spacing_rule.has_same_metal = idb_spacing_eol.get()->get_parallel_edge().value().is_same_metal();
        }

        end_of_line_spacing_rule.has_enclose_cut = idb_spacing_eol.get()->get_enclose_cut().has_value();
        if (idb_spacing_eol.get()->get_enclose_cut().has_value()) {
          end_of_line_spacing_rule.has_below
              = idb_spacing_eol.get()->get_enclose_cut().value().get_direction() == idb::routinglayer::Lef58SpacingEol::Direction::kBelow;
          end_of_line_spacing_rule.has_above
              = idb_spacing_eol.get()->get_enclose_cut().value().get_direction() == idb::routinglayer::Lef58SpacingEol::Direction::kAbove;
          end_of_line_spacing_rule.enclosed_dist = idb_spacing_eol.get()->get_enclose_cut().value().get_enclose_dist();
          end_of_line_spacing_rule.cut_to_metal_spacing = idb_spacing_eol.get()->get_enclose_cut().value().get_cut_to_metal_space();
          end_of_line_spacing_rule.has_all_cuts = idb_spacing_eol.get()->get_enclose_cut().value().is_all_cuts();
        }
        end_of_line_spacing_rule_list.push_back(end_of_line_spacing_rule);
      }
      exist_rule_set.insert(ViolationType::kEndOfLineSpacing);
    }
  }
  // MaximumWidthRule
  {
    MaximumWidthRule& maximum_width_rule = routing_layer.get_maximum_width_rule();
    int32_t max_width = INT32_MAX;
    if (idb_layer->get_max_width() != -1) {
      max_width = idb_layer->get_max_width();
    }
    maximum_width_rule.max_width = max_width;
    exist_rule_set.insert(ViolationType::kMaximumWidth);
  }
  // MinHoleRule
  {
    MinHoleRule& min_hole_rule = routing_layer.get_min_hole_rule();
    std::vector<IdbMinEncloseArea>& min_area_list = idb_layer->get_min_enclose_area_list()->get_min_area_list();
    if (!min_area_list.empty()) {
      min_hole_rule.min_hole_area = min_area_list.front()._area;
      exist_rule_set.insert(ViolationType::kMinHole);
    }
  }
  // MinimumAreaRule
  {
    MinimumAreaRule& minimum_area_rule = routing_layer.get_minimum_area_rule();
    minimum_area_rule.min_area = idb_layer->get_area();
    exist_rule_set.insert(ViolationType::kMinimumArea);
  }
  // MinimumCutRule
  {
    std::vector<MinimumCutRule>& minimum_cut_rule_list = routing_layer.get_minimum_cut_rule_list();
    if (!idb_layer->get_lef58_minimum_cut().empty()) {
      for (std::shared_ptr<idb::routinglayer::Lef58MinimumCut>& idb_minimum_cut : idb_layer->get_lef58_minimum_cut()) {
        MinimumCutRule minimum_cut_rule;
        if (idb_minimum_cut.get()->get_num_cuts().has_value()) {
          minimum_cut_rule.num_cuts = idb_minimum_cut.get()->get_num_cuts().value();
        } else {
          for (const idb::routinglayer::Lef58MinimumCut::CutClass& idb_cut_class : idb_minimum_cut.get()->get_cut_classes()) {
            if (idb_cut_class.get_class_name() == "VSINGLECUT") {
              minimum_cut_rule.num_cuts = idb_cut_class.get_num_cuts();
              break;
            }
          }
        }
        minimum_cut_rule.width = idb_minimum_cut.get()->get_width();
        minimum_cut_rule.has_within_cut_distance = idb_minimum_cut.get()->get_within_cut_distance().has_value();
        if (idb_minimum_cut.get()->get_within_cut_distance().has_value()) {
          minimum_cut_rule.within_cut_distance = idb_minimum_cut.get()->get_within_cut_distance().value();
        }
        minimum_cut_rule.has_from_above = idb_minimum_cut.get()->get_orient() == idb::routinglayer::Lef58MinimumCut::Orient::kFromAbove ? true : false;
        minimum_cut_rule.has_from_below = idb_minimum_cut.get()->get_orient() == idb::routinglayer::Lef58MinimumCut::Orient::kFromBelow ? true : false;
        minimum_cut_rule.has_length = idb_minimum_cut.get()->get_length().has_value();
        if (idb_minimum_cut.get()->get_length().has_value()) {
          minimum_cut_rule.length = idb_minimum_cut.get()->get_length().value().get_length();
          minimum_cut_rule.distance = idb_minimum_cut.get()->get_length().value().get_distance();
        }
        minimum_cut_rule_list.push_back(minimum_cut_rule);
      }
      exist_rule_set.insert(ViolationType::kMinimumCut);
    }
  }
  // MinimumWidthRule
  {
    MinimumWidthRule& minimum_width_rule = routing_layer.get_minimum_width_rule();
    minimum_width_rule.min_width = idb_layer->get_min_width();
    exist_rule_set.insert(ViolationType::kMinimumWidth);
  }
  // MinStepRule
  {
    MinStepRule& min_step_rule = routing_layer.get_min_step_rule();
    idb::IdbMinStep* idb_min_step = idb_layer->get_min_step().get();
    std::vector<std::shared_ptr<idb::routinglayer::Lef58MinStep>>& idb_lef58_min_step_list = idb_layer->get_lef58_min_step();
    if (idb_min_step != nullptr && !idb_lef58_min_step_list.empty()) {
      min_step_rule.min_step = idb_min_step->get_min_step_length();
      min_step_rule.max_edges = idb_min_step->get_max_edges();
      for (std::shared_ptr<idb::routinglayer::Lef58MinStep>& idb_lef58_min_step : idb_layer->get_lef58_min_step()) {
        min_step_rule.lef58_min_step = idb_lef58_min_step.get()->get_min_step_length();
        min_step_rule.lef58_min_adjacent_length = idb_lef58_min_step.get()->get_min_adjacent_length().value().get_min_adj_length();
        break;
      }
      exist_rule_set.insert(ViolationType::kMinStep);
    }
  }
  // NonsufficientMetalOverlapRule
  {
    NonsufficientMetalOverlapRule& nonsufficient_metal_overlap_rule = routing_layer.get_nonsufficient_metal_overlap_rule();
    nonsufficient_metal_overlap_rule.min_width = idb_layer->get_min_width();
    exist_rule_set.insert(ViolationType::kNonsufficientMetalOverlap);
  }
  // NotchSpacingRule
  {
    NotchSpacingRule& notch_spacing_rule = routing_layer.get_notch_spacing_rule();
    IdbLayerSpacingNotchLength& idb_notch = idb_layer->get_spacing_notchlength();
    idb::routinglayer::Lef58SpacingNotchlength* idb_lef58_notch = idb_layer->get_lef58_spacing_notchlength().get();
    if (idb_notch.exist()) {
      notch_spacing_rule.notch_spacing = idb_notch.get_min_spacing();
      notch_spacing_rule.notch_length = idb_notch.get_notch_length();
      exist_rule_set.insert(ViolationType::kNotchSpacing);
    } else if (idb_lef58_notch != nullptr) {
      notch_spacing_rule.notch_spacing = idb_lef58_notch->get_min_spacing();
      notch_spacing_rule.notch_length = idb_lef58_notch->get_min_notch_length();
      notch_spacing_rule.concave_ends = idb_lef58_notch->get_concave_ends_side_of_notch_width();
      exist_rule_set.insert(ViolationType::kNotchSpacing);
    }
  }
  // ParallelRunLengthSpacingRule
  {
    ParallelRunLengthSpacingRule& parallel_run_length_spacing_rule = routing_layer.get_parallel_run_length_spacing_rule();
    std::shared_ptr<idb::IdbParallelSpacingTable> idb_spacing_table;
    bool exist_spacing_table = false;
    if (idb_layer->get_spacing_table().get()->get_parallel().get() != nullptr && idb_layer->get_spacing_table().get()->is_parallel()) {
      idb_spacing_table = idb_layer->get_spacing_table()->get_parallel();
      exist_spacing_table = true;
    } else if (idb_layer->get_spacing_list() != nullptr && !idb_layer->get_spacing_table().get()->is_parallel()) {
      idb_spacing_table = idb_layer->get_spacing_table_from_spacing_list()->get_parallel();
      exist_spacing_table = true;
    }
    if (exist_spacing_table) {
      std::vector<int32_t>& width_list = parallel_run_length_spacing_rule.width_list;
      std::vector<int32_t>& parallel_length_list = parallel_run_length_spacing_rule.parallel_length_list;
      GridMap<int32_t>& width_parallel_length_map = parallel_run_length_spacing_rule.width_parallel_length_map;

      width_list = idb_spacing_table->get_width_list();
      parallel_length_list = idb_spacing_table->get_parallel_length_list();
      width_parallel_length_map.init(width_list.size(), parallel_length_list.size());
      for (int32_t x = 0; x < width_parallel_length_map.get_x_size(); x++) {
        for (int32_t y = 0; y < width_parallel_length_map.get_y_size(); y++) {
          width_parallel_length_map[x][y] = idb_spacing_table->get_spacing_table()[x][y];
        }
      }
      exist_rule_set.insert(ViolationType::kParallelRunLengthSpacing);
    }
  }
}

void DRCInterface::wrapCutDesignRule(CutLayer& cut_layer, idb::IdbLayerCut* idb_layer)
{
  std::set<ViolationType>& exist_rule_set = DRCDM.getDatabase().get_exist_rule_set();

  // default
  {
    exist_rule_set.insert(ViolationType::kCutShort);
  }
  // CutEOLSpacingRule
  {
    CutEOLSpacingRule& cut_eol_spacing_rule = cut_layer.get_cut_eol_spacing_rule();
    if (idb_layer->get_lef58_eol_spacing().get() != nullptr) {
      idb::cutlayer::Lef58EolSpacing* idb_eol_spacing = idb_layer->get_lef58_eol_spacing().get();
      cut_eol_spacing_rule.eol_spacing = idb_eol_spacing->get_cut_spacing1();
      cut_eol_spacing_rule.eol_prl = idb_eol_spacing->get_prl();
      cut_eol_spacing_rule.eol_prl_spacing = idb_eol_spacing->get_cut_spacing2();
      cut_eol_spacing_rule.eol_width = idb_eol_spacing->get_eol_width();
      cut_eol_spacing_rule.smaller_overhang = idb_eol_spacing->get_smaller_overhang();
      cut_eol_spacing_rule.equal_overhang = idb_eol_spacing->get_equal_overhang();
      cut_eol_spacing_rule.side_ext = idb_eol_spacing->get_side_ext();
      cut_eol_spacing_rule.backward_ext = idb_eol_spacing->get_backward_ext();
      cut_eol_spacing_rule.span_length = idb_eol_spacing->get_span_length();
      exist_rule_set.insert(ViolationType::kCutEOLSpacing);
    }
  }
  // DifferentLayerCutSpacingRule
  {
    DifferentLayerCutSpacingRule& different_layer_cut_spacing_rule = cut_layer.get_different_layer_cut_spacing_rule();
    if (!idb_layer->get_lef58_spacing_table().empty()) {
      idb::cutlayer::Lef58SpacingTable* spacing_table = nullptr;
      for (std::shared_ptr<idb::cutlayer::Lef58SpacingTable>& spacing_table_ptr : idb_layer->get_lef58_spacing_table()) {
        if (!spacing_table_ptr.get()->get_second_layer().has_value()) {
          continue;
        }
        spacing_table = spacing_table_ptr.get();
      }
      if (spacing_table != nullptr) {
        idb::cutlayer::Lef58SpacingTable::CutSpacing cut_spacing = spacing_table->get_cutclass().get_cut_spacing(0, 0);

        int32_t below_spacing = cut_spacing.get_cut_spacing1().value();
        int32_t below_prl = spacing_table->get_prl().value().get_prl();
        int32_t below_prl_spacing = cut_spacing.get_cut_spacing2().value();
        different_layer_cut_spacing_rule.below_spacing = below_spacing;
        different_layer_cut_spacing_rule.below_prl = below_prl;
        different_layer_cut_spacing_rule.below_prl_spacing = below_prl_spacing;
        exist_rule_set.insert(ViolationType::kDifferentLayerCutSpacing);
      }
    }
  }
  // EnclosureEdgeRule
  {
    std::vector<EnclosureEdgeRule>& enclosure_edge_rule_list = cut_layer.get_enclosure_edge_rule_list();
    if (!idb_layer->get_lef58_enclosure_edge_list().empty()) {
      for (std::shared_ptr<idb::cutlayer::Lef58EnclosureEdge>& idb_enclosure_edge : idb_layer->get_lef58_enclosure_edge_list()) {
        if (idb_enclosure_edge.get()->get_convex_corners().has_value()) {
          continue;
        }
        EnclosureEdgeRule enclosure_edge_rule;
        enclosure_edge_rule.has_above = (idb_enclosure_edge.get()->get_direction() == idb::cutlayer::Lef58EnclosureEdge::Direction::kAbove);
        enclosure_edge_rule.has_below = (idb_enclosure_edge.get()->get_direction() == idb::cutlayer::Lef58EnclosureEdge::Direction::kBelow);
        enclosure_edge_rule.overhang = idb_enclosure_edge.get()->get_overhang();
        enclosure_edge_rule.min_width = idb_enclosure_edge.get()->get_min_width().value();
        enclosure_edge_rule.par_length = idb_enclosure_edge.get()->get_par_length().value();
        enclosure_edge_rule.par_within = idb_enclosure_edge.get()->get_par_within().value();
        enclosure_edge_rule.has_except_two_edges = idb_enclosure_edge.get()->has_except_twoedges();
        enclosure_edge_rule_list.push_back(enclosure_edge_rule);
      }
      exist_rule_set.insert(ViolationType::kEnclosureEdge);
    }
  }
  // EnclosureParallelRule
  {
    EnclosureParallelRule& enclosure_parallel_rule = cut_layer.get_enclosure_parallel_rule();
    if (idb_layer->get_lef58_eol_enclosure().get() != nullptr) {
      idb::cutlayer::Lef58EolEnclosure* idb_eol_enclosure = idb_layer->get_lef58_eol_enclosure().get();
      enclosure_parallel_rule.eol_width = idb_eol_enclosure->get_eol_width();
      enclosure_parallel_rule.has_above = (idb_eol_enclosure->get_direction() == idb::cutlayer::Lef58EolEnclosure::Direction::kAbove);
      enclosure_parallel_rule.has_below = (idb_eol_enclosure->get_direction() == idb::cutlayer::Lef58EolEnclosure::Direction::kBelow);
      enclosure_parallel_rule.overhang = idb_eol_enclosure->get_overhang();
      if (idb_eol_enclosure->get_par_space().has_value()) {
        enclosure_parallel_rule.par_spacing = idb_eol_enclosure->get_par_space().value();
      }
      if (idb_eol_enclosure->get_extension().has_value()) {
        enclosure_parallel_rule.backward_ext = idb_eol_enclosure->get_extension().value().get_backward_ext();
        enclosure_parallel_rule.forward_ext = idb_eol_enclosure->get_extension().value().get_forward_ext();
      }
      enclosure_parallel_rule.has_min_length = idb_eol_enclosure->get_min_length().has_value();
      if (idb_eol_enclosure->get_min_length().has_value()) {
        enclosure_parallel_rule.min_length = idb_eol_enclosure->get_min_length().value();
      }
      exist_rule_set.insert(ViolationType::kEnclosureParallel);
    }
  }
  // SameLayerCutSpacingRule
  {
    SameLayerCutSpacingRule& same_layer_cut_spacing_rule = cut_layer.get_same_layer_cut_spacing_rule();
    if (!idb_layer->get_spacings().empty()) {
      same_layer_cut_spacing_rule.curr_spacing = idb_layer->get_spacings().front()->get_spacing();
      same_layer_cut_spacing_rule.curr_prl = 0;
      same_layer_cut_spacing_rule.curr_prl_spacing = idb_layer->get_spacings().front()->get_spacing();
      exist_rule_set.insert(ViolationType::kSameLayerCutSpacing);
    } else if (!idb_layer->get_lef58_spacing_table().empty()) {
      idb::cutlayer::Lef58SpacingTable* spacing_table = nullptr;
      for (std::shared_ptr<idb::cutlayer::Lef58SpacingTable>& spacing_table_ptr : idb_layer->get_lef58_spacing_table()) {
        if (spacing_table_ptr.get()->get_second_layer().has_value()) {
          continue;
        }
        spacing_table = spacing_table_ptr.get();
      }
      if (spacing_table != nullptr) {
        idb::cutlayer::Lef58SpacingTable::CutSpacing cut_spacing = spacing_table->get_cutclass().get_cut_spacing(0, 0);

        int32_t curr_spacing = cut_spacing.get_cut_spacing1().value();
        int32_t curr_prl = spacing_table->get_prl().value().get_prl();
        int32_t curr_prl_spacing = cut_spacing.get_cut_spacing2().value();
        same_layer_cut_spacing_rule.curr_spacing = curr_spacing;
        same_layer_cut_spacing_rule.curr_prl = curr_prl;
        same_layer_cut_spacing_rule.curr_prl_spacing = curr_prl_spacing;
        exist_rule_set.insert(ViolationType::kSameLayerCutSpacing);
      }
    }
  }
}

void DRCInterface::wrapLayerInfo()
{
  std::map<int32_t, int32_t>& routing_idb_layer_id_to_idx_map = DRCDM.getDatabase().get_routing_idb_layer_id_to_idx_map();
  std::map<int32_t, int32_t>& cut_idb_layer_id_to_idx_map = DRCDM.getDatabase().get_cut_idb_layer_id_to_idx_map();
  std::map<std::string, int32_t>& routing_layer_name_to_idx_map = DRCDM.getDatabase().get_routing_layer_name_to_idx_map();
  std::map<std::string, int32_t>& cut_layer_name_to_idx_map = DRCDM.getDatabase().get_cut_layer_name_to_idx_map();

  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  for (size_t i = 0; i < routing_layer_list.size(); i++) {
    routing_idb_layer_id_to_idx_map[routing_layer_list[i].get_layer_idx()] = static_cast<int32_t>(i);
    routing_layer_name_to_idx_map[routing_layer_list[i].get_layer_name()] = static_cast<int32_t>(i);
  }
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  for (size_t i = 0; i < cut_layer_list.size(); i++) {
    cut_idb_layer_id_to_idx_map[cut_layer_list[i].get_layer_idx()] = static_cast<int32_t>(i);
    cut_layer_name_to_idx_map[cut_layer_list[i].get_layer_name()] = static_cast<int32_t>(i);
  }
}

Direction DRCInterface::getDRCDirectionByDB(idb::IdbLayerDirection idb_direction)
{
  if (idb_direction == idb::IdbLayerDirection::kHorizontal) {
    return Direction::kHorizontal;
  } else if (idb_direction == idb::IdbLayerDirection::kVertical) {
    return Direction::kVertical;
  } else {
    return Direction::kOblique;
  }
}

#endif

#if 1  // output

void DRCInterface::output()
{
}

#endif

#endif

#if 1  // check

std::vector<ids::Shape> DRCInterface::buildEnvShapeList()
{
  std::vector<ids::Shape> env_shape_list;
  Monitor monitor;
  DRCLOG.info(Loc::current(), "Starting...");

  std::vector<idb::IdbInstance*>& idb_instance_list = dmInst->get_idb_def_service()->get_design()->get_instance_list()->get_instance_list();
  std::vector<idb::IdbSpecialNet*>& idb_special_net_list = dmInst->get_idb_def_service()->get_design()->get_special_net_list()->get_net_list();
  std::vector<idb::IdbPin*>& idb_io_pin_list = dmInst->get_idb_def_service()->get_design()->get_io_pin_list()->get_pin_list();

  size_t total_env_shape_num = 0;
  {
    // instance
    for (idb::IdbInstance* idb_instance : idb_instance_list) {
      // instance obs
      for (idb::IdbLayerShape* obs_box : idb_instance->get_obs_box_list()) {
        total_env_shape_num += obs_box->get_rect_list().size();
      }
      // instance pin without net
      for (idb::IdbPin* idb_pin : idb_instance->get_pin_list()->get_pin_list()) {
        for (idb::IdbLayerShape* port_box : idb_pin->get_port_box_list()) {
          total_env_shape_num += port_box->get_rect_list().size();
        }
        for (idb::IdbVia* idb_via : idb_pin->get_via_list()) {
          total_env_shape_num += 2;
          total_env_shape_num += idb_via->get_cut_layer_shape().get_rect_list().size();
        }
      }
    }
    // special net
    for (idb::IdbSpecialNet* idb_net : idb_special_net_list) {
      for (idb::IdbSpecialWire* idb_wire : idb_net->get_wire_list()->get_wire_list()) {
        for (idb::IdbSpecialWireSegment* idb_segment : idb_wire->get_segment_list()) {
          if (idb_segment->is_via()) {
            total_env_shape_num += idb_segment->get_via()->get_top_layer_shape().get_rect_list().size();
            total_env_shape_num += idb_segment->get_via()->get_bottom_layer_shape().get_rect_list().size();
            total_env_shape_num += idb_segment->get_via()->get_cut_layer_shape().get_rect_list().size();
          } else {
            total_env_shape_num += 1;
          }
        }
      }
    }
    // io pin
    for (idb::IdbPin* idb_io_pin : idb_io_pin_list) {
      for (idb::IdbLayerShape* port_box : idb_io_pin->get_port_box_list()) {
        total_env_shape_num += port_box->get_rect_list().size();
      }
    }
  }
  env_shape_list.reserve(total_env_shape_num);
  {
    // instance
    for (idb::IdbInstance* idb_instance : idb_instance_list) {
      // instance obs
      for (idb::IdbLayerShape* obs_box : idb_instance->get_obs_box_list()) {
        for (idb::IdbRect* rect : obs_box->get_rect_list()) {
          ids::Shape ids_shape;
          ids_shape.net_idx = -1;
          ids_shape.ll_x = rect->get_low_x();
          ids_shape.ll_y = rect->get_low_y();
          ids_shape.ur_x = rect->get_high_x();
          ids_shape.ur_y = rect->get_high_y();
          ids_shape.layer_idx = obs_box->get_layer()->get_id();
          ids_shape.is_routing = obs_box->get_layer()->is_routing();
          env_shape_list.push_back(ids_shape);
        }
      }
      // instance pin without net
      for (idb::IdbPin* idb_pin : idb_instance->get_pin_list()->get_pin_list()) {
        int32_t net_idx = -1;
        if (!isSkipping(idb_pin->get_net())) {
          net_idx = static_cast<int32_t>(idb_pin->get_net()->get_id());
        }
        for (idb::IdbLayerShape* port_box : idb_pin->get_port_box_list()) {
          for (idb::IdbRect* rect : port_box->get_rect_list()) {
            ids::Shape ids_shape;
            ids_shape.net_idx = net_idx;
            ids_shape.ll_x = rect->get_low_x();
            ids_shape.ll_y = rect->get_low_y();
            ids_shape.ur_x = rect->get_high_x();
            ids_shape.ur_y = rect->get_high_y();
            ids_shape.layer_idx = port_box->get_layer()->get_id();
            ids_shape.is_routing = port_box->get_layer()->is_routing();
            env_shape_list.push_back(ids_shape);
          }
        }
        for (idb::IdbVia* idb_via : idb_pin->get_via_list()) {
          {
            idb::IdbLayerShape idb_shape_top = idb_via->get_top_layer_shape();
            idb::IdbRect idb_box_top = idb_shape_top.get_bounding_box();

            ids::Shape ids_shape;
            ids_shape.net_idx = net_idx;
            ids_shape.ll_x = idb_box_top.get_low_x();
            ids_shape.ll_y = idb_box_top.get_low_y();
            ids_shape.ur_x = idb_box_top.get_high_x();
            ids_shape.ur_y = idb_box_top.get_high_y();
            ids_shape.layer_idx = idb_shape_top.get_layer()->get_id();
            ids_shape.is_routing = true;
            env_shape_list.push_back(ids_shape);
          }
          {
            idb::IdbLayerShape idb_shape_bottom = idb_via->get_bottom_layer_shape();
            idb::IdbRect idb_box_bottom = idb_shape_bottom.get_bounding_box();

            ids::Shape ids_shape;
            ids_shape.net_idx = net_idx;
            ids_shape.ll_x = idb_box_bottom.get_low_x();
            ids_shape.ll_y = idb_box_bottom.get_low_y();
            ids_shape.ur_x = idb_box_bottom.get_high_x();
            ids_shape.ur_y = idb_box_bottom.get_high_y();
            ids_shape.layer_idx = idb_shape_bottom.get_layer()->get_id();
            ids_shape.is_routing = true;
            env_shape_list.push_back(ids_shape);
          }
          idb::IdbLayerShape idb_shape_cut = idb_via->get_cut_layer_shape();
          for (idb::IdbRect* idb_rect : idb_shape_cut.get_rect_list()) {
            ids::Shape ids_shape;
            ids_shape.net_idx = net_idx;
            ids_shape.ll_x = idb_rect->get_low_x();
            ids_shape.ll_y = idb_rect->get_low_y();
            ids_shape.ur_x = idb_rect->get_high_x();
            ids_shape.ur_y = idb_rect->get_high_y();
            ids_shape.layer_idx = idb_shape_cut.get_layer()->get_id();
            ids_shape.is_routing = false;
            env_shape_list.push_back(ids_shape);
          }
        }
      }
    }
    // special net
    for (idb::IdbSpecialNet* idb_net : idb_special_net_list) {
      for (idb::IdbSpecialWire* idb_wire : idb_net->get_wire_list()->get_wire_list()) {
        for (idb::IdbSpecialWireSegment* idb_segment : idb_wire->get_segment_list()) {
          if (idb_segment->is_via()) {
            for (idb::IdbLayerShape layer_shape : {idb_segment->get_via()->get_top_layer_shape(), idb_segment->get_via()->get_bottom_layer_shape()}) {
              for (idb::IdbRect* rect : layer_shape.get_rect_list()) {
                ids::Shape ids_shape;
                ids_shape.net_idx = -1;
                ids_shape.ll_x = rect->get_low_x();
                ids_shape.ll_y = rect->get_low_y();
                ids_shape.ur_x = rect->get_high_x();
                ids_shape.ur_y = rect->get_high_y();
                ids_shape.layer_idx = layer_shape.get_layer()->get_id();
                ids_shape.is_routing = true;
                env_shape_list.push_back(ids_shape);
              }
            }
            idb::IdbLayerShape cut_layer_shape = idb_segment->get_via()->get_cut_layer_shape();
            for (idb::IdbRect* rect : cut_layer_shape.get_rect_list()) {
              ids::Shape ids_shape;
              ids_shape.net_idx = -1;
              ids_shape.ll_x = rect->get_low_x();
              ids_shape.ll_y = rect->get_low_y();
              ids_shape.ur_x = rect->get_high_x();
              ids_shape.ur_y = rect->get_high_y();
              ids_shape.layer_idx = cut_layer_shape.get_layer()->get_id();
              ids_shape.is_routing = false;
              env_shape_list.push_back(ids_shape);
            }
          } else {
            idb::IdbRect* idb_rect = idb_segment->get_bounding_box();
            ids::Shape ids_shape;
            ids_shape.net_idx = -1;
            ids_shape.ll_x = idb_rect->get_low_x();
            ids_shape.ll_y = idb_rect->get_low_y();
            ids_shape.ur_x = idb_rect->get_high_x();
            ids_shape.ur_y = idb_rect->get_high_y();
            ids_shape.layer_idx = idb_segment->get_layer()->get_id();
            ids_shape.is_routing = true;
            env_shape_list.push_back(ids_shape);
          }
        }
      }
    }
    // io pin
    for (idb::IdbPin* idb_io_pin : idb_io_pin_list) {
      int32_t net_idx = -1;
      if (!isSkipping(idb_io_pin->get_net())) {
        net_idx = static_cast<int32_t>(idb_io_pin->get_net()->get_id());
      }
      for (idb::IdbLayerShape* port_box : idb_io_pin->get_port_box_list()) {
        for (idb::IdbRect* rect : port_box->get_rect_list()) {
          ids::Shape ids_shape;
          ids_shape.net_idx = net_idx;
          ids_shape.ll_x = rect->get_low_x();
          ids_shape.ll_y = rect->get_low_y();
          ids_shape.ur_x = rect->get_high_x();
          ids_shape.ur_y = rect->get_high_y();
          ids_shape.layer_idx = port_box->get_layer()->get_id();
          ids_shape.is_routing = port_box->get_layer()->is_routing();
          env_shape_list.push_back(ids_shape);
        }
      }
    }
  }
  DRCLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
  return env_shape_list;
}

bool DRCInterface::isSkipping(idb::IdbNet* idb_net)
{
  if (idb_net == nullptr) {
    return true;
  }
  bool has_io_pin = false;
  if (idb_net->has_io_pins() && idb_net->get_io_pins()->get_pin_num() == 1) {
    has_io_pin = true;
  }
  bool has_io_cell = false;
  std::vector<idb::IdbInstance*>& instance_list = idb_net->get_instance_list()->get_instance_list();
  if (instance_list.size() == 1 && instance_list.front()->get_cell_master()->is_pad()) {
    has_io_cell = true;
  }
  if (has_io_pin && has_io_cell) {
    return true;
  }

  int32_t pin_num = 0;
  for (idb::IdbPin* idb_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
    if (idb_pin->get_term()->get_port_number() <= 0) {
      continue;
    }
    pin_num++;
  }
  for (idb::IdbPin* idb_pin : idb_net->get_io_pins()->get_pin_list()) {
    if (idb_pin->get_term()->get_port_number() <= 0) {
      continue;
    }
    pin_num++;
  }
  if (pin_num <= 1) {
    return true;
  }
  return false;
}

std::vector<ids::Shape> DRCInterface::buildResultShapeList()
{
  std::vector<ids::Shape> result_shape_list;
  Monitor monitor;
  DRCLOG.info(Loc::current(), "Starting...");

  std::vector<idb::IdbNet*>& idb_net_list = dmInst->get_idb_def_service()->get_design()->get_net_list()->get_net_list();

  size_t total_result_shape_num = 0;
  {
    for (idb::IdbNet* idb_net : idb_net_list) {
      for (idb::IdbRegularWire* idb_wire : idb_net->get_wire_list()->get_wire_list()) {
        for (idb::IdbRegularWireSegment* idb_segment : idb_wire->get_segment_list()) {
          if (idb_segment->get_point_number() >= 2) {
            total_result_shape_num += 1;
          }
          if (idb_segment->is_via()) {
            for (idb::IdbVia* idb_via : idb_segment->get_via_list()) {
              total_result_shape_num += idb_via->get_top_layer_shape().get_rect_list().size();
              total_result_shape_num += idb_via->get_bottom_layer_shape().get_rect_list().size();
              total_result_shape_num += idb_via->get_cut_layer_shape().get_rect_list().size();
            }
          }
          if (idb_segment->is_rect()) {
            total_result_shape_num += 1;
          }
        }
      }
    }
  }
  result_shape_list.reserve(total_result_shape_num);
  // net
  for (idb::IdbNet* idb_net : idb_net_list) {
    for (idb::IdbRegularWire* idb_wire : idb_net->get_wire_list()->get_wire_list()) {
      for (idb::IdbRegularWireSegment* idb_segment : idb_wire->get_segment_list()) {
        if (idb_segment->get_point_number() >= 2) {
          PlanarCoord first_coord(idb_segment->get_point_start()->get_x(), idb_segment->get_point_start()->get_y());
          PlanarCoord second_coord(idb_segment->get_point_second()->get_x(), idb_segment->get_point_second()->get_y());
          int32_t half_width = dynamic_cast<IdbLayerRouting*>(idb_segment->get_layer())->get_width() / 2;
          PlanarRect rect = DRCUTIL.getEnlargedRect(first_coord, second_coord, half_width);
          ids::Shape ids_shape;
          ids_shape.net_idx = static_cast<int32_t>(idb_net->get_id());
          ids_shape.ll_x = rect.get_ll_x();
          ids_shape.ll_y = rect.get_ll_y();
          ids_shape.ur_x = rect.get_ur_x();
          ids_shape.ur_y = rect.get_ur_y();
          ids_shape.layer_idx = idb_segment->get_layer()->get_id();
          ids_shape.is_routing = true;
          result_shape_list.push_back(ids_shape);
        }
        if (idb_segment->is_via()) {
          for (idb::IdbVia* idb_via : idb_segment->get_via_list()) {
            for (idb::IdbLayerShape layer_shape : {idb_via->get_top_layer_shape(), idb_via->get_bottom_layer_shape()}) {
              for (idb::IdbRect* rect : layer_shape.get_rect_list()) {
                ids::Shape ids_shape;
                ids_shape.net_idx = static_cast<int32_t>(idb_net->get_id());
                ids_shape.ll_x = rect->get_low_x();
                ids_shape.ll_y = rect->get_low_y();
                ids_shape.ur_x = rect->get_high_x();
                ids_shape.ur_y = rect->get_high_y();
                ids_shape.layer_idx = layer_shape.get_layer()->get_id();
                ids_shape.is_routing = true;
                result_shape_list.push_back(ids_shape);
              }
            }
            idb::IdbLayerShape cut_layer_shape = idb_via->get_cut_layer_shape();
            for (idb::IdbRect* rect : cut_layer_shape.get_rect_list()) {
              ids::Shape ids_shape;
              ids_shape.net_idx = static_cast<int32_t>(idb_net->get_id());
              ids_shape.ll_x = rect->get_low_x();
              ids_shape.ll_y = rect->get_low_y();
              ids_shape.ur_x = rect->get_high_x();
              ids_shape.ur_y = rect->get_high_y();
              ids_shape.layer_idx = cut_layer_shape.get_layer()->get_id();
              ids_shape.is_routing = false;
              result_shape_list.push_back(ids_shape);
            }
          }
        }
        if (idb_segment->is_rect()) {
          PlanarCoord offset_coord(idb_segment->get_point_start()->get_x(), idb_segment->get_point_start()->get_y());
          PlanarRect delta_rect(idb_segment->get_delta_rect()->get_low_x(), idb_segment->get_delta_rect()->get_low_y(),
                                idb_segment->get_delta_rect()->get_high_x(), idb_segment->get_delta_rect()->get_high_y());
          PlanarRect rect = DRCUTIL.getOffsetRect(delta_rect, offset_coord);
          ids::Shape ids_shape;
          ids_shape.net_idx = static_cast<int32_t>(idb_net->get_id());
          ids_shape.ll_x = rect.get_ll_x();
          ids_shape.ll_y = rect.get_ll_y();
          ids_shape.ur_x = rect.get_ur_x();
          ids_shape.ur_y = rect.get_ur_y();
          ids_shape.layer_idx = idb_segment->get_layer()->get_id();
          ids_shape.is_routing = true;
          result_shape_list.push_back(ids_shape);
        }
      }
    }
  }
  DRCLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
  return result_shape_list;
}

void DRCInterface::printSummary(std::map<std::string, std::vector<ids::Violation>>& type_violation_map)
{
  std::string& golden_directory_path = DRCDM.getConfig().golden_directory_path;
  if (golden_directory_path != "null") {
    return;
  }
  int32_t total_violation_num = 0;
  for (auto& [type, violation_list] : type_violation_map) {
    total_violation_num += static_cast<int32_t>(violation_list.size());
  }
  fort::char_table type_violation_map_table;
  {
    type_violation_map_table.set_cell_text_align(fort::text_align::right);
    type_violation_map_table << fort::header << "violation_type"
                             << "violation_num" << "prop" << fort::endr;
    for (auto& [type, violation_list] : type_violation_map) {
      type_violation_map_table << type << violation_list.size() << DRCUTIL.getPercentage(violation_list.size(), total_violation_num) << fort::endr;
    }
    type_violation_map_table << fort::header << "Total" << total_violation_num << DRCUTIL.getPercentage(total_violation_num, total_violation_num) << fort::endr;
  }
  DRCUTIL.printTableList({type_violation_map_table});
}

void DRCInterface::outputViolationJson(std::map<std::string, std::vector<ids::Violation>>& type_violation_map)
{
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();
  std::string& temp_directory_path = DRCDM.getConfig().temp_directory_path;

  std::vector<idb::IdbNet*>& idb_net_list = dmInst->get_idb_def_service()->get_design()->get_net_list()->get_net_list();

  std::vector<nlohmann::json> violation_json_list;
  for (auto& [type, violation_list] : type_violation_map) {
    for (ids::Violation& violation : violation_list) {
      nlohmann::json violation_json;
      violation_json["type"] = violation.violation_type;

      int32_t layer_idx = violation.layer_idx;
      if (!violation.is_routing) {
        std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[layer_idx];
        layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
      }
      violation_json["shape"] = {violation.ll_x, violation.ll_y, violation.ur_x, violation.ur_y, routing_layer_list[layer_idx].get_layer_name()};
      for (int32_t net_idx : violation.violation_net_set) {
        if (net_idx != -1) {
          violation_json["net"].push_back(idb_net_list[net_idx]->get_net_name());
        } else {
          violation_json["net"].push_back("obs");
        }
      }
      violation_json_list.push_back(violation_json);
    }
  }
  std::string violation_json_file_path = DRCUTIL.getString(temp_directory_path, "violation_map.json");
  std::ofstream* violation_json_file = DRCUTIL.getOutputFileStream(violation_json_file_path);
  (*violation_json_file) << violation_json_list;
  DRCUTIL.closeFileStream(violation_json_file);
  sendNotification(DRCUTIL.getString("DRC_violation_map"), violation_json_file_path);
}

void DRCInterface::outputSummary(std::map<std::string, std::vector<ids::Violation>>& type_violation_map)
{
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();

  featureInst->get_type_layer_violation_map().clear();
  for (auto& [type, violation_list] : type_violation_map) {
    for (ids::Violation& violation : violation_list) {
      std::string layer_name;
      if (violation.is_routing) {
        layer_name = routing_layer_list[violation.layer_idx].get_layer_name();
      } else {
        layer_name = cut_layer_list[violation.layer_idx].get_layer_name();
      }
      featureInst->get_type_layer_violation_map()[type][layer_name].push_back(violation);
    }
  }
}

DRCShape DRCInterface::convertToDRCShape(const ids::Shape& ids_shape)
{
  DRCShape drc_shape;
  drc_shape.set_net_idx(ids_shape.net_idx);
  drc_shape.set_ll(ids_shape.ll_x, ids_shape.ll_y);
  drc_shape.set_ur(ids_shape.ur_x, ids_shape.ur_y);
  drc_shape.set_layer_idx(ids_shape.layer_idx);
  drc_shape.set_is_routing(ids_shape.is_routing);
  return drc_shape;
}

#endif

#if 1  // ecos

void DRCInterface::sendNotification(std::string stage, std::string json_path)
{
}

#endif

#endif

// private

DRCInterface* DRCInterface::_drc_interface_instance = nullptr;

}  // namespace idrc

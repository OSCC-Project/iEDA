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
#include "RuleChecker.hpp"
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

void DRCInterface::initDRC(std::map<std::string, std::any> config_map)
{
  Logger::initInst();
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

  DRCLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void DRCInterface::checkDef()
{
  std::vector<ids::Shape> ids_env_shape_list;
  std::vector<ids::Shape> ids_result_shape_list;
  buildEnvShapeList(ids_env_shape_list);
  buildResultShapeList(ids_result_shape_list);
  getViolationList(ids_env_shape_list, ids_result_shape_list, false);
}

void DRCInterface::destroyDRC()
{
  Monitor monitor;
  DRCLOG.info(Loc::current(), "Starting...");

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

std::vector<ids::Violation> DRCInterface::getViolationList(std::vector<ids::Shape>& ids_env_shape_list, std::vector<ids::Shape>& ids_result_shape_list,
                                                           bool enable_quiet)
{
  if (enable_quiet) {
    DRCLOG.enableQuiet();
  }
  std::vector<DRCShape> drc_env_shape_list;
  drc_env_shape_list.reserve(ids_env_shape_list.size());
  for (ids::Shape& ids_env_shape : ids_env_shape_list) {
    drc_env_shape_list.push_back(convertToDRCShape(ids_env_shape));
  }
  std::vector<DRCShape> drc_result_shape_list;
  drc_result_shape_list.reserve(ids_result_shape_list.size());
  for (ids::Shape& ids_result_shape : ids_result_shape_list) {
    drc_result_shape_list.push_back(convertToDRCShape(ids_result_shape));
  }
  RuleChecker::initInst();
  std::vector<ids::Violation> ids_violation_list;
  for (Violation& violation : DRCRC.check(drc_env_shape_list, drc_result_shape_list)) {
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
  RuleChecker::destroyInst();
  if (enable_quiet) {
    DRCLOG.disableQuiet();
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
  DRCDM.getConfig().temp_directory_path = DRCUTIL.getConfigValue<std::string>(config_map, "-temp_directory_path", "./rt_temp_directory");
  DRCDM.getConfig().thread_number = DRCUTIL.getConfigValue<int32_t>(config_map, "-thread_number", 128);
  omp_set_num_threads(std::max(DRCDM.getConfig().thread_number, 1));
  /////////////////////////////////////////////
}

void DRCInterface::wrapDatabase()
{
  wrapManufactureGrid();
  wrapLayerList();
  wrapLayerInfo();
}

void DRCInterface::wrapManufactureGrid()
{
  DRCDM.getDatabase().set_manufacture_grid(dmInst->get_idb_lef_service()->get_layout()->get_munufacture_grid());
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
      routing_layer.set_min_width(idb_routing_layer->get_min_width());
      routing_layer.set_min_area(idb_routing_layer->get_area());
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
  // prl
  {
    std::shared_ptr<idb::IdbParallelSpacingTable> idb_spacing_table;
    if (idb_layer->get_spacing_table().get()->get_parallel().get() != nullptr && idb_layer->get_spacing_table().get()->is_parallel()) {
      idb_spacing_table = idb_layer->get_spacing_table()->get_parallel();
    } else if (idb_layer->get_spacing_list() != nullptr && !idb_layer->get_spacing_table().get()->is_parallel()) {
      idb_spacing_table = idb_layer->get_spacing_table_from_spacing_list()->get_parallel();
    } else {
      DRCLOG.warn(Loc::current(), "The idb layer ", idb_layer->get_name(), " spacing table is empty!");
      idb_spacing_table = std::make_shared<idb::IdbParallelSpacingTable>(1, 1);
      idb_spacing_table.get()->set_parallel_length(0, 0);
      idb_spacing_table.get()->set_width(0, 0);
      idb_spacing_table.get()->set_spacing(0, 0, 0);
    }

    SpacingTable& prl_spacing_table = routing_layer.get_prl_spacing_table();
    std::vector<int32_t>& width_list = prl_spacing_table.get_width_list();
    std::vector<int32_t>& parallel_length_list = prl_spacing_table.get_parallel_length_list();
    GridMap<int32_t>& width_parallel_length_map = prl_spacing_table.get_width_parallel_length_map();

    width_list = idb_spacing_table->get_width_list();
    parallel_length_list = idb_spacing_table->get_parallel_length_list();
    width_parallel_length_map.init(width_list.size(), parallel_length_list.size());
    for (int32_t x = 0; x < width_parallel_length_map.get_x_size(); x++) {
      for (int32_t y = 0; y < width_parallel_length_map.get_y_size(); y++) {
        width_parallel_length_map[x][y] = idb_spacing_table->get_spacing_table()[x][y];
      }
    }
  }
  // eol
  {
    if (!idb_layer->get_lef58_spacing_eol_list().empty()) {
      routinglayer::Lef58SpacingEol* idb_spacing_eol = idb_layer->get_lef58_spacing_eol_list().front().get();

      int32_t eol_spacing = idb_spacing_eol->get_eol_space();
      int32_t eol_ete = 0;
      if (idb_spacing_eol->get_end_to_end().has_value()) {
        eol_ete = idb_spacing_eol->get_end_to_end().value().get_end_to_end_space();
      }
      int32_t eol_within = idb_spacing_eol->get_eol_within().value();
      routing_layer.set_eol_spacing(eol_spacing);
      routing_layer.set_eol_ete(eol_ete);
      routing_layer.set_eol_within(eol_within);
    } else {
      DRCLOG.warn(Loc::current(), "The idb layer ", idb_layer->get_name(), " eol_spacing is empty!");
      routing_layer.set_eol_spacing(0);
      routing_layer.set_eol_ete(0);
      routing_layer.set_eol_within(0);
    }
  }
}

void DRCInterface::wrapCutDesignRule(CutLayer& cut_layer, idb::IdbLayerCut* idb_layer)
{
  // curr layer
  {
    // prl
    if (!idb_layer->get_spacings().empty()) {
      cut_layer.set_curr_spacing(idb_layer->get_spacings().front()->get_spacing());
      cut_layer.set_curr_prl(0);
      cut_layer.set_curr_prl_spacing(idb_layer->get_spacings().front()->get_spacing());
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
        int32_t curr_prl = -1 * spacing_table->get_prl().value().get_prl();
        int32_t curr_prl_spacing = cut_spacing.get_cut_spacing2().value();
        cut_layer.set_curr_spacing(curr_spacing);
        cut_layer.set_curr_prl(curr_prl);
        cut_layer.set_curr_prl_spacing(curr_prl_spacing);
      } else {
        DRCLOG.warn(Loc::current(), "The idb layer ", idb_layer->get_name(), " curr layer spacing is empty!");
        cut_layer.set_curr_spacing(0);
        cut_layer.set_curr_prl(0);
        cut_layer.set_curr_prl_spacing(0);
      }
    } else {
      DRCLOG.warn(Loc::current(), "The idb layer ", idb_layer->get_name(), " curr layer spacing is empty!");
      cut_layer.set_curr_spacing(0);
      cut_layer.set_curr_prl(0);
      cut_layer.set_curr_prl_spacing(0);
    }
    // eol
    if (idb_layer->get_lef58_eol_spacing().get() != nullptr) {
      idb::cutlayer::Lef58EolSpacing* idb_eol_spacing = idb_layer->get_lef58_eol_spacing().get();

      int32_t curr_eol_spacing = idb_eol_spacing->get_cut_spacing1();
      int32_t curr_eol_prl = -1 * idb_eol_spacing->get_prl();
      int32_t curr_eol_prl_spacing = idb_eol_spacing->get_cut_spacing2();
      cut_layer.set_curr_eol_spacing(curr_eol_spacing);
      cut_layer.set_curr_eol_prl(curr_eol_prl);
      cut_layer.set_curr_eol_prl_spacing(curr_eol_prl_spacing);
    } else {
      DRCLOG.warn(Loc::current(), "The idb layer ", idb_layer->get_name(), " eol_spacing is empty!");
      cut_layer.set_curr_eol_spacing(0);
      cut_layer.set_curr_eol_prl(0);
      cut_layer.set_curr_eol_prl_spacing(0);
    }
  }
  // below layer
  {
    // prl
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
        int32_t below_prl = -1 * spacing_table->get_prl().value().get_prl();
        int32_t below_prl_spacing = cut_spacing.get_cut_spacing2().value();
        cut_layer.set_below_spacing(below_spacing);
        cut_layer.set_below_prl(below_prl);
        cut_layer.set_below_prl_spacing(below_prl_spacing);
      } else {
        cut_layer.set_below_spacing(0);
        cut_layer.set_below_prl(0);
        cut_layer.set_below_prl_spacing(0);
        DRCLOG.warn(Loc::current(), "The idb layer ", idb_layer->get_name(), " below layer spacing is empty!");
      }
    } else {
      cut_layer.set_below_spacing(0);
      cut_layer.set_below_prl(0);
      cut_layer.set_below_prl_spacing(0);
      DRCLOG.warn(Loc::current(), "The idb layer ", idb_layer->get_name(), " below layer spacing is empty!");
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

#endif

#if 1  // output

void DRCInterface::output()
{
}

#endif

#endif

#if 1  // form def

void DRCInterface::buildEnvShapeList(std::vector<ids::Shape>& env_shape_list)
{
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
          net_idx = idb_pin->get_net()->get_id();
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
        net_idx = idb_io_pin->get_net()->get_id();
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

void DRCInterface::buildResultShapeList(std::vector<ids::Shape>& result_shape_list)
{
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
          ids_shape.net_idx = idb_net->get_id();
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
                ids_shape.net_idx = idb_net->get_id();
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
              ids_shape.net_idx = idb_net->get_id();
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
          ids_shape.net_idx = idb_net->get_id();
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
}

#endif

#if 1  // form tool

DRCShape DRCInterface::convertToDRCShape(ids::Shape& ids_shape)
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

#endif

// private

DRCInterface* DRCInterface::_drc_interface_instance = nullptr;

}  // namespace idrc

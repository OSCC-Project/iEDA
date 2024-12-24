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
#include "RTInterface.hpp"

#include "DRCEngine.hpp"
#include "DetailedRouter.hpp"
#include "EarlyRouter.hpp"
#include "GDSPlotter.hpp"
#include "GlobalRouter.hpp"
#include "LSAssigner4iEDA/ls_assigner/LSAssigner.h"
#include "LayerAssigner.hpp"
#include "Monitor.hpp"
#include "PinAccessor.hpp"
#include "RTInterface.hpp"
#include "SupplyAnalyzer.hpp"
#include "TopologyGenerator.hpp"
#include "TrackAssigner.hpp"
#include "ViolationRepairer.hpp"
#include "api/PowerEngine.hh"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "feature_irt.h"
#include "feature_manager.h"
#include "flute3/flute.h"
#include "idm.h"
#include "idrc_api.h"
#include "tool_api/ista_io/ista_io.h"

namespace irt {

// public

RTInterface& RTInterface::getInst()
{
  if (_rt_interface_instance == nullptr) {
    _rt_interface_instance = new RTInterface();
  }
  return *_rt_interface_instance;
}

void RTInterface::destroyInst()
{
  if (_rt_interface_instance != nullptr) {
    delete _rt_interface_instance;
    _rt_interface_instance = nullptr;
  }
}

#if 1  // 外部调用RT的API

#if 1  // iRT

void RTInterface::initRT(std::map<std::string, std::any> config_map)
{
  Logger::initInst();
  // clang-format off
  RTLOG.info(Loc::current(), ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
  RTLOG.info(Loc::current(), "_____ ________ ________     _______________________ ________ ________  ");
  RTLOG.info(Loc::current(), "___(_)___  __ \\___  __/     __  ___/___  __/___    |___  __ \\___  __/");
  RTLOG.info(Loc::current(), "__  / __  /_/ /__  /        _____ \\ __  /   __  /| |__  /_/ /__  /    ");
  RTLOG.info(Loc::current(), "_  /  _  _, _/ _  /         ____/ / _  /    _  ___ |_  _, _/ _  /      ");
  RTLOG.info(Loc::current(), "/_/   /_/ |_|  /_/          /____/  /_/     /_/  |_|/_/ |_|  /_/       ");
  RTLOG.info(Loc::current(), ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
  // clang-format on
  RTLOG.printLogFilePath();
  //////////////////////////////////////////////////////
  //////////////////////////////////////////////////////
  //////////////////////////////////////////////////////
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  DataManager::initInst();
  RTDM.input(config_map);
  DRCEngine::initInst();
  GDSPlotter::initInst();
  initFlute();

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void RTInterface::runEGR()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  RTGP.init();

  SupplyAnalyzer::initInst();
  RTSA.analyze();
  SupplyAnalyzer::destroyInst();

  EarlyRouter::initInst();
  RTER.route();
  EarlyRouter::destroyInst();

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void RTInterface::runRT()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  RTGP.init();
  RTDE.init();

  PinAccessor::initInst();
  RTPA.access();
  PinAccessor::destroyInst();

  SupplyAnalyzer::initInst();
  RTSA.analyze();
  SupplyAnalyzer::destroyInst();

  TopologyGenerator::initInst();
  RTTG.generate();
  TopologyGenerator::destroyInst();

  LayerAssigner::initInst();
  RTLA.route();
  LayerAssigner::destroyInst();

  GlobalRouter::initInst();
  RTGR.route();
  GlobalRouter::destroyInst();

  TrackAssigner::initInst();
  RTTA.assign();
  TrackAssigner::destroyInst();

  DetailedRouter::initInst();
  RTDR.route();
  DetailedRouter::destroyInst();

  ViolationRepairer::initInst();
  RTVR.repair();
  ViolationRepairer::destroyInst();

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void RTInterface::destroyRT()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  destroyFlute();
  GDSPlotter::destroyInst();
  DRCEngine::destroyInst();
  RTDM.output();
  DataManager::destroyInst();

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());

  RTLOG.printLogFilePath();
  // clang-format off
  RTLOG.info(Loc::current(), ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
  RTLOG.info(Loc::current(), "_____ ________ ________     _______________________   ________________________  __  ");
  RTLOG.info(Loc::current(), "___(_)___  __ \\___  __/     ___  ____/____  _/___  | / /____  _/__  ___/___  / / / ");
  RTLOG.info(Loc::current(), "__  / __  /_/ /__  /        __  /_     __  /  __   |/ /  __  /  _____ \\ __  /_/ /  ");
  RTLOG.info(Loc::current(), "_  /  _  _, _/ _  /         _  __/    __/ /   _  /|  /  __/ /   ____/ / _  __  /    ");
  RTLOG.info(Loc::current(), "/_/   /_/ |_|  /_/          /_/       /___/   /_/ |_/   /___/   /____/  /_/ /_/     ");
  RTLOG.info(Loc::current(), ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
  // clang-format on
  Logger::destroyInst();
}

void RTInterface::clearDef()
{
  idb::IdbPins* idb_pin_list = dmInst->get_idb_def_service()->get_design()->get_io_pin_list();
  IdbNetList* idb_net_list = dmInst->get_idb_def_service()->get_design()->get_net_list();

  //////////////////////////////////////////
  // 删除net内所有的wire
  for (idb::IdbNet* idb_net : idb_net_list->get_net_list()) {
    idb_net->clear_wire_list();
  }
  // 删除net内所有的wire
  //////////////////////////////////////////

  //////////////////////////////////////////
  // 删除net内所有的virtual
  for (idb::IdbNet* idb_net : idb_net_list->get_net_list()) {
    for (idb::IdbRegularWire* wire : idb_net->get_wire_list()->get_wire_list()) {
      std::vector<idb::IdbRegularWireSegment*> del_segment_list;
      for (idb::IdbRegularWireSegment* segment : wire->get_segment_list()) {
        if (segment->is_virtual(segment->get_point_second())) {
          del_segment_list.push_back(segment);
        }
      }
      for (idb::IdbRegularWireSegment* segment : del_segment_list) {
        wire->delete_seg(segment);
      }
    }
  }
  // 删除net内所有的virtual
  //////////////////////////////////////////

  //////////////////////////////////////////
  // 删除net内所有的patch
  for (idb::IdbNet* idb_net : idb_net_list->get_net_list()) {
    for (idb::IdbRegularWire* wire : idb_net->get_wire_list()->get_wire_list()) {
      std::vector<idb::IdbRegularWireSegment*> del_segment_list;
      for (idb::IdbRegularWireSegment* segment : wire->get_segment_list()) {
        if (segment->is_rect()) {
          del_segment_list.push_back(segment);
        }
      }
      for (idb::IdbRegularWireSegment* segment : del_segment_list) {
        wire->delete_seg(segment);
      }
    }
  }
  // 删除net内所有的patch
  //////////////////////////////////////////

  //////////////////////////////////////////
  // 删除net: 虚拟的io_pin与io_cell连接的PAD
  std::vector<std::string> remove_net_list;
  for (idb::IdbNet* idb_net : idb_net_list->get_net_list()) {
    bool has_io_pin = false;
    if (idb_net->get_io_pins() != nullptr) {
      has_io_pin = true;
    }
    bool has_io_cell = false;
    for (idb::IdbInstance* instance : idb_net->get_instance_list()->get_instance_list()) {
      if (instance->get_cell_master()->is_pad()) {
        has_io_cell = true;
        break;
      }
    }
    if (has_io_pin && has_io_cell) {
      RTLOG.info(Loc::current(), "The net '", idb_net->get_net_name(), "' connects PAD and io_pin! removing...");
      remove_net_list.push_back(idb_net->get_net_name());
    }
  }
  for (std::string remove_net : remove_net_list) {
    idb_net_list->remove_net(remove_net);
  }
  // 删除net: 虚拟的io_pin与io_cell连接的PAD
  //////////////////////////////////////////

  //////////////////////////////////////////
  // 删除虚空的io_pin
  std::vector<idb::IdbPin*> remove_pin_list;
  for (idb::IdbPin* io_pin : idb_pin_list->get_pin_list()) {
    if (io_pin->get_port_box_list().empty()) {
      RTLOG.info(Loc::current(), "del io_pin: ", io_pin->get_pin_name());
      remove_pin_list.push_back(io_pin);
    }
  }
  for (idb::IdbPin* io_pin : remove_pin_list) {
    idb_pin_list->remove_pin(io_pin);
  }
  // 删除虚空的io_pin
  //////////////////////////////////////////
}

#endif

#endif

#if 1  // RT调用外部的API

#if 1  // TopData

#if 1  // input

void RTInterface::input(std::map<std::string, std::any>& config_map)
{
  wrapConfig(config_map);
  wrapDatabase();
}

void RTInterface::wrapConfig(std::map<std::string, std::any>& config_map)
{
  /////////////////////////////////////////////
  RTDM.getConfig().temp_directory_path = RTUTIL.getConfigValue<std::string>(config_map, "-temp_directory_path", "./rt_temp_directory");
  RTDM.getConfig().thread_number = RTUTIL.getConfigValue<int32_t>(config_map, "-thread_number", 128);
  omp_set_num_threads(std::max(RTDM.getConfig().thread_number, 1));
  RTDM.getConfig().bottom_routing_layer = RTUTIL.getConfigValue<std::string>(config_map, "-bottom_routing_layer", "");
  RTDM.getConfig().top_routing_layer = RTUTIL.getConfigValue<std::string>(config_map, "-top_routing_layer", "");
  RTDM.getConfig().output_inter_result = RTUTIL.getConfigValue<int32_t>(config_map, "-output_inter_result", 0);
  RTDM.getConfig().enable_timing = RTUTIL.getConfigValue<int32_t>(config_map, "-enable_timing", 0);
  RTDM.getConfig().enable_lsa = RTUTIL.getConfigValue<int32_t>(config_map, "-enable_lsa", 0);
  /////////////////////////////////////////////
}

void RTInterface::wrapDatabase()
{
  wrapDBInfo();
  wrapMicronDBU();
  wrapManufactureGrid();
  wrapDie();
  wrapRow();
  wrapLayerList();
  wrapLayerInfo();
  wrapLayerViaMasterList();
  wrapObstacleList();
  wrapNetList();
}

void RTInterface::wrapDBInfo()
{
  RTDM.getDatabase().set_design_name(dmInst->get_idb_def_service()->get_design()->get_design_name());
  RTDM.getDatabase().set_lef_file_path_list(dmInst->get_idb_lef_service()->get_lef_files());
  RTDM.getDatabase().set_def_file_path(dmInst->get_idb_def_service()->get_def_file());
}

void RTInterface::wrapMicronDBU()
{
  RTDM.getDatabase().set_micron_dbu(dmInst->get_idb_def_service()->get_design()->get_units()->get_micron_dbu());
}

void RTInterface::wrapManufactureGrid()
{
  RTDM.getDatabase().set_manufacture_grid(dmInst->get_idb_lef_service()->get_layout()->get_munufacture_grid());
}

void RTInterface::wrapDie()
{
  idb::IdbDie* die = dmInst->get_idb_lef_service()->get_layout()->get_die();

  EXTPlanarRect& die_box = RTDM.getDatabase().get_die();
  die_box.set_real_ll(die->get_llx(), die->get_lly());
  die_box.set_real_ur(die->get_urx(), die->get_ury());
}

void RTInterface::wrapRow()
{
  int32_t start_x = INT32_MAX;
  int32_t start_y = INT32_MAX;
  for (idb::IdbRow* idb_row : dmInst->get_idb_def_service()->get_layout()->get_rows()->get_row_list()) {
    start_x = std::min(start_x, idb_row->get_original_coordinate()->get_x());
    start_y = std::min(start_y, idb_row->get_original_coordinate()->get_y());
  }
  Row& row = RTDM.getDatabase().get_row();
  row.set_start_x(start_x);
  row.set_start_y(start_y);
  row.set_height(dmInst->get_idb_def_service()->get_layout()->get_rows()->get_row_height());
}

void RTInterface::wrapLayerList()
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();

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
      routing_layer.set_prefer_direction(getRTDirectionByDB(idb_routing_layer->get_direction()));
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

void RTInterface::wrapTrackAxis(RoutingLayer& routing_layer, idb::IdbLayerRouting* idb_layer)
{
  ScaleAxis& track_axis = routing_layer.get_track_axis();

  for (idb::IdbTrackGrid* idb_track_grid : idb_layer->get_track_grid_list()) {
    idb::IdbTrack* idb_track = idb_track_grid->get_track();

    ScaleGrid track_grid;
    track_grid.set_start_line(static_cast<int32_t>(idb_track->get_start()));
    track_grid.set_step_length(static_cast<int32_t>(idb_track->get_pitch()));
    track_grid.set_step_num(static_cast<int32_t>(idb_track_grid->get_track_num()) - 1);

    if (idb_track->get_direction() == idb::IdbTrackDirection::kDirectionX) {
      track_axis.get_x_grid_list().push_back(track_grid);
    } else if (idb_track->get_direction() == idb::IdbTrackDirection::kDirectionY) {
      track_axis.get_y_grid_list().push_back(track_grid);
    }
  }
}

void RTInterface::wrapRoutingDesignRule(RoutingLayer& routing_layer, idb::IdbLayerRouting* idb_layer)
{
  // prl
  {
    std::shared_ptr<idb::IdbParallelSpacingTable> idb_spacing_table;
    if (idb_layer->get_spacing_table().get()->get_parallel().get() != nullptr && idb_layer->get_spacing_table().get()->is_parallel()) {
      idb_spacing_table = idb_layer->get_spacing_table()->get_parallel();
    } else if (idb_layer->get_spacing_list() != nullptr && !idb_layer->get_spacing_table().get()->is_parallel()) {
      idb_spacing_table = idb_layer->get_spacing_table_from_spacing_list()->get_parallel();
    } else {
      RTLOG.warn(Loc::current(), "The idb layer ", idb_layer->get_name(), " spacing table is empty!");
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
      RTLOG.warn(Loc::current(), "The idb layer ", idb_layer->get_name(), " eol_spacing is empty!");
      routing_layer.set_eol_spacing(0);
      routing_layer.set_eol_ete(0);
      routing_layer.set_eol_within(0);
    }
  }
}

void RTInterface::wrapCutDesignRule(CutLayer& cut_layer, idb::IdbLayerCut* idb_layer)
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
        RTLOG.warn(Loc::current(), "The idb layer ", idb_layer->get_name(), " curr layer spacing is empty!");
        cut_layer.set_curr_spacing(0);
        cut_layer.set_curr_prl(0);
        cut_layer.set_curr_prl_spacing(0);
      }
    } else {
      RTLOG.warn(Loc::current(), "The idb layer ", idb_layer->get_name(), " curr layer spacing is empty!");
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
      RTLOG.warn(Loc::current(), "The idb layer ", idb_layer->get_name(), " eol_spacing is empty!");
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
        RTLOG.warn(Loc::current(), "The idb layer ", idb_layer->get_name(), " below layer spacing is empty!");
      }
    } else {
      cut_layer.set_below_spacing(0);
      cut_layer.set_below_prl(0);
      cut_layer.set_below_prl_spacing(0);
      RTLOG.warn(Loc::current(), "The idb layer ", idb_layer->get_name(), " below layer spacing is empty!");
    }
  }
}

void RTInterface::wrapLayerInfo()
{
  std::map<int32_t, int32_t>& routing_idb_layer_id_to_idx_map = RTDM.getDatabase().get_routing_idb_layer_id_to_idx_map();
  std::map<int32_t, int32_t>& cut_idb_layer_id_to_idx_map = RTDM.getDatabase().get_cut_idb_layer_id_to_idx_map();
  std::map<std::string, int32_t>& routing_layer_name_to_idx_map = RTDM.getDatabase().get_routing_layer_name_to_idx_map();
  std::map<std::string, int32_t>& cut_layer_name_to_idx_map = RTDM.getDatabase().get_cut_layer_name_to_idx_map();

  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  for (size_t i = 0; i < routing_layer_list.size(); i++) {
    routing_idb_layer_id_to_idx_map[routing_layer_list[i].get_layer_idx()] = static_cast<int32_t>(i);
    routing_layer_name_to_idx_map[routing_layer_list[i].get_layer_name()] = static_cast<int32_t>(i);
  }
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  for (size_t i = 0; i < cut_layer_list.size(); i++) {
    cut_idb_layer_id_to_idx_map[cut_layer_list[i].get_layer_idx()] = static_cast<int32_t>(i);
    cut_layer_name_to_idx_map[cut_layer_list[i].get_layer_name()] = static_cast<int32_t>(i);
  }
}

void RTInterface::wrapLayerViaMasterList()
{
  idb::IdbVias* idb_via_list_lib = dmInst->get_idb_lef_service()->get_layout()->get_via_list();
  if (idb_via_list_lib == nullptr) {
    RTLOG.error(Loc::current(), "Via list in tech lef is empty!");
  }

  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  std::vector<idb::IdbLayer*> idb_routing_layers = dmInst->get_idb_lef_service()->get_layout()->get_layers()->get_routing_layers();
  layer_via_master_list.resize(idb_routing_layers.size());

  std::vector<idb::IdbVia*>& idb_via_list = idb_via_list_lib->get_via_list();
  for (idb::IdbVia* idb_via : idb_via_list) {
    if (idb_via == nullptr) {
      RTLOG.error(Loc::current(), "The via is empty!");
    }
    if (!idb_via->get_instance()->is_default()) {
      continue;
    }
    ViaMaster via_master;
    via_master.set_via_name(idb_via->get_name());
    idb::IdbViaMaster* idb_via_master = idb_via->get_instance();
    // top enclosure
    idb::IdbLayerShape* idb_shape_top = idb_via_master->get_top_layer_shape();
    idb::IdbLayerRouting* idb_layer_top = dynamic_cast<idb::IdbLayerRouting*>(idb_shape_top->get_layer());
    idb::IdbRect idb_box_top = idb_shape_top->get_bounding_box();
    LayerRect above_enclosure(idb_box_top.get_low_x(), idb_box_top.get_low_y(), idb_box_top.get_high_x(), idb_box_top.get_high_y(),
                              idb_layer_top->get_id());
    via_master.set_above_enclosure(above_enclosure);
    via_master.set_above_direction(getRTDirectionByDB(idb_layer_top->get_direction()));
    // bottom enclosure
    idb::IdbLayerShape* idb_shape_bottom = idb_via_master->get_bottom_layer_shape();
    idb::IdbLayerRouting* idb_layer_bottom = dynamic_cast<idb::IdbLayerRouting*>(idb_shape_bottom->get_layer());
    idb::IdbRect idb_box_bottom = idb_shape_bottom->get_bounding_box();
    LayerRect below_enclosure(idb_box_bottom.get_low_x(), idb_box_bottom.get_low_y(), idb_box_bottom.get_high_x(),
                              idb_box_bottom.get_high_y(), idb_layer_bottom->get_id());
    via_master.set_below_enclosure(below_enclosure);
    via_master.set_below_direction(getRTDirectionByDB(idb_layer_bottom->get_direction()));
    // cut shape
    idb::IdbLayerShape idb_shape_cut = idb_via->get_cut_layer_shape();
    std::vector<PlanarRect>& cut_shape_list = via_master.get_cut_shape_list();
    for (idb::IdbRect* idb_rect : idb_shape_cut.get_rect_list()) {
      PlanarRect cut_shape;
      cut_shape.set_ll(idb_rect->get_low_x(), idb_rect->get_low_y());
      cut_shape.set_ur(idb_rect->get_high_x(), idb_rect->get_high_y());
      cut_shape_list.push_back(std::move(cut_shape));
    }
    via_master.set_cut_layer_idx(idb_shape_cut.get_layer()->get_id());
    layer_via_master_list.front().push_back(std::move(via_master));
  }
}

void RTInterface::wrapObstacleList()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<Obstacle>& routing_obstacle_list = RTDM.getDatabase().get_routing_obstacle_list();
  std::vector<Obstacle>& cut_obstacle_list = RTDM.getDatabase().get_cut_obstacle_list();
  std::vector<idb::IdbInstance*>& instance_list = dmInst->get_idb_def_service()->get_design()->get_instance_list()->get_instance_list();
  std::vector<idb::IdbSpecialNet*>& idb_special_net_list
      = dmInst->get_idb_def_service()->get_design()->get_special_net_list()->get_net_list();
  std::vector<idb::IdbPin*>& idb_io_pin_list = dmInst->get_idb_def_service()->get_design()->get_io_pin_list()->get_pin_list();

  int32_t total_routing_obstacle_num = 0;
  int32_t total_cut_obstacle_num = 0;
  {
    // instance
    for (idb::IdbInstance* instance : instance_list) {
      // instance obs
      for (idb::IdbLayerShape* obs_box : instance->get_obs_box_list()) {
        if (obs_box->get_layer()->is_routing()) {
          total_routing_obstacle_num += obs_box->get_rect_list().size();
        } else if (obs_box->get_layer()->is_cut()) {
          total_cut_obstacle_num += obs_box->get_rect_list().size();
        }
      }
      // instance pin without net
      for (idb::IdbPin* idb_pin : instance->get_pin_list()->get_pin_list()) {
        if (!isSkipping(idb_pin->get_net(), false)) {
          continue;
        }
        for (idb::IdbLayerShape* port_box : idb_pin->get_port_box_list()) {
          if (port_box->get_layer()->is_routing()) {
            total_routing_obstacle_num += port_box->get_rect_list().size();
          } else if (port_box->get_layer()->is_cut()) {
            total_cut_obstacle_num += port_box->get_rect_list().size();
          }
        }
        for (idb::IdbVia* idb_via : idb_pin->get_via_list()) {
          total_routing_obstacle_num += 2;
          total_cut_obstacle_num += idb_via->get_cut_layer_shape().get_rect_list().size();
        }
      }
    }
    // special net
    for (idb::IdbSpecialNet* idb_net : idb_special_net_list) {
      for (idb::IdbSpecialWire* idb_wire : idb_net->get_wire_list()->get_wire_list()) {
        for (idb::IdbSpecialWireSegment* idb_segment : idb_wire->get_segment_list()) {
          if (idb_segment->is_via()) {
            total_routing_obstacle_num += idb_segment->get_via()->get_top_layer_shape().get_rect_list().size();
            total_routing_obstacle_num += idb_segment->get_via()->get_bottom_layer_shape().get_rect_list().size();
            total_cut_obstacle_num += idb_segment->get_via()->get_cut_layer_shape().get_rect_list().size();
          } else {
            total_routing_obstacle_num += 1;
          }
        }
      }
    }
    // io pin
    for (idb::IdbPin* idb_io_pin : idb_io_pin_list) {
      if (!isSkipping(idb_io_pin->get_net(), false)) {
        continue;
      }
      for (idb::IdbLayerShape* port_box : idb_io_pin->get_port_box_list()) {
        if (port_box->get_layer()->is_routing()) {
          total_routing_obstacle_num += port_box->get_rect_list().size();
        } else if (port_box->get_layer()->is_cut()) {
          total_cut_obstacle_num += port_box->get_rect_list().size();
        }
      }
    }
  }
  routing_obstacle_list.reserve(total_routing_obstacle_num);
  cut_obstacle_list.reserve(total_cut_obstacle_num);
  {
    // instance
    for (idb::IdbInstance* instance : instance_list) {
      // instance obs
      for (idb::IdbLayerShape* obs_box : instance->get_obs_box_list()) {
        for (idb::IdbRect* rect : obs_box->get_rect_list()) {
          Obstacle obstacle;
          obstacle.set_real_ll(rect->get_low_x(), rect->get_low_y());
          obstacle.set_real_ur(rect->get_high_x(), rect->get_high_y());
          obstacle.set_layer_idx(obs_box->get_layer()->get_id());
          if (obs_box->get_layer()->is_routing()) {
            routing_obstacle_list.push_back(std::move(obstacle));
          } else if (obs_box->get_layer()->is_cut()) {
            cut_obstacle_list.push_back(std::move(obstacle));
          }
        }
      }
      // instance pin without net
      for (idb::IdbPin* idb_pin : instance->get_pin_list()->get_pin_list()) {
        if (!isSkipping(idb_pin->get_net(), false)) {
          continue;
        }
        for (idb::IdbLayerShape* port_box : idb_pin->get_port_box_list()) {
          for (idb::IdbRect* rect : port_box->get_rect_list()) {
            Obstacle obstacle;
            obstacle.set_real_ll(rect->get_low_x(), rect->get_low_y());
            obstacle.set_real_ur(rect->get_high_x(), rect->get_high_y());
            obstacle.set_layer_idx(port_box->get_layer()->get_id());
            if (port_box->get_layer()->is_routing()) {
              routing_obstacle_list.push_back(std::move(obstacle));
            } else if (port_box->get_layer()->is_cut()) {
              cut_obstacle_list.push_back(std::move(obstacle));
            }
          }
        }
        for (idb::IdbVia* idb_via : idb_pin->get_via_list()) {
          {
            idb::IdbLayerShape idb_shape_top = idb_via->get_top_layer_shape();
            idb::IdbRect idb_box_top = idb_shape_top.get_bounding_box();
            Obstacle obstacle;
            obstacle.set_real_ll(idb_box_top.get_low_x(), idb_box_top.get_low_y());
            obstacle.set_real_ur(idb_box_top.get_high_x(), idb_box_top.get_high_y());
            obstacle.set_layer_idx(idb_shape_top.get_layer()->get_id());
            routing_obstacle_list.push_back(std::move(obstacle));
          }
          {
            idb::IdbLayerShape idb_shape_bottom = idb_via->get_bottom_layer_shape();
            idb::IdbRect idb_box_bottom = idb_shape_bottom.get_bounding_box();
            Obstacle obstacle;
            obstacle.set_real_ll(idb_box_bottom.get_low_x(), idb_box_bottom.get_low_y());
            obstacle.set_real_ur(idb_box_bottom.get_high_x(), idb_box_bottom.get_high_y());
            obstacle.set_layer_idx(idb_shape_bottom.get_layer()->get_id());
            routing_obstacle_list.push_back(std::move(obstacle));
          }
          idb::IdbLayerShape idb_shape_cut = idb_via->get_cut_layer_shape();
          for (idb::IdbRect* idb_rect : idb_shape_cut.get_rect_list()) {
            Obstacle obstacle;
            obstacle.set_real_ll(idb_rect->get_low_x(), idb_rect->get_low_y());
            obstacle.set_real_ur(idb_rect->get_high_x(), idb_rect->get_high_y());
            obstacle.set_layer_idx(idb_shape_cut.get_layer()->get_id());
            cut_obstacle_list.push_back(std::move(obstacle));
          }
        }
      }
    }
    // special net
    for (idb::IdbSpecialNet* idb_net : idb_special_net_list) {
      for (idb::IdbSpecialWire* idb_wire : idb_net->get_wire_list()->get_wire_list()) {
        for (idb::IdbSpecialWireSegment* idb_segment : idb_wire->get_segment_list()) {
          if (idb_segment->is_via()) {
            for (idb::IdbLayerShape layer_shape :
                 {idb_segment->get_via()->get_top_layer_shape(), idb_segment->get_via()->get_bottom_layer_shape()}) {
              for (idb::IdbRect* rect : layer_shape.get_rect_list()) {
                Obstacle obstacle;
                obstacle.set_real_ll(rect->get_low_x(), rect->get_low_y());
                obstacle.set_real_ur(rect->get_high_x(), rect->get_high_y());
                obstacle.set_layer_idx(layer_shape.get_layer()->get_id());
                routing_obstacle_list.push_back(std::move(obstacle));
              }
            }
            idb::IdbLayerShape cut_layer_shape = idb_segment->get_via()->get_cut_layer_shape();
            for (idb::IdbRect* rect : cut_layer_shape.get_rect_list()) {
              Obstacle obstacle;
              obstacle.set_real_ll(rect->get_low_x(), rect->get_low_y());
              obstacle.set_real_ur(rect->get_high_x(), rect->get_high_y());
              obstacle.set_layer_idx(cut_layer_shape.get_layer()->get_id());
              cut_obstacle_list.push_back(std::move(obstacle));
            }
          } else {
            idb::IdbRect* idb_rect = idb_segment->get_bounding_box();
            // wire
            Obstacle obstacle;
            obstacle.set_real_ll(idb_rect->get_low_x(), idb_rect->get_low_y());
            obstacle.set_real_ur(idb_rect->get_high_x(), idb_rect->get_high_y());
            obstacle.set_layer_idx(idb_segment->get_layer()->get_id());
            routing_obstacle_list.push_back(std::move(obstacle));
          }
        }
      }
    }
    // io pin
    for (idb::IdbPin* idb_io_pin : idb_io_pin_list) {
      if (!isSkipping(idb_io_pin->get_net(), false)) {
        continue;
      }
      for (idb::IdbLayerShape* port_box : idb_io_pin->get_port_box_list()) {
        for (idb::IdbRect* rect : port_box->get_rect_list()) {
          Obstacle obstacle;
          obstacle.set_real_ll(rect->get_low_x(), rect->get_low_y());
          obstacle.set_real_ur(rect->get_high_x(), rect->get_high_y());
          obstacle.set_layer_idx(port_box->get_layer()->get_id());
          if (port_box->get_layer()->is_routing()) {
            routing_obstacle_list.push_back(std::move(obstacle));
          } else if (port_box->get_layer()->is_cut()) {
            cut_obstacle_list.push_back(std::move(obstacle));
          }
        }
      }
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void RTInterface::wrapNetList()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();
  std::vector<idb::IdbNet*>& idb_net_list = dmInst->get_idb_def_service()->get_design()->get_net_list()->get_net_list();

  std::vector<idb::IdbNet*> valid_idb_net_list;
  {
    valid_idb_net_list.reserve(idb_net_list.size());
    for (idb::IdbNet* idb_net : idb_net_list) {
      if (isSkipping(idb_net, true)) {
        continue;
      }
      valid_idb_net_list.push_back(idb_net);
    }
  }
  net_list.resize(valid_idb_net_list.size());
#pragma omp parallel for
  for (size_t i = 0; i < valid_idb_net_list.size(); i++) {
    idb::IdbNet* valid_idb_net = valid_idb_net_list[i];
    Net& net = net_list[i];
    net.set_net_name(valid_idb_net->get_net_name());
    net.set_connect_type(getRTConnectTypeByDB(valid_idb_net->get_connect_type()));
    wrapPinList(net, valid_idb_net);
    wrapDrivenPin(net, valid_idb_net);
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

bool RTInterface::isSkipping(idb::IdbNet* idb_net, bool with_log)
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
  } else if (pin_num >= 500) {
    if (with_log) {
      RTLOG.warn(Loc::current(), "The ultra large net: ", idb_net->get_net_name(), " has ", pin_num, " pins!");
    }
  }
  return false;
}

void RTInterface::wrapPinList(Net& net, idb::IdbNet* idb_net)
{
  std::vector<Pin>& pin_list = net.get_pin_list();

  for (idb::IdbPin* idb_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
    if (idb_pin->get_term()->get_port_number() <= 0) {
      continue;
    }
    Pin pin;
    pin.set_pin_name(RTUTIL.getString(idb_pin->get_instance()->get_name(), ":", idb_pin->get_pin_name()));
    wrapPinShapeList(pin, idb_pin);
    pin_list.push_back(std::move(pin));
  }
  for (idb::IdbPin* idb_pin : idb_net->get_io_pins()->get_pin_list()) {
    if (idb_pin->get_term()->get_port_number() <= 0) {
      continue;
    }
    Pin pin;
    pin.set_pin_name(idb_pin->get_pin_name());
    wrapPinShapeList(pin, idb_pin);
    pin_list.push_back(std::move(pin));
  }
}

void RTInterface::wrapPinShapeList(Pin& pin, idb::IdbPin* idb_pin)
{
  std::vector<EXTLayerRect>& routing_shape_list = pin.get_routing_shape_list();
  std::vector<EXTLayerRect>& cut_shape_list = pin.get_cut_shape_list();

  for (idb::IdbLayerShape* layer_shape : idb_pin->get_port_box_list()) {
    for (idb::IdbRect* rect : layer_shape->get_rect_list()) {
      EXTLayerRect pin_shape;
      pin_shape.set_real_ll(rect->get_low_x(), rect->get_low_y());
      pin_shape.set_real_ur(rect->get_high_x(), rect->get_high_y());
      pin_shape.set_layer_idx(layer_shape->get_layer()->get_id());
      if (layer_shape->get_layer()->is_routing()) {
        routing_shape_list.push_back(std::move(pin_shape));
      } else if (layer_shape->get_layer()->is_cut()) {
        cut_shape_list.push_back(std::move(pin_shape));
      }
    }
  }
  for (idb::IdbVia* idb_via : idb_pin->get_via_list()) {
    {
      idb::IdbLayerShape idb_shape_top = idb_via->get_top_layer_shape();
      idb::IdbRect idb_box_top = idb_shape_top.get_bounding_box();
      EXTLayerRect pin_shape;
      pin_shape.set_real_ll(idb_box_top.get_low_x(), idb_box_top.get_low_y());
      pin_shape.set_real_ur(idb_box_top.get_high_x(), idb_box_top.get_high_y());
      pin_shape.set_layer_idx(idb_shape_top.get_layer()->get_id());
      routing_shape_list.push_back(std::move(pin_shape));
    }
    {
      idb::IdbLayerShape idb_shape_bottom = idb_via->get_bottom_layer_shape();
      idb::IdbRect idb_box_bottom = idb_shape_bottom.get_bounding_box();
      EXTLayerRect pin_shape;
      pin_shape.set_real_ll(idb_box_bottom.get_low_x(), idb_box_bottom.get_low_y());
      pin_shape.set_real_ur(idb_box_bottom.get_high_x(), idb_box_bottom.get_high_y());
      pin_shape.set_layer_idx(idb_shape_bottom.get_layer()->get_id());
      routing_shape_list.push_back(std::move(pin_shape));
    }
    idb::IdbLayerShape idb_shape_cut = idb_via->get_cut_layer_shape();
    for (idb::IdbRect* idb_rect : idb_shape_cut.get_rect_list()) {
      EXTLayerRect pin_shape;
      pin_shape.set_real_ll(idb_rect->get_low_x(), idb_rect->get_low_y());
      pin_shape.set_real_ur(idb_rect->get_high_x(), idb_rect->get_high_y());
      pin_shape.set_layer_idx(idb_shape_cut.get_layer()->get_id());
      cut_shape_list.push_back(std::move(pin_shape));
    }
  }
}

void RTInterface::wrapDrivenPin(Net& net, idb::IdbNet* idb_net)
{
  idb::IdbPin* idb_driving_pin = idb_net->get_driving_pin();
  if (idb_driving_pin == nullptr) {
    return;
  }
  std::string driven_pin_name = idb_driving_pin->get_pin_name();
  if (!idb_driving_pin->is_io_pin()) {
    driven_pin_name = RTUTIL.getString(idb_driving_pin->get_instance()->get_name(), ":", driven_pin_name);
  }
  for (Pin& pin : net.get_pin_list()) {
    if (pin.get_pin_name() == driven_pin_name) {
      pin.set_is_driven(true);
    }
  }
}

Direction RTInterface::getRTDirectionByDB(idb::IdbLayerDirection idb_direction)
{
  if (idb_direction == idb::IdbLayerDirection::kHorizontal) {
    return Direction::kHorizontal;
  } else if (idb_direction == idb::IdbLayerDirection::kVertical) {
    return Direction::kVertical;
  } else {
    return Direction::kOblique;
  }
}

ConnectType RTInterface::getRTConnectTypeByDB(idb::IdbConnectType idb_connect_type)
{
  ConnectType connect_type;
  switch (idb_connect_type) {
    case idb::IdbConnectType::kClock:
      connect_type = ConnectType::kClock;
      break;
    default:
      connect_type = ConnectType::kSignal;
      break;
  }
  return connect_type;
}

#endif

#if 1  // output

void RTInterface::output()
{
  outputTrackGrid();
  outputGCellGrid();
  outputNetList();
  outputSummary();
}

void RTInterface::outputTrackGrid()
{
  idb::IdbLayers* idb_layer_list = dmInst->get_idb_def_service()->get_layout()->get_layers();
  idb::IdbTrackGridList* idb_track_grid_list = dmInst->get_idb_def_service()->get_layout()->get_track_grid_list();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  idb_track_grid_list->reset();

  for (int32_t i = static_cast<int32_t>(routing_layer_list.size()) - 1; i >= 0; --i) {
    RoutingLayer& routing_layer = routing_layer_list[i];

    std::string layer_name = routing_layer.get_layer_name();
    idb::IdbLayer* idb_layer = idb_layer_list->find_layer(layer_name);
    if (idb_layer == nullptr) {
      RTLOG.error(Loc::current(), "Can not find idb layer ", layer_name);
    }
    idb::IdbLayerRouting* idb_routing_layer = dynamic_cast<idb::IdbLayerRouting*>(idb_layer);
    idb_routing_layer->get_track_grid_list().clear();

    std::map<Direction, std::vector<ScaleGrid>> direction_scale_grid_list_map;
    for (ScaleGrid& x_grid : routing_layer.get_track_axis().get_x_grid_list()) {
      direction_scale_grid_list_map[Direction::kVertical].push_back(x_grid);
    }
    for (ScaleGrid& y_grid : routing_layer.get_track_axis().get_y_grid_list()) {
      direction_scale_grid_list_map[Direction::kHorizontal].push_back(y_grid);
    }
    for (auto& [direction, scale_grid_list] : direction_scale_grid_list_map) {
      for (ScaleGrid& scale_grid : scale_grid_list) {
        idb::IdbTrackGrid* idb_track_grid = idb_track_grid_list->add_track_grid();
        idb::IdbTrack* idb_track = idb_track_grid->get_track();
        if (direction == Direction::kVertical) {
          idb_track->set_direction(idb::IdbTrackDirection::kDirectionX);
        } else if (direction == Direction::kHorizontal) {
          idb_track->set_direction(idb::IdbTrackDirection::kDirectionY);
        }
        idb_track->set_start(scale_grid.get_start_line());
        idb_track->set_pitch(scale_grid.get_step_length());
        idb_track_grid->set_track_number(scale_grid.get_step_num() + 1);
        idb_track_grid->add_layer_list(idb_layer);
        idb_routing_layer->add_track_grid(idb_track_grid);
      }
    }
  }
}

void RTInterface::outputGCellGrid()
{
  idb::IdbGCellGridList* idb_gcell_grid_list = dmInst->get_idb_lef_service()->get_layout()->get_gcell_grid_list();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();

  idb_gcell_grid_list->clear();

  for (idb::IdbTrackDirection idb_track_direction : {idb::IdbTrackDirection::kDirectionX, idb::IdbTrackDirection::kDirectionY}) {
    std::vector<ScaleGrid> gcell_grid_list;
    if (idb_track_direction == idb::IdbTrackDirection::kDirectionX) {
      gcell_grid_list = gcell_axis.get_x_grid_list();
    } else {
      gcell_grid_list = gcell_axis.get_y_grid_list();
    }
    for (ScaleGrid& gcell_grid : gcell_grid_list) {
      idb::IdbGCellGrid* idb_gcell_grid = new idb::IdbGCellGrid();
      idb_gcell_grid->set_start(gcell_grid.get_start_line());
      idb_gcell_grid->set_space(gcell_grid.get_step_length());
      idb_gcell_grid->set_num(gcell_grid.get_step_num() + 1);
      idb_gcell_grid->set_direction(idb_track_direction);
      idb_gcell_grid_list->add_gcell_grid(idb_gcell_grid);
    }
  }
}

void RTInterface::outputNetList()
{
  Die& die = RTDM.getDatabase().get_die();
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();

  std::map<int32_t, std::vector<idb::IdbRegularWireSegment*>> net_idb_segment_map;
  for (auto& [net_idx, segment_set] : RTDM.getNetFinalResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      net_idb_segment_map[net_idx].push_back(getIDBSegmentByNetResult(net_idx, *segment));
    }
  }
  for (auto& [net_idx, patch_set] : RTDM.getNetFinalPatchMap(die)) {
    for (EXTLayerRect* patch : patch_set) {
      net_idb_segment_map[net_idx].push_back(getIDBSegmentByNetPatch(net_idx, *patch));
    }
  }
  idb::IdbNetList* idb_net_list = dmInst->get_idb_def_service()->get_design()->get_net_list();
  if (idb_net_list == nullptr) {
    RTLOG.error(Loc::current(), "The idb net list is empty!");
  }
  for (idb::IdbNet* idb_net : idb_net_list->get_net_list()) {
    idb_net->clear_wire_list();
  }
  for (auto& [net_idx, idb_segment_list] : net_idb_segment_map) {
    std::string net_name = net_list[net_idx].get_net_name();
    idb::IdbNet* idb_net = idb_net_list->find_net(net_name);
    if (idb_net == nullptr) {
      RTLOG.info(Loc::current(), "The idb net named ", net_name, " cannot be found!");
      continue;
    }
    idb_net->clear_wire_list();
    idb::IdbRegularWireList* idb_wire_list = idb_net->get_wire_list();
    if (idb_wire_list == nullptr) {
      RTLOG.error(Loc::current(), "The idb wire list is empty!");
    }
    idb::IdbRegularWire* idb_wire = idb_wire_list->add_wire();
    idb_wire->set_wire_state(idb::IdbWiringStatement::kRouted);

    int32_t print_new = false;
    for (idb::IdbRegularWireSegment* idb_segment : idb_segment_list) {
      idb_wire->add_segment(idb_segment);
      if (print_new == false) {
        idb_segment->set_layer_as_new();
        print_new = true;
      }
    }
  }
}

void RTInterface::outputSummary()
{
  ieda_feature::RTSummary& top_rt_summary = featureInst->get_summary()->get_summary_irt();

  Summary& rt_summary = RTDM.getDatabase().get_summary();

  // pa_summary
  {
    top_rt_summary.pa_summary.routing_access_point_num_map = rt_summary.pa_summary.routing_access_point_num_map;
    top_rt_summary.pa_summary.total_access_point_num = rt_summary.pa_summary.total_access_point_num;
  }
  // sa_summary
  {
    top_rt_summary.sa_summary.routing_supply_map = rt_summary.sa_summary.routing_supply_map;
    top_rt_summary.sa_summary.total_supply = rt_summary.sa_summary.total_supply;
  }
  // tg_summary
  {
    top_rt_summary.tg_summary.total_demand = rt_summary.tg_summary.total_demand;
    top_rt_summary.tg_summary.total_overflow = rt_summary.tg_summary.total_overflow;
    top_rt_summary.tg_summary.total_wire_length = rt_summary.tg_summary.total_wire_length;
    for (auto& [clock_name, timing_map] : rt_summary.tg_summary.clock_timing) {
      ieda_feature::ClockTiming clock_timing;
      clock_timing.clock_name = clock_name;
      clock_timing.setup_tns = timing_map["TNS"];
      clock_timing.setup_wns = timing_map["WNS"];
      clock_timing.suggest_freq = timing_map["Freq(MHz)"];
      top_rt_summary.tg_summary.clocks_timing.push_back(clock_timing);
    }
    top_rt_summary.tg_summary.power_info
        = {rt_summary.tg_summary.power_map["static_power"], rt_summary.tg_summary.power_map["dynamic_power"]};
  }
  // la_summary
  {
    top_rt_summary.la_summary.routing_demand_map = rt_summary.la_summary.routing_demand_map;
    top_rt_summary.la_summary.total_demand = rt_summary.la_summary.total_demand;
    top_rt_summary.la_summary.routing_overflow_map = rt_summary.la_summary.routing_overflow_map;
    top_rt_summary.la_summary.total_overflow = rt_summary.la_summary.total_overflow;
    top_rt_summary.la_summary.routing_wire_length_map = rt_summary.la_summary.routing_wire_length_map;
    top_rt_summary.la_summary.total_wire_length = rt_summary.la_summary.total_wire_length;
    top_rt_summary.la_summary.cut_via_num_map = rt_summary.la_summary.cut_via_num_map;
    top_rt_summary.la_summary.total_via_num = rt_summary.la_summary.total_via_num;
    for (auto& [clock_name, timing_map] : rt_summary.la_summary.clock_timing) {
      ieda_feature::ClockTiming clock_timing;
      clock_timing.clock_name = clock_name;
      clock_timing.setup_tns = timing_map["TNS"];
      clock_timing.setup_wns = timing_map["WNS"];
      clock_timing.suggest_freq = timing_map["Freq(MHz)"];
      top_rt_summary.la_summary.clocks_timing.push_back(clock_timing);
    }
    top_rt_summary.la_summary.power_info
        = {rt_summary.la_summary.power_map["static_power"], rt_summary.la_summary.power_map["dynamic_power"]};
  }
  // gr_summary
  {
    for (auto& [iter, gr_summary] : rt_summary.iter_gr_summary_map) {
      ieda_feature::GRSummary& top_gr_summary = top_rt_summary.iter_gr_summary_map[iter];
      top_gr_summary.routing_demand_map = gr_summary.routing_demand_map;
      top_gr_summary.total_demand = gr_summary.total_demand;
      top_gr_summary.routing_overflow_map = gr_summary.routing_overflow_map;
      top_gr_summary.total_overflow = gr_summary.total_overflow;
      top_gr_summary.routing_wire_length_map = gr_summary.routing_wire_length_map;
      top_gr_summary.total_wire_length = gr_summary.total_wire_length;
      top_gr_summary.cut_via_num_map = gr_summary.cut_via_num_map;
      top_gr_summary.total_via_num = gr_summary.total_via_num;
      for (auto& [clock_name, timing_map] : gr_summary.clock_timing) {
        ieda_feature::ClockTiming clock_timing;
        clock_timing.clock_name = clock_name;
        clock_timing.setup_tns = timing_map["TNS"];
        clock_timing.setup_wns = timing_map["WNS"];
        clock_timing.suggest_freq = timing_map["Freq(MHz)"];
        top_gr_summary.clocks_timing.push_back(clock_timing);
      }
      top_gr_summary.power_info = {gr_summary.power_map["static_power"], gr_summary.power_map["dynamic_power"]};
    }
  }
  // ta_summary
  {
    top_rt_summary.ta_summary.routing_wire_length_map = rt_summary.ta_summary.routing_wire_length_map;
    top_rt_summary.ta_summary.total_wire_length = rt_summary.ta_summary.total_wire_length;
    top_rt_summary.ta_summary.routing_violation_num_map = rt_summary.ta_summary.routing_violation_num_map;
    top_rt_summary.ta_summary.total_violation_num = rt_summary.ta_summary.total_violation_num;
  }
  // dr_summary
  {
    for (auto& [iter, dr_summary] : rt_summary.iter_dr_summary_map) {
      ieda_feature::DRSummary& top_dr_summary = top_rt_summary.iter_dr_summary_map[iter];
      top_dr_summary.routing_wire_length_map = dr_summary.routing_wire_length_map;
      top_dr_summary.total_wire_length = dr_summary.total_wire_length;
      top_dr_summary.cut_via_num_map = dr_summary.cut_via_num_map;
      top_dr_summary.total_via_num = dr_summary.total_via_num;
      top_dr_summary.routing_violation_num_map = dr_summary.routing_violation_num_map;
      top_dr_summary.total_violation_num = dr_summary.total_violation_num;

      for (auto& [clock_name, timing_map] : dr_summary.clock_timing) {
        ieda_feature::ClockTiming clock_timing;
        clock_timing.clock_name = clock_name;
        clock_timing.setup_tns = timing_map["TNS"];
        clock_timing.setup_wns = timing_map["WNS"];
        clock_timing.suggest_freq = timing_map["Freq(MHz)"];
        top_dr_summary.clocks_timing.push_back(clock_timing);
      }
      top_dr_summary.power_info = {dr_summary.power_map["static_power"], dr_summary.power_map["dynamic_power"]};
    }
  }
  // er_summary
  {
    top_rt_summary.er_summary.routing_demand_map = rt_summary.er_summary.routing_demand_map;
    top_rt_summary.er_summary.total_demand = rt_summary.er_summary.total_demand;
    top_rt_summary.er_summary.routing_overflow_map = rt_summary.er_summary.routing_overflow_map;
    top_rt_summary.er_summary.total_overflow = rt_summary.er_summary.total_overflow;
    top_rt_summary.er_summary.routing_wire_length_map = rt_summary.er_summary.routing_wire_length_map;
    top_rt_summary.er_summary.total_wire_length = rt_summary.er_summary.total_wire_length;
    top_rt_summary.er_summary.cut_via_num_map = rt_summary.er_summary.cut_via_num_map;
    top_rt_summary.er_summary.total_via_num = rt_summary.er_summary.total_via_num;
    for (auto& [clock_name, timing_map] : rt_summary.er_summary.clock_timing) {
      ieda_feature::ClockTiming clock_timing;
      clock_timing.clock_name = clock_name;
      clock_timing.setup_tns = timing_map["TNS"];
      clock_timing.setup_wns = timing_map["WNS"];
      clock_timing.suggest_freq = timing_map["Freq(MHz)"];
      top_rt_summary.er_summary.clocks_timing.push_back(clock_timing);
    }
    top_rt_summary.er_summary.power_info
        = {rt_summary.er_summary.power_map["static_power"], rt_summary.er_summary.power_map["dynamic_power"]};
  }
}

#endif

#if 1  // 获得IdbSegment

idb::IdbLayerShape* RTInterface::getIDBLayerShapeByFixedRect(EXTLayerRect* fixed_rect, bool is_routing)
{
  idb::IdbLayers* idb_layer_list = dmInst->get_idb_def_service()->get_layout()->get_layers();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();

  std::string layer_name;
  if (is_routing) {
    layer_name = routing_layer_list[fixed_rect->get_layer_idx()].get_layer_name();
  } else {
    layer_name = cut_layer_list[fixed_rect->get_layer_idx()].get_layer_name();
  }
  idb::IdbLayer* idb_layer = idb_layer_list->find_layer(layer_name);
  if (idb_layer == nullptr) {
    RTLOG.error(Loc::current(), "Can not find idb layer ", layer_name);
  }
  PlanarRect& real_rect = fixed_rect->get_real_rect();

  idb::IdbLayerShape* idb_shape = new idb::IdbLayerShape();
  idb_shape->set_type_rect();
  idb_shape->add_rect(real_rect.get_ll_x(), real_rect.get_ll_y(), real_rect.get_ur_x(), real_rect.get_ur_y());
  idb_shape->set_layer(idb_layer);
  return idb_shape;
}

idb::IdbRegularWireSegment* RTInterface::getIDBSegmentByNetResult(int32_t net_idx, Segment<LayerCoord>& segment)
{
  if (segment.get_first().get_layer_idx() == segment.get_second().get_layer_idx()) {
    return getIDBWire(net_idx, segment);
  } else {
    return getIDBVia(net_idx, segment);
  }
}

idb::IdbRegularWireSegment* RTInterface::getIDBSegmentByNetPatch(int32_t net_idx, EXTLayerRect& ext_layer_rect)
{
  idb::IdbLayers* idb_layer_list = dmInst->get_idb_def_service()->get_layout()->get_layers();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  std::string layer_name = routing_layer_list[ext_layer_rect.get_layer_idx()].get_layer_name();
  idb::IdbLayer* idb_layer = idb_layer_list->find_layer(layer_name);
  if (idb_layer == nullptr) {
    RTLOG.error(Loc::current(), "Can not find idb layer ", layer_name);
  }
  PlanarRect& real_rect = ext_layer_rect.get_real_rect();

  idb::IdbRegularWireSegment* idb_segment = new idb::IdbRegularWireSegment();
  idb_segment->set_layer(idb_layer);
  idb_segment->set_is_rect(true);
  idb_segment->add_point(real_rect.get_ll_x(), real_rect.get_ll_y());
  idb_segment->set_delta_rect(0, 0, real_rect.get_ur_x() - real_rect.get_ll_x(), real_rect.get_ur_y() - real_rect.get_ll_y());
  return idb_segment;
}

idb::IdbRegularWireSegment* RTInterface::getIDBWire(int32_t net_idx, Segment<LayerCoord>& segment)
{
  idb::IdbLayers* idb_layer_list = dmInst->get_idb_def_service()->get_layout()->get_layers();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  LayerCoord& first_coord = segment.get_first();
  LayerCoord& second_coord = segment.get_second();
  int32_t layer_idx = first_coord.get_layer_idx();

  if (RTUTIL.isOblique(first_coord, second_coord)) {
    RTLOG.error(Loc::current(), "The wire is oblique!");
  }
  std::string layer_name = routing_layer_list[layer_idx].get_layer_name();
  idb::IdbLayer* idb_layer = idb_layer_list->find_layer(layer_name);
  if (idb_layer == nullptr) {
    RTLOG.error(Loc::current(), "Can not find idb layer ", layer_name);
  }
  idb::IdbRegularWireSegment* idb_segment = new idb::IdbRegularWireSegment();
  idb_segment->set_layer(idb_layer);
  idb_segment->add_point(first_coord.get_x(), first_coord.get_y());
  idb_segment->add_point(second_coord.get_x(), second_coord.get_y());
  return idb_segment;
}

idb::IdbRegularWireSegment* RTInterface::getIDBVia(int32_t net_idx, Segment<LayerCoord>& segment)
{
  idb::IdbVias* lef_via_list = dmInst->get_idb_lef_service()->get_layout()->get_via_list();
  idb::IdbVias* def_via_list = dmInst->get_idb_def_service()->get_design()->get_via_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();

  LayerCoord& first_coord = segment.get_first();
  LayerCoord& second_coord = segment.get_second();
  int32_t below_layer_idx = std::min(first_coord.get_layer_idx(), second_coord.get_layer_idx());

  if (below_layer_idx < 0 || below_layer_idx >= static_cast<int32_t>(layer_via_master_list.size())) {
    RTLOG.error(Loc::current(), "The via below_layer_idx is illegal!");
  }
  std::string via_name = layer_via_master_list[below_layer_idx].front().get_via_name();
  idb::IdbVia* idb_via = lef_via_list->find_via(via_name);
  if (idb_via == nullptr) {
    idb_via = def_via_list->find_via(via_name);
  }
  if (idb_via == nullptr) {
    RTLOG.error(Loc::current(), "Can not find idb via ", via_name, "!");
  }
  idb::IdbLayer* idb_layer_top = idb_via->get_instance()->get_top_layer_shape()->get_layer();
  if (idb_layer_top == nullptr) {
    RTLOG.error(Loc::current(), "Can not find layer from idb via ", via_name, "!");
  }
  idb::IdbRegularWireSegment* idb_segment = new idb::IdbRegularWireSegment();
  idb_segment->set_layer(idb_layer_top);
  idb_segment->set_is_via(true);
  idb_segment->add_point(first_coord.get_x(), first_coord.get_y());
  idb::IdbVia* idb_via_new = idb_segment->copy_via(idb_via);
  idb_via_new->set_coordinate(first_coord.get_x(), first_coord.get_y());
  return idb_segment;
}

#endif

#endif

#if 1  // iDRC

std::vector<Violation> RTInterface::getViolationList(std::vector<std::pair<EXTLayerRect*, bool>>& env_shape_list,
                                                     std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>>& net_pin_shape_map,
                                                     std::map<int32_t, std::vector<Segment<LayerCoord>*>>& net_result_map,
                                                     std::map<int32_t, std::vector<EXTLayerRect*>>& net_patch_map)
{
  std::vector<idb::IdbLayerShape*> idb_env_shape_list;
  for (std::pair<EXTLayerRect*, bool>& env_shape : env_shape_list) {
    idb_env_shape_list.push_back(getIDBLayerShapeByFixedRect(env_shape.first, env_shape.second));
  }
  std::map<int32_t, std::vector<idb::IdbLayerShape*>> idb_net_pin_shape_map;
  for (auto& [net_idx, pin_shape_list] : net_pin_shape_map) {
    for (std::pair<EXTLayerRect*, bool>& pin_shape : pin_shape_list) {
      idb_net_pin_shape_map[net_idx].push_back(getIDBLayerShapeByFixedRect(pin_shape.first, pin_shape.second));
    }
  }
  std::map<int32_t, std::vector<idb::IdbRegularWireSegment*>> idb_net_result_map;
  for (auto& [net_idx, segment_list] : net_result_map) {
    for (Segment<LayerCoord>* segment : segment_list) {
      idb_net_result_map[net_idx].push_back(getIDBSegmentByNetResult(net_idx, *segment));
    }
  }
  for (auto& [net_idx, patch_set] : net_patch_map) {
    for (EXTLayerRect* patch : patch_set) {
      idb_net_result_map[net_idx].push_back(getIDBSegmentByNetPatch(net_idx, *patch));
    }
  }
  std::vector<Violation> violation_list = getViolationList(idb_env_shape_list, idb_net_pin_shape_map, idb_net_result_map);
  // free memory
  {
    for (idb::IdbLayerShape* idb_env_shape : idb_env_shape_list) {
      delete idb_env_shape;
      idb_env_shape = nullptr;
    }
    for (auto& [net_idx, pin_shape_list] : idb_net_pin_shape_map) {
      for (idb::IdbLayerShape* pin_shape : pin_shape_list) {
        delete pin_shape;
        pin_shape = nullptr;
      }
    }
    for (auto& [net_idx, segment_list] : idb_net_result_map) {
      for (idb::IdbRegularWireSegment* segment : segment_list) {
        delete segment;
        segment = nullptr;
      }
    }
  }
  return violation_list;
}

std::vector<Violation> RTInterface::getViolationList(std::vector<idb::IdbLayerShape*>& env_shape_list,
                                                     std::map<int32_t, std::vector<idb::IdbLayerShape*>>& net_pin_shape_map,
                                                     std::map<int32_t, std::vector<idb::IdbRegularWireSegment*>>& net_result_map)
{
  std::map<std::string, int32_t>& routing_layer_name_to_idx_map = RTDM.getDatabase().get_routing_layer_name_to_idx_map();

  idrc::DrcApi drc_api;
  drc_api.init();

  std::vector<Violation> violation_list;
  for (auto& [idrc_violation_type, idrc_violation_list] : drc_api.check(env_shape_list, net_pin_shape_map, net_result_map)) {
    ViolationType violation_type;
    {
      switch (idrc_violation_type) {
        case idrc::ViolationEnumType::kArea:
          violation_type = ViolationType::kMinimumArea;
          break;
        case idrc::ViolationEnumType::kAreaEnclosed:
          violation_type = ViolationType::kMinimumArea;
          break;
        case idrc::ViolationEnumType::kShort:
          violation_type = ViolationType::kMetalShort;
          break;
        case idrc::ViolationEnumType::kDefaultSpacing:
          violation_type = ViolationType::kParallelRunLengthSpacing;
          break;
        case idrc::ViolationEnumType::kPRLSpacing:
          violation_type = ViolationType::kParallelRunLengthSpacing;
          break;
        case idrc::ViolationEnumType::kJogToJog:
          violation_type = ViolationType::kJogToJogSpacing;
          break;
        case idrc::ViolationEnumType::kEOL:
          violation_type = ViolationType::kEndOfLineSpacing;
          break;
        case idrc::ViolationEnumType::kWidth:
          violation_type = ViolationType::kMinimumWidth;
          break;
        case idrc::ViolationEnumType::kMinStep:
          violation_type = ViolationType::kMinStep;
          break;
        case idrc::ViolationEnumType::kNotch:
          violation_type = ViolationType::kNotchSpacing;
          break;
        case idrc::ViolationEnumType::kCornerFill:
          violation_type = ViolationType::kCornerFillSpacing;
          break;
        default:
          RTLOG.warn(Loc::current(), "Unknow rule!");
          violation_type = ViolationType::kNone;
          break;
      }
    }
    for (idrc::DrcViolation* idrc_violation : idrc_violation_list) {
      EXTLayerRect ext_layer_rect;
      {
        if (idrc_violation->is_rect()) {
          idrc::DrcViolationRect* idrc_violation_rect = static_cast<idrc::DrcViolationRect*>(idrc_violation);
          ext_layer_rect.set_real_ll(idrc_violation_rect->get_llx(), idrc_violation_rect->get_lly());
          ext_layer_rect.set_real_ur(idrc_violation_rect->get_urx(), idrc_violation_rect->get_ury());
        } else {
          RTLOG.error(Loc::current(), "Not supported!");
        }
        if (idrc_violation->get_layer()->is_routing()) {
          ext_layer_rect.set_layer_idx(routing_layer_name_to_idx_map[idrc_violation->get_layer()->get_name()]);
        } else {
          RTLOG.error(Loc::current(), "Not supported!");
        }
      }
      std::set<int32_t> violation_net_set;
      {
        violation_net_set = idrc_violation->get_net_ids();
        if (violation_net_set.size() > 2) {
          RTLOG.error(Loc::current(), "The violation_net_set size > 2!");
        }
      }
      Violation violation;
      violation.set_violation_type(violation_type);
      violation.set_violation_shape(ext_layer_rect);
      violation.set_is_routing(true);
      violation.set_violation_net_set(violation_net_set);
      violation.set_required_size(0);
      violation_list.push_back(violation);
    }
  }
  return violation_list;
}

#endif

#if 1  // iSTA

void RTInterface::updateTimingAndPower(std::vector<std::map<std::string, std::vector<LayerCoord>>>& real_pin_coord_map_list,
                                       std::vector<std::vector<Segment<LayerCoord>>>& routing_segment_list_list,
                                       std::map<std::string, std::map<std::string, double>>& clock_timing,
                                       std::map<std::string, double>& power)
{
#if 1  // 数据结构定义
  struct RCPin
  {
    RCPin() = default;
    RCPin(LayerCoord coord, bool is_real_pin, std::string pin_name)
    {
      _coord = coord;
      _is_real_pin = is_real_pin;
      _pin_name = pin_name;
    }
    RCPin(LayerCoord coord, bool is_real_pin, int32_t fake_pin_id)
    {
      _coord = coord;
      _is_real_pin = is_real_pin;
      _fake_pin_id = fake_pin_id;
    }
    ~RCPin() = default;

    LayerCoord _coord;
    bool _is_real_pin = false;
    std::string _pin_name;
    int32_t _fake_pin_id = -1;
  };
#endif

#if 1  // 函数定义
  auto initTimingEngine = [](std::string workspace) {
    ista::TimingEngine* timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
    if (!timing_engine->isBuildGraph()) {
      timing_engine->set_design_work_space(workspace.c_str());
      timing_engine->readLiberty(dmInst->get_config().get_lib_paths());
      auto db_adapter = std::make_unique<ista::TimingIDBAdapter>(timing_engine->get_ista());
      db_adapter->set_idb(dmInst->get_idb_builder());
      db_adapter->convertDBToTimingNetlist();
      timing_engine->set_db_adapter(std::move(db_adapter));
      timing_engine->readSdc(dmInst->get_config().get_sdc_path().c_str());
      timing_engine->buildGraph();
    }
    timing_engine->initRcTree();
    return timing_engine;
  };
  auto initPowerEngine = [](std::string workspace) {
    auto* power_engine = ipower::PowerEngine::getOrCreatePowerEngine();
    if (!power_engine->isBuildGraph()) {
      power_engine->get_power()->set_design_work_space(workspace.c_str());
      power_engine->get_power()->initPowerGraphData();
      power_engine->get_power()->initToggleSPData();
    }
    return power_engine;
  };
  auto getRCSegmentList = [](std::map<LayerCoord, std::vector<std::string>, CmpLayerCoordByXASC>& coord_real_pin_map,
                             std::vector<Segment<LayerCoord>>& routing_segment_list) {
    // 预处理 对名字去重
    for (auto& [coord, real_pin_list] : coord_real_pin_map) {
      std::sort(real_pin_list.begin(), real_pin_list.end());
      real_pin_list.erase(std::unique(real_pin_list.begin(), real_pin_list.end()), real_pin_list.end());
    }
    // 构建coord_fake_pin_map
    std::map<LayerCoord, int32_t, CmpLayerCoordByXASC> coord_fake_pin_map;
    {
      int32_t fake_id = 0;
      for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
        LayerCoord& first_coord = routing_segment.get_first();
        LayerCoord& second_coord = routing_segment.get_second();

        if (!RTUTIL.exist(coord_real_pin_map, first_coord) && !RTUTIL.exist(coord_fake_pin_map, first_coord)) {
          coord_fake_pin_map[first_coord] = fake_id++;
        }
        if (!RTUTIL.exist(coord_real_pin_map, second_coord) && !RTUTIL.exist(coord_fake_pin_map, second_coord)) {
          coord_fake_pin_map[second_coord] = fake_id++;
        }
      }
    }
    std::vector<Segment<RCPin>> rc_segment_list;
    {
      // 生成线长为0的线段
      for (auto& [coord, real_pin_list] : coord_real_pin_map) {
        for (size_t i = 1; i < real_pin_list.size(); i++) {
          RCPin first_rc_pin(coord, true, RTUTIL.escapeBackslash(real_pin_list[i - 1]));
          RCPin second_rc_pin(coord, true, RTUTIL.escapeBackslash(real_pin_list[i]));
          rc_segment_list.emplace_back(first_rc_pin, second_rc_pin);
        }
      }
      // 生成线长大于0的线段
      for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
        auto getRCPin = [&](LayerCoord& coord) {
          RCPin rc_pin;
          if (RTUTIL.exist(coord_real_pin_map, coord)) {
            rc_pin = RCPin(coord, true, RTUTIL.escapeBackslash(coord_real_pin_map[coord].front()));
          } else if (RTUTIL.exist(coord_fake_pin_map, coord)) {
            rc_pin = RCPin(coord, false, coord_fake_pin_map[coord]);
          } else {
            RTLOG.error(Loc::current(), "The coord is not exist!");
          }
          return rc_pin;
        };
        rc_segment_list.emplace_back(getRCPin(routing_segment.get_first()), getRCPin(routing_segment.get_second()));
      }
    }
    return rc_segment_list;
  };
  auto getRctNode = [](ista::TimingEngine* timing_engine, ista::Netlist* sta_net_list, ista::Net* ista_net, RCPin& rc_pin) {
    ista::RctNode* rct_node = nullptr;
    if (rc_pin._is_real_pin) {
      ista::DesignObject* pin_port = nullptr;
      auto pin_port_list = sta_net_list->findPin(rc_pin._pin_name.c_str(), false, false);
      if (!pin_port_list.empty()) {
        pin_port = pin_port_list.front();
      } else {
        pin_port = sta_net_list->findPort(rc_pin._pin_name.c_str());
      }
      rct_node = timing_engine->makeOrFindRCTreeNode(pin_port);
    } else {
      rct_node = timing_engine->makeOrFindRCTreeNode(ista_net, rc_pin._fake_pin_id);
    }
    return rct_node;
  };
#endif

#if 1  // 预处理流程
  // 每个pin只留一个连通的坐标
  for (size_t i = 0; i < real_pin_coord_map_list.size(); i++) {
    std::vector<Segment<LayerCoord>>& routing_segment_list = routing_segment_list_list[i];
    for (auto& [pin_name, coord_list] : real_pin_coord_map_list[i]) {
      if (coord_list.size() < 2) {
        continue;
      }
      if (routing_segment_list.empty()) {
        coord_list.erase(coord_list.begin() + 1, coord_list.end());
      } else {
        for (LayerCoord& coord : coord_list) {
          bool is_exist = false;
          for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
            if (coord == routing_segment.get_first() || coord == routing_segment.get_second()) {
              is_exist = true;
              break;
            }
          }
          if (is_exist) {
            coord_list[0] = coord;
            coord_list.erase(coord_list.begin() + 1, coord_list.end());
            break;
          }
        }
      }
      if (coord_list.size() > 2) {
        RTLOG.error(Loc::current(), "The pin ", pin_name, " is not in segment_list");
      }
    }
  }
  // coord_real_pin_map_list
  std::vector<std::map<LayerCoord, std::vector<std::string>, CmpLayerCoordByXASC>> coord_real_pin_map_list;
  coord_real_pin_map_list.resize(real_pin_coord_map_list.size());
  for (size_t i = 0; i < real_pin_coord_map_list.size(); i++) {
    for (auto& [real_pin, coord_list] : real_pin_coord_map_list[i]) {
      for (LayerCoord& coord : coord_list) {
        coord_real_pin_map_list[i][coord].push_back(real_pin);
      }
    }
  }
#endif

#if 1  // 主流程
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();
  std::string& temp_directory_path = RTDM.getConfig().temp_directory_path;

  ista::TimingEngine* timing_engine = initTimingEngine(RTUTIL.getString(temp_directory_path, "sta/"));
  ista::Netlist* sta_net_list = timing_engine->get_netlist();

  for (size_t net_idx = 0; net_idx < coord_real_pin_map_list.size(); net_idx++) {
    ista::Net* ista_net = sta_net_list->findNet(RTUTIL.escapeBackslash(net_list[net_idx].get_net_name()).c_str());
    timing_engine->resetRcTree(ista_net);
    for (Segment<RCPin>& segment : getRCSegmentList(coord_real_pin_map_list[net_idx], routing_segment_list_list[net_idx])) {
      RCPin& first_rc_pin = segment.get_first();
      RCPin& second_rc_pin = segment.get_second();

      double cap = 0;
      double res = 0;
      if (first_rc_pin._coord.get_layer_idx() == second_rc_pin._coord.get_layer_idx()) {
        int32_t distance = RTUTIL.getManhattanDistance(first_rc_pin._coord, second_rc_pin._coord);
        int32_t unit = dmInst->get_idb_def_service()->get_design()->get_units()->get_micron_dbu();
        std::optional<double> width = std::nullopt;
        cap = dynamic_cast<ista::TimingIDBAdapter*>(timing_engine->get_db_adapter())
                  ->getCapacitance(first_rc_pin._coord.get_layer_idx() + 1, distance / 1.0 / unit, width);
        res = dynamic_cast<ista::TimingIDBAdapter*>(timing_engine->get_db_adapter())
                  ->getResistance(first_rc_pin._coord.get_layer_idx() + 1, distance / 1.0 / unit, width);
      }

      ista::RctNode* first_node = getRctNode(timing_engine, sta_net_list, ista_net, first_rc_pin);
      ista::RctNode* second_node = getRctNode(timing_engine, sta_net_list, ista_net, second_rc_pin);
      timing_engine->makeResistor(ista_net, first_node, second_node, res);
      timing_engine->incrCap(first_node, cap / 2, true);
      timing_engine->incrCap(second_node, cap / 2, true);
    }
    timing_engine->updateRCTreeInfo(ista_net);
    // auto* rc_tree = timing_engine->get_ista()->getRcNet(ista_net)->rct();
    // rc_tree->printGraphViz();
    // int32_t a = 0;
    // dot -Tpdf tree.dot -o tree.pdf
  }
  timing_engine->updateTiming();
  timing_engine->reportTiming();

  auto clk_list = timing_engine->getClockList();
  std::ranges::for_each(clk_list, [&](ista::StaClock* clk) {
    auto clk_name = clk->get_clock_name();
    auto setup_tns = timing_engine->getTNS(clk_name, AnalysisMode::kMax);
    auto setup_wns = timing_engine->getWNS(clk_name, AnalysisMode::kMax);
    auto suggest_freq = 1000.0 / (clk->getPeriodNs() - setup_wns);
    clock_timing[clk_name]["TNS"] = setup_tns;
    clock_timing[clk_name]["WNS"] = setup_wns;
    clock_timing[clk_name]["Freq(MHz)"] = suggest_freq;
  });
  ipower::PowerEngine* power_engine = initPowerEngine(RTUTIL.getString(temp_directory_path, "pwr/"));
  power_engine->get_power()->updatePower();
  power_engine->get_power()->reportPower();

  double static_power = 0;
  for (const auto& data : power_engine->get_power()->get_leakage_powers()) {
    static_power += data->get_leakage_power();
  }
  double dynamic_power = 0;
  for (const auto& data : power_engine->get_power()->get_internal_powers()) {
    dynamic_power += data->get_internal_power();
  }
  for (const auto& data : power_engine->get_power()->get_switch_powers()) {
    dynamic_power += data->get_switch_power();
  }
  power["static_power"] = static_power;
  power["dynamic_power"] = dynamic_power;
#endif
}

#endif

#if 1  // flute

void RTInterface::initFlute()
{
  Flute::readLUT();
}

void RTInterface::destroyFlute()
{
  Flute::deleteLUT();
}

std::vector<Segment<PlanarCoord>> RTInterface::getPlanarTopoList(std::vector<PlanarCoord> planar_coord_list)
{
  std::vector<Segment<PlanarCoord>> planar_topo_list;
  if (planar_coord_list.size() > 1) {
    size_t point_num = planar_coord_list.size();
    Flute::DTYPE* x_list = (Flute::DTYPE*) malloc(sizeof(Flute::DTYPE) * (point_num));
    Flute::DTYPE* y_list = (Flute::DTYPE*) malloc(sizeof(Flute::DTYPE) * (point_num));
    for (size_t i = 0; i < point_num; i++) {
      x_list[i] = planar_coord_list[i].get_x();
      y_list[i] = planar_coord_list[i].get_y();
    }
    Flute::Tree flute_tree = Flute::flute(point_num, x_list, y_list, FLUTE_ACCURACY);
    free(x_list);
    free(y_list);

    for (int32_t i = 0; i < 2 * flute_tree.deg - 2; i++) {
      int32_t n_id = flute_tree.branch[i].n;
      PlanarCoord first_coord(flute_tree.branch[i].x, flute_tree.branch[i].y);
      PlanarCoord second_coord(flute_tree.branch[n_id].x, flute_tree.branch[n_id].y);
      if (first_coord != second_coord) {
        planar_topo_list.emplace_back(first_coord, second_coord);
      }
    }
    Flute::free_tree(flute_tree);
  }
  return planar_topo_list;
}

#endif

#if 1  // lsa

void RTInterface::routeTAPanel(TAPanel& ta_panel)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  TAPanelId& ta_panel_id = ta_panel.get_ta_panel_id();
  RoutingLayer& routing_layer = routing_layer_list[ta_panel_id.get_layer_idx()];
  int32_t half_width = routing_layer.get_min_width() / 2;

  // 构造ls_panel
  lsa::LSPanel ls_panel;
  {
    ls_panel.layer_id = ta_panel_id.get_layer_idx();
    ls_panel.panel_id = ta_panel_id.get_panel_idx();
    ls_panel.ll_x = ta_panel.get_panel_rect().get_real_ll_x();
    ls_panel.ll_y = ta_panel.get_panel_rect().get_real_ll_y();
    ls_panel.ur_x = ta_panel.get_panel_rect().get_real_ur_x();
    ls_panel.ur_y = ta_panel.get_panel_rect().get_real_ur_y();
    ls_panel.prefer_direction = (routing_layer.isPreferH() ? "H" : "V");

    // track_list
    for (ScaleGrid& x_grid : ta_panel.get_panel_track_axis().get_x_grid_list()) {
      lsa::LSTrack ls_track;
      ls_track.axis = "X";
      ls_track.start = x_grid.get_start_line();
      ls_track.step_length = x_grid.get_step_length();
      ls_track.end = x_grid.get_end_line();
      ls_panel.track_list.push_back(ls_track);
    }
    for (ScaleGrid& y_grid : ta_panel.get_panel_track_axis().get_y_grid_list()) {
      lsa::LSTrack ls_track;
      ls_track.axis = "Y";
      ls_track.start = y_grid.get_start_line();
      ls_track.step_length = y_grid.get_step_length();
      ls_track.end = y_grid.get_end_line();
      ls_panel.track_list.push_back(ls_track);
    }
    // wire_list
    for (TATask* ta_task : ta_panel.get_ta_task_list()) {
      std::vector<TAGroup>& ta_group_list = ta_task->get_ta_group_list();
      LayerCoord first_coord = ta_group_list.front().get_coord_list().front();
      LayerCoord second_coord = ta_group_list.back().get_coord_list().front();
      if (routing_layer.isPreferH()) {
        first_coord.set_y(half_width);
        second_coord.set_y(half_width);
      } else {
        first_coord.set_x(half_width);
        second_coord.set_x(half_width);
      }
      LayerRect rect(RTUTIL.getEnlargedRect(first_coord, second_coord, half_width), ta_panel_id.get_layer_idx());
      lsa::LSShape ls_shape;
      ls_shape.net_id = ta_task->get_net_idx();
      ls_shape.task_id = ta_task->get_task_idx();
      ls_shape.ll_x = rect.get_ll_x();
      ls_shape.ll_y = rect.get_ll_y();
      ls_shape.ur_x = rect.get_ur_x();
      ls_shape.ur_y = rect.get_ur_y();
      ls_panel.wire_list.push_back(ls_shape);
    }
    // hard_shape_list
    for (auto& [net_idx, fixed_rect_set] : ta_panel.get_net_fixed_rect_map()) {
      for (auto& fixed_rect : fixed_rect_set) {
        lsa::LSShape ls_shape;
        ls_shape.net_id = net_idx;
        ls_shape.ll_x = fixed_rect->get_real_ll_x();
        ls_shape.ll_y = fixed_rect->get_real_ll_y();
        ls_shape.ur_x = fixed_rect->get_real_ur_x();
        ls_shape.ur_y = fixed_rect->get_real_ur_y();
        ls_panel.hard_shape_list.push_back(ls_shape);
      }
    }
    for (auto& [net_idx, rect_list] : ta_panel.get_net_access_result_map()) {
      for (auto& rect : rect_list) {
        lsa::LSShape ls_shape;
        ls_shape.net_id = net_idx;
        ls_shape.ll_x = rect.get_ll_x();
        ls_shape.ll_y = rect.get_ll_y();
        ls_shape.ur_x = rect.get_ur_x();
        ls_shape.ur_y = rect.get_ur_y();
        ls_panel.hard_shape_list.push_back(ls_shape);
      }
    }
    for (auto& [net_idx, rect_list] : ta_panel.get_net_detailed_result_map()) {
      for (auto& rect : rect_list) {
        lsa::LSShape ls_shape;
        ls_shape.net_id = net_idx;
        ls_shape.ll_x = rect.get_ll_x();
        ls_shape.ll_y = rect.get_ll_y();
        ls_shape.ur_x = rect.get_ur_x();
        ls_shape.ur_y = rect.get_ur_y();
        ls_panel.hard_shape_list.push_back(ls_shape);
      }
    }
  }
  // 将结果存回ls_panel
  {
    lsa::LSAssigner ls_assigner;
    ls_panel = ls_assigner.getResult(ls_panel);
  }
  // 写回ta_panel
  {
    std::map<int32_t, std::vector<Segment<LayerCoord>>> task_segment_map;
    for (lsa::LSShape& wire : ls_panel.wire_list) {
      Segment<LayerCoord> routing_segment(
          LayerCoord(static_cast<int32_t>(wire.ll_x + half_width), static_cast<int32_t>(wire.ll_y + half_width), ls_panel.layer_id),
          LayerCoord(static_cast<int32_t>(wire.ur_x - half_width), static_cast<int32_t>(wire.ur_y - half_width), ls_panel.layer_id));
      if (RTUTIL.isOblique(routing_segment.get_first(), routing_segment.get_second())) {
        RTLOG.error(Loc::current(), "The segment is oblique");
      }
      task_segment_map[wire.task_id].push_back(routing_segment);
    }
    std::vector<TATask*>& ta_task_list = ta_panel.get_ta_task_list();
    std::sort(ta_task_list.begin(), ta_task_list.end(), [](TATask* a, TATask* b) { return a->get_task_idx() < b->get_task_idx(); });
    for (auto& [task_idx, routing_segment_list] : task_segment_map) {
      TATask* ta_task = ta_task_list[task_idx];
      if (ta_task->get_task_idx() != task_idx) {
        RTLOG.error(Loc::current(), "The task idx is not equal!");
      }
      std::vector<LayerCoord> candidate_root_coord_list;
      std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
      std::vector<TAGroup>& ta_group_list = ta_task->get_ta_group_list();
      for (size_t i = 0; i < ta_group_list.size(); i++) {
        for (LayerCoord& coord : ta_group_list[i].get_coord_list()) {
          candidate_root_coord_list.push_back(coord);
          key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
        }
      }
      MTree<LayerCoord> coord_tree = RTUTIL.getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, key_coord_pin_map);
      for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
        ta_panel.get_net_task_result_map()[ta_task->get_net_idx()][task_idx].emplace_back(coord_segment.get_first()->value(),
                                                                                          coord_segment.get_second()->value());
      }
    }
  }
}

#endif

#endif

// private

RTInterface* RTInterface::_rt_interface_instance = nullptr;

}  // namespace irt

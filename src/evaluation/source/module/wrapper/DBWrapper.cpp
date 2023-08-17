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
#include "DBWrapper.hpp"

#include <regex>

namespace eval {

DBWrapper::DBWrapper(Config* config) : _eval_db(new EvalDB())
{
  _db_config = config->get_db_config();
  _cong_config = config->get_cong_config();
  _drc_config = config->get_drc_config();
  _gds_wrapper_config = config->get_gds_wrapper_config();
  _power_config = config->get_power_config();
  _timing_config = config->get_timing_config();
  _wl_config = config->get_wl_config();
  initIDB();
  wrapIDBData();
}

DBWrapper::~DBWrapper()
{
  delete _eval_db;
}

void DBWrapper::initIDB()
{
  auto* idb_bulider = _eval_db->_idb_builder;

  std::vector<std::string> lef_files = _db_config.get_lef_file_list();
  idb_bulider->buildLef(lef_files);

  std::string def_file = _db_config.get_def_file();
  idb_bulider->buildDef(def_file);
}

void DBWrapper::wrapIDBData()
{
  IdbDefService* idb_def_service = _eval_db->_idb_builder->get_def_service();
  IdbLayout* idb_layout = idb_def_service->get_layout();
  IdbDesign* idb_design = idb_def_service->get_design();

  bool is_gds_wrapper_eval = _gds_wrapper_config.enable_eval();
  if (is_gds_wrapper_eval == true) {
    LOG_INFO << "Convert lef/def data to GDS.";
    wrapGDSNetlist(_eval_db->_idb_builder);
  } else {
    wrapLayout(idb_layout);
    wrapDesign(idb_design);
  }
}

void DBWrapper::wrapLayout(IdbLayout* idb_layout)
{
  auto* ieval_layout = _eval_db->_layout;

  // set dbu
  int32_t database_unit = idb_layout->get_units()->get_micron_dbu();
  ieval_layout->set_database_unit(database_unit);

  // set die shape.
  IdbDie* idb_die = idb_layout->get_die();
  ieval_layout->set_die_shape(Rectangle<int32_t>(idb_die->get_llx(), idb_die->get_lly(), idb_die->get_urx(), idb_die->get_ury()));

  // set core shape.
  IdbCore* idb_core = idb_layout->get_core();
  IdbRect* idb_core_rect = idb_core->get_bounding_box();
  ieval_layout->set_core_shape(
      Rectangle<int32_t>(idb_core_rect->get_low_x(), idb_core_rect->get_low_y(), idb_core_rect->get_high_x(), idb_core_rect->get_high_y()));

  // set tileGrid
  ieval_layout->set_tile_grid(_cong_config.get_tile_size_x(), _cong_config.get_tile_size_y(),
                              idb_layout->get_layers()->get_routing_layers_number());

  // set binGrid for idb
  ieval_layout->set_cong_grid(_cong_config.get_bin_cnt_x(), _cong_config.get_bin_cnt_y(), idb_layout->get_layers());

  // set bingrid for predictor
  // ieval_layout->set_cong_grid_for_predictor(_cong_config.get_tile_size_x(), _cong_config.get_tile_size_y(), idb_layout->get_layers());
}

void DBWrapper::wrapDesign(IdbDesign* idb_design)
{
  auto* ieval_design = _eval_db->_design;

  // set design name.
  const std::string& design_name = idb_design->get_design_name();
  ieval_design->set_design_name(design_name);

  // set netlists, must be after wrapInstances()
  bool is_wirelength_eval = _wl_config.enable_eval();
  bool is_congestion_eval = _cong_config.enable_eval();

  if (is_wirelength_eval == true) {
    LOG_INFO << "Setting netlist to WLNet.";
    wrapWLNetlists(idb_design);
  }

  if (is_congestion_eval == true) {
    LOG_INFO << "Setting netlist to CongNet.";
    wrapInstances(idb_design);
    wrapCongNetlists(idb_design);
  }
}

void DBWrapper::wrapInstances(IdbDesign* idb_design)
{
  auto* ieval_design = _eval_db->_design;

  for (auto* idb_inst : idb_design->get_instance_list()->get_instance_list()) {
    CongInst* inst_ptr = new CongInst();
    inst_ptr->set_name(idb_inst->get_name());

    // set instance coordinate.
    auto bbox = idb_inst->get_bounding_box();
    inst_ptr->set_shape(bbox->get_low_x(), bbox->get_low_y(), bbox->get_high_x(), bbox->get_high_y());

    // set type
    if (!isCoreOverlap(idb_inst)) {
      inst_ptr->set_loc_type(INSTANCE_LOC_TYPE::kOutside);
    } else {
      inst_ptr->set_loc_type(INSTANCE_LOC_TYPE::kNormal);
    }

    ieval_design->add_instance(inst_ptr);
  }
}

bool DBWrapper::isCoreOverlap(IdbInstance* idb_inst)
{
  Point<int32_t> die_lower = this->get_layout()->get_die_shape().get_lower_left();
  Point<int32_t> die_upper = this->get_layout()->get_die_shape().get_upper_right();
  Point<int32_t> core_lower = this->get_layout()->get_core_shape().get_lower_left();
  Point<int32_t> core_upper = this->get_layout()->get_core_shape().get_upper_right();

  if ((idb_inst->get_bounding_box()->get_low_x() >= die_lower.get_x() && idb_inst->get_bounding_box()->get_high_x() <= core_lower.get_x())
      || (idb_inst->get_bounding_box()->get_low_x() >= core_upper.get_x()
          && idb_inst->get_bounding_box()->get_high_x() <= die_upper.get_x())
      || (idb_inst->get_bounding_box()->get_low_y() >= die_lower.get_y()
          && idb_inst->get_bounding_box()->get_high_y() <= core_lower.get_y())
      || (idb_inst->get_bounding_box()->get_low_y() >= core_upper.get_y()
          && idb_inst->get_bounding_box()->get_high_y() <= die_upper.get_y())) {
    return false;
  } else {
    return true;
  }
}

void DBWrapper::wrapWLNetlists(IdbDesign* idb_design)
{
  auto* ieval_design = _eval_db->_design;

  for (auto* idb_net : idb_design->get_net_list()->get_net_list()) {
    std::string net_name = fixSlash(idb_net->get_net_name());

    WLNet* net_ptr = new WLNet();
    net_ptr->set_name(net_name);
    // set net type.
    auto connect_type = idb_net->get_connect_type();
    if (connect_type == IdbConnectType::kSignal) {
      net_ptr->set_type(NET_TYPE::kSignal);
    } else if (connect_type == IdbConnectType::kClock) {
      net_ptr->set_type(NET_TYPE::kClock);
    } else if (connect_type == IdbConnectType::kReset) {
      net_ptr->set_type(NET_TYPE::kReset);
    } else {
      net_ptr->set_type(NET_TYPE::kNone);
    }  // fengzhuang
    // set pins.
    auto* idb_driving_pin = idb_net->get_driving_pin();
    if (idb_driving_pin) {
      WLPin* pin_ptr = wrapWLPin(idb_driving_pin);
      net_ptr->add_pin(pin_ptr);
      net_ptr->set_driver_pin(pin_ptr);
    }
    for (auto* idb_load_pin : idb_net->get_load_pins()) {
      WLPin* pin_ptr = wrapWLPin(idb_load_pin);
      net_ptr->add_pin(pin_ptr);
      net_ptr->add_sink_pin(pin_ptr);
    }
    net_ptr->set_real_wirelength(idb_net->wireLength());
    ieval_design->add_net(net_ptr);
    _eval_db->_idb_net_map.emplace(idb_net, net_ptr);
    _eval_db->_net_idb_map.emplace(net_ptr, idb_net);
  }
}

void DBWrapper::wrapCongNetlists(IdbDesign* idb_design)
{
  auto* ieval_design = _eval_db->_design;

  for (auto* idb_net : idb_design->get_net_list()->get_net_list()) {
    std::string net_name = fixSlash(idb_net->get_net_name());

    CongNet* net_ptr = new CongNet();
    net_ptr->set_name(net_name);

    auto* idb_driving_pin = idb_net->get_driving_pin();
    if (idb_driving_pin) {
      CongPin* pin_ptr = wrapCongPin(idb_driving_pin);
      net_ptr->add_pin(pin_ptr);
    }
    for (auto* idb_load_pin : idb_net->get_load_pins()) {
      CongPin* pin_ptr = wrapCongPin(idb_load_pin);
      net_ptr->add_pin(pin_ptr);
    }
    ieval_design->add_net(net_ptr);
  }
}

void DBWrapper::wrapGDSNetlist(IdbBuilder* idb_builder)
{
  auto* ieval_design = _eval_db->_design;
  IdbDesign* idb_design = idb_builder->get_def_service()->get_design();

  for (auto* idb_net : idb_design->get_net_list()->get_net_list()) {
    std::string net_name = fixSlash(idb_net->get_net_name());

    GDSNet* net_ptr = new GDSNet();
    net_ptr->set_name(net_name);

    auto* idb_driving_pin = idb_net->get_driving_pin();
    if (idb_driving_pin) {
      GDSPin* pin_ptr = wrapGDSPin(idb_driving_pin);
      net_ptr->add_pin(pin_ptr);
    }
    for (auto* idb_load_pin : idb_net->get_load_pins()) {
      GDSPin* pin_ptr = wrapGDSPin(idb_load_pin);
      net_ptr->add_pin(pin_ptr);
    }

    auto idb_wire_list = idb_net->get_wire_list()->get_wire_list();
    for (size_t i = 0; i < idb_wire_list.size(); ++i) {
      auto& idb_segment_list = idb_wire_list[i]->get_segment_list();
      for (size_t j = 0; j < idb_segment_list.size(); ++j) {
        auto idb_segment = idb_segment_list[j];
        if (idb_segment->is_via()) {
          GDSViaNodes* gds_via_nodes = new GDSViaNodes();
          auto idb_via = idb_segment->get_via_list()[0];
          auto idb_via_master = idb_via->get_instance();

          int32_t up_idx = idb_via_master->get_top_layer_shape()->get_layer()->get_id();
          int32_t down_idx = idb_via_master->get_bottom_layer_shape()->get_layer()->get_id();
          gds_via_nodes->set_above_layer_idx(up_idx);
          gds_via_nodes->set_below_layer_idx(down_idx);

          int32_t x_coord = idb_via->get_coordinate()->get_x();
          int32_t y_coord = idb_via->get_coordinate()->get_y();
          gds_via_nodes->set_real_coord(Point<int32_t>(x_coord, y_coord));

          int32_t lx, ly, ux, uy;
          auto above_shape = idb_via_master->get_top_layer_shape()->get_bounding_box();
          lx = above_shape.get_low_x();
          ly = above_shape.get_low_y();
          ux = above_shape.get_high_x();
          uy = above_shape.get_high_y();
          gds_via_nodes->set_above_shape(Rectangle<int32_t>(lx, ly, ux, uy));
          auto below_shape = idb_via_master->get_bottom_layer_shape()->get_bounding_box();
          lx = below_shape.get_low_x();
          ly = below_shape.get_low_y();
          ux = below_shape.get_high_x();
          uy = below_shape.get_high_y();
          gds_via_nodes->set_below_shape(Rectangle<int32_t>(lx, ly, ux, uy));

          net_ptr->add_via_nodes(gds_via_nodes);
        } else {
          GDSWireNodes* gds_wire_nodes = new GDSWireNodes();

          int32_t layer_idx = idb_segment->get_layer()->get_id();
          gds_wire_nodes->set_layer_idx(layer_idx);

          IdbLayers* idb_layer_list = idb_builder->get_def_service()->get_layout()->get_layers();
          auto routing_layer = dynamic_cast<IdbLayerRouting*>(idb_layer_list->find_routing_layer(layer_idx));
          int32_t width = routing_layer->get_width();
          gds_wire_nodes->set_width(width);

          int32_t first_x = idb_segment->get_point_start()->get_x();
          int32_t first_y = idb_segment->get_point_start()->get_y();
          gds_wire_nodes->set_first(Point<int32_t>(first_x, first_y));

          int32_t second_x = idb_segment->get_point_end()->get_x();
          int32_t second_y = idb_segment->get_point_end()->get_y();
          gds_wire_nodes->set_second(Point<int32_t>(second_x, second_y));

          net_ptr->add_wire_nodes(gds_wire_nodes);
        }
      }
    }
    ieval_design->add_net(net_ptr);
  }
}

WLPin* DBWrapper::wrapWLPin(IdbPin* idb_pin)
{
  auto* ieval_design = _eval_db->_design;
  auto* idb_inst = idb_pin->get_instance();
  WLPin* pin_ptr = nullptr;

  if (!idb_inst) {
    pin_ptr = new WLPin();
    pin_ptr->set_name(idb_pin->get_pin_name());
    pin_ptr->set_type(PIN_TYPE::kIOPort);
  } else {
    std::string pin_name = idb_inst->get_name() + _db_config.get_separator() + idb_pin->get_pin_name();
    pin_ptr = new WLPin();
    pin_ptr->set_name(idb_pin->get_pin_name());
    pin_ptr->set_type(PIN_TYPE::kInstancePort);
  }

  LOG_ERROR_IF(!pin_ptr) << "Fail on creating ieval PIN!";

  // set pin io type.
  auto pin_direction = idb_pin->get_term()->get_direction();
  if (pin_direction == IdbConnectDirection::kInput) {
    pin_ptr->set_io_type(PIN_IO_TYPE::kInput);
  } else if (pin_direction == IdbConnectDirection::kOutput) {
    pin_ptr->set_io_type(PIN_IO_TYPE::kOutput);
  } else if (pin_direction == IdbConnectDirection::kInOut) {
    pin_ptr->set_io_type(PIN_IO_TYPE::kInputOutput);
  } else {
    pin_ptr->set_io_type(PIN_IO_TYPE::kNone);
  }

  // set pin center coordinate.
  pin_ptr->set_x(idb_pin->get_average_coordinate()->get_x());
  pin_ptr->set_y(idb_pin->get_average_coordinate()->get_y());

  ieval_design->add_pin(pin_ptr);
  _eval_db->_idb_pin_map.emplace(idb_pin, pin_ptr);
  _eval_db->_pin_idb_map.emplace(pin_ptr, idb_pin);

  return pin_ptr;
}

CongPin* DBWrapper::wrapCongPin(IdbPin* idb_pin)
{
  auto* ieval_design = _eval_db->_design;
  auto* idb_inst = idb_pin->get_instance();
  CongPin* pin_ptr = nullptr;

  if (!idb_inst) {
    pin_ptr = new CongPin();
    pin_ptr->set_name(idb_pin->get_pin_name());
    pin_ptr->set_type(PIN_TYPE::kIOPort);
  } else {
    std::string pin_name = idb_inst->get_name() + _db_config.get_separator() + idb_pin->get_pin_name();
    pin_ptr = new CongPin();
    pin_ptr->set_name(idb_pin->get_pin_name());
    pin_ptr->set_type(PIN_TYPE::kInstancePort);
    // set instance
    CongInst* inst = ieval_design->find_cong_inst(idb_inst->get_name());
    LOG_ERROR_IF(!inst) << idb_inst->get_name() << " is not found in ieval design!";
    inst->add_pin(pin_ptr);
  }

  LOG_ERROR_IF(!pin_ptr) << "Fail on creating ieval PIN!";

  pin_ptr->set_x(idb_pin->get_average_coordinate()->get_x());
  pin_ptr->set_y(idb_pin->get_average_coordinate()->get_y());
  ieval_design->add_pin(pin_ptr);

  return pin_ptr;
}

GDSPin* DBWrapper::wrapGDSPin(IdbPin* idb_pin)
{
  auto* idb_inst = idb_pin->get_instance();

  GDSPin* pin_ptr = nullptr;
  if (!idb_inst) {
    pin_ptr = new GDSPin();
    pin_ptr->set_name(idb_pin->get_pin_name());
  } else {
    std::string pin_name = idb_inst->get_name() + _db_config.get_separator() + idb_pin->get_pin_name();
    pin_ptr = new GDSPin();
    pin_ptr->set_name(idb_pin->get_pin_name());
  }

  pin_ptr->set_idx(idb_pin->get_id());

  int32_t coord_x = idb_pin->get_average_coordinate()->get_x();
  int32_t coord_y = idb_pin->get_average_coordinate()->get_y();
  pin_ptr->set_coord(Point<int32_t>(coord_x, coord_y));

  auto idb_port_list = idb_pin->get_port_box_list();
  for (size_t i = 0; i < idb_port_list.size(); ++i) {
    GDSPort* gds_port = new GDSPort();
    int32_t layer_idx = idb_port_list[i]->get_layer()->get_id();
    gds_port->set_layer_idx(layer_idx);
    auto idb_rect_list = idb_port_list[i]->get_rect_list();
    for (size_t j = 0; j < idb_rect_list.size(); ++j) {
      int32_t lx = idb_rect_list[j]->get_low_x();
      int32_t ly = idb_rect_list[j]->get_low_y();
      int32_t ux = idb_rect_list[j]->get_high_x();
      int32_t uy = idb_rect_list[j]->get_high_y();
      gds_port->add_rect(Rectangle<int32_t>(lx, ly, ux, uy));
    }
    pin_ptr->add_port(gds_port);
  }
  return pin_ptr;
}

std::string DBWrapper::fixSlash(std::string raw_str)
{
  std::regex re(R"(\\)");
  return std::regex_replace(raw_str, re, "");
}

}  // namespace eval

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
 * @Date: 2022-01-21 15:24:20
 * @LastEditTime: 2022-12-10 20:45:57
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/wrapper/IDBWrapper.cc
 * Contact : https://github.com/sjchanson
 */

#include "IDBWrapper.hh"

#include <cstdlib>
#include <regex>

#include "utility/Utility.hh"

namespace ipl {

IDBWrapper::IDBWrapper(IdbBuilder* idb_builder) : _idbw_database(new IDBWDatabase())
{
  _idbw_database->_idb_builder = idb_builder;
  wrapIDBData();
  correctInstanceOrient();
}

IDBWrapper::~IDBWrapper()
{
  delete _idbw_database;
}

void IDBWrapper::updateFromSourceDataBase()
{
  auto* ipl_design = _idbw_database->_design;
  auto* idb_design = _idbw_database->_idb_builder->get_def_service()->get_design();

  for (auto* idb_inst : idb_design->get_instance_list()->get_instance_list()) {
    std::string inst_name = fixSlash(idb_inst->get_name());

    auto* pl_inst = ipl_design->find_instance(inst_name);
    if (pl_inst) {
      // skip fixed inst
      if (pl_inst->isFixed()) {
        continue;
      }

      updatePLInstanceInfo(idb_inst, pl_inst);
      continue;
    }

    wrapIdbInstance(idb_inst);
  }

  correctInstanceOrient();

  for (auto* idb_net : idb_design->get_net_list()->get_net_list()) {
    wrapIdbNet(idb_net);
  }
}

void IDBWrapper::updateFromSourceDataBase(std::vector<std::string> inst_list)
{
  std::set<IdbInstance*> record_insts;
  std::set<IdbNet*> record_nets;

  // search for relative inst and net
  auto* ipl_design = _idbw_database->_design;
  auto* idb_design = _idbw_database->_idb_builder->get_def_service()->get_design();
  auto* idb_inst_list = idb_design->get_instance_list();
  for (std::string inst_name : inst_list) {
    IdbInstance* idb_inst = idb_inst_list->find_instance(inst_name);
    auto* pl_inst = ipl_design->find_instance(inst_name);

    if (pl_inst) {
      updatePLInstanceInfo(idb_inst, pl_inst);
      continue;
    }

    if (!idb_inst) {
      LOG_ERROR << "Instance: " << inst_name << " has not been add to IDB";
    }

    for (auto* inst_pin : idb_inst->get_pin_list()->get_pin_list()) {
      auto* idb_net = inst_pin->get_net();
      if (!idb_net) {
        continue;
      }

      record_nets.emplace(inst_pin->get_net());
    }
    record_insts.emplace(idb_inst);
  }

  for (auto* idb_inst : record_insts) {
    wrapIdbInstance(idb_inst);
  }

  correctInstanceOrient();

  for (auto* idb_net : record_nets) {
    wrapIdbNet(idb_net);
  }
}

bool IDBWrapper::wrapPartOfInstances(std::vector<std::string> inst_list)
{
  bool flag = true;
  auto* idb_design = _idbw_database->_idb_builder->get_def_service()->get_design();
  auto* idb_inst_list = idb_design->get_instance_list();
  auto* ipl_design = _idbw_database->_design;

  int32_t wrap_inst_cnt = 0;
  for (std::string inst_name : inst_list) {
    auto* ipl_inst = ipl_design->find_instance(inst_name);
    if (ipl_inst) {
      continue;
    }

    IdbInstance* idb_inst = idb_inst_list->find_instance(inst_name);
    if (!idb_inst) {
      flag = false;
      return flag;
    }
    wrap_inst_cnt++;
    wrapIdbInstance(idb_inst);
  }
  // LOG_INFO << "Succeed to add Insntace Count : " << wrap_inst_cnt;
  return flag;
}

void IDBWrapper::updatePLInstanceInfo(IdbInstance* idb_inst, Instance* pl_inst)
{
  // set orient.
  const IdbOrient idb_orient = idb_inst->get_orient();
  if (idb_orient == IdbOrient::kN_R0) {
    pl_inst->set_orient(Orient::kN_R0);
  } else if (idb_orient == IdbOrient::kS_R180) {
    pl_inst->set_orient(Orient::kS_R180);
  } else if (idb_orient == IdbOrient::kW_R90) {
    pl_inst->set_orient(Orient::kW_R90);
  } else if (idb_orient == IdbOrient::kE_R270) {
    pl_inst->set_orient(Orient::kE_R270);
  } else if (idb_orient == IdbOrient::kFN_MY) {
    pl_inst->set_orient(Orient::kFN_MY);
  } else if (idb_orient == IdbOrient::kFS_MX) {
    pl_inst->set_orient(Orient::kFS_MX);
  } else if (idb_orient == IdbOrient::kFW_MX90) {
    pl_inst->set_orient(Orient::kFW_MX90);
  } else if (idb_orient == IdbOrient::kFE_MY90) {
    pl_inst->set_orient(Orient::kFE_MY90);
  } else {
    // pl_inst->set_orient(Orient::kNone);
    pl_inst->set_orient(Orient::kN_R0);
  }

  // set coordi.
  if (!checkInCore(idb_inst) && !(idb_inst->get_status() == idb::IdbPlacementStatus::kFixed)) {
    pl_inst->update_coordi(-1, -1);
  } else {
    auto bbox = idb_inst->get_bounding_box();
    pl_inst->update_coordi(bbox->get_low_x(), bbox->get_low_y());
  }
}

bool IDBWrapper::wrapPartOfNetlists(std::vector<std::string> net_list)
{
  bool flag = true;
  auto* idb_design = _idbw_database->_idb_builder->get_def_service()->get_design();
  auto* idb_net_list = idb_design->get_net_list();
  auto* ipl_design = _idbw_database->_design;

  int32_t wrap_net_cnt = 0;
  for (std::string net_name : net_list) {
    IdbNet* idb_net = idb_net_list->find_net(net_name);
    if (!idb_net) {
      flag = false;
      return flag;
    }
    if (!ipl_design->find_net(net_name)) {
      wrap_net_cnt++;
    }

    wrapIdbNet(idb_net);
  }
  // LOG_INFO << "Succeed to add Net Count : " << wrap_net_cnt;

  return flag;
}

void IDBWrapper::updatePLNetInfo(IdbNet* idb_net, Net* pl_net)
{
  //
}

void IDBWrapper::wrapIDBData()
{
  IdbDefService* idb_def_service = _idbw_database->_idb_builder->get_def_service();
  IdbLayout* idb_layout = idb_def_service->get_layout();
  IdbDesign* idb_design = idb_def_service->get_design();

  wrapLayout(idb_layout);
  wrapDesign(idb_design);

  initInstancesForFragmentedRow();
}

void IDBWrapper::wrapLayout(IdbLayout* idb_layout)
{
  auto* ipl_layout = _idbw_database->_layout;

  // set dbu.
  int32_t database_unit = idb_layout->get_units()->get_micron_dbu();
  ipl_layout->set_database_unit(database_unit);

  // set die shape.
  IdbDie* idb_die = idb_layout->get_die();
  ipl_layout->set_die_shape(Rectangle<int32_t>(idb_die->get_llx(), idb_die->get_lly(), idb_die->get_urx(), idb_die->get_ury()));

  // set core shape.
  IdbCore* idb_core = idb_layout->get_core();
  IdbRect* idb_core_rect = idb_core->get_bounding_box();
  ipl_layout->set_core_shape(
      Rectangle<int32_t>(idb_core_rect->get_low_x(), idb_core_rect->get_low_y(), idb_core_rect->get_high_x(), idb_core_rect->get_high_y()));

  // wrap rows.
  wrapRows(idb_layout);

  // wrap cells.
  wrapCells(idb_layout);

  // wrap routing info.
  wrapRoutingInfo(idb_layout);
}

void IDBWrapper::wrapRows(IdbLayout* idb_layout)
{
  auto* ipl_layout = _idbw_database->_layout;

  IdbRows* idb_rows = idb_layout->get_rows();
  int32_t row_id = 0;

  Row* ipl_origin_row = nullptr;
  for (auto* idb_row : idb_rows->get_row_list()) {
    Row* ipl_row = new Row(idb_row->get_name());
    ipl_row->set_row_id(row_id++);
    IdbRect* idb_row_rect = idb_row->get_bounding_box();
    ipl_row->set_shape(
        Rectangle<int32_t>(idb_row_rect->get_low_x(), idb_row_rect->get_low_y(), idb_row_rect->get_high_x(), idb_row_rect->get_high_y()));

    // set site.
    Site* site_ptr = new Site(idb_row->get_site()->get_name());
    site_ptr->set_width(idb_row->get_site()->get_width());
    site_ptr->set_height(idb_row->get_site()->get_height());

    // set site orient.
    const IdbOrient idb_orient = idb_row->get_site()->get_orient();
    if (idb_orient == IdbOrient::kN_R0) {
      site_ptr->set_orient(Orient::kN_R0);
    } else if (idb_orient == IdbOrient::kS_R180) {
      site_ptr->set_orient(Orient::kS_R180);
    } else if (idb_orient == IdbOrient::kW_R90) {
      site_ptr->set_orient(Orient::kW_R90);
    } else if (idb_orient == IdbOrient::kE_R270) {
      site_ptr->set_orient(Orient::kE_R270);
    } else if (idb_orient == IdbOrient::kFN_MY) {
      site_ptr->set_orient(Orient::kFN_MY);
    } else if (idb_orient == IdbOrient::kFS_MX) {
      site_ptr->set_orient(Orient::kFS_MX);
    } else if (idb_orient == IdbOrient::kFW_MX90) {
      site_ptr->set_orient(Orient::kFW_MX90);
    } else if (idb_orient == IdbOrient::kFE_MY90) {
      site_ptr->set_orient(Orient::kFE_MY90);
    } else {
      site_ptr->set_orient(Orient::kNone);
    }

    ipl_row->set_site(site_ptr);
    ipl_row->set_site_num(idb_row->get_site_count());
    ipl_layout->add_row(ipl_row);

    if (ipl_origin_row == nullptr || ipl_row->get_coordi().get_y() < ipl_origin_row->get_coordi().get_y()) {
      ipl_origin_row = ipl_row;
    }
  }

  Orient even_orient = ipl_origin_row->get_orient();
  Orient odd_orient = (even_orient == Orient::kN_R0) ? Orient::kFS_MX : Orient::kN_R0;
  int32_t row_cnt_y = ipl_layout->get_core_height() / ipl_layout->get_row_height();
  for (int32_t i = 0; i < row_cnt_y; i++) {
    if (i % 2 == 0) {
      ipl_layout->add_row_orient(even_orient);
    } else {
      ipl_layout->add_row_orient(odd_orient);
    }
  }
}

void IDBWrapper::wrapCells(IdbLayout* idb_layout)
{
  auto* ipl_layout = _idbw_database->_layout;

  for (auto* idb_cell : idb_layout->get_cell_master_list()->get_cell_master()) {
    idb_cell->get_term_list();
    Cell* cell_ptr = new Cell(idb_cell->get_name());

    cell_ptr->set_width(idb_cell->get_width());
    cell_ptr->set_height(idb_cell->get_height());

    // set cell type.
    if (idb_cell->is_core()) {
      cell_ptr->set_type(CELL_TYPE::kLogic);
    } else if (idb_cell->is_block()) {
      cell_ptr->set_type(CELL_TYPE::kMacro);
    } else if (idb_cell->is_pad_filler() || idb_cell->is_endcap() || idb_cell->is_core_filler()) {
      cell_ptr->set_type(CELL_TYPE::kPhysicalFiller);
    } else {
      cell_ptr->set_type(CELL_TYPE::kNone);
    }

    // lable io cell from idb.
    if (idb_cell->is_pad() || idb_cell->is_endcap()) {
      cell_ptr->set_type(CELL_TYPE::kIOCell);
    }

    // set inpin/outpin name list
    for (auto* idb_term : idb_cell->get_term_list()) {
      auto idb_direction = idb_term->get_direction();
      if (idb_direction == IdbConnectDirection::kInput) {
        cell_ptr->add_inpin_name(idb_term->get_name());
      }

      if (idb_direction == IdbConnectDirection::kOutput) {
        cell_ptr->add_outpin_name(idb_term->get_name());
      }
    }

    ipl_layout->add_cell(cell_ptr);
  }
}

void IDBWrapper::wrapRoutingInfo(IdbLayout* idb_layout)
{
  auto* ipl_layout = _idbw_database->_layout;

  int route_cap_h = 0;
  int route_cap_v = 0;
  int partial_route_cap_h = 0;
  int partial_route_cap_v = 0;
  int count_h = 0;
  int count_v = 0;

  IdbLayers* idb_layers = idb_layout->get_layers();
  int route_layer_num = idb_layers->get_routing_layers_number();
  for (int i = 0; i < route_layer_num; ++i) {
    IdbLayerRouting* layer = dynamic_cast<IdbLayerRouting*>(idb_layers->find_routing_layer(i));
    bool is_hor = layer->is_horizontal();
    if (is_hor) {
      count_h++;
      route_cap_h += layer->get_prefer_track_grid()->get_track_num();
      if (count_h == 2) {
        partial_route_cap_h = route_cap_h;
      }
    } else {
      count_v++;
      route_cap_v += layer->get_prefer_track_grid()->get_track_num();
      if (count_v == 2) {
        partial_route_cap_v = route_cap_v;
      }
    }
  }

  ipl_layout->set_route_cap_h(route_cap_h);
  ipl_layout->set_route_cap_v(route_cap_v);
  ipl_layout->set_partial_route_cap_h(partial_route_cap_h);
  ipl_layout->set_partial_route_cap_v(partial_route_cap_v);
}

void IDBWrapper::wrapDesign(IdbDesign* idb_design)
{
  auto* ipl_design = _idbw_database->_design;

  // set design name.
  const std::string& design_name = idb_design->get_design_name();
  ipl_design->set_design_name(design_name);

  // set instances.
  wrapInstances(idb_design);

  // set netlists.
  wrapNetlists(idb_design);
  searchForDontCareNet();

  // FOR DEBUG.
  // deleteInstsForTest();

  // set regions.
  wrapRegions(idb_design);
}

void IDBWrapper::deleteInstsForTest()
{
  std::vector<Instance*> empty_inst_record;
  std::map<ipl::Instance*, idb::IdbInstance*> new_map;

  _idbw_database->_design->deleteInstForTest();
  for (auto pair : _idbw_database->_idb_inst_map) {
    auto* inst = pair.first;
    if (inst->get_pins().size() == 0) {
      empty_inst_record.push_back(inst);
    } else {
      new_map.emplace(inst, pair.second);
    }
  }

  _idbw_database->_idb_inst_map = new_map;

  for (auto* inst : empty_inst_record) {
    delete inst;
  }
}

void IDBWrapper::wrapIdbInstance(IdbInstance* idb_inst)
{
  auto* ipl_design = _idbw_database->_design;
  auto* ipl_layout = _idbw_database->_layout;

  std::string inst_name = fixSlash(idb_inst->get_name());
  Instance* inst_ptr = new Instance(inst_name);

  Cell* cell_ptr = ipl_layout->find_cell(idb_inst->get_cell_master()->get_name());
  LOG_ERROR_IF(!cell_ptr) << "Cell Master has not been created!";
  inst_ptr->set_cell_master(cell_ptr);
  if ("IO_BOND_pad_vddio_e0" == inst_name) {
    LOG_INFO << "IO_BOND_pad_vddio_e0";
  }
  // set instace coordinate.
  auto bbox = idb_inst->get_bounding_box();
  inst_ptr->set_shape(bbox->get_low_x(), bbox->get_low_y(), bbox->get_high_x(), bbox->get_high_y());

  // set instance state.
  if (idb_inst->is_unplaced() || idb_inst->get_status() == IdbPlacementStatus::kNone) {
    inst_ptr->set_shape(-1, -1, -1, -1);  // set an unlegal coordinate.
    inst_ptr->set_instance_state(INSTANCE_STATE::KUnPlaced);
  } else if (idb_inst->is_placed()) {
    inst_ptr->set_instance_state(INSTANCE_STATE::KUnPlaced);  // tmp for read placed case.
    // inst_ptr->set_instance_state(INSTANCE_STATE::kPlaced);
  } else if (idb_inst->is_fixed()) {
    inst_ptr->set_instance_state(INSTANCE_STATE::kFixed);
  } else {
    inst_ptr->set_instance_state(INSTANCE_STATE::kNone);
  }

  // set type.
  if (!isCoreOverlap(idb_inst)) {
    if (idb_inst->is_fixed()) {
      inst_ptr->set_instance_type(INSTANCE_TYPE::kOutside);
    } else {
      if (cell_ptr->isIOCell()) {
        inst_ptr->set_instance_type(INSTANCE_TYPE::kOutside);
      } else if (!checkInCore(idb_inst)) {
        inst_ptr->set_shape(-1, -1, -1, -1);  // set an unlegal coordinate.
      }
    }
  } else {
    if (idb_inst->is_fixed() && isCrossInst(idb_inst)) {
      inst_ptr->set_instance_type(INSTANCE_TYPE::kCross);
      Rectangle<int32_t> cross_shape = obtainCrossRect(idb_inst, ipl_layout->get_core_shape());
      inst_ptr->set_shape(cross_shape.get_ll_x(), cross_shape.get_ll_y(), cross_shape.get_ur_x(), cross_shape.get_ur_y());
      cell_ptr->set_height(cross_shape.get_ur_y() - cross_shape.get_ll_y());
      cell_ptr->set_width(cross_shape.get_ur_x() - cross_shape.get_ll_x());
    } else {
      inst_ptr->set_instance_type(INSTANCE_TYPE::kNormal);
    }
  }

  if (cell_ptr->isIOCell()) {
    if (!inst_ptr->isOutsideInstance()) {
      LOG_WARNING << "Inst: " << inst_ptr->get_name() << " is io_cell but it is not fixed outside. " << "Shape: "
                  << " (" << bbox->get_low_x() << "," << bbox->get_low_y() << ") "
                  << " (" << bbox->get_high_x() << "," << bbox->get_high_y() << ")"
                  << " iPL regard as "
                  << " (" << inst_ptr->get_coordi().get_x() << "," << inst_ptr->get_coordi().get_y() << ") "
                  << " (" << inst_ptr->get_shape().get_ur_x() << "," << inst_ptr->get_shape().get_ur_y() << ")";
    }
  }

  // set orient.
  const IdbOrient idb_orient = idb_inst->get_orient();
  if (idb_orient == IdbOrient::kN_R0) {
    inst_ptr->set_orient(Orient::kN_R0);
  } else if (idb_orient == IdbOrient::kS_R180) {
    inst_ptr->set_orient(Orient::kS_R180);
  } else if (idb_orient == IdbOrient::kW_R90) {
    inst_ptr->set_orient(Orient::kW_R90);
  } else if (idb_orient == IdbOrient::kE_R270) {
    inst_ptr->set_orient(Orient::kE_R270);
  } else if (idb_orient == IdbOrient::kFN_MY) {
    inst_ptr->set_orient(Orient::kFN_MY);
  } else if (idb_orient == IdbOrient::kFS_MX) {
    inst_ptr->set_orient(Orient::kFS_MX);
  } else if (idb_orient == IdbOrient::kFW_MX90) {
    inst_ptr->set_orient(Orient::kFW_MX90);
  } else if (idb_orient == IdbOrient::kFE_MY90) {
    inst_ptr->set_orient(Orient::kFE_MY90);
  } else {
    // inst_ptr->set_orient(Orient::kNone);
    inst_ptr->set_orient(Orient::kN_R0);
  }

  // cover cell type.
  // TODO : where is clock buffer?
  if (idb_inst->get_cell_master()->is_block()) {
    cell_ptr->set_type(CELL_TYPE::kMacro);
  } else if (inst_ptr->isOutsideInstance()) {
    cell_ptr->set_type(CELL_TYPE::kIOCell);
  } else if (idb_inst->is_flip_flop()) {
    cell_ptr->set_type(CELL_TYPE::kFlipflop);
  } else if (idb_inst->get_type() == IdbInstanceType::kDist) {
    cell_ptr->set_type(CELL_TYPE::kPhysicalFiller);
  }

  // tmp set the halo as blockage.
  if (idb_inst->has_halo()) {
    auto core_shape = ipl_layout->get_core_shape();
    auto* inst_halo = idb_inst->get_halo();
    int32_t ll_x = inst_ptr->get_shape().get_ll_x() - inst_halo->get_extend_lef();
    int32_t ll_y = inst_ptr->get_shape().get_ll_y() - inst_halo->get_extend_bottom();
    int32_t ur_x = inst_ptr->get_shape().get_ur_x() + inst_halo->get_extend_right();
    int32_t ur_y = inst_ptr->get_shape().get_ur_y() + inst_halo->get_extend_top();

    ll_x < core_shape.get_ll_x() ? ll_x = core_shape.get_ll_x() : ll_x;
    ll_y < core_shape.get_ll_y() ? ll_y = core_shape.get_ll_y() : ll_y;
    ur_x > core_shape.get_ur_x() ? ur_x = core_shape.get_ur_x() : ur_x;
    ur_y > core_shape.get_ur_y() ? ur_y = core_shape.get_ur_y() : ur_y;

    if ((ll_x < ur_x) && (ll_y < ur_y)) {
      Region* region_ptr = new Region(idb_inst->get_name() + "_HALO");
      region_ptr->set_type(REGION_TYPE::kFence);
      region_ptr->add_boundary(Rectangle<int32_t>(ll_x, ll_y, ur_x, ur_y));
      ipl_design->add_region(region_ptr);
    }
  }

  ipl_design->add_instance(inst_ptr);
  _idbw_database->_ipl_inst_map.emplace(idb_inst, inst_ptr);
  _idbw_database->_idb_inst_map.emplace(inst_ptr, idb_inst);
}

void IDBWrapper::wrapInstances(IdbDesign* idb_design)
{
  for (auto* idb_inst : idb_design->get_instance_list()->get_instance_list()) {
    wrapIdbInstance(idb_inst);
  }
}

void IDBWrapper::wrapIdbNet(IdbNet* idb_net)
{
  auto* ipl_design = _idbw_database->_design;
  std::string net_name = fixSlash(idb_net->get_net_name());

  auto* ipl_net = ipl_design->find_net(net_name);
  if (ipl_net) {
    ipl_net->disConnectLoadPins();
    for (auto* idb_load_pin : idb_net->get_load_pins()) {
      Pin* pin_ptr = wrapPin(idb_load_pin);
      pin_ptr->set_net(ipl_net);
      ipl_net->add_sink_pin(pin_ptr);
    }
  } else {
    Net* net_ptr = new Net(net_name);

    // set net type.
    auto connect_type = idb_net->get_connect_type();
    if (connect_type == IdbConnectType::kSignal) {
      net_ptr->set_net_type(NET_TYPE::kSignal);
    } else if (connect_type == IdbConnectType::kClock) {
      net_ptr->set_net_type(NET_TYPE::kClock);
    } else if (connect_type == IdbConnectType::kReset) {
      net_ptr->set_net_type(NET_TYPE::kReset);
    } else {
      // net_ptr->set_net_type(NET_TYPE::kNone);
      net_ptr->set_net_type(NET_TYPE::kSignal);
    }

    // set net state.
    net_ptr->set_net_state(NET_STATE::kNormal);

    // set pins.
    auto* idb_driving_pin = idb_net->get_driving_pin();
    if (idb_driving_pin) {
      Pin* pin_ptr = wrapPin(idb_driving_pin);
      pin_ptr->set_net(net_ptr);
      net_ptr->set_driver_pin(pin_ptr);
    }

    for (auto* idb_load_pin : idb_net->get_load_pins()) {
      Pin* pin_ptr = wrapPin(idb_load_pin);
      pin_ptr->set_net(net_ptr);
      net_ptr->add_sink_pin(pin_ptr);
    }

    ipl_design->add_net(net_ptr);
    _idbw_database->_ipl_net_map.emplace(idb_net, net_ptr);
    _idbw_database->_idb_net_map.emplace(net_ptr, idb_net);
  }
}

void IDBWrapper::wrapNetlists(IdbDesign* idb_design)
{
  for (auto* idb_net : idb_design->get_net_list()->get_net_list()) {
    wrapIdbNet(idb_net);
  }
}

Pin* IDBWrapper::wrapPin(IdbPin* idb_pin)
{
  auto* ipl_design = _idbw_database->_design;
  auto* idb_inst = idb_pin->get_instance();
  Pin* pin_ptr = nullptr;

  if (!idb_inst) {
    std::string pin_name = idb_pin->get_pin_name();

    auto* ipl_pin = ipl_design->find_pin(pin_name);
    if (ipl_pin) {
      return ipl_pin;
    }

    pin_ptr = new Pin(idb_pin->get_pin_name());
    // set pin type.
    pin_ptr->set_pin_type(PIN_TYPE::kIOPort);
    // set pin offset coordinate.
    pin_ptr->set_offset_coordi(0, 0);

  } else {
    std::string inst_name = fixSlash(idb_inst->get_name());
    std::string pin_name = inst_name + ":" + idb_pin->get_pin_name();

    auto* ipl_pin = ipl_design->find_pin(pin_name);
    if (ipl_pin) {
      return ipl_pin;
    }

    pin_ptr = new Pin(pin_name);
    // set pin type.
    pin_ptr->set_pin_type(PIN_TYPE::kInstancePort);

    // set instance.
    Instance* ipl_inst = ipl_design->find_instance(inst_name);
    LOG_ERROR_IF(!ipl_inst) << idb_inst->get_name() + " has not been wraped by IDBWrapper!";
    pin_ptr->set_instance(ipl_inst);
    ipl_inst->add_pin(pin_ptr);

    // set pin offset coordinate.
    auto term_avg_position = idb_pin->get_term()->get_average_position();
    pin_ptr->set_offset_coordi(term_avg_position.get_x() - ipl_inst->get_cell_master()->get_width() / 2,
                               term_avg_position.get_y() - ipl_inst->get_cell_master()->get_height() / 2);
  }

  LOG_ERROR_IF(!pin_ptr) << "Fail on creating ipl PIN!";

  // set pin io type.
  auto pin_direction = idb_pin->get_term()->get_direction();
  if (pin_direction == IdbConnectDirection::kInput) {
    pin_ptr->set_pin_io_type(PIN_IO_TYPE::kInput);
  } else if (pin_direction == IdbConnectDirection::kOutput) {
    pin_ptr->set_pin_io_type(PIN_IO_TYPE::kOutput);
  } else if (pin_direction == IdbConnectDirection::kInOut) {
    pin_ptr->set_pin_io_type(PIN_IO_TYPE::kInputOutput);
  } else {
    pin_ptr->set_pin_io_type(PIN_IO_TYPE::kNone);
  }

  // set pin center coordinate.
  pin_ptr->set_center_coordi(idb_pin->get_average_coordinate()->get_x(), idb_pin->get_average_coordinate()->get_y());

  ipl_design->add_pin(pin_ptr);
  _idbw_database->_ipl_pin_map.emplace(idb_pin, pin_ptr);
  _idbw_database->_idb_pin_map.emplace(pin_ptr, idb_pin);

  return pin_ptr;
}

void IDBWrapper::wrapRegions(IdbDesign* idb_design)
{
  auto* ipl_design = _idbw_database->_design;
  Rectangle<int32_t> core_shape = this->get_layout()->get_core_shape();

  // Region.
  for (auto* idb_region : idb_design->get_region_list()->get_region_list()) {
    Region* region_ptr = new Region(idb_region->get_name());

    // set type.
    auto region_type = idb_region->get_type();
    if (region_type == IdbRegionType::kFence) {
      region_ptr->set_type(REGION_TYPE::kFence);
    } else if (region_type == IdbRegionType::kGuide) {
      region_ptr->set_type(REGION_TYPE::kGuide);
    } else {
      region_ptr->set_type(REGION_TYPE::kNone);
    }

    // add boundaries.
    for (auto* boundary : idb_region->get_boundary()) {
      region_ptr->add_boundary(
          Rectangle<int32_t>(boundary->get_low_x(), boundary->get_low_y(), boundary->get_high_x(), boundary->get_high_y()));
    }

    // add instances.
    for (auto* idb_inst : idb_region->get_instance_list()) {
      Instance* ipl_inst = ipl_design->find_instance(idb_inst->get_name());
      LOG_ERROR_IF(!ipl_inst) << idb_inst->get_name() + " has not been wraped by IDBWrapper!";
      region_ptr->add_instance(ipl_inst);
      ipl_inst->set_belong_region(region_ptr);
    }

    ipl_design->add_region(region_ptr);
  }

  // Blockage.
  int blockage_num = 0;
  for (auto* blockage_list : idb_design->get_blockage_list()->get_blockage_list()) {
    if (blockage_list->is_palcement_blockage()) {
      Region* blockage_ptr = new Region("blockage_list" + std::to_string(blockage_num++));
      blockage_ptr->set_type(REGION_TYPE::kFence);

      // add boundaries.
      for (auto* rect : blockage_list->get_rect_list()) {
        // fix out of core blockage.
        int32_t lx = (rect->get_low_x() < core_shape.get_ll_x() ? core_shape.get_ll_x() : rect->get_low_x());
        int32_t ly = (rect->get_low_y() < core_shape.get_ll_y() ? core_shape.get_ll_y() : rect->get_low_y());
        int32_t ux = (rect->get_high_x() > core_shape.get_ur_x() ? core_shape.get_ur_x() : rect->get_high_x());
        int32_t uy = (rect->get_high_y() > core_shape.get_ur_y() ? core_shape.get_ur_y() : rect->get_high_y());

        blockage_ptr->add_boundary(Rectangle<int32_t>(lx, ly, ux, uy));
      }

      ipl_design->add_region(blockage_ptr);
    }
  }
}

void IDBWrapper::writeBackSourceDatabase()
{
  for (auto* inst : _idbw_database->_design->get_instance_list()) {
    if (inst->isFakeInstance()) {
      continue;
    }

    // iPL should not change fixed instances.
    if (inst->isFixed()) {
      continue;
    }

    IdbInstance* idb_inst = nullptr;
    auto idb_inst_iter = _idbw_database->_idb_inst_map.find(inst);
    if (idb_inst_iter != _idbw_database->_idb_inst_map.end()) {
      idb_inst = idb_inst_iter->second;
    }

    if (idb_inst) {
      // set state.
      if (inst->isFixed()) {
        idb_inst->set_status(IdbPlacementStatus::kFixed);
      } else if (inst->isUnPlaced() || inst->isPlaced()) {
        idb_inst->set_status(IdbPlacementStatus::kPlaced);
      } else {
        idb_inst->set_status(IdbPlacementStatus::kNone);
      }
      // if (inst->isUnPlaced()) {
      //   // idb_inst->set_status(IdbPlacementStatus::kUnplaced);
      // } else if (inst->isPlaced()) {
      //   idb_inst->set_status(IdbPlacementStatus::kPlaced);
      // } else if (inst->isFixed()) {
      //   idb_inst->set_status(IdbPlacementStatus::kFixed);
      // } else {
      //   idb_inst->set_status(IdbPlacementStatus::kNone);
      // }

      // set orient.
      auto inst_orient = inst->get_orient();
      if (inst_orient == Orient::kN_R0) {
        idb_inst->set_orient(IdbOrient::kN_R0);
      } else if (inst_orient == Orient::kS_R180) {
        idb_inst->set_orient(IdbOrient::kS_R180);
      } else if (inst_orient == Orient::kW_R90) {
        idb_inst->set_orient(IdbOrient::kW_R90);
      } else if (inst_orient == Orient::kE_R270) {
        idb_inst->set_orient(IdbOrient::kE_R270);
      } else if (inst_orient == Orient::kFN_MY) {
        idb_inst->set_orient(IdbOrient::kFN_MY);
      } else if (inst_orient == Orient::kFS_MX) {
        idb_inst->set_orient(IdbOrient::kFS_MX);
      } else if (inst_orient == Orient::kFW_MX90) {
        idb_inst->set_orient(IdbOrient::kFW_MX90);
      } else if (inst_orient == Orient::kFE_MY90) {
        idb_inst->set_orient(IdbOrient::kFE_MY90);
      } else {
        idb_inst->set_orient(IdbOrient::kNone);
      }

      // set coordi.
      idb_inst->set_coodinate(inst->get_coordi().get_x(), inst->get_coordi().get_y());
    } else {
      auto* idb_new_inst
          = _idbw_database->get_idb_builder()->get_def_service()->get_design()->get_instance_list()->add_instance(inst->get_name());
      auto* idb_cell_master = _idbw_database->get_idb_builder()->get_lef_service()->get_layout()->get_cell_master_list()->find_cell_master(
          inst->get_cell_master()->get_name());
      // set cell master.
      idb_new_inst->set_cell_master(idb_cell_master);

      // set cell name

      // set state.
      if (inst->isFixed()) {
        idb_new_inst->set_status(IdbPlacementStatus::kFixed);
      } else if (inst->isUnPlaced() || inst->isPlaced()) {
        idb_new_inst->set_status(IdbPlacementStatus::kPlaced);
      } else {
        idb_new_inst->set_status(IdbPlacementStatus::kNone);
      }

      // set orient.
      auto inst_orient = inst->get_orient();
      if (inst_orient == Orient::kN_R0) {
        idb_new_inst->set_orient(IdbOrient::kN_R0);
      } else if (inst_orient == Orient::kS_R180) {
        idb_new_inst->set_orient(IdbOrient::kS_R180);
      } else if (inst_orient == Orient::kW_R90) {
        idb_new_inst->set_orient(IdbOrient::kW_R90);
      } else if (inst_orient == Orient::kE_R270) {
        idb_new_inst->set_orient(IdbOrient::kE_R270);
      } else if (inst_orient == Orient::kFN_MY) {
        idb_new_inst->set_orient(IdbOrient::kFN_MY);
      } else if (inst_orient == Orient::kFS_MX) {
        idb_new_inst->set_orient(IdbOrient::kFS_MX);
      } else if (inst_orient == Orient::kFW_MX90) {
        idb_new_inst->set_orient(IdbOrient::kFW_MX90);
      } else if (inst_orient == Orient::kFE_MY90) {
        idb_new_inst->set_orient(IdbOrient::kFE_MY90);
      } else {
      }

      // set coordi.
      idb_new_inst->set_coodinate(inst->get_coordi().get_x(), inst->get_coordi().get_y());
    }
  }
}

void IDBWrapper::writeDef(std::string file_name = "")
{
  if (const auto ret = _idbw_database->_idb_builder->saveDef("./" + file_name); !ret) {
    LOG_FATAL << "Fail to write DEF file!";
  }
}

std::string IDBWrapper::fixSlash(std::string raw_str)
{
  std::regex re(R"(\\)");
  return std::regex_replace(raw_str, re, "");
}

Point<int32_t> IDBWrapper::calMedianOffsetFromCellCenter(IdbPin* idb_pin)
{
  auto* idb_inst = idb_pin->get_instance();
  if (!idb_inst) {
    LOG_ERROR << "Current pin: " + idb_pin->get_pin_name() + " does not belong to any instance.";
  }

  auto* cell_master = idb_inst->get_cell_master();
  int32_t cell_half_x = cell_master->get_width() / 2;
  int32_t cell_half_y = cell_master->get_height() / 2;

  int32_t offset_lower_x = INT32_MAX;
  int32_t offset_lower_y = INT32_MAX;
  int32_t offset_upper_x = INT32_MIN;
  int32_t offset_upper_y = INT32_MIN;
  for (auto* port : idb_pin->get_term()->get_port_list()) {
    for (auto shape : port->get_layer_shape()) {
      for (auto idb_rect : shape->get_rect_list()) {
        offset_lower_x = std::min(idb_rect->get_low_x(), offset_lower_x);
        offset_lower_y = std::min(idb_rect->get_low_y(), offset_lower_y);
        offset_upper_x = std::max(idb_rect->get_high_x(), offset_upper_x);
        offset_upper_y = std::max(idb_rect->get_high_y(), offset_upper_y);
      }
    }
  }

  int32_t offset_center_x = (offset_lower_x + offset_upper_x) / 2 - cell_half_x;
  int32_t offset_center_y = (offset_lower_y + offset_upper_y) / 2 - cell_half_y;

  return Point<int32_t>(offset_center_x, offset_center_y);
}

Point<int32_t> IDBWrapper::calAverageOffsetFromCellCenter(IdbPin* idb_pin)
{
  auto* idb_inst = idb_pin->get_instance();
  if (!idb_inst) {
    LOG_ERROR << "Current pin: " + idb_pin->get_pin_name() + " does not belong to any instance.";
  }

  auto* cell_master = idb_inst->get_cell_master();
  int32_t cell_half_x = cell_master->get_width() / 2;
  int32_t cell_half_y = cell_master->get_height() / 2;

  int32_t coordinate_x = 0;
  int32_t coordinate_y = 0;
  int32_t layer_num = 0;
  for (auto* port : idb_pin->get_term()->get_port_list()) {
    for (auto shape : port->get_layer_shape()) {
      for (auto idb_rect : shape->get_rect_list()) {
        coordinate_x += (idb_rect->get_low_x() + idb_rect->get_high_x()) / 2;
        coordinate_y += (idb_rect->get_low_y() + idb_rect->get_high_y()) / 2;
        layer_num++;
      }
    }
  }

  int32_t offset_center_x = coordinate_x / layer_num - cell_half_x;
  int32_t offset_center_y = coordinate_y / layer_num - cell_half_y;

  return Point<int32_t>(offset_center_x, offset_center_y);
}

bool IDBWrapper::isCoreOverlap(IdbInstance* idb_inst)
{
  Point<int32_t> die_lower = this->get_layout()->get_die_shape().get_lower_left();
  Point<int32_t> die_upper = this->get_layout()->get_die_shape().get_upper_right();
  Point<int32_t> core_lower = this->get_layout()->get_core_shape().get_lower_left();
  Point<int32_t> core_upper = this->get_layout()->get_core_shape().get_upper_right();

  auto* bounding_box = idb_inst->get_bounding_box();
  if ((bounding_box->get_low_x() >= die_lower.get_x() && bounding_box->get_high_x() <= core_lower.get_x())
      || (bounding_box->get_low_x() >= core_upper.get_x() && bounding_box->get_high_x() <= die_upper.get_x())
      || (bounding_box->get_low_y() >= die_lower.get_y() && bounding_box->get_high_y() <= core_lower.get_y())
      || (bounding_box->get_low_y() >= core_upper.get_y() && bounding_box->get_high_y() <= die_upper.get_y())) {
    return false;
  } else {
    if (bounding_box->get_low_x() >= die_upper.get_x() || bounding_box->get_high_x() <= die_lower.get_x()
        || bounding_box->get_low_y() >= die_upper.get_y() || bounding_box->get_high_y() <= die_lower.get_y()) {
      return false;
    }
    return true;
  }
}

bool IDBWrapper::isCrossInst(IdbInstance* idb_inst)
{
  Point<int32_t> core_lower = this->get_layout()->get_core_shape().get_lower_left();
  Point<int32_t> core_upper = this->get_layout()->get_core_shape().get_upper_right();
  auto* bounding_box = idb_inst->get_bounding_box();

  if ((bounding_box->get_low_x() < core_lower.get_x() && bounding_box->get_high_x() > core_lower.get_x())
      || (bounding_box->get_low_y() < core_lower.get_y() && bounding_box->get_high_y() > core_lower.get_y())
      || (bounding_box->get_high_x() > core_upper.get_x() && bounding_box->get_low_x() < core_upper.get_x())
      || (bounding_box->get_high_y() > core_upper.get_y() && bounding_box->get_low_y() < core_upper.get_y())) {
    return true;
  } else {
    return false;
  }
}

Rectangle<int32_t> IDBWrapper::obtainCrossRect(IdbInstance* idb_inst, Rectangle<int32_t> core_shape)
{
  auto* bounding_box = idb_inst->get_bounding_box();

  int32_t min_x = std::max(bounding_box->get_low_x(), core_shape.get_ll_x());
  int32_t min_y = std::max(bounding_box->get_low_y(), core_shape.get_ll_y());
  int32_t max_x = std::min(bounding_box->get_high_x(), core_shape.get_ur_x());
  int32_t max_y = std::min(bounding_box->get_high_y(), core_shape.get_ur_y());

  if (min_x > max_x || min_y > max_y) {
    LOG_ERROR << "Cross instance is out of core boundary.";
    exit(0);
  }
  return Rectangle<int32_t>(min_x, min_y, max_x, max_y);
}

bool IDBWrapper::checkInCore(IdbInstance* idb_inst)
{
  Point<int32_t> core_lower = this->get_layout()->get_core_shape().get_lower_left();
  Point<int32_t> core_upper = this->get_layout()->get_core_shape().get_upper_right();

  auto* bounding_box = idb_inst->get_bounding_box();
  if (bounding_box->get_low_x() >= core_lower.get_x() && bounding_box->get_low_y() >= core_lower.get_y()
      && bounding_box->get_high_x() <= core_upper.get_x() && bounding_box->get_high_y() <= core_upper.get_y()) {
    return true;
  } else {
    return false;
  }
}

void IDBWrapper::searchForDontCareNet()
{
  for (auto* net : _idbw_database->_design->get_net_list()) {
    for (auto* pin : net->get_pins()) {
      if (pin->isIOPort()) {
        // detect if all other insts is fixed.
        bool is_fixed_connection = true;
        for (auto* target_pin : net->get_pins()) {
          auto* pin_inst = target_pin->get_instance();
          if (pin_inst && !pin_inst->isFixed()) {
            is_fixed_connection = false;
            break;
          }
        }

        if (is_fixed_connection) {
          net->set_net_state(NET_STATE::kDontCare);
          net->set_netweight(0.0f);
        }

        break;
      }
    }
  }
}

void IDBWrapper::saveVerilogForDebug(std::string path)
{
  // _idbw_database->_idb_builder->saveVerilog(path.c_str(),false,false);
}

void IDBWrapper::initInstancesForFragmentedRow()
{
  auto* ipl_layout = _idbw_database->_layout;
  auto* ipl_design = _idbw_database->_design;
  Utility utility;

  Point<int32_t> core_lower = ipl_layout->get_core_shape().get_lower_left();
  int32_t row_height = ipl_layout->get_row_height();
  int32_t site_width = ipl_layout->get_site_width();
  int32_t site_count_x = ipl_layout->get_core_width() / site_width;
  int32_t site_count_y = ipl_layout->get_core_height() / row_height;

  enum SiteRecord
  {
    kEmpty,
    kRowHolding,
    kFixedInst
  };
  std::vector<SiteRecord> site_grid(site_count_x * site_count_y, SiteRecord::kEmpty);

  // fill in rows area.
  std::vector<Row*> row_list = ipl_layout->get_row_list();
  for (auto* row : row_list) {
    Rectangle<int32_t> row_shape = row->get_shape();
    std::pair<int, int> pair_x = utility.obtainMinMaxIdx(core_lower.get_x(), site_width, row_shape.get_ll_x(), row_shape.get_ur_x());
    std::pair<int, int> pair_y = utility.obtainMinMaxIdx(core_lower.get_y(), row_height, row_shape.get_ll_y(), row_shape.get_ur_y());

    for (int i = pair_x.first; i < pair_x.second; ++i) {
      for (int j = pair_y.first; j < pair_y.second; ++j) {
        LOG_FATAL_IF((j * site_count_x + i) >= static_cast<int32_t>(site_grid.size()))
            << "Row : " << row->get_name() << " is out of core boundary.";
        site_grid.at(j * site_count_x + i) = kRowHolding;
      }
    }
  }

  // fill in fixed instance area.
  for (auto inst : ipl_design->get_instance_list()) {
    if (inst->isOutsideInstance()) {
      continue;
    }

    if (inst->isFixed()) {
      Rectangle<int32_t> inst_shape = inst->get_shape();
      std::pair<int, int> pair_x = utility.obtainMinMaxIdx(core_lower.get_x(), site_width, inst_shape.get_ll_x(), inst_shape.get_ur_x());
      std::pair<int, int> pair_y = utility.obtainMinMaxIdx(core_lower.get_y(), row_height, inst_shape.get_ll_y(), inst_shape.get_ur_y());

      // keep legal range

      for (int i = pair_x.first; i < pair_x.second; ++i) {
        for (int j = pair_y.first; j < pair_y.second; ++j) {
          LOG_FATAL_IF((j * site_count_x + i) >= static_cast<int32_t>(site_grid.size()))
              << "Inst : " << inst->get_name() << " (" << inst->get_shape().get_ll_x() << "," << inst->get_shape().get_ll_y() << " "
              << inst->get_shape().get_ur_x() << "," << inst->get_shape().get_ur_y() << ")"
              << " is out of core boundary "
              << " (" << ipl_layout->get_core_shape().get_ll_x() << "," << ipl_layout->get_core_shape().get_ll_y() << " "
              << ipl_layout->get_core_shape().get_ur_x() << "," << ipl_layout->get_core_shape().get_ur_y() << ")";
          site_grid.at(j * site_count_x + i) = kFixedInst;
        }
      }
    }
  }

  int dummy_count = 0;
  for (int j = 0; j < site_count_y; ++j) {
    for (int i = 0; i < site_count_x; ++i) {
      if (site_grid.at(j * site_count_x + i) == kEmpty) {
        int start_x = i;
        while (i < site_count_x && site_grid.at(j * site_count_x + i) == kEmpty) {
          ++i;
        }
        int end_x = i;
        Instance* dummy = new Instance("dummy_" + std::to_string(dummy_count++));

        int32_t dummy_llx = core_lower.get_x() + site_width * start_x;
        int32_t dummy_lly = core_lower.get_y() + row_height * j;
        int32_t dummy_urx = core_lower.get_x() + site_width * end_x;
        int32_t dummy_ury = core_lower.get_y() + row_height * (j + 1);

        dummy->set_shape(dummy_llx, dummy_lly, dummy_urx, dummy_ury);
        dummy->set_instance_type(INSTANCE_TYPE::kFakeInstance);
        dummy->set_instance_state(INSTANCE_STATE::kFixed);

        ipl_design->add_instance(dummy);
      }
    }
  }
}

void IDBWrapper::correctInstanceOrient()
{
  for (auto* inst : _idbw_database->_design->get_instance_list()) {
    if (inst->isFixed() || inst->isOutsideInstance() || inst->isFakeInstance()) {
      continue;
    }

    auto orient = inst->get_orient();
    if (orient == Orient::kW_R90 || orient == Orient::kE_R270 || orient == Orient::kFW_MX90 || orient == Orient::kFE_MY90) {
      inst->set_orient(Orient::kN_R0);
      inst->update_coordi(inst->get_coordi().get_x(), inst->get_coordi().get_y());
    }
  }
}

}  // namespace ipl

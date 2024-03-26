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

#include "IDBParserEngine.hh"

#include "Block.hh"
#include "Instance.hh"
#include "Layout.hh"
#include "Net.hh"
#include "Netlist.hh"
#include "Pin.hh"
#include "idm.h"
// #include "utility/Utility.hh"
namespace imp {
IdbOrient orientTransform(Orient orient)
{
  if (orient == Orient::kN_R0) {
    return IdbOrient::kN_R0;
  } else if (orient == Orient::kS_R180) {
    return IdbOrient::kS_R180;
  } else if (orient == Orient::kW_R90) {
    return IdbOrient::kW_R90;
  } else if (orient == Orient::kE_R270) {
    return IdbOrient::kE_R270;
  } else if (orient == Orient::kFN_MY) {
    return IdbOrient::kFN_MY;
  } else if (orient == Orient::kFS_MX) {
    return IdbOrient::kFS_MX;
  } else if (orient == Orient::kFW_MX90) {
    return IdbOrient::kFW_MX90;
  } else if (orient == Orient::kFE_MY90) {
    return IdbOrient::kFE_MY90;
  } else {
    return IdbOrient::kNone;
  }
}
Orient orientTransform(IdbOrient idb_orient)
{
  if (idb_orient == IdbOrient::kN_R0) {
    return Orient::kN_R0;
  } else if (idb_orient == IdbOrient::kS_R180) {
    return Orient::kS_R180;
  } else if (idb_orient == IdbOrient::kW_R90) {
    return Orient::kW_R90;
  } else if (idb_orient == IdbOrient::kE_R270) {
    return Orient::kE_R270;
  } else if (idb_orient == IdbOrient::kFN_MY) {
    return Orient::kFN_MY;
  } else if (idb_orient == IdbOrient::kFS_MX) {
    return Orient::kFS_MX;
  } else if (idb_orient == IdbOrient::kFW_MX90) {
    return Orient::kFW_MX90;
  } else if (idb_orient == IdbOrient::kFE_MY90) {
    return Orient::kFE_MY90;
  } else {
    return Orient::kNone;
  }
}

IDBParser::IDBParser(idb::IdbBuilder* idb_builder)
{
  setIdbBuilder(idb_builder);
}

bool IDBParser::read()
{
  for (auto&& [idb_inst, inst] : _idb2inst) {
    if (read(idb_inst, inst))
      continue;
    return false;
  }
  return true;
}

bool IDBParser::write()
{
  for (auto&& [inst, idb_inst] : _inst2idb) {
    if (write(inst, idb_inst))
      continue;
    return false;
  }
  return true;
}

void IDBParser::setIdbBuilder(idb::IdbBuilder* idb_builder)
{
  _idb_builder = idb_builder;
  initNetlist();
}

void IDBParser::initNetlist()
{
  auto idb_def_service = _idb_builder->get_def_service();
  _idb_layout = idb_def_service->get_layout();
  _idb_design = idb_def_service->get_design();
  _design = std::make_shared<Block>(_idb_design->get_design_name(), std::make_shared<Netlist>(transform(_idb_layout)));
  auto core_shape = _design->netlist().property()->get_core_shape();
  _design->set_shape_curve(core_shape);
  _design->set_min_corner(core_shape.min_corner());
  initRows();
  initCells();

  // Init instances
  std::unordered_map<std::string, size_t> name2pos;
  for (auto* idb_inst : _idb_design->get_instance_list()->get_instance_list()) {
    auto inst_ptr = transform(idb_inst);
    assert(!_idb2inst.contains(idb_inst) || !_inst2idb.contains(inst_ptr));
    _idb2inst[idb_inst] = inst_ptr;
    _inst2idb[inst_ptr] = idb_inst;
    _instances[inst_ptr->get_name()] = inst_ptr;
    name2pos[inst_ptr->get_name()] = add_object(_design->netlist(), inst_ptr);
  }

  // Init nets
  size_t terminal_num = 0;
  for (auto* idb_net : _idb_design->get_net_list()->get_net_list()) {
    if (idb_net->is_clock() || idb_net->is_pdn() || idb_net->is_power() || idb_net->is_ground() || idb_net->get_pin_number() > 300) {
      continue;
    }
    std::vector<std::shared_ptr<Pin>> pins;
    std::vector<size_t> inst_pos;
    for (auto* idb_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
      auto pin = transform(idb_pin);
      pins.push_back(pin);
      // if (!pin->isIOPort()) {
      assert(_instances.contains(idb_pin->get_instance()->get_name()));
      assert(name2pos.contains(idb_pin->get_instance()->get_name()));
      size_t pos = name2pos.at(idb_pin->get_instance()->get_name());
      inst_pos.push_back(pos);
      // }
    }

    if (idb_net->has_io_pins()) {
      for (auto* idb_pin : idb_net->get_io_pins()->get_pin_list()) {
        auto pin = transform(idb_pin);
        pins.push_back(pin);
        terminal_num++;
        // create Pseudo IO-Cell for IO-Pin, use Pin-name as Instance Name
        auto io = std::make_shared<Instance>(pin->get_name(), _cells["Pseudo_IO_Cell"], _design);
        io->set_type(INSTANCE_TYPE::kPseudo);
        io->set_state(INSTANCE_STATE::kFixed);
        io->set_min_corner(idb_pin->get_location()->get_x(), idb_pin->get_location()->get_y());
        size_t pos = add_object(_design->netlist(), io);
        inst_pos.push_back(pos);
      }
    }

    if (!pins.empty()) {
      auto net = transform(idb_net);
      _net2idb[net] = idb_net;
      add_net(_design->netlist(), inst_pos, pins, net);
    }
  }
}

void IDBParser::initRows()
{
  IdbRows* idb_rows = _idb_layout->get_rows();
  for (auto* idb_row : idb_rows->get_row_list()) {
    auto row = std::make_shared<Row>(idb_row->get_name());
    IdbRect* idb_row_rect = idb_row->get_bounding_box();
    row->set_shape(
        geo::make_box(idb_row_rect->get_low_x(), idb_row_rect->get_low_y(), idb_row_rect->get_high_x(), idb_row_rect->get_high_y()));

    // set site.
    auto site_ptr = std::make_shared<Site>(idb_row->get_site()->get_name());
    site_ptr->set_width(idb_row->get_site()->get_width());
    site_ptr->set_height(idb_row->get_site()->get_height());

    // set site orient.
    const IdbOrient idb_orient = idb_row->get_site()->get_orient();
    site_ptr->set_orient(orientTransform(idb_orient));

    row->set_site(site_ptr);
    row->set_site_num(idb_row->get_site_count());
    _rows[idb_row->get_name()] = row;
  }
}

void IDBParser::initCells()
{
  for (auto* idb_cell : _idb_layout->get_cell_master_list()->get_cell_master()) {
    idb_cell->get_term_list();
    auto cell_ptr = std::make_shared<Cell>(idb_cell->get_name());
    int32_t width = idb_cell->get_width();
    int32_t height = idb_cell->get_height();
    cell_ptr->set_shape(geo::make_box(0, 0, width, height));
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
    _cells[idb_cell->get_name()] = cell_ptr;
  }
  // create Pseudo cell-master for IO-Pin
  auto io_cell_ptr = std::make_shared<Cell>("Pseudo_IO_Cell");
  io_cell_ptr->set_shape(geo::make_box(0, 0, 0, 0));  // io-cell has no shape;
  io_cell_ptr->set_type(CELL_TYPE::kIOCell);
  _cells[io_cell_ptr->get_name()] = io_cell_ptr;
}

std::shared_ptr<Layout> IDBParser::transform(idb::IdbLayout* idb_layout)
{
  auto layout = std::make_shared<Layout>();
  int32_t database_unit = idb_layout->get_units()->get_micron_dbu();
  layout->set_database_unit(database_unit);

  // set die shape.
  IdbDie* idb_die = idb_layout->get_die();
  layout->set_die_shape(geo::make_box(idb_die->get_llx(), idb_die->get_lly(), idb_die->get_urx(), idb_die->get_ury()));

  // set core shape.
  IdbCore* idb_core = idb_layout->get_core();
  IdbRect* idb_core_rect = idb_core->get_bounding_box();
  layout->set_core_shape(
      geo::make_box(idb_core_rect->get_low_x(), idb_core_rect->get_low_y(), idb_core_rect->get_high_x(), idb_core_rect->get_high_y()));
  return layout;
}

std::shared_ptr<Instance> IDBParser::transform(idb::IdbInstance* idb_inst)
{
  auto cell_ptr = _cells[idb_inst->get_cell_master()->get_name()];
  auto inst_ptr = std::make_shared<Instance>(idb_inst->get_name(), cell_ptr, _design);

  // set instace coordinate.
  auto idb_box = idb_inst->get_bounding_box();
  auto bbox = geo::make_box(idb_box->get_low_x(), idb_box->get_low_y(), idb_box->get_high_x(), idb_box->get_high_y());
  inst_ptr->set_min_corner(bbox.min_corner());

  // set instance state.
  if (idb_inst->is_unplaced() || idb_inst->get_status() == IdbPlacementStatus::kNone) {
    inst_ptr->set_min_corner(-1, -1);  // set an unlegal coordinate.
    inst_ptr->set_state(INSTANCE_STATE::kUnPlaced);
  } else if (idb_inst->is_placed()) {
    inst_ptr->set_state(INSTANCE_STATE::kPlaced);
  } else if (idb_inst->is_fixed()) {
    inst_ptr->set_state(INSTANCE_STATE::kFixed);
  } else {
    inst_ptr->set_state(INSTANCE_STATE::kNone);
  }

  // set type.
  const auto& layout_ = get_layout(_design->netlist());
  if (geo::within(bbox, layout_->get_die_shape()) && !geo::within(bbox, layout_->get_core_shape()) && idb_inst->is_fixed()) {
    inst_ptr->set_type(INSTANCE_TYPE::kOutside);
  } else if (idb_inst->is_fixed() && geo::overlaps(bbox, layout_->get_core_shape())) {
    inst_ptr->set_type(INSTANCE_TYPE::kCross);
  } else {
    inst_ptr->set_type(INSTANCE_TYPE::kNormal);
  }

  // set orient.
  const IdbOrient idb_orient = idb_inst->get_orient();
  inst_ptr->set_orient(orientTransform(idb_orient));

  // cover cell type.
  // TODO : where is clock buffer?
  if (idb_inst->get_cell_master()->is_block()) {
    cell_ptr->set_type(CELL_TYPE::kMacro);
    // } else if (idb_inst->is_io_instance() || inst_ptr->isOutside()) {
    //   cell_ptr->set_type(CELL_TYPE::kIOCell);
  } else if (idb_inst->is_flip_flop()) {
    cell_ptr->set_type(CELL_TYPE::kFlipflop);
  } else if (idb_inst->get_type() == IdbInstanceType::kDist) {
    cell_ptr->set_type(CELL_TYPE::kPhysicalFiller);
  }

  // add halo
  if (idb_inst->has_halo()) {
    auto* halo = idb_inst->get_halo();
    inst_ptr->set_extend(halo->get_extend_lef(), halo->get_extend_right(), halo->get_extend_bottom(), halo->get_extend_top());
  }

  return inst_ptr;
}

std::shared_ptr<Net> IDBParser::transform(idb::IdbNet* idb_net)
{
  auto net_ptr = std::make_shared<Net>(idb_net->get_net_name());
  auto connect_type = idb_net->get_connect_type();
  if (connect_type == IdbConnectType::kSignal) {
    net_ptr->set_net_type(NET_TYPE::kSignal);
  } else if (connect_type == IdbConnectType::kClock) {
    net_ptr->set_net_type(NET_TYPE::kClock);
  } else if (connect_type == IdbConnectType::kReset) {
    net_ptr->set_net_type(NET_TYPE::kReset);
  } else {
    net_ptr->set_net_type(NET_TYPE::kSignal);
  }
  if (idb_net->has_io_pins()) {
    net_ptr->set_io_net();
  }
  // set net state.
  net_ptr->set_net_state(NET_STATE::kNormal);
  return net_ptr;
}

std::shared_ptr<Pin> IDBParser::transform(idb::IdbPin* idb_pin)
{
  auto pin_ptr = std::make_shared<Pin>(idb_pin->get_pin_name());

  auto* idb_inst = idb_pin->get_instance();
  if (!idb_inst) {
    // set pin type.
    pin_ptr->set_pin_type(PIN_TYPE::kIOPort);
    // set pin offset coordinate.
    pin_ptr->set_offset(0, 0);

  } else {
    assert(_idb2inst.contains(idb_inst));
    // set pin type.
    pin_ptr->set_pin_type(PIN_TYPE::kInstancePort);

    // set pin offset coordinate.
    auto term_avg_position = idb_pin->get_term()->get_average_position();
    pin_ptr->set_offset(term_avg_position.get_x(), term_avg_position.get_y());
  }
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
  pin_ptr->set_coordi(idb_pin->get_average_coordinate()->get_x(), idb_pin->get_average_coordinate()->get_y());
  return pin_ptr;
}

bool IDBParser::read(idb::IdbInstance* idb_inst, std::shared_ptr<Instance> inst)
{
  if (idb_inst->get_name() == inst->get_name())
    return false;
  auto coordi = idb_inst->get_coordinate();
  auto orient = orientTransform(idb_inst->get_orient());
  inst->set_min_corner(coordi->get_x(), coordi->get_y());
  inst->set_orient(orient);
  return true;
}

bool IDBParser::write(std::shared_ptr<Instance> inst, idb::IdbInstance* idb_inst)
{
  if (idb_inst->get_name() == inst->get_name())
    return false;
  auto coordi = inst->get_min_corner();
  auto orient = orientTransform(inst->get_orient());
  idb_inst->set_coodinate(coordi.x(), coordi.y());
  idb_inst->set_orient(orient);
  return true;
}

}  // namespace imp

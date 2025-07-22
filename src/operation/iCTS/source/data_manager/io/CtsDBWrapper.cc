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
/**
 * @file CtsDBWrapper.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "CtsDBWrapper.hh"

#include "CTSAPI.hh"

namespace icts {

CtsDBWrapper::CtsDBWrapper(IdbBuilder* idb)
{
  _idb = idb;
  _idb_design = _idb->get_def_service()->get_design();
  _idb_layout = _idb->get_lef_service()->get_layout();
}

void CtsDBWrapper::writeDef()
{
  auto* config = CTSAPIInst.get_config();
  _idb->saveDef(config->get_output_def_path());
}

void CtsDBWrapper::read()
{
  auto* design = CTSAPIInst.get_design();
  auto* idb_net_list = _idb_design->get_net_list();

  auto& clock_net_names = design->get_clock_net_names();
  for (auto& clock_net_name : clock_net_names) {
    auto& clock_name = clock_net_name.first;
    auto& net_name = clock_net_name.second;

    auto* idb_net = idb_net_list->find_net(net_name);

    CtsNet* net = idbToCts(idb_net);
    net->set_is_newly(false);
    design->addNet(net);
    for (auto* idb_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
      auto* idb_inst = idb_pin->get_instance();
      auto* inst = idbToCts(idb_inst);
      auto* pin = idbToCts(idb_pin);

      design->addInstance(inst);
      design->addPin(pin);
    }

    /// add io pin
    auto* io_pins = idb_net->get_io_pins();
    for (auto* io_pin : io_pins->get_pin_list()) {
      auto* pin = idbToCts(io_pin);
      design->addPin(pin);
    }

    design->addClockNet(clock_name, net);
    /// ensure the net type of idb_net is clock type after read the net name
    /// from config
    idb_net->set_connect_type(IdbConnectType::kClock);
  }
  // check mux
  std::ranges::for_each(design->get_insts(), [](CtsInstance* inst) {
    if (inst->isSink()) {
      return;
    }
    auto pins = inst->get_pin_list();
    int load_num = 0;
    std::ranges::for_each(pins, [&load_num](CtsPin* pin) {
      if (pin->get_pin_type() == CtsPinType::kIn) {
        ++load_num;
      }
    });
    if (load_num > 1) {
      inst->set_type(CtsInstanceType::kMux);
    }
  });
}

void CtsDBWrapper::linkIdb(CtsInstance* inst)
{
  auto* idb_inst = makeIdbInstance(inst);
  // link coordinate and synchronize to cts db
  linkInstanceCood(inst, idb_inst);
}

void CtsDBWrapper::updateCell(CtsInstance* inst)
{
  auto cell = inst->get_cell_master();
  auto* idb_inst = ctsToIdb(inst);
  IdbCellMasterList* master_list = _idb_layout->get_cell_master_list();
  IdbCellMaster* master = master_list->find_cell_master(cell);
  LOG_FATAL_IF(master == nullptr) << inst->get_name() << " can't find cell master: " << cell;
  idb_inst->set_cell_master(master);
}

CtsInstance* CtsDBWrapper::makeInstance(const string& name, const string& cell_name)
{
  IdbCellMasterList* master_list = _idb_layout->get_cell_master_list();
  IdbCellMaster* master = master_list->find_cell_master(cell_name);
  if (!master) {
    return nullptr;
  }
  IdbInstance* idb_inst = new IdbInstance();
  idb_inst->set_name(name);
  idb_inst->set_cell_master(master);

  _idb_design->get_instance_list()->add_instance(idb_inst);

  return idbToCts(idb_inst);
}

CtsNet* CtsDBWrapper::makeNet(const string& name)
{
  IdbNetList* idb_net_list = _idb_design->get_net_list();
  IdbNet* idb_net = idb_net_list->add_net(name, idb::IdbConnectType::kClock);

  return idbToCts(idb_net);
}

IdbInstance* CtsDBWrapper::ctsToIdb(CtsInstance* inst)
{
  if (_cts2idbInst.find(inst) != _cts2idbInst.end()) {
    return _cts2idbInst[inst];
  }
  return nullptr;
}
IdbPin* CtsDBWrapper::ctsToIdb(CtsPin* pin)
{
  IdbPin* idb_pin = nullptr;
  if (_cts2idbPin.find(pin) != _cts2idbPin.end()) {
    idb_pin = _cts2idbPin[pin];
  }
  return idb_pin;
}
IdbNet* CtsDBWrapper::ctsToIdb(CtsNet* net)
{
  IdbNet* idb_net = nullptr;
  if (_cts2idbNet.find(net) != _cts2idbNet.end()) {
    idb_net = _cts2idbNet[net];
    idb_net->set_connect_type(IdbConnectType::kClock);
  }
  return idb_net;
}

IdbCoordinate<int32_t> CtsDBWrapper::ctsToIdb(const Point& loc) const
{
  return IdbCoordinate<int32_t>(loc.x(), loc.y());
}

bool CtsDBWrapper::isValidPin(IdbPin* idb_pin) const
{
  auto idb_pin_type = idb_pin->get_term()->get_type();
  bool is_flip_flop = isFlipFlop(idb_pin->get_instance());

  if (is_flip_flop) {
    return isClockPin(idb_pin) ? true : false;
  } else {
    if (idb_pin_type != IdbConnectType::kPower && idb_pin_type != IdbConnectType::kGround) {
      return true;
    } else {
      return false;
    }
  }
}

CtsInstance* CtsDBWrapper::idbToCts(IdbInstance* idb_inst, bool b_virtual)
{
  if (_idb2ctsInst.find(idb_inst) == _idb2ctsInst.end()) {
    CtsInstance* inst = new CtsInstance(idb_inst->get_name());
    inst->set_is_newly(false);
    inst->set_cell_master(b_virtual ? idb_inst->get_name() : idb_inst->get_cell_master()->get_name());
    inst->set_location(idbToCts(*idb_inst->get_coordinate()));
    inst->set_type(b_virtual || isFlipFlop(idb_inst) ? CtsInstanceType::kSink : CtsInstanceType::kBuffer);
    inst->set_virtual(b_virtual);

    crossRef(inst, idb_inst);
  }
  return _idb2ctsInst[idb_inst];
}

CtsInstance* CtsDBWrapper::idbToCts(IdbInstance* idb_inst, CtsInstanceType inst_type)
{
  if (_idb2ctsInst.find(idb_inst) == _idb2ctsInst.end()) {
    CtsInstance* inst = new CtsInstance(idb_inst->get_name());
    inst->set_is_newly(false);
    inst->set_cell_master(idb_inst->get_cell_master()->get_name());
    inst->set_location(idbToCts(*idb_inst->get_coordinate()));
    inst->set_type(inst_type);
    crossRef(inst, idb_inst);
  }
  return _idb2ctsInst[idb_inst];
}
CtsPin* CtsDBWrapper::idbToCts(IdbPin* idb_pin)
{
  if (_idb2ctsPin.find(idb_pin) == _idb2ctsPin.end()) {
    CtsPin* pin = new CtsPin(idb_pin->get_pin_name());
    CtsPinType pin_type = idbToCts(idb_pin->get_term()->get_type(), idb_pin->get_term()->get_direction());
    Point loc = idbToCts(*idb_pin->get_average_coordinate());
    pin->set_pin_type(pin_type);
    pin->set_location(loc);
    pin->set_io(idb_pin->is_io_pin());

    crossRef(pin, idb_pin);
  }
  return _idb2ctsPin[idb_pin];
}

bool CtsDBWrapper::isClockPin(IdbPin* idb_pin) const
{
  auto idb_pin_name = idb_pin->get_pin_name();
  return idb_pin_name == "CK" || idb_pin_name == "CLK";
}
bool CtsDBWrapper::isFlipFlop(IdbInstance* idb_inst) const
{
  return CTSAPIInst.isFlipFlop(idb_inst->get_name());
}

CtsNet* CtsDBWrapper::idbToCts(IdbNet* idb_net)
{
  if (_idb2ctsNet.find(idb_net) == _idb2ctsNet.end()) {
    CtsNet* net = new CtsNet(idb_net->get_net_name());

    for (auto* idb_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
      auto* pin = idbToCts(idb_pin);
      net->addPin(pin);

      auto* idb_inst = idb_pin->get_instance();
      auto* inst = idbToCts(idb_inst);
      inst->addPin(pin);
    }

    auto* io_pins = idb_net->get_io_pins();
    for (auto* io_pin : io_pins->get_pin_list()) {
      auto* pin = idbToCts(io_pin);
      net->addPin(pin);

      /// make a virtual instance

      IdbInstance* idb_inst_new = new IdbInstance();
      IdbCellMasterList* cell_master_list = _idb_layout->get_cell_master_list();
      IdbCellMaster* cell_master_new = cell_master_list->set_cell_master(pin->get_pin_name());
      cell_master_new->set_height(io_pin->get_bounding_box()->get_height());
      cell_master_new->set_width(io_pin->get_bounding_box()->get_width());

      idb_inst_new->set_cell_master(cell_master_new);
      idb_inst_new->set_coodinate(*(io_pin->get_average_coordinate()), false);
      idb_inst_new->set_name(pin->get_pin_name());
      auto* inst = idbToCts(idb_inst_new, true);
      inst->addPin(pin);
    }

    crossRef(net, idb_net);
  }
  return _idb2ctsNet[idb_net];
}

CtsPinType CtsDBWrapper::idbToCts(IdbConnectType idb_pin_type, IdbConnectDirection idb_pin_direction) const
{
  CtsPinType pin_type;

  if (idb_pin_type == IdbConnectType::kClock) {
    pin_type = CtsPinType::kClock;
  } else if (idb_pin_direction == IdbConnectDirection::kOutput) {
    pin_type = CtsPinType::kOut;
  } else if (idb_pin_direction == IdbConnectDirection::kInput) {
    pin_type = CtsPinType::kIn;
  } else if (idb_pin_direction == IdbConnectDirection::kInOut) {
    pin_type = CtsPinType::kInOut;
  } else {
    pin_type = CtsPinType::kOther;
  }
  return pin_type;
}

Point CtsDBWrapper::idbToCts(IdbCoordinate<int32_t>& coord) const
{
  return Point(coord.get_x(), coord.get_y());
}

IdbInstance* CtsDBWrapper::makeIdbInstance(CtsInstance* inst)
{
  auto* design_list = _idb_design->get_instance_list();
  IdbInstance* idb_inst = design_list->find_instance(inst->get_name());
  if (idb_inst) {
    return idb_inst;
  }
  idb_inst = new IdbInstance();
  idb_inst->set_name(inst->get_name());
  auto* master_list = _idb_layout->get_cell_master_list();
  auto* master = master_list->find_cell_master(inst->get_cell_master());
  LOG_FATAL_IF(master == nullptr) << inst->get_name() << " can't find cell master: " << inst->get_cell_master();
  idb_inst->set_cell_master(master);
  design_list->add_instance(idb_inst);
  return idb_inst;
}

IdbNet* CtsDBWrapper::makeIdbNet(CtsNet* net)
{
  auto* idb_net_list = _idb_design->get_net_list();
  auto* idb_net = idb_net_list->add_net(net->get_net_name(), idb::IdbConnectType::kClock);

  crossRef(net, idb_net);
  return idb_net;
}

void CtsDBWrapper::linkInstanceCood(CtsInstance* inst, IdbInstance* idb_inst)
{
  if (ctsToIdb(inst)) {
    return;
  }

  auto loc = inst->get_location();
  idb_inst->set_coodinate(loc.x(), loc.y());
  idb_inst->set_status_placed();

  // LOG_FATAL_IF(!withinCore(inst->get_location()))
  //     << "Instance " << inst->get_name() << " (" << inst->get_location().x() << ", " << inst->get_location().y() << ")"
  //     << " is not within core";

  // IdbRow* row = findRow(inst->get_location());
  // LOG_FATAL_IF(!row) << "Cannot find row for instance " << inst->get_name() << " (" << inst->get_location().x() << ", "
  //                    << inst->get_location().y() << ")";
  // IdbOrient row_orient = row->get_site()->get_orient();
  // idb_inst->set_orient(row_orient);
  auto* rows = _idb_layout->get_rows();
  auto orient = rows->get_row_list().front()->get_site()->get_orient();
  idb_inst->set_orient(orient);

  for (auto& idb_pin : idb_inst->get_pin_list()->get_pin_list()) {
    CtsPin* pin = idbToCts(idb_pin);
    pin->set_is_newly(true);
    inst->addPin(pin);
  }

  crossRef(inst, idb_inst);
}

bool CtsDBWrapper::ctsConnect(CtsInstance* inst, CtsPin* pin, CtsNet* net)
{
  if (inst->is_virtual()) {
    return true;
  }
  net->addPin(pin);
  pin->set_net(net);
  return true;
}
bool CtsDBWrapper::idbConnect(CtsPin* pin, CtsNet* net)
{
  auto* inst = pin->get_instance();
  ctsConnect(inst, pin, net);
  if (inst->is_virtual()) {
    return true;
  }
  IdbNet* idb_net = ctsToIdb(net);
  IdbInstance* idb_inst = ctsToIdb(inst);

  auto* idb_pin_list = idb_inst->get_pin_list();
  if (idb_pin_list == nullptr) {
    return false;
  }

  for (auto* idb_pin : idb_pin_list->get_pin_list()) {
    if (idb_pin->get_pin_name() == pin->get_pin_name()) {
      idb_net->add_instance_pin(idb_pin);
      idb_pin->set_net(idb_net);
      return true;
    }
  }
  return false;
}

bool CtsDBWrapper::ctsDisconnect(CtsPin* pin)
{
  CtsNet* net = pin->get_net();
  if (!net) {
    return true;
  }
  net->removePin(pin);
  pin->set_net(nullptr);
  return true;
}

bool CtsDBWrapper::idbDisconnect(CtsPin* pin)
{
  ctsDisconnect(pin);

  IdbPin* idb_pin = ctsToIdb(pin);
  if (!idb_pin) {
    LOG_WARNING << "can't found pin: " << pin->get_full_name();
    return false;
  }
  auto* idb_net = idb_pin->get_net();
  if (!idb_net) {
    return true;
  }
  idb_net->remove_pin(idb_pin);
  return true;
}

IdbRow* CtsDBWrapper::findRow(const Point& loc) const
{
  auto* rows = _idb_layout->get_rows();
  for (auto row : rows->get_row_list()) {
    auto bounding_box = row->get_bounding_box();
    if (bounding_box->get_low_y() == loc.y()) {
      return row;
    }
  }
  return nullptr;
}

IdbRect CtsDBWrapper::get_bounding_box(CtsInstance* inst) const
{
  int lx = inst->get_location().x();
  int ly = inst->get_location().y();

  auto cell_master_name = inst->get_cell_master();
  auto* cell_master_list = _idb_layout->get_cell_master_list();
  auto* cell_master = cell_master_list->find_cell_master(cell_master_name);

  if (cell_master != nullptr) {
    return IdbRect(lx, ly, lx + cell_master->get_width(), ly + cell_master->get_height());
  } else {
    return IdbRect(lx, ly, lx, ly);
  }
}

IdbRect* CtsDBWrapper::get_core_bounding_box() const
{
  auto* idb_core_box = _idb_layout->get_core()->get_bounding_box();
  return idb_core_box;
}

int CtsDBWrapper::get_site_width() const
{
  auto& rows = _idb_layout->get_rows()->get_row_list();
  IdbRow* row = rows.front();
  IdbSite* site = row->get_site();
  return site->get_width();
}

int CtsDBWrapper::get_site_height() const
{
  auto& rows = _idb_layout->get_rows()->get_row_list();
  IdbRow* row = rows.front();
  IdbSite* site = row->get_site();
  return site->get_height();
}

int CtsDBWrapper::get_row_num() const
{
  // auto &rows = _idb_layout->get_rows()->get_row_list();
  // return rows.size();
  auto* bound = get_core_bounding_box();
  return bound->get_height() / get_site_height();
}

int CtsDBWrapper::get_site_num() const
{
  // auto& rows = _idb_layout->get_rows()->get_row_list();
  // IdbRow* row = rows.front();
  // return row->get_site_count();
  auto* bound = get_core_bounding_box();
  return bound->get_width() / get_site_width();
}

Point CtsDBWrapper::getPinLoc(CtsPin* pin)
{
  auto* idb_pin = ctsToIdb(pin);
  auto* layer_shape = idb_pin->get_port_box_list().front();
  auto rect = layer_shape->get_rect_list().front();
  return icts::Point((rect->get_low_x() + rect->get_high_x()) / 2, (rect->get_low_y() + rect->get_high_y()) / 2);
}

std::string CtsDBWrapper::getPinLayer(CtsPin* pin)
{
  auto* idb_pin = ctsToIdb(pin);
  auto* layer_shape = idb_pin->get_port_box_list().front();
  auto inst_layer = layer_shape->get_layer()->get_name();
  return inst_layer;
}

bool CtsDBWrapper::withinCore(const Point& loc) const
{
  auto* idb_core = _idb_layout->get_core();
  auto* core_box = idb_core->get_bounding_box();
  auto pt = ctsToIdb(loc);
  return core_box->containPoint(pt);
}

}  // namespace icts
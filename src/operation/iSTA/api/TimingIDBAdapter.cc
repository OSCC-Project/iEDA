// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file TimingIDBAdapter.hh
 * @author longshy (longshy@pcl.ac.cn)
 * @brief idb and ista data adapter.
 * @version 0.1
 * @date 2021-10-11
 */

#include "TimingIDBAdapter.hh"

#include <memory>
#include <regex>

// #include "idm.h"
#include "log/Log.hh"

namespace ista {

bool TimingIDBAdapter::isPlaced(DesignObject* pin_or_port) {
  IdbPlacementStatus status = IdbPlacementStatus::kUnplaced;
  if (pin_or_port->isPin()) {
    IdbPin* idb_pin = staToDb(dynamic_cast<Pin*>(pin_or_port));
    if (idb_pin) {
      IdbInstance* idb_inst = idb_pin->get_instance();
      status = idb_inst->get_status();
    }
  } else {
    LOG_FATAL_IF(!pin_or_port->isPort());
    IdbPin* idb_pin = staToDb(dynamic_cast<Port*>(pin_or_port));
    if (idb_pin) {
      IdbInstance* idb_inst = idb_pin->get_instance();
      status = idb_inst->get_status();
    }
  }
  return (status == IdbPlacementStatus::kPlaced ||
          status == IdbPlacementStatus::kCover);
}

/**
 * @brief dbu to meter.
 *
 * @param dist
 * @return double
 */
double TimingIDBAdapter::dbuToMeters(int distance) const {
  IdbLayout* idb_layout = _idb_lef_service->get_layout();
  int dbu = idb_layout->get_units()->get_micron_dbu();
  return distance / (dbu * 1e+6);
}

/**
 * @brief Get the pin location.
 *
 * @param pin
 * @param x
 * @param y
 * @param exists
 */
void TimingIDBAdapter::location(DesignObject* pin_or_port,
                                // Return values.
                                double& x, double& y, bool& exists) {
  if (isPlaced(pin_or_port)) {
    IdbCoordinate<int32_t>* coordinate = idbLocation(pin_or_port);
    x = dbuToMeters(coordinate->get_x());
    y = dbuToMeters(coordinate->get_y());
    exists = true;
  } else {
    x = 0;
    y = 0;
    exists = false;
  }
}

/**
 * @brief Get the pin location.
 *
 * @param pin
 * @return IdbCoordinate<int32_t>*
 */
IdbCoordinate<int32_t>* TimingIDBAdapter::idbLocation(
    DesignObject* pin_or_port) {
  IdbCoordinate<int32_t>* coordinate = nullptr;
  IdbPin* dpin = pin_or_port->isPin()
                     ? staToDb(dynamic_cast<Pin*>(pin_or_port))
                     : staToDb(dynamic_cast<Port*>(pin_or_port));
  if (dpin) {
    coordinate = dpin->is_io_pin() ? dpin->get_location()
                                   : dpin->get_average_coordinate();
  }

  return coordinate;
}

/**
 * @brief get segment resistance.
 *
 * @param num_layer layer number = target routing layer id - first routing layer
 * id by data config
 * @param segment_length unit is um (micro meter)
 * @param segment_width unit is um (micro meter)
 * @return double Ω
 */
double TimingIDBAdapter::getResistance(int num_layer, double segment_length,
                                       std::optional<double> segment_width) {
  double segment_resistance = 0;
  IdbLayout* idb_layout = _idb_lef_service->get_layout();
  vector<IdbLayer*>& routing_layers =
      idb_layout->get_layers()->get_routing_layers();

  int routing_layer_1st = 0;
  int routing_layer_id = num_layer - 1 + routing_layer_1st;
  int routing_layer_size = routing_layers.size();

  if (num_layer >= routing_layer_size ||
      routing_layer_id >= routing_layer_size || num_layer < 0) {
    LOG_FATAL << "Layer id error = " << num_layer;
    return 0;
  }

  IdbLayerRouting* routing_layer =
      dynamic_cast<IdbLayerRouting*>(routing_layers[routing_layer_id]);

  if (!segment_width) {
    segment_width = (double)routing_layer->get_width() /
                    idb_layout->get_units()->get_micron_dbu();
  }

  double lef_resistance = routing_layer->get_resistance();

  segment_resistance = lef_resistance * segment_length / *segment_width;

  // _debug_csv_file << lef_resistance << "," << segment_length << ","
  //           << *segment_width << "," << num_layer << ","
  //           << segment_resistance << "\n";

  return segment_resistance;
}

/**
 * @brief get segment capacitance.
 *
 * @param num_layer  layer number = target routing layer id - first routing layer
 * id by data config
 * @param segment_length unit is um (micro meter)
 * @param segment_width unit is um (micro meter)
 * @return double cap unit is pf
 */
double TimingIDBAdapter::getCapacitance(int num_layer, double segment_length,
                                        std::optional<double> segment_width) {
  double segment_capacitance = 0;
  IdbLayout* idb_layout = _idb_lef_service->get_layout();
  vector<IdbLayer*>& routing_layers =
      idb_layout->get_layers()->get_routing_layers();

  int routing_layer_1st = 0;  // dmInst->get_routing_layer_1st();
  int routing_layer_id = num_layer - 1 + routing_layer_1st;
  int routing_layer_size = routing_layers.size();

  if (num_layer >= routing_layer_size ||
      routing_layer_id >= routing_layer_size || num_layer < 0) {
    LOG_FATAL << "Layer id error = " << num_layer;
    return 0;
  }

  IdbLayerRouting* routing_layer =
      dynamic_cast<IdbLayerRouting*>(routing_layers[routing_layer_id]);

  if (!segment_width) {
    segment_width = (double)routing_layer->get_width() /
                    idb_layout->get_units()->get_micron_dbu();
  }

  double lef_capacitance = routing_layer->get_capacitance();
  double lef_edge_capacitance = routing_layer->get_edge_capacitance();

  segment_capacitance =
      (lef_capacitance * segment_length * (*segment_width)) +
      (lef_edge_capacitance * 2 * (segment_length + (*segment_width)));

  return segment_capacitance;
}

/**
 * @brief get unit capacitance.
 *
 * @param segment_width unit is um (micro meter)
 * @return double
 */
double TimingIDBAdapter::getAverageResistance(
    std::optional<double>& segment_width) {
  IdbLayout* idb_layout = _idb_lef_service->get_layout();
  vector<IdbLayer*>& routing_layers =
      idb_layout->get_layers()->get_routing_layers();

  double layers_resistance = 0;
  for (unsigned int i = 0; i < routing_layers.size(); i++) {
    IdbLayerRouting* routing_layer =
        dynamic_cast<IdbLayerRouting*>(routing_layers[i]);

    if (!segment_width) {
      segment_width = (double)routing_layer->get_width() /
                      idb_layout->get_units()->get_micron_dbu();
    }

    double lef_resistance;
    if (idb_layout->get_units()->get_ohms() == -1) {
      lef_resistance = routing_layer->get_resistance();
    } else {
      lef_resistance = routing_layer->get_resistance();
    }

    layers_resistance += lef_resistance / *segment_width;
  }
  double unit_resistance = layers_resistance / routing_layers.size();

  return unit_resistance;
}

/**
 * @brief get unit resistance.
 *
 * @param segment_width unit is um (micro meter)
 * @return double
 */
double TimingIDBAdapter::getAverageCapacitance(
    std::optional<double>& segment_width) {
  IdbLayout* idb_layout = _idb_lef_service->get_layout();
  vector<IdbLayer*>& routing_layers =
      idb_layout->get_layers()->get_routing_layers();

  double layers_capacitance = 0;
  for (unsigned int i = 0; i < routing_layers.size(); i++) {
    IdbLayerRouting* routing_layer =
        dynamic_cast<IdbLayerRouting*>(routing_layers[i]);

    if (!segment_width) {
      segment_width = (double)routing_layer->get_width() /
                      idb_layout->get_units()->get_micron_dbu();
    }

    double lef_capacitance = routing_layer->get_capacitance();
    double lef_edge_capacitance = routing_layer->get_edge_capacitance();

    layers_capacitance += (lef_capacitance * (*segment_width)) +
                          (lef_edge_capacitance * 2 * (1 + (*segment_width)));
    ;
  }
  double unit_capacitance = layers_capacitance / routing_layers.size();
  return unit_capacitance;
}

/**
 * @brief Return the wire length corresponding to the capacitance value
 *
 * @param num_layer
 * @param cap
 * @param segment_width
 * @return double
 */
double TimingIDBAdapter::capToLength(int num_layer, double cap,
                                     std::optional<double>& segment_width) {
  IdbLayout* idb_layout = _idb_lef_service->get_layout();
  vector<IdbLayer*>& routing_layers =
      idb_layout->get_layers()->get_routing_layers();
  double length = 0;

  int routing_layer_1st = 0;  // dmInst->get_routing_layer_1st();
  int routing_layer_id = num_layer - 1 + routing_layer_1st;
  int routing_layer_size = routing_layers.size();

  if (num_layer >= routing_layer_size ||
      routing_layer_id >= routing_layer_size || num_layer < 0) {
    LOG_FATAL << "Layer id error = " << num_layer;
    return 0;
  }

  IdbLayerRouting* routing_layer =
      dynamic_cast<IdbLayerRouting*>(routing_layers[routing_layer_id]);

  if (!segment_width) {
    segment_width = (double)routing_layer->get_width() /
                    idb_layout->get_units()->get_micron_dbu();
  }
  double lef_capacitance = routing_layer->get_capacitance();
  double lef_edge_capacitance = routing_layer->get_edge_capacitance();

  length = (cap - 2 * lef_edge_capacitance * (*segment_width)) /
           (lef_capacitance * (*segment_width) + 2 * lef_edge_capacitance);

  return length;
}

/**
 * @brief convert db type to sta port type.
 *
 * @param sig_type
 * @param io_type
 * @return PortDir
 */
PortDir TimingIDBAdapter::dbToSta(IdbConnectType sig_type,
                                  IdbConnectDirection io_type) const {
  if (sig_type == IdbConnectType::kPower) {
    return PortDir::kOther;
  } else if (sig_type == IdbConnectType::kGround) {
    return PortDir::kOther;
  } else if (io_type == IdbConnectDirection::kInput) {
    return PortDir::kIn;
  } else if (io_type == IdbConnectDirection::kOutput) {
    return PortDir::kOut;
  } else if (io_type == IdbConnectDirection::kInOut) {
    return PortDir::kInOut;
  } else if (io_type == IdbConnectDirection::kFeedThru) {
    return PortDir::kOther;
  } else {
    LOG_FATAL << "not support.";
    return PortDir::kOther;
  }
}

LibCell* TimingIDBAdapter::dbToSta(IdbCellMaster* master) {
  std::string liberty_cell_name = master->get_name();
  return _ista->findLibertyCell(liberty_cell_name.c_str());
}

/**
 * @brief
 *
 * @param cell
 * @return dbMaster*
 */
IdbCellMaster* TimingIDBAdapter::staToDb(const LibCell* cell) const {
  IdbLayout* idb_layout = _idb_lef_service->get_layout();
  IdbCellMasterList* idb_master_list = idb_layout->get_cell_master_list();
  return idb_master_list->find_cell_master(cell->get_cell_name());
}

LibPort* TimingIDBAdapter::dbToSta(IdbTerm* idb_term) {
  IdbCellMaster* idb_master = idb_term->get_cell_master();
  auto* liberty_cell = dbToSta(idb_master);
  return liberty_cell->get_cell_port_or_port_bus(idb_term->get_name().c_str());
}

IdbTerm* TimingIDBAdapter::staToDb(LibPort* port) const {
  const LibCell* cell = port->get_ower_cell();
  IdbCellMaster* master = staToDb(cell);
  vector<IdbTerm*> terms = master->get_term_list();
  for (IdbTerm* term : terms) {
    if (term->get_name() == port->get_port_name()) {
      return term;
    }
  }
  return nullptr;
}

/**
 * @brief Create instance in db and timing netlist.
 *
 */
Instance* TimingIDBAdapter::createInstance(LibCell* cell, const char* name) {
  const char* cell_name = cell->get_cell_name();
  IdbLayout* idb_layout = _idb_lef_service->get_layout();
  IdbCellMasterList* master_list = idb_layout->get_cell_master_list();
  IdbCellMaster* master = master_list->find_cell_master(cell_name);
  if (master) {
    IdbInstance* idb_inst = new IdbInstance();
    idb_inst->set_name(name);
    idb_inst->set_cell_master(master);
    IdbDesign* idb_design = _idb_def_service->get_design();
    idb_design->get_instance_list()->add_instance(idb_inst);

    Instance sta_inst(name, cell);
    LibPort* library_port;
    FOREACH_CELL_PORT(cell, library_port) {
      const char* pin_name = library_port->get_port_name();
      // May be need add pin bus, fixme.
      auto* inst_pin = sta_inst.addPin(pin_name, library_port);
      auto idb_pin = idb_inst->get_pin_list()->find_pin_by_term(pin_name);
      crossRef(inst_pin, idb_pin);
    }
    auto* design_netlist = getNetlist();
    auto& created_inst = design_netlist->addInstance(std::move(sta_inst));
    crossRef(&created_inst, idb_inst);

    return &created_inst;
  }
  return nullptr;
}

/**
 * @brief remove the instance.
 *
 * @param instance_name
 */
void TimingIDBAdapter::deleteInstance(const char* instance_name) {
  auto* design_netlist = getNetlist();
  auto* the_instance = design_netlist->findInstance(instance_name);

  IdbDesign* idb_design = _idb_def_service->get_design();

  Pin* pin;
  FOREACH_INSTANCE_PIN(the_instance, pin) {
    auto* db_pin = staToDb(pin);
    removeCrossRef(pin, db_pin);
  }

  design_netlist->removeInstance(instance_name);
  std::string idb_inst_name = staToDb(the_instance)->get_name();
  idb_design->get_instance_list()->remove_instance(idb_inst_name);
  IdbInstance* idb_instance = staToDb(the_instance);
  removeCrossRef(the_instance, idb_instance);
}

/**
 * @brief  Replace the db inst cell.
 *
 * @param inst
 * @param cell
 */
void TimingIDBAdapter::substituteCell(Instance* inst, LibCell* cell) {
  IdbCellMaster* idb_master = staToDb(cell);
  IdbInstance* idb_inst = staToDb(inst);
  idb_inst->set_cell_master(idb_master);  // TODO: dinst->swapMaster(master)
}

/**
 * @brief Connect the port on the inst to the net.
 *
 * @param inst
 * @param port
 * @param net
 * @return Pin*
 */
Pin* TimingIDBAdapter::attach(Instance* inst, const char* port_name, Net* net) {
  IdbNet* dnet = staToDb(net);
  if (!dnet) {
    dnet = _idb_design->get_net_list()->find_net(net->get_name());
    if (!dnet) {
      std::string sta_net_name = net->get_name();
      std::string idb_net_name = changeStaBusNetNameToIdb(sta_net_name);
      dnet = _idb_design->get_net_list()->find_net(idb_net_name);
      LOG_FATAL_IF(!dnet) << "idb net " << net->get_name() << "is not found.";
    }
  }
  // const char* port_name = port->get_port_name();
  Pin* pin = nullptr;
  IdbInstance* dinst = staToDb(inst);
  if (!dinst) {
    dinst = _idb_design->get_instance_list()->find_instance(inst->get_name());
  }
  auto& dpin_list = dinst->get_pin_list()->get_pin_list();
  for (auto dpin : dpin_list) {
    if (dpin->get_pin_name() == port_name) {
      if (dpin->is_io_pin()) {
        dnet->add_io_pin(dpin);
        dpin->set_net(dnet);
      } else {
        dnet->add_instance_pin(dpin);
        dpin->set_net(dnet);
      }
      pin = dbToStaPin(dpin);
      net->addPinPort(pin);
      break;
    }
  }

  return pin;
}

/**
 * @brief connet the port to the net.
 *
 * @param port
 * @param port_name
 * @param net
 * @return Port*
 */
Port* TimingIDBAdapter::attach(Port* port, const char* port_name, Net* net) {
  IdbNet* dnet = staToDb(net);
  if (!dnet) {
    dnet = _idb_design->get_net_list()->find_net(net->get_name());
  }
  // const char* port_name = port->get_port_name();
  IdbPin* dport = staToDb(port);

  if (dport->get_pin_name() == port_name) {
    dnet->add_io_pin(dport);
    dport->set_net(dnet);
    net->addPinPort(port);
  }

  return port;
}

/**
 * @brief Disconnect net of pin.
 *
 * @param pin
 */
void TimingIDBAdapter::disattachPin(Pin* pin) {
  auto* sta_net = pin->get_net();
  sta_net->removePinPort(pin);
  IdbPin* dpin = staToDb(pin);

  if (!dpin) {
    auto* dnet = _idb_design->get_net_list()->find_net(sta_net->get_name());
    if (!dnet) {
      std::string sta_net_name = sta_net->get_name();
      std::string idb_net_name = changeStaBusNetNameToIdb(sta_net_name);
      dnet = _idb_design->get_net_list()->find_net(idb_net_name);
      LOG_FATAL_IF(!dnet) << "idb net " << sta_net->get_name()
                          << "is not found.";
    }
    auto* idb_instance = _idb_design->get_instance_list()->find_instance(
        pin->get_own_instance()->getFullName());
    dpin = idb_instance->get_pin_list()->find_pin_by_term(pin->get_name());
    dnet->remove_pin(dpin);

  } else {
    auto* dnet = dpin->get_net();
    dnet->remove_pin(dpin);
  }
}

/**
 * @brief Disconnect net of port.
 *
 * @param pin
 */
void TimingIDBAdapter::disattachPinPort(DesignObject* pin_or_port) {
  if (pin_or_port->isPin()) {
    disattachPin(dynamic_cast<Pin*>(pin_or_port));
  } else {
    // port
    IdbPin* dpin = staToDb(dynamic_cast<Port*>(pin_or_port));

    if (!dpin) {
      dpin = _idb_design->get_io_pin_list()->find_pin(pin_or_port->get_name());
    }

    LOG_FATAL_IF(!dpin) << "dpin " << pin_or_port->get_name()
                        << " is not found.";

    auto* dnet = dpin->get_net();
    dnet->remove_pin(dpin);

    pin_or_port->get_net()->removePinPort(pin_or_port);
  }
}

/**
 * @brief reconnet the pin to net.
 *
 * @param pin
 * @param net
 */
void TimingIDBAdapter::reattachPin(Net* net, Pin* old_connect_pin,
                                   std::vector<Pin*> new_connect_pins) {
  IdbNet* dnet = staToDb(net);
  IdbPin* old_dpin = staToDb(old_connect_pin);
  old_dpin->set_net(nullptr);

  old_connect_pin->set_net(nullptr);
  net->removePinPort(old_connect_pin);

  dnet->remove_pin(old_dpin);
  for (auto* new_connect_pin : new_connect_pins) {
    IdbPin* new_dpin = staToDb(new_connect_pin);
    new_dpin->set_net(dnet);

    if (new_dpin->is_io_pin()) {
      dnet->add_io_pin(new_dpin);
    } else {
      dnet->add_instance_pin(new_dpin);
    }

    net->addPinPort(new_connect_pin);
  }
}

/**
 * @brief Create a net with net_name,own_instance.
 *
 * @param name
 * @param parent
 * @return Net*
 */
Net* TimingIDBAdapter::createNet(const char* name, Instance* /*parent*/) {
  std::string str_name = name;
  IdbNetList* dbnet_list = _idb_design->get_net_list();
  IdbNet* dnet = dbnet_list->add_net(str_name, idb::IdbConnectType::kClock);
  auto* design_netlist = getNetlist();
  auto& created_net = design_netlist->addNet(Net(name));
  crossRef(&created_net, dnet);
  return &created_net;
}

/**
 * @brief Create a net with net_name,own_instance,net_connect_type.
 *
 * @param name
 * @param parent
 * @return Net*
 */
Net* TimingIDBAdapter::createNet(const char* name, Instance* /*parent*/,
                                 idb::IdbConnectType connect_type) {
  std::string str_name = name;
  IdbNetList* dbnet_list = _idb_design->get_net_list();
  IdbNet* dnet = dbnet_list->add_net(str_name, connect_type);
  auto* design_netlist = getNetlist();
  auto& created_net = design_netlist->addNet(Net(name));
  crossRef(&created_net, dnet);
  return &created_net;
}

/**
 * @brief Create a net with net_name,connect_pins,net_connect_type.
 *
 * @param name
 * @param sink_pin_list
 * @param connect_type
 * @return Net*
 */
Net* TimingIDBAdapter::createNet(const char* name,
                                 std::vector<std::string>& sink_pin_list,
                                 idb::IdbConnectType connect_type) {
  std::string str_name = name;
  IdbNetList* dbnet_list = _idb_design->get_net_list();
  IdbNet* dnet = dbnet_list->add_net(str_name, connect_type);
  auto* design_netlist = getNetlist();
  Net new_net = Net(name);
  for (const auto& sink_pin_name : sink_pin_list) {
    auto [instance_name, instance_pin_name] =
        Str::splitTwoPart(sink_pin_name.c_str(), "/:");
    if (!instance_pin_name.empty()) {
      Instance* instance = design_netlist->findInstance(instance_name.c_str());
      LOG_FATAL_IF(!instance)
          << "instance: " << instance->getFullName() << " not found.";
      std::optional<Pin*> pin = instance->getPin(instance_pin_name.c_str());
      if (pin) {
        new_net.addPinPort(*pin);
      } else {
        LOG_FATAL << "pin: " << (*pin)->getFullName() << " not found.";
      }

      auto* idb_instance =
          _idb_design->get_instance_list()->find_instance(instance_name);
      auto* idb_pin =
          idb_instance->get_pin_list()->find_pin_by_term(instance_pin_name);

      dnet->add_instance_pin(idb_pin);
      idb_pin->set_net(dnet);
    } else {
      auto* idb_pin = _idb_design->get_io_pin_list()->find_pin(sink_pin_name);
      dnet->add_io_pin(idb_pin);
      idb_pin->set_net(dnet);
    }
  }
  auto& created_net = design_netlist->addNet(std::move(new_net));
  crossRef(&created_net, dnet);
  return &created_net;
}

/**
 * @brief remove net.
 *
 * @param sta_net
 */
void TimingIDBAdapter::deleteNet(Net* sta_net) {
  IdbNetList* dbnet_list = _idb_design->get_net_list();
  IdbNet* dnet = dbnet_list->find_net(sta_net->get_name());

  auto* design_netlist = getNetlist();
  design_netlist->removeNet(sta_net);
  removeCrossRef(sta_net, dnet);

  std::string dnet_name = dnet->get_net_name();
  dbnet_list->remove_net(dnet_name);
}

/**
 * @brief config sta the need link cell to speed up load liberty.
 *
 */
void TimingIDBAdapter::configStaLinkCells() {
  std::set<std::string> link_cells;
  auto db_inst_list = _idb_design->get_instance_list()->get_instance_list();
  for (auto* db_inst : db_inst_list) {
    std::string liberty_cell_name = db_inst->get_cell_master()->get_name();
    link_cells.insert(std::move(liberty_cell_name));
  }

  _ista->addLinkCells(std::move(link_cells));
}

/**
 * @brief convert the idb to timing netlist.
 *
 * @return unsigned
 */
unsigned TimingIDBAdapter::convertDBToTimingNetlist(bool link_all_cell) {
  // reset all net to rc net
  _ista->resetAllRcNet();

  _ista->resetNetlist();
  Netlist& design_netlist = *(_ista->get_netlist());

  auto* def_service = _idb->get_def_service();
  if (!def_service) {
    return 0;
  }

  // link liberty lazy to build netlist.
  if (!link_all_cell) {
    configStaLinkCells();
  }

  _ista->linkLibertys();

  _ista->set_design_name(_idb_design->get_design_name().c_str());
  int dbu = _idb_design->get_units()->get_micron_dbu();
  set_dbu(dbu);
  double width = _idb_design->get_layout()->get_die()->get_width() /
                 static_cast<double>(dbu);
  double height = _idb_design->get_layout()->get_die()->get_height() /
                  static_cast<double>(dbu);
  design_netlist.set_core_size(width, height);

  LOG_INFO << "core area width " << width << "um"
           << " height " << height << "um";

  auto build_insts = [this, &design_netlist, dbu]() {
    // build insts
    auto db_inst_list = _idb_design->get_instance_list()->get_instance_list();
    for (auto* db_inst : db_inst_list) {
      std::string raw_name = db_inst->get_name();
      std::regex re(R"(\\)");
      std::string inst_name = std::regex_replace(raw_name, re, "");

      std::string liberty_cell_name = db_inst->get_cell_master()->get_name();
      auto* inst_cell = _ista->findLibertyCell(liberty_cell_name.c_str());

      if (!inst_cell) {
        LOG_INFO_FIRST_N(10)
            << "liberty cell " << liberty_cell_name << " is not exist.";
        continue;
      }

      Instance sta_inst(inst_name.c_str(), inst_cell);

      double x = db_inst->get_coordinate()->get_x() / static_cast<double>(dbu);
      double y = db_inst->get_coordinate()->get_y() / static_cast<double>(dbu);

      sta_inst.set_coordinate(x, y);

      // build inst pin
      auto db_inst_pin_list = db_inst->get_pin_list()->get_pin_list();
      for (auto* db_inst_pin : db_inst_pin_list) {
        if ((db_inst_pin->get_term()->get_type() == IdbConnectType::kPower) ||
            (db_inst_pin->get_term()->get_type() == IdbConnectType::kGround)) {
          continue;
        }
        std::string cell_port_name = db_inst_pin->get_term_name();
        auto [port_base_name, index] =
            Str::matchBusName(cell_port_name.c_str());
        auto* library_port_or_port_bus =
            inst_cell->get_cell_port_or_port_bus(port_base_name.c_str());

        LOG_INFO_IF(!library_port_or_port_bus)
            << cell_port_name << " port is not found in lib cell "
            << inst_cell->get_cell_name() << " of "
            << inst_cell->get_owner_lib()->get_file_name();

        std::unique_ptr<PinBus> pin_bus;
        PinBus* found_pin_bus = nullptr;
        if (library_port_or_port_bus) {
          LibPort* library_port = nullptr;

          if (!library_port_or_port_bus->isLibertyPortBus()) {
            library_port = library_port_or_port_bus;
          } else {
            // port bus
            auto* library_port_bus =
                dynamic_cast<LibPortBus*>(library_port_or_port_bus);
            library_port = (*library_port_bus)[index.value()];

            found_pin_bus = sta_inst.findPinBus(port_base_name);

            if (!found_pin_bus) {
              auto bus_size =
                  dynamic_cast<LibPortBus*>(library_port_bus)->getBusSize();
              LOG_FATAL_IF(!bus_size)
                  << library_port_bus->get_port_name() << " bus size is empty.";
              pin_bus = std::make_unique<PinBus>(port_base_name.c_str(),
                                                 bus_size, 0, bus_size);
            }
          }

          auto* inst_pin =
              sta_inst.addPin(cell_port_name.c_str(), library_port);
          crossRef(inst_pin, db_inst_pin);

          if (pin_bus) {
            pin_bus->addPin(index.value(), inst_pin);
            sta_inst.addPinBus(std::move(pin_bus));
          } else if (found_pin_bus) {
            found_pin_bus->addPin(index.value(), inst_pin);
          }
        }
      }
      auto& created_inst = design_netlist.addInstance(std::move(sta_inst));
      crossRef(&created_inst, db_inst);

      LOG_INFO_EVERY_N(10000)
          << "build inst num: " << design_netlist.getInstanceNum();
    }
  };

  auto build_ports = [this, &design_netlist]() {
    //  build ports
    auto db_ports = _idb_design->get_io_pin_list()->get_pin_list();
    for (auto* db_port : db_ports) {
      std::string port_name = db_port->get_term_name();
      auto io_type = dbToSta(db_port->get_term()->get_type(),
                             db_port->get_term()->get_direction());
      Port sta_port(port_name.c_str(), io_type);
      auto& created_port = design_netlist.addPort(std::move(sta_port));
      crossRef(&created_port, db_port);
    }
  };

  auto build_nets = [this, &design_netlist]() {
    // build nets

    auto process_net = [this, &design_netlist]<typename T>(T* db_net) {
      std::string raw_name = db_net->get_net_name();
      if ((db_net->get_connect_type() == IdbConnectType::kPower) ||
          (db_net->get_connect_type() == IdbConnectType::kGround)) {
        return;
      }

      std::regex re(R"(\\)"); 
      std::string net_name = std::regex_replace(raw_name, re, "");
      Net* sta_net = design_netlist.findNet(net_name.c_str());

      auto instance_pin_list = db_net->get_instance_pin_list()->get_pin_list();
      for (auto* instance_pin : instance_pin_list) {
        std::string cell_port_name = instance_pin->get_term_name();
        auto* db_inst = instance_pin->get_instance();
        std::string raw_name = db_inst->get_name();
        std::regex re(R"(\\)");
        std::string inst_name = std::regex_replace(raw_name, re, "");

        auto* sta_inst = design_netlist.findInstance(inst_name.c_str());
        LOG_FATAL_IF(!sta_inst) << "Instance " << inst_name << " not found";
        auto inst_pin = sta_inst->getPin(cell_port_name.c_str());
        LOG_FATAL_IF(!inst_pin)
            << "Instance " << sta_inst->getFullName() << " cell Pin "
            << cell_port_name << " not found for cell "
            << sta_inst->get_inst_cell()->get_cell_name();

        if (sta_net) {
          sta_net->addPinPort(*inst_pin);
        } else {
          // DLOG_INFO << "create net " << net_name;
          auto& created_net = design_netlist.addNet(Net(net_name.c_str()));

          created_net.addPinPort(*inst_pin);
          sta_net = &created_net;
          crossRef(sta_net, db_net);
        }
      }

      auto* io_pins = db_net->get_io_pins();
      for (auto* io_pin : io_pins->get_pin_list()) {
        std::string port_name = io_pin->get_term_name();
        if (auto* design_port = design_netlist.findPort(port_name.c_str());
            design_port) {
          if (sta_net) {
            sta_net->addPinPort(design_port);
          } else {
            // DLOG_INFO << "create net " << net_name;
            auto& created_net = design_netlist.addNet(Net(net_name.c_str()));

            created_net.addPinPort(design_port);
            sta_net = &created_net;
            crossRef(sta_net, db_net);
          }
        }
      }

      LOG_INFO_EVERY_N(10000)
          << "build net num: " << design_netlist.getNetNum();
    };

    auto db_net_list = _idb_design->get_net_list()->get_net_list();
    for (auto* db_net : db_net_list) {
      process_net(db_net);
    }

    auto db_special_nets = _idb_design->get_special_net_list()->get_net_list();
    for (auto* db_special_net : db_special_nets) {
      process_net(db_special_net);
    }
  };

  build_insts();
  build_ports();
  build_nets();

  LOG_INFO << "build instance num: " << design_netlist.getInstanceNum();
  LOG_INFO << "build port num: " << design_netlist.getPortNum();
  LOG_INFO << "build net num: " << design_netlist.getNetNum();

  return 1;
}

/**
 * @brief sta bus net do not contain \[\], need change [] to match idb net
 * name.
 *
 * @param sta_net_name
 * @return std::string
 */
std::string TimingIDBAdapter::changeStaBusNetNameToIdb(
    std::string sta_net_name) {
  return Str::addBackslash(sta_net_name);
}

}  // namespace ista

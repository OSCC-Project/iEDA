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
 * @project		iDB
 * @file		IdbDesign.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe def.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbDesign.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace idb {

namespace {

std::string jsonEscape(const std::string& text)
{
  std::ostringstream os;
  for (char ch : text) {
    switch (ch) {
      case '\\':
        os << "\\\\";
        break;
      case '"':
        os << "\\\"";
        break;
      case '\n':
        os << "\\n";
        break;
      case '\r':
        os << "\\r";
        break;
      case '\t':
        os << "\\t";
        break;
      default:
        os << ch;
        break;
    }
  }
  return os.str();
}

std::string pinDisplayName(IdbPin* pin)
{
  if (pin == nullptr) {
    return "<null>";
  }
  if (pin->get_instance() != nullptr) {
    return pin->get_instance()->get_name() + "/" + pin->get_pin_name();
  }
  return pin->get_pin_name();
}

void pushLimitedMessage(std::vector<std::string>& messages, const std::string& message)
{
  if (messages.size() < 128) {
    messages.emplace_back(message);
  }
}

void refreshPinNetName(IdbPin* pin)
{
  if (pin == nullptr) {
    return;
  }
  if (pin->get_net() != nullptr) {
    pin->set_net_name(pin->get_net()->get_net_name());
  } else if (pin->get_special_net() != nullptr) {
    pin->set_net_name(pin->get_special_net()->get_net_name());
  } else {
    pin->set_net_name("");
  }
}

}  // namespace

IdbDesign::IdbDesign(IdbLayout* layout)
{
  //!----tbd----------
  _layout = layout;
  // new method will be replaced in future
  _design_name = "";
  _units = new IdbUnits();

  //  _row = new IdbRow();

  _instance_list = new IdbInstanceList();
  _io_pin_list = new IdbPins();
  _net_list = new IdbNetList();
  _via_list = new IdbVias();
  _blockage_list = new IdbBlockageList();
  _slot_list = new IdbSlotList();
  _group_list = new IdbGroupList();
  _special_net_list = new IdbSpecialNetList();
  _region_list = new IdbRegionList();
  _fill_list = new IdbFillList();
  _bus_bit_chars = new IdbBusBitChars();

  _bus_list = new IdbBusList();
}

IdbDesign::~IdbDesign()
{
  if (_instance_list != nullptr) {
    delete _instance_list;
    _instance_list = nullptr;
  }
  if (_io_pin_list != nullptr) {
    delete _io_pin_list;
    _io_pin_list = nullptr;
  }
  if (_net_list != nullptr) {
    delete _net_list;
    _net_list = nullptr;
  }
  if (_via_list != nullptr) {
    delete _via_list;
    _via_list = nullptr;
  }
  if (_blockage_list != nullptr) {
    delete _blockage_list;
    _blockage_list = nullptr;
  }
  if (_slot_list != nullptr) {
    delete _slot_list;
    _slot_list = nullptr;
  }
  if (_group_list != nullptr) {
    delete _group_list;
    _group_list = nullptr;
  }
  if (_special_net_list != nullptr) {
    delete _special_net_list;
    _special_net_list = nullptr;
  }
  if (_region_list != nullptr) {
    delete _region_list;
    _region_list = nullptr;
  }
  if (_fill_list != nullptr) {
    delete _fill_list;
    _fill_list = nullptr;
  }
  if (_bus_bit_chars != nullptr) {
    delete _bus_bit_chars;
    _bus_bit_chars = nullptr;
  }
  if (_bus_list != nullptr) {
    delete _bus_list;
    _bus_list = nullptr;
  }
}

// void IdbDesign::createDefaultVias(IdbLayers* layers)
// {
//   for (IdbLayer* layer : layers->get_cut_layers()) {
//     IdbLayerCut* cut_layer = dynamic_cast<IdbLayerCut*>(layer);
//     string via_name = cut_layer->get_name() + "_Generate";
//     _via_list->createViaDefault(via_name, cut_layer);
//   }
// }

IdbPin* IdbDesign::createOrFindIoPin(const std::string& pin_name, IdbCreatePolicy policy)
{
  if (_io_pin_list == nullptr || pin_name.empty()) {
    return nullptr;
  }

  IdbPin* existed_pin = _io_pin_list->find_pin(pin_name);
  if (existed_pin != nullptr) {
    if (policy == IdbCreatePolicy::kErrorIfExists) {
      return nullptr;
    }
    if (policy == IdbCreatePolicy::kReturnExisting) {
      return existed_pin;
    }
    disconnectPinFromNet(existed_pin);
    disconnectPinFromSpecialNet(existed_pin);
    _io_pin_list->delete_pin(existed_pin);
  }

  IdbPin* pin = _io_pin_list->add_pin_list(pin_name);
  if (pin != nullptr) {
    pin->set_as_io();
  }
  return pin;
}

bool IdbDesign::removeIoPinSafe(const std::string& pin_name)
{
  if (_io_pin_list == nullptr) {
    return false;
  }

  return removeIoPinSafe(_io_pin_list->find_pin(pin_name));
}

bool IdbDesign::removeIoPinSafe(IdbPin* pin)
{
  if (_io_pin_list == nullptr || pin == nullptr) {
    return false;
  }

  disconnectPinFromNet(pin);
  disconnectPinFromSpecialNet(pin);
  return _io_pin_list->delete_pin(pin);
}

IdbInstance* IdbDesign::createInstance(const std::string& inst_name, const std::string& master_name, IdbInstanceType type,
                                       IdbPlacementStatus status, IdbOrient orient, int32_t coord_x, int32_t coord_y,
                                       IdbCreatePolicy policy)
{
  if (_layout == nullptr || _layout->get_cell_master_list() == nullptr || _instance_list == nullptr || inst_name.empty()
      || master_name.empty()) {
    return nullptr;
  }

  IdbCellMaster* cell_master = _layout->get_cell_master_list()->find_cell_master(master_name);
  if (cell_master == nullptr) {
    return nullptr;
  }

  IdbInstance* existed_inst = _instance_list->find_instance(inst_name);
  if (existed_inst != nullptr) {
    if (policy == IdbCreatePolicy::kErrorIfExists) {
      return nullptr;
    }
    if (policy == IdbCreatePolicy::kReplaceExisting) {
      removeInstanceSafe(inst_name);
    } else {
      return existed_inst;
    }
  }

  IdbInstance* instance = _instance_list->add_instance(inst_name);
  if (instance == nullptr) {
    return nullptr;
  }

  instance->set_cell_master(cell_master);
  instance->set_type(type);
  instance->set_orient(orient, false);
  instance->set_coodinate(coord_x, coord_y, false);
  instance->set_status(status);
  if (orient != IdbOrient::kNone && status != IdbPlacementStatus::kNone) {
    instance->set_bounding_box();
    instance->set_pin_list_coodinate();
    instance->set_halo_coodinate();
    instance->set_obs_box_list();
  }

  return instance;
}

bool IdbDesign::placeInstance(const std::string& inst_name, int32_t coord_x, int32_t coord_y, IdbOrient orient, IdbPlacementStatus status)
{
  if (_instance_list == nullptr || orient == IdbOrient::kNone) {
    return false;
  }

  IdbInstance* instance = _instance_list->find_instance(inst_name);
  if (instance == nullptr || instance->get_cell_master() == nullptr) {
    return false;
  }

  instance->set_orient(orient, false);
  instance->set_coodinate(coord_x, coord_y, false);
  instance->set_status(status);
  instance->set_bounding_box();
  instance->set_pin_list_coodinate();
  instance->set_halo_coodinate();
  instance->set_obs_box_list();
  return true;
}

bool IdbDesign::replaceInstanceMaster(const std::string& inst_name, const std::string& master_name, bool preserve_connection)
{
  if (_layout == nullptr || _layout->get_cell_master_list() == nullptr || _instance_list == nullptr || inst_name.empty()
      || master_name.empty()) {
    return false;
  }

  IdbInstance* instance = _instance_list->find_instance(inst_name);
  IdbCellMaster* cell_master = _layout->get_cell_master_list()->find_cell_master(master_name);
  if (instance == nullptr || cell_master == nullptr) {
    return false;
  }

  struct PinConnection
  {
    std::string pin_name;
    std::string term_name;
    IdbNet* regular_net = nullptr;
    IdbSpecialNet* special_net = nullptr;
  };

  std::vector<PinConnection> pin_connections;
  if (preserve_connection && instance->get_pin_list() != nullptr) {
    for (auto* pin : instance->get_pin_list()->get_pin_list()) {
      if (pin == nullptr) {
        continue;
      }
      PinConnection connection;
      connection.pin_name = pin->get_pin_name();
      if (pin->get_term() != nullptr) {
        connection.term_name = pin->get_term_name();
      }
      connection.regular_net = pin->get_net();
      connection.special_net = pin->get_special_net();
      pin_connections.emplace_back(connection);
    }
  }

  if (instance->get_pin_list() != nullptr) {
    std::vector<IdbPin*> old_pins = instance->get_pin_list()->get_pin_list();
    for (auto* pin : old_pins) {
      disconnectPinFromNet(pin);
      disconnectPinFromSpecialNet(pin);
    }
    instance->get_pin_list()->reset();
  }

  instance->set_cell_master(cell_master);
  if (preserve_connection && instance->get_pin_list() != nullptr) {
    for (const auto& connection : pin_connections) {
      IdbPin* new_pin = nullptr;
      if (!connection.term_name.empty()) {
        new_pin = instance->get_pin_by_term(connection.term_name);
      }
      if (new_pin == nullptr && !connection.pin_name.empty()) {
        new_pin = instance->get_pin(connection.pin_name);
      }
      if (new_pin == nullptr) {
        continue;
      }
      if (connection.regular_net != nullptr) {
        connectPinToNet(new_pin, connection.regular_net);
      }
      if (connection.special_net != nullptr) {
        connectPinToSpecialNet(new_pin, connection.special_net);
      }
    }
  }

  if (instance->get_orient() != IdbOrient::kNone && instance->get_status() != IdbPlacementStatus::kNone) {
    instance->set_bounding_box();
    instance->set_pin_list_coodinate();
    instance->set_halo_coodinate();
    instance->set_obs_box_list();
  }

  return true;
}

std::string IdbDesign::makeUniqueInstanceName(const std::string& prefix) const
{
  if (_instance_list == nullptr) {
    return prefix;
  }

  std::string base_name = prefix.empty() ? "inst" : prefix;
  if (_instance_list->find_instance(base_name) == nullptr) {
    return base_name;
  }

  for (uint64_t index = 0;; ++index) {
    std::string candidate = base_name + std::to_string(index);
    if (_instance_list->find_instance(candidate) == nullptr) {
      return candidate;
    }
  }
}

bool IdbDesign::removeInstanceSafe(const std::string& inst_name)
{
  if (_instance_list == nullptr) {
    return false;
  }

  IdbInstance* instance = _instance_list->find_instance(inst_name);
  if (instance == nullptr) {
    return false;
  }

  std::vector<IdbPin*> pin_list = instance->get_pin_list()->get_pin_list();
  for (auto* pin : pin_list) {
    disconnectPinFromNet(pin);
    disconnectPinFromSpecialNet(pin);
  }

  if (_region_list != nullptr) {
    for (auto* region : _region_list->get_region_list()) {
      if (region != nullptr) {
        region->remove_instance(instance);
      }
    }
  }
  instance->set_region(nullptr);

  if (_group_list != nullptr) {
    for (auto* group : _group_list->get_group_list()) {
      if (group != nullptr) {
        group->remove_instance(instance);
      }
    }
  }

  return _instance_list->remove_instance(inst_name);
}

IdbNet* IdbDesign::createOrFindNet(const std::string& net_name, IdbConnectType type, IdbCreatePolicy policy)
{
  if (_net_list == nullptr || net_name.empty()) {
    return nullptr;
  }

  IdbNet* existed_net = _net_list->find_net(net_name);
  if (existed_net != nullptr) {
    if (policy == IdbCreatePolicy::kErrorIfExists) {
      return nullptr;
    }
    if (policy == IdbCreatePolicy::kReplaceExisting) {
      removeNetSafe(net_name);
    } else {
      if (type != IdbConnectType::kNone) {
        existed_net->set_connect_type(type);
      }
      return existed_net;
    }
  }

  return _net_list->add_net(net_name, type);
}

std::string IdbDesign::makeUniqueNetName(const std::string& prefix) const
{
  if (_net_list == nullptr) {
    return prefix;
  }

  std::string base_name = prefix.empty() ? "net" : prefix;
  if (_net_list->find_net(base_name) == nullptr) {
    return base_name;
  }

  for (uint64_t index = 0;; ++index) {
    std::string candidate = base_name + std::to_string(index);
    if (_net_list->find_net(candidate) == nullptr) {
      return candidate;
    }
  }
}

bool IdbDesign::setNetConnectType(const std::string& net_name, IdbConnectType type)
{
  if (_net_list == nullptr) {
    return false;
  }

  IdbNet* net = _net_list->find_net(net_name);
  if (net == nullptr) {
    return false;
  }

  net->set_connect_type(type);
  return true;
}

bool IdbDesign::disconnectPinFromNet(IdbPin* pin)
{
  if (pin == nullptr) {
    return false;
  }

  IdbNet* net = pin->get_net();
  if (net == nullptr) {
    refreshPinNetName(pin);
    return true;
  }

  net->erase_pin_ref(pin);

  IdbInstance* instance = pin->get_instance();
  if (instance != nullptr) {
    bool keep_instance = false;
    for (auto* net_pin : net->get_instance_pin_list()->get_pin_list()) {
      if (net_pin != nullptr && net_pin->get_instance() == instance) {
        keep_instance = true;
        break;
      }
    }
    if (!keep_instance) {
      net->erase_instance_ref(instance);
    }
  }

  pin->remove_net();
  refreshPinNetName(pin);
  return true;
}

bool IdbDesign::connectPinToNet(IdbPin* pin, IdbNet* net)
{
  if (pin == nullptr || net == nullptr) {
    return false;
  }

  if (pin->get_net() != nullptr && pin->get_net() != net) {
    disconnectPinFromNet(pin);
  }

  if (pin->is_io_pin()) {
    net->add_io_pin_unique(pin);
  } else {
    net->add_instance_pin_unique(pin);
    IdbInstance* instance = pin->get_instance();
    if (instance != nullptr) {
      net->get_instance_list()->add_instance_ref(instance);
    }
  }

  pin->set_net(net);
  pin->set_net_name(net->get_net_name());
  return true;
}

bool IdbDesign::connectIoPinToNet(const std::string& io_pin_name, const std::string& net_name)
{
  if (_io_pin_list == nullptr || _net_list == nullptr) {
    return false;
  }

  return connectPinToNet(_io_pin_list->find_pin(io_pin_name), _net_list->find_net(net_name));
}

bool IdbDesign::connectInstancePinToNet(const std::string& inst_name, const std::string& pin_name, const std::string& net_name)
{
  if (_instance_list == nullptr || _net_list == nullptr) {
    return false;
  }

  IdbInstance* instance = _instance_list->find_instance(inst_name);
  IdbNet* net = _net_list->find_net(net_name);
  if (instance == nullptr || net == nullptr) {
    return false;
  }

  IdbPin* pin = instance->get_pin(pin_name);
  if (pin == nullptr) {
    pin = instance->get_pin_by_term(pin_name);
  }

  return connectPinToNet(pin, net);
}

bool IdbDesign::removeNetSafe(const std::string& net_name)
{
  if (_net_list == nullptr) {
    return false;
  }

  IdbNet* net = _net_list->find_net(net_name);
  if (net == nullptr) {
    return false;
  }

  std::vector<IdbPin*> pin_list;
  auto& io_pins = net->get_io_pins()->get_pin_list();
  auto& inst_pins = net->get_instance_pin_list()->get_pin_list();
  pin_list.insert(pin_list.end(), io_pins.begin(), io_pins.end());
  pin_list.insert(pin_list.end(), inst_pins.begin(), inst_pins.end());
  for (auto* pin : pin_list) {
    disconnectPinFromNet(pin);
  }
  net->clear_wire_list();

  return _net_list->remove_net_only(net_name);
}

bool IdbDesign::renameNet(IdbNet* net, const std::string& new_name)
{
  if (_net_list == nullptr) {
    return false;
  }

  return _net_list->rename_net(net, new_name);
}

bool IdbDesign::mergeNetInto(const std::string& target_net_name, const std::string& source_net_name, bool move_wires)
{
  if (_net_list == nullptr || target_net_name == source_net_name) {
    return false;
  }

  IdbNet* target_net = _net_list->find_net(target_net_name);
  IdbNet* source_net = _net_list->find_net(source_net_name);
  if (target_net == nullptr || source_net == nullptr) {
    return false;
  }

  std::vector<IdbPin*> pin_list;
  auto& io_pins = source_net->get_io_pins()->get_pin_list();
  auto& inst_pins = source_net->get_instance_pin_list()->get_pin_list();
  pin_list.insert(pin_list.end(), io_pins.begin(), io_pins.end());
  pin_list.insert(pin_list.end(), inst_pins.begin(), inst_pins.end());
  for (auto* pin : pin_list) {
    connectPinToNet(pin, target_net);
  }

  if (move_wires && target_net->get_wire_list() != nullptr && source_net->get_wire_list() != nullptr) {
    auto& target_wires = target_net->get_wire_list()->get_wire_list();
    auto& source_wires = source_net->get_wire_list()->get_wire_list();
    target_wires.insert(target_wires.end(), source_wires.begin(), source_wires.end());
    source_wires.clear();
  }

  return removeNetSafe(source_net_name);
}

IdbSpecialNet* IdbDesign::createOrFindSpecialNet(const std::string& net_name, IdbConnectType type, IdbCreatePolicy policy)
{
  if (_special_net_list == nullptr || net_name.empty()) {
    return nullptr;
  }

  IdbSpecialNet* existed_net = _special_net_list->find_net(net_name);
  if (existed_net != nullptr) {
    if (policy == IdbCreatePolicy::kErrorIfExists) {
      return nullptr;
    }
    if (policy == IdbCreatePolicy::kReplaceExisting) {
      removeSpecialNetSafe(net_name);
    } else {
      if (type != IdbConnectType::kNone) {
        existed_net->set_connect_type(type);
      }
      return existed_net;
    }
  }

  IdbSpecialNet* net = _special_net_list->add_net(net_name);
  if (net != nullptr && type != IdbConnectType::kNone) {
    net->set_connect_type(type);
  }
  return net;
}

bool IdbDesign::disconnectPinFromSpecialNet(IdbPin* pin)
{
  if (pin == nullptr) {
    return false;
  }

  IdbSpecialNet* net = pin->get_special_net();
  if (net == nullptr) {
    refreshPinNetName(pin);
    return true;
  }

  net->erase_pin_ref(pin);
  IdbInstance* instance = pin->get_instance();
  if (instance != nullptr) {
    bool keep_instance = false;
    for (auto* net_pin : net->get_instance_pin_list()->get_pin_list()) {
      if (net_pin != nullptr && net_pin->get_instance() == instance) {
        keep_instance = true;
        break;
      }
    }
    if (!keep_instance) {
      net->erase_instance_ref(instance);
    }
  }

  pin->remove_special_net();
  refreshPinNetName(pin);
  return true;
}

bool IdbDesign::connectPinToSpecialNet(IdbPin* pin, IdbSpecialNet* net)
{
  if (pin == nullptr || net == nullptr) {
    return false;
  }

  if (pin->get_special_net() != nullptr && pin->get_special_net() != net) {
    disconnectPinFromSpecialNet(pin);
  }

  if (pin->is_io_pin()) {
    net->add_io_pin(pin);
  } else {
    net->add_instance_pin(pin);
    IdbInstance* instance = pin->get_instance();
    if (instance != nullptr) {
      net->add_instance(instance);
    }
  }
  pin->set_special_net(net);
  refreshPinNetName(pin);
  return true;
}

IdbSpecialNet* IdbDesign::findSpecialNetForInstancePin(IdbPin* pin) const
{
  if (pin == nullptr || pin->is_io_pin()) {
    return nullptr;
  }

  if (pin->get_special_net() != nullptr) {
    return pin->get_special_net();
  }

  if (_special_net_list == nullptr || pin->get_term() == nullptr) {
    return nullptr;
  }

  const std::string term_name = pin->get_term_name();
  for (auto* net : _special_net_list->get_net_list()) {
    if (net != nullptr && net->matches_wildcard_instance_pin(term_name)) {
      return net;
    }
  }

  return nullptr;
}

int32_t IdbDesign::connectInstancePinsToSpecialNet(const std::vector<std::string>& pin_name_list, IdbSpecialNet* net)
{
  if (_instance_list == nullptr || net == nullptr || pin_name_list.empty()) {
    return 0;
  }

  std::unordered_set<std::string> pin_names(pin_name_list.begin(), pin_name_list.end());
  int32_t number = 0;
  for (auto* instance : _instance_list->get_instance_list()) {
    if (instance == nullptr || instance->get_pin_list() == nullptr) {
      continue;
    }

    for (auto* pin : instance->get_pin_list()->get_pin_list()) {
      if (pin == nullptr) {
        continue;
      }

      if (pin->get_term() != nullptr && pin_names.find(pin->get_term_name()) != pin_names.end() && connectPinToSpecialNet(pin, net)) {
        ++number;
      }
    }
  }

  return number;
}

int32_t IdbDesign::materializeSpecialNetWildcardPins(IdbSpecialNet* net)
{
  if (_instance_list == nullptr || net == nullptr || !net->has_wildcard_instance_pins()) {
    return 0;
  }

  std::unordered_set<std::string> wildcard_terms(net->get_pin_string_list().begin(), net->get_pin_string_list().end());
  std::unordered_map<std::string, std::vector<IdbPin*>> pins_by_term;

  for (auto* instance : _instance_list->get_instance_list()) {
    if (instance == nullptr || instance->get_pin_list() == nullptr) {
      continue;
    }
    for (auto* pin : instance->get_pin_list()->get_pin_list()) {
      if (pin == nullptr || pin->get_term() == nullptr) {
        continue;
      }
      const std::string term_name = pin->get_term_name();
      if (wildcard_terms.find(term_name) != wildcard_terms.end()) {
        pins_by_term[term_name].emplace_back(pin);
      }
    }
  }

  int32_t number = 0;
  for (const auto& term : wildcard_terms) {
    auto iter = pins_by_term.find(term);
    if (iter == pins_by_term.end()) {
      continue;
    }
    for (auto* pin : iter->second) {
      if (!net->has_instance_pin(pin) && connectPinToSpecialNet(pin, net)) {
        ++number;
      }
    }
  }

  return number;
}

int32_t IdbDesign::materializeAllSpecialNetWildcardPins()
{
  if (_special_net_list == nullptr || _instance_list == nullptr) {
    return 0;
  }

  std::unordered_map<std::string, std::vector<IdbSpecialNet*>> nets_by_term;
  for (auto* net : _special_net_list->get_net_list()) {
    if (net == nullptr || !net->has_wildcard_instance_pins()) {
      continue;
    }
    for (const auto& term : net->get_pin_string_list()) {
      nets_by_term[term].emplace_back(net);
    }
  }

  if (nets_by_term.empty()) {
    return 0;
  }

  int32_t number = 0;
  for (auto* instance : _instance_list->get_instance_list()) {
    if (instance == nullptr || instance->get_pin_list() == nullptr) {
      continue;
    }
    for (auto* pin : instance->get_pin_list()->get_pin_list()) {
      if (pin == nullptr || pin->get_term() == nullptr) {
        continue;
      }
      auto iter = nets_by_term.find(pin->get_term_name());
      if (iter == nets_by_term.end()) {
        continue;
      }
      for (auto* net : iter->second) {
        if (net != nullptr && !net->has_instance_pin(pin) && connectPinToSpecialNet(pin, net)) {
          ++number;
          break;
        }
      }
    }
  }

  return number;
}

bool IdbDesign::removeSpecialNetSafe(const std::string& net_name)
{
  if (_special_net_list == nullptr) {
    return false;
  }

  IdbSpecialNet* net = _special_net_list->find_net(net_name);
  if (net == nullptr) {
    return false;
  }

  std::vector<IdbPin*> pin_list;
  auto& io_pins = net->get_io_pin_list()->get_pin_list();
  auto& inst_pins = net->get_instance_pin_list()->get_pin_list();
  pin_list.insert(pin_list.end(), io_pins.begin(), io_pins.end());
  pin_list.insert(pin_list.end(), inst_pins.begin(), inst_pins.end());
  for (auto* pin : pin_list) {
    disconnectPinFromSpecialNet(pin);
  }

  return _special_net_list->remove_net(net_name);
}

void IdbDesign::mergeAllNetWires()
{
  if (_net_list == nullptr) {
    return;
  }

#pragma omp parallel for schedule(dynamic)
  for (auto* net : _net_list->get_net_list()) {
    if (net != nullptr) {
      net->mergeWireSegments();
    }
  }
}

IdbPlacementBlockage* IdbDesign::addPlacementBlockage(int32_t llx, int32_t lly, int32_t urx, int32_t ury)
{
  if (_layout == nullptr || _blockage_list == nullptr) {
    return nullptr;
  }

  auto idb_core = _layout->get_core();
  IdbPlacementBlockage* pl_blockage = _blockage_list->add_blockage_placement();
  if (idb_core == nullptr || pl_blockage == nullptr) {
    return pl_blockage;
  }

  IdbRect* core = idb_core->get_bounding_box();
  if (core != nullptr && core->isIntersection(IdbRect(llx, lly, urx, ury))) {
    std::vector<int32_t> rect_x{llx, urx, core->get_low_x(), core->get_high_x()};
    std::vector<int32_t> rect_y{lly, ury, core->get_low_y(), core->get_high_y()};
    std::sort(rect_x.begin(), rect_x.end());
    std::sort(rect_y.begin(), rect_y.end());
    pl_blockage->add_rect(rect_x[1], rect_y[1], rect_x[2], rect_y[2]);
  } else {
    pl_blockage->add_rect(llx, lly, urx, ury);
  }

  return pl_blockage;
}

void IdbDesign::addPlacementHalo(const std::string& instance_name, int32_t distance_top, int32_t distance_bottom, int32_t distance_left,
                                 int32_t distance_right)
{
  if (_instance_list == nullptr) {
    return;
  }

  auto add_halo = [&](IdbInstance* instance) {
    if (instance == nullptr || instance->is_unplaced()) {
      return;
    }
    instance->set_bounding_box();
    int blk_llx = instance->get_bounding_box()->get_low_x() - distance_left;
    int blk_urx = instance->get_bounding_box()->get_high_x() + distance_right;
    int blk_lly = instance->get_bounding_box()->get_low_y() - distance_bottom;
    int blk_ury = instance->get_bounding_box()->get_high_y() + distance_top;
    addPlacementBlockage(blk_llx, blk_lly, blk_urx, blk_ury);
  };

  if (!instance_name.empty() && instance_name != "all") {
    add_halo(_instance_list->find_instance(instance_name));
  } else if (instance_name == "all") {
    for (auto* instance : _instance_list->get_instance_list()) {
      add_halo(instance);
    }
  }
}

void IdbDesign::removeBlockageExceptPGNet()
{
  if (_blockage_list != nullptr) {
    _blockage_list->removeExceptPgNetBlockageList();
  }
}

void IdbDesign::clearBlockage(std::string type)
{
  if (_blockage_list == nullptr) {
    return;
  }

  if (type == "routing") {
    _blockage_list->clearRoutingBlockage();
  } else if (type == "placement") {
    _blockage_list->clearPlacementBlockage();
  } else {
    _blockage_list->reset();
  }
}

void IdbDesign::addRoutingBlockage(int32_t llx, int32_t lly, int32_t urx, int32_t ury, const std::vector<std::string>& layers,
                                  const bool& is_except_pgnet)
{
  if (_layout == nullptr || _blockage_list == nullptr || _layout->get_layers() == nullptr) {
    return;
  }

  auto layer_list = _layout->get_layers();
  for (const std::string& layer : layers) {
    IdbRoutingBlockage* rt_blockage = _blockage_list->add_blockage_routing(layer);
    if (rt_blockage == nullptr) {
      continue;
    }
    rt_blockage->set_layer(layer_list->find_layer(layer));
    rt_blockage->add_rect(llx, lly, urx, ury);
    rt_blockage->set_except_pgnet(is_except_pgnet);
  }
}

void IdbDesign::addRoutingHalo(const std::string& instance_name, const std::vector<std::string>& layers, int32_t distance_top,
                               int32_t distance_bottom, int32_t distance_left, int32_t distance_right, const bool& is_except_pgnet)
{
  if (_instance_list == nullptr) {
    return;
  }

  auto add_halo = [&](IdbInstance* instance) {
    if (instance == nullptr || instance->is_unplaced()) {
      return;
    }
    instance->set_bounding_box();
    int blk_llx = instance->get_bounding_box()->get_low_x() - distance_left;
    int blk_urx = instance->get_bounding_box()->get_high_x() + distance_right;
    int blk_lly = instance->get_bounding_box()->get_low_y() - distance_bottom;
    int blk_ury = instance->get_bounding_box()->get_high_y() + distance_top;
    addRoutingBlockage(blk_llx, blk_lly, blk_urx, blk_ury, layers, is_except_pgnet);
  };

  if (!instance_name.empty() && instance_name != "all") {
    add_halo(_instance_list->find_instance(instance_name));
  } else if (instance_name == "all") {
    for (auto* instance : _instance_list->get_instance_list()) {
      add_halo(instance);
    }
  }
}

IdbConnectivityCheckResult IdbDesign::validateConnectivity(bool check_floating) const
{
  IdbConnectivityCheckResult result;
  std::unordered_set<IdbInstance*> design_instances;
  std::unordered_set<IdbPin*> owner_pins;

  if (_instance_list != nullptr) {
    std::set<std::string> inst_names;
    for (auto* inst : _instance_list->get_instance_list()) {
      if (inst == nullptr) {
        continue;
      }
      design_instances.insert(inst);
      if (!inst_names.insert(inst->get_name()).second) {
        result.duplicate_instance_count++;
        pushLimitedMessage(result.messages, "duplicate instance: " + inst->get_name());
      }
      if (inst->get_pin_list() != nullptr) {
        for (auto* pin : inst->get_pin_list()->get_pin_list()) {
          if (pin != nullptr) {
            owner_pins.insert(pin);
          }
        }
      }
    }
  }

  if (_io_pin_list != nullptr) {
    std::set<std::string> io_pin_names;
    for (auto* pin : _io_pin_list->get_pin_list()) {
      if (pin == nullptr) {
        continue;
      }
      owner_pins.insert(pin);
      if (!io_pin_names.insert(pin->get_pin_name()).second) {
        result.duplicate_io_pin_count++;
        pushLimitedMessage(result.messages, "duplicate io pin: " + pin->get_pin_name());
      }
      if (check_floating && pin->get_net() == nullptr && pin->get_special_net() == nullptr) {
        result.floating_pin_count++;
      }
    }
  }

  if (_net_list != nullptr) {
    std::set<std::string> net_names;
    for (auto* net : _net_list->get_net_list()) {
      if (net == nullptr) {
        continue;
      }
      if (!net_names.insert(net->get_net_name()).second) {
        result.duplicate_net_count++;
        pushLimitedMessage(result.messages, "duplicate net: " + net->get_net_name());
      }

      std::set<IdbPin*> pin_refs;
      std::unordered_set<IdbInstance*> insts_from_pins;
      std::vector<IdbPin*> pin_list;
      if (net->get_io_pins() != nullptr) {
        auto& io_pins = net->get_io_pins()->get_pin_list();
        pin_list.insert(pin_list.end(), io_pins.begin(), io_pins.end());
      }
      if (net->get_instance_pin_list() != nullptr) {
        auto& inst_pins = net->get_instance_pin_list()->get_pin_list();
        pin_list.insert(pin_list.end(), inst_pins.begin(), inst_pins.end());
      }
      for (auto* pin : pin_list) {
        if (pin == nullptr) {
          continue;
        }
        if (!owner_pins.contains(pin)) {
          result.stale_regular_pin_ref_count++;
          pushLimitedMessage(result.messages, "regular net " + net->get_net_name() + " has non-owner pin ref " + pinDisplayName(pin));
        }
        if (!pin_refs.insert(pin).second) {
          result.duplicate_pin_ref_count++;
          pushLimitedMessage(result.messages, "regular net " + net->get_net_name() + " has duplicate pin ref " + pinDisplayName(pin));
        }
        if (pin->get_net() != net) {
          result.stale_pin_net_ref_count++;
          pushLimitedMessage(result.messages, "regular pin reverse net mismatch: " + pinDisplayName(pin) + " in " + net->get_net_name());
        }
        if (!pin->get_net_name().empty() && pin->get_net_name() != net->get_net_name()) {
          result.pin_reverse_mismatch_count++;
          pushLimitedMessage(result.messages, "regular pin net name mismatch: " + pinDisplayName(pin) + " has " + pin->get_net_name()
                                                + " but is listed in " + net->get_net_name());
        }
        if (!pin->is_io_pin()) {
          auto* instance = pin->get_instance();
          if (instance == nullptr || !design_instances.contains(instance)) {
            result.stale_regular_pin_ref_count++;
            pushLimitedMessage(result.messages, "regular net " + net->get_net_name() + " has stale instance pin " + pinDisplayName(pin));
          } else {
            insts_from_pins.insert(instance);
          }
        }
      }

      std::unordered_set<IdbInstance*> insts_from_index;
      if (net->get_instance_list() != nullptr) {
        for (auto* inst : net->get_instance_list()->get_instance_list()) {
          if (inst == nullptr) {
            continue;
          }
          insts_from_index.insert(inst);
          if (!insts_from_pins.contains(inst)) {
            result.net_instance_mismatch_count++;
            pushLimitedMessage(result.messages, "regular net " + net->get_net_name() + " has extra instance index " + inst->get_name());
          }
        }
      }
      for (auto* inst : insts_from_pins) {
        if (!insts_from_index.contains(inst)) {
          result.net_instance_mismatch_count++;
          pushLimitedMessage(result.messages, "regular net " + net->get_net_name() + " misses instance index " + inst->get_name());
        }
      }
    }
  }

  if (_special_net_list != nullptr) {
    for (auto* net : _special_net_list->get_net_list()) {
      if (net == nullptr) {
        continue;
      }
      std::set<IdbPin*> pin_refs;
      std::vector<IdbPin*> pin_list;
      if (net->get_io_pin_list() != nullptr) {
        auto& io_pins = net->get_io_pin_list()->get_pin_list();
        pin_list.insert(pin_list.end(), io_pins.begin(), io_pins.end());
      }
      if (net->get_instance_pin_list() != nullptr) {
        auto& inst_pins = net->get_instance_pin_list()->get_pin_list();
        pin_list.insert(pin_list.end(), inst_pins.begin(), inst_pins.end());
      }
      for (auto* pin : pin_list) {
        if (pin == nullptr) {
          continue;
        }
        if (!owner_pins.contains(pin)) {
          result.stale_special_pin_ref_count++;
          pushLimitedMessage(result.messages, "special net " + net->get_net_name() + " has non-owner pin ref " + pinDisplayName(pin));
        }
        if (!pin_refs.insert(pin).second) {
          result.duplicate_pin_ref_count++;
          pushLimitedMessage(result.messages, "special net " + net->get_net_name() + " has duplicate pin ref " + pinDisplayName(pin));
        }
        IdbSpecialNet* pin_special_net = pin->is_io_pin() ? pin->get_special_net() : findSpecialNetForInstancePin(pin);
        if (pin_special_net != net) {
          result.stale_special_pin_ref_count++;
          pushLimitedMessage(result.messages, "special pin reverse net mismatch: " + pinDisplayName(pin) + " in " + net->get_net_name());
        }
      }
    }
  }

  if (check_floating && _instance_list != nullptr) {
    for (auto* inst : _instance_list->get_instance_list()) {
      if (inst == nullptr || inst->get_pin_list() == nullptr) {
        continue;
      }
      for (auto* pin : inst->get_pin_list()->get_pin_list()) {
        if (pin != nullptr && pin->get_net() == nullptr && findSpecialNetForInstancePin(pin) == nullptr) {
          result.floating_pin_count++;
        }
      }
    }
  }

  result.ok = result.duplicate_net_count == 0 && result.duplicate_instance_count == 0 && result.duplicate_io_pin_count == 0
              && result.stale_regular_pin_ref_count == 0 && result.stale_pin_net_ref_count == 0
              && result.stale_special_pin_ref_count == 0 && result.pin_reverse_mismatch_count == 0
              && result.net_instance_mismatch_count == 0 && result.duplicate_pin_ref_count == 0
              && (!check_floating || result.floating_pin_count == 0);
  return result;
}

bool IdbDesign::writeConnectivitySnapshot(const std::string& path, bool check_floating) const
{
  std::ofstream out(path);
  if (!out.is_open()) {
    return false;
  }

  auto result = validateConnectivity(check_floating);
  out << "{\n";
  out << "  \"ok\": " << (result.ok ? "true" : "false") << ",\n";
  out << "  \"duplicate_net_count\": " << result.duplicate_net_count << ",\n";
  out << "  \"duplicate_instance_count\": " << result.duplicate_instance_count << ",\n";
  out << "  \"duplicate_io_pin_count\": " << result.duplicate_io_pin_count << ",\n";
  out << "  \"stale_regular_pin_ref_count\": " << result.stale_regular_pin_ref_count << ",\n";
  out << "  \"stale_pin_net_ref_count\": " << result.stale_pin_net_ref_count << ",\n";
  out << "  \"stale_special_pin_ref_count\": " << result.stale_special_pin_ref_count << ",\n";
  out << "  \"pin_reverse_mismatch_count\": " << result.pin_reverse_mismatch_count << ",\n";
  out << "  \"net_instance_mismatch_count\": " << result.net_instance_mismatch_count << ",\n";
  out << "  \"duplicate_pin_ref_count\": " << result.duplicate_pin_ref_count << ",\n";
  out << "  \"floating_pin_count\": " << result.floating_pin_count << ",\n";
  out << "  \"net_count\": " << (_net_list == nullptr ? 0 : _net_list->get_net_list().size()) << ",\n";
  out << "  \"special_net_count\": " << (_special_net_list == nullptr ? 0 : _special_net_list->get_net_list().size()) << ",\n";
  out << "  \"instance_count\": " << (_instance_list == nullptr ? 0 : _instance_list->get_instance_list().size()) << ",\n";
  out << "  \"io_pin_count\": " << (_io_pin_list == nullptr ? 0 : _io_pin_list->get_pin_list().size()) << ",\n";
  out << "  \"messages\": [";
  for (size_t i = 0; i < result.messages.size(); ++i) {
    out << (i == 0 ? "" : ", ") << "\"" << jsonEscape(result.messages[i]) << "\"";
  }
  out << "]\n";
  out << "}\n";
  return true;
}

bool IdbDesign::connectIOPinToPowerStripe(vector<IdbCoordinate<int32_t>*>& point_list, IdbLayer* layer)
{
  if (point_list.size() < _POINT_MAX_ || layer == nullptr) {
    return false;
  }

  /// find the IO pin that covered by the point list
  IdbPin* pin = _io_pin_list->find_pin_by_coordinate_list(point_list, layer);
  if (pin == nullptr) {
    std::cout << "Error : no IO pin covered by point list." << std::endl;
    for (IdbCoordinate<int32_t>* pt : point_list) {
      std::cout << " ( " << pt->get_x() << " , " << pt->get_y() << " )";
    }
    std::cout << std::endl;
    return false;
  }

  /// if point list is not horizontal or vertical, gernerate the correct points and adjust the order
  if (point_list.size() == _POINT_MAX_ && point_list[0]->get_x() != point_list[1]->get_x()
      && point_list[0]->get_y() != point_list[1]->get_y()) {
    /// original value
    int32_t start_x = point_list[0]->get_x();
    int32_t start_y = point_list[0]->get_y();
    int32_t end_x = point_list[1]->get_x();
    int32_t end_y = point_list[1]->get_y();
    /// get the middle coordinate
    int32_t mid_x = (start_x + end_x) / 2;
    int32_t mid_y = (start_y + end_y) / 2;

    IdbCore* core = _layout->get_core();

    if (core->is_side_left_or_right(point_list[0]) || core->is_side_left_or_right(point_list[1])) {
      /// make horizontal
      IdbCoordinate<int32_t>* new_coordinate = new IdbCoordinate<int32_t>(mid_x, start_y);
      point_list.insert(point_list.begin() + 1, new_coordinate);
      new_coordinate = new IdbCoordinate<int32_t>(mid_x, end_y);
      point_list.insert(point_list.begin() + 2, new_coordinate);

    } else if (core->is_side_top_or_bottom(point_list[0]) || core->is_side_top_or_bottom(point_list[1])) {
      /// vertical
      IdbCoordinate<int32_t>* new_coordinate = new IdbCoordinate<int32_t>(start_x, mid_y);
      point_list.insert(point_list.begin() + 1, new_coordinate);
      new_coordinate = new IdbCoordinate<int32_t>(end_x, mid_y);
      point_list.insert(point_list.begin() + 2, new_coordinate);
    } else {
      std::cout << "Error : illegal point list." << std::endl;
      return false;
    }
  }

  return _special_net_list->connectIO(point_list, layer);
}

bool IdbDesign::connectPowerStripe(vector<IdbCoordinate<int32_t>*>& point_list, string net_name, string layer_name)
{
  return _special_net_list->addPowerStripe(point_list, net_name, layer_name);
}

}  // namespace idb

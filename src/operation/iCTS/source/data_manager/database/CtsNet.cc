#include "CtsNet.h"

#include "log/Log.hh"
namespace icts {
class CtsInstance;

void CtsNet::addPin(CtsPin *pin) {
  pin->set_net(this);
  _pins.push_back(pin);
}

CtsPin *CtsNet::get_driver_pin() const {
  for (auto *pin : _pins) {
    if (pin->is_io()) {
      // if (pin->get_pin_type() == CtsPinType::kIn) {
      return pin;
      // }
    } else {
      if (pin->get_pin_type() == CtsPinType::kOut) {
        return pin;
      }
    }
  }
  return nullptr;
}

CtsInstance *CtsNet::get_driver_inst(CtsPin *pin) const {
  CtsPin *this_pin = pin != nullptr ? pin : get_driver_pin();

  return this_pin ? this_pin->get_instance() : nullptr;
}

CtsPin *CtsNet::findPin(const string &pin_name) {
  for (auto pin : _pins) {
    if (pin_name == pin->get_pin_name()) {
      return pin;
    }
  }
  return nullptr;
}

vector<CtsInstance *> CtsNet::get_instances() const {
  vector<CtsInstance *> insts;
  for (auto *pin : _pins) {
    auto *inst = pin->get_instance();
    insts.emplace_back(inst);
  }
  return insts;
}

vector<CtsPin *> CtsNet::get_load_pins() const {
  vector<CtsPin *> load_pins;

  for (auto *pin : _pins) {
    if (pin->get_pin_type() != CtsPinType::kOut && !pin->is_io()) {
      load_pins.push_back(pin);
    }
  }
  return load_pins;
}

vector<CtsInstance *> CtsNet::get_load_insts() const {
  vector<CtsInstance *> insts;
  for (auto *pin : get_load_pins()) {
    insts.push_back(pin->get_instance());
  }
  return insts;
}

void CtsNet::removePin(CtsPin *pin) {
  auto iter = std::find(_pins.begin(), _pins.end(), pin);
  if (iter != _pins.end()) {
    _pins.erase(iter);
  }
}

vector<CtsSignalWire> &CtsNet::get_signal_wires() {
  if (_signal_wires.empty()) {
    const auto *driver_pin = get_driver_pin();
    // TBD check InOut
    if (!driver_pin) {
      for (auto *pin : _pins) {
        if (pin->get_pin_type() == CtsPinType::kInOut) {
          driver_pin = pin;
          break;
        }
      }
    }
    LOG_FATAL_IF(driver_pin == nullptr)
        << "No driver pin found for net " << _net_name;
    auto *driver_inst = driver_pin->get_instance();
    Endpoint first = {driver_inst->get_name(), driver_inst->get_location(),
                      kPin};
    for (auto *load_pin : get_load_pins()) {
      auto *load_inst = load_pin->get_instance();
      Endpoint second = {load_inst->get_name(), load_inst->get_location(),
                         kPin};
      addSignalWire(CtsSignalWire(first, second));
    }
  }
  return _signal_wires;
}
CtsInstance *CtsNet::findInstance(const string &inst_name) {
  for (auto *pin : _pins) {
    if (pin->get_instance() && pin->get_instance()->get_name() == inst_name) {
      return pin->get_instance();
    }
  }
  return nullptr;
}
}  // namespace icts
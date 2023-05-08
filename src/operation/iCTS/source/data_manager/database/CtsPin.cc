#include "CtsPin.h"

namespace icts {
CtsPin::CtsPin() : _pin_name(""), _instance(nullptr), _net(nullptr) {}

CtsPin::CtsPin(const string &pin_name)
    : _pin_name(pin_name), _instance(nullptr), _net(nullptr) {}

CtsPin::CtsPin(const string &pin_name, CtsPinType pin_type,
               CtsInstance *instance, CtsNet *net)
    : _pin_name(pin_name),
      _pin_type(pin_type),
      _instance(instance),
      _net(net) {}

string CtsPin::get_full_name() const {
  return _instance ? _instance->get_name() + "/" + _pin_name : _pin_name;
}
}  // namespace icts
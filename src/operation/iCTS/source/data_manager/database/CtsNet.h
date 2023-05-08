#pragma once

#include <string>
#include <vector>

#include "CtsInstance.h"
#include "CtsPin.h"
#include "CtsSignalWire.h"
#include "DesignObject.h"
#include "pgl.h"

namespace icts {
using std::string;
using std::vector;

class CtsPin;
class CtsInstance;

struct LogicNetTag {};

class CtsNet : public DesignObject {
 public:
  CtsNet() = default;
  explicit CtsNet(const string &net_name) : _net_name(net_name) {}
  CtsNet(const CtsNet &net) = default;
  ~CtsNet() = default;

  // getter
  const string &get_net_name() const { return _net_name; }
  CtsPin *get_driver_pin() const;
  vector<CtsPin *> get_load_pins() const;
  CtsInstance *get_driver_inst(CtsPin *pin = nullptr) const;
  vector<CtsInstance *> get_load_insts() const;
  vector<CtsPin *> &get_pins() { return _pins; }
  vector<CtsInstance *> get_instances() const;
  vector<CtsSignalWire> &get_signal_wires();
  // setter
  void set_net_name(const string &net_name) { _net_name = net_name; }
  template <typename WireIterator>
  void setSignalWire(WireIterator begin, WireIterator end) {
    _signal_wires.clear();
    for (auto itr = begin; itr != end; itr++) {
      addSignalWire(*itr);
    }
  }

  // operator
  bool isClockRouted() const { return _is_clock_routed; }
  void setClockRouted(const bool &is_clock_routed = true) {
    _is_clock_routed = is_clock_routed;
  }
  CtsPin *findPin(const string &pin_name);
  void addPin(CtsPin *pin);
  void removePin(CtsPin *pin);
  void addSignalWire(const CtsSignalWire &signal_wire) {
    _signal_wires.push_back(signal_wire);
  }
  void clearWires() { _signal_wires.clear(); }

  CtsInstance *findInstance(const string &inst_name);

 private:
  string _net_name;
  vector<CtsPin *> _pins;
  vector<CtsSignalWire> _signal_wires;
  bool _is_clock_routed = false;
};

}  // namespace icts
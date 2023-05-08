#pragma once
#include <string>
#include <unordered_map>

#include "CtsInstance.h"
#include "CtsNet.h"
#include "CtsSignalWire.h"

namespace icts {

class OptiNet {
 public:
  typedef int id_type;

  OptiNet(CtsNet *clk_net);
  ~OptiNet() = default;

  CtsNet *get_clk_net() const { return _clk_net; }
  CtsInstance *get_driver() const { return _clk_net->get_driver_inst(); }
  vector<CtsInstance *> get_loads() const { return _clk_net->get_load_insts(); }
  vector<CtsSignalWire> get_signal_wires() const {
    return _clk_net->get_signal_wires();
  }

  int getId(const std::string &name) const {
    if (_name_to_id.count(name) != 0) {
      return _name_to_id.at(name);
    }
    return -1;
  }

 private:
  CtsNet *_clk_net;
  std::unordered_map<std::string, id_type> _name_to_id;
};

}  // namespace icts
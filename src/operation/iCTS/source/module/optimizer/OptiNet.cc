#include "OptiNet.h"

#include <set>

namespace icts {

OptiNet::OptiNet(CtsNet *clk_net) : _clk_net(clk_net) {
  std::set<std::string> node_names;
  auto &signal_wires = clk_net->get_signal_wires();
  for (const auto &signal_wire : signal_wires) {
    node_names.insert(signal_wire.get_first()._name);
    node_names.insert(signal_wire.get_second()._name);
  }
  id_type id = 0;
  for (auto itr = node_names.begin(); itr != node_names.end(); ++itr) {
    if (_name_to_id.count(*itr) == 0) {
      _name_to_id.insert(std::make_pair(*itr, id++));
    }
  }
}

}  // namespace icts
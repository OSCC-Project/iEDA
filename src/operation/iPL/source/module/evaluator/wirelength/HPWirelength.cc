/*
 * @Author: S.J Chen
 * @Date: 2022-03-09 15:09:04
 * @LastEditTime: 2022-04-06 12:01:32
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/evaluator/wirelength/HPWirelength.cc
 * Contact : https://github.com/sjchanson
 */

#include "HPWirelength.hh"

namespace ipl {

int64_t HPWirelength::obtainTotalWirelength()
{
  int64_t total_hpwl = 0;
  for (auto* network : _topology_manager->get_network_list()) {
    Rectangle<int32_t> network_shape = std::move(network->obtainNetWorkShape());
    total_hpwl += network_shape.get_half_perimeter();
  }
  return total_hpwl;
}

int64_t HPWirelength::obtainNetWirelength(std::string net_name)
{
  int64_t hpwl = 0;

  auto* network = _topology_manager->findNetwork(net_name);
  if (network) {
    hpwl = network->obtainNetWorkShape().get_half_perimeter();
  }

  return hpwl;
}

int64_t HPWirelength::obtainPartOfNetWirelength(std::string net_name, std::string sink_pin_name)
{
  int64_t wirelength = 0;

  auto* network = _topology_manager->findNetwork(net_name);
  if (network) {
    auto* transmitter = network->get_transmitter();
    if (transmitter) {
      for (auto* receiver : network->get_receiver_list()) {
        if (receiver->get_name() == sink_pin_name) {
          wirelength = (std::abs(transmitter->get_location().get_x() - receiver->get_location().get_x()) + std::abs(transmitter->get_location().get_y() - receiver->get_location().get_y()));
        }
      }
    }
  }

  return wirelength;
}

}  // namespace ipl
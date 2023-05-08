#ifndef SRC_PLATFORM_EVALUATOR_DATA_GDSNET_HPP_
#define SRC_PLATFORM_EVALUATOR_DATA_GDSNET_HPP_

#include "GDSPin.hpp"
#include "GDSViaNodes.hpp"
#include "GDSWireNodes.hpp"

namespace eval {

class GDSNet
{
 public:
  GDSNet() = default;
  ~GDSNet() = default;

  std::string get_name() const { return _name; }
  std::vector<GDSPin*> get_pin_list() const { return _gds_pin_list; }
  std::vector<GDSViaNodes*> get_via_node_list() const { return _gds_via_nodes_list; }
  std::vector<GDSWireNodes*> get_wire_node_list() const { return _gds_wire_nodes_list; }

  void set_name(const std::string& name) { _name = name; }
  void add_pin(GDSPin* pin) { _gds_pin_list.push_back(pin); }
  void add_via_nodes(GDSViaNodes* via_nodes) { _gds_via_nodes_list.push_back(via_nodes); }
  void add_wire_nodes(GDSWireNodes* wire_nodes) { _gds_wire_nodes_list.push_back(wire_nodes); }

 private:
  std::string _name;
  std::vector<GDSPin*> _gds_pin_list;
  std::vector<GDSViaNodes*> _gds_via_nodes_list;
  std::vector<GDSWireNodes*> _gds_wire_nodes_list;
};

}  // namespace eval

#endif  // SRC_PLATFORM_EVALUATOR_DATA_GDSNET_HPP_

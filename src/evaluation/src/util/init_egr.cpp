/*
 * @FilePath: init_egr.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#include "init_egr.h"

#include <any>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

#include "RTInterface.hpp"
#include "idm.h"
namespace ieval {

InitEGR* InitEGR::_init_egr = nullptr;

InitEGR::InitEGR()
{
  std::string base = dmInst->get_config().get_output_path();
  if (base.empty()) {
    base = ".";
  }
  _egr_dir_path = base + "/rt/rt_temp_directory";
  // ieval::createDirectoryRecursive(_egr_dir_path);
}

InitEGR::~InitEGR()
{
}

InitEGR* InitEGR::getInst()
{
  if (_init_egr == nullptr) {
    _init_egr = new InitEGR();
  }
  return _init_egr;
}

void InitEGR::destroyInst()
{
  if (_init_egr != nullptr) {
    delete _init_egr;
    _init_egr = nullptr;
  }
}

void InitEGR::runEGR(bool enable_timing)
{
  irt::RTInterface& rt_interface = irt::RTInterface::getInst();
  auto* idb_layout = dmInst->get_idb_lef_service()->get_layout();
  auto routing_layers = idb_layout->get_layers()->get_routing_layers();
  auto logic_layer_name = routing_layers.size() >= 2 ? routing_layers[1]->get_name() : routing_layers[0]->get_name();
  auto clock_layer_name = routing_layers.size() >= 4 ? routing_layers[routing_layers.size() - 4]->get_name() : logic_layer_name;
  std::string bottom = _bottom_layer_override.empty() ? logic_layer_name : _bottom_layer_override;
  std::string top = _top_layer_override.empty() ? clock_layer_name : _top_layer_override;
  bool use_timing = _enable_timing_override_set ? _enable_timing_override : enable_timing;
  std::map<std::string, std::any> config_map;
  config_map.insert({"-temp_directory_path", _egr_dir_path});
  config_map.insert({"-bottom_routing_layer", bottom});
  config_map.insert({"-top_routing_layer", top});
  if (_thread_number_override_set && _thread_number_override > 0) {
    config_map.insert({"-thread_number", _thread_number_override});
  }
  if (_output_inter_result_override_set) {
    config_map.insert({"-output_inter_result", _output_inter_result_override});
  }
  if (use_timing) {
    config_map.insert({"-enable_timing", 1});
  }
  rt_interface.initRT(config_map);
  std::map<std::string, std::any> ert_config_map;
  if (!_stage_override.empty()) {
    // stage：egr2D, egr3D, edr
    ert_config_map.insert({"-stage", _stage_override});
  }
  if (!_resolve_congestion_override.empty()) {
    // resolve_congestion: low, high
    ert_config_map.insert({"-resolve_congestion", _resolve_congestion_override});
  }
  rt_interface.runERT(ert_config_map);
  rt_interface.destroyRT();
}

double InitEGR::parseEGRWL(std::string guide_path)
{
  parseGuideFile(guide_path);
  double total_egr_wl = 0.0;
  for (const auto& net : _net_lengths) {
    total_egr_wl += net.second;
  }
  return total_egr_wl;
}

void InitEGR::parseGuideFile(const std::string& guide_path)
{
  _net_lengths.clear();

  std::ifstream file(guide_path);
  std::string line;
  std::string current_net;

  struct Wire
  {
    float rx1, ry1, rx2, ry2;
  };

  for (int i = 0; i < 4; ++i) {
    std::getline(file, line);
  }

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string type;
    iss >> type;

    if (type == "guide") {
      iss >> current_net;
      if (_net_lengths.find(current_net) == _net_lengths.end()) {
        _net_lengths[current_net] = 0.0;
      }
    } else if (type == "wire") {
      Wire wire;
      int gx1, gy1, gx2, gy2;
      std::string layer;
      iss >> gx1 >> gy1 >> gx2 >> gy2 >> wire.rx1 >> wire.ry1 >> wire.rx2 >> wire.ry2 >> layer;
      float wire_length = std::abs(wire.rx1 - wire.rx2) + std::abs(wire.ry1 - wire.ry2);
      _net_lengths[current_net] += wire_length;
    }
  }
}

float InitEGR::parseNetEGRWL(std::string guide_path, std::string net_name)
{
  float net_egr_wl = 0;

  std::ifstream file(guide_path);
  std::string line;
  std::string current_net;

  struct Wire
  {
    int gx1, gy1, gx2, gy2;
    float rx1, ry1, rx2, ry2;
    std::string layer;
  };

  for (int i = 0; i < 4; ++i) {
    std::getline(file, line);
  }

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string type;
    iss >> type;

    if (type == "guide") {
      iss >> current_net;
    } else if (type == "wire" && current_net == net_name) {
      Wire wire;
      iss >> wire.gx1 >> wire.gy1 >> wire.gx2 >> wire.gy2 >> wire.rx1 >> wire.ry1 >> wire.rx2 >> wire.ry2 >> wire.layer;
      float wire_length = fabs(wire.rx1 - wire.rx2) + fabs(wire.ry1 - wire.ry2);
      net_egr_wl += wire_length;
    }
  }
  return net_egr_wl;
}

float InitEGR::parsePathEGRWL(std::string guide_path, std::string net_name, std::string load_name)
{
  float path_egr_wl = 0;

  std::ifstream file(guide_path);
  std::string line;
  std::string target_net = net_name;
  bool is_target_net = false;
  struct Pin
  {
    int gx, gy;
    double rx, ry;
    std::string layer;
    std::string energy;
    std::string name;
  };
  struct Wire
  {
    int gx1, gy1, gx2, gy2;
    float rx1, ry1, rx2, ry2;
    std::string layer;
  };
  Pin driven_pin;
  Pin load_pin;
  std::stack<Wire> wire_stack;

  for (int i = 0; i < 4; ++i) {
    std::getline(file, line);
  }

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string type;
    iss >> type;

    if (type == "guide") {
      std::string net_name;
      iss >> net_name;
      is_target_net = (net_name == target_net);
    } else if (is_target_net) {
      if (type == "pin") {
        Pin pin;
        iss >> pin.gx >> pin.gy >> pin.rx >> pin.ry >> pin.layer >> pin.energy >> pin.name;
        if (pin.energy == "driven") {
          driven_pin = pin;
        } else if (pin.energy == "load" && pin.name == load_name) {
          load_pin = pin;
        }
      } else if (type == "wire") {
        Wire wire;
        iss >> wire.gx1 >> wire.gy1 >> wire.gx2 >> wire.gy2 >> wire.rx1 >> wire.ry1 >> wire.rx2 >> wire.ry2 >> wire.layer;
        wire_stack.push(wire);
      }
    }
  }

  int current_x = driven_pin.gx, current_y = driven_pin.gy;
  std::stack<Wire> temp_stack;

  while (!wire_stack.empty()) {
    bool wire_found = false;
    while (!wire_stack.empty()) {
      Wire wire = wire_stack.top();
      wire_stack.pop();

      if (wire.gx1 == current_x && wire.gy1 == current_y) {
        path_egr_wl += std::fabs(wire.rx1 - wire.rx2) + std::fabs(wire.ry1 - wire.ry2);
        current_x = wire.gx2;
        current_y = wire.gy2;
        wire_found = true;

        if (current_x == load_pin.gx && current_y == load_pin.gy) {
          return path_egr_wl;
        }
        break;
      } else {
        temp_stack.push(wire);
      }
    }

    if (!wire_found) {
      std::cout << "Error: Path interrupted. Unable to reach load pin." << std::endl;
      return -1;
    }

    while (!temp_stack.empty()) {
      wire_stack.push(temp_stack.top());
      temp_stack.pop();
    }
  }

  std::cout << "Error: Reached end of wires without finding load pin." << std::endl;
  return -1;
}

std::unordered_map<std::string, LayerDirection> InitEGR::parseLayerDirection(std::string guide_path)
{
  std::ifstream file(guide_path);
  std::string line;
  std::unordered_map<std::string, LayerDirection> layer_directions;

  struct Wire
  {
    int gx1, gy1, gx2, gy2;
    float rx1, ry1, rx2, ry2;
    std::string layer;
  };

  for (int i = 0; i < 4; ++i) {
    std::getline(file, line);
  }

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string type;
    iss >> type;

    if (type == "wire") {
      Wire wire;
      iss >> wire.gx1 >> wire.gy1 >> wire.gx2 >> wire.gy2 >> wire.rx1 >> wire.ry1 >> wire.rx2 >> wire.ry2 >> wire.layer;
      if (layer_directions.find(wire.layer) == layer_directions.end()) {
        if (wire.gx1 == wire.gx2) {
          layer_directions[wire.layer] = LayerDirection::Vertical;
        } else if (wire.gy1 == wire.gy2) {
          layer_directions[wire.layer] = LayerDirection::Horizontal;
        }
      }
    }
  }
  return layer_directions;
}

float InitEGR::getNetEGRWL(std::string net_name)
{
  auto it = _net_lengths.find(net_name);
  if (it != _net_lengths.end()) {
    return it->second;
  }
  return 0.0;
}

}  // namespace ieval
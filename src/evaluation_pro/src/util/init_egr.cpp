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

namespace ieval {

InitEGR::InitEGR()
{
}

InitEGR::~InitEGR()
{
}

void InitEGR::runEGR()
{
  irt::RTInterface& rtInterface = irt::RTInterface::getInst();
  std::map<std::string, std::any> config_map;
  config_map.insert({"-output_csv", 1});
  rtInterface.initRT(config_map);
  rtInterface.runEGR();
  rtInterface.destroyRT();
}

float InitEGR::parseEGRWL(std::string guide_path)
{
  float egr_wl = 0;

  std::ifstream file(guide_path);
  std::string line;
  std::map<std::string, float> netLengths;
  std::string currentNet;

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
      iss >> currentNet;
      if (netLengths.find(currentNet) == netLengths.end()) {
        netLengths[currentNet] = 0.0;
      }
    } else if (type == "wire") {
      Wire wire;
      iss >> wire.gx1 >> wire.gy1 >> wire.gx2 >> wire.gy2 >> wire.rx1 >> wire.ry1 >> wire.rx2 >> wire.ry2 >> wire.layer;
      float wireLength = fabs(wire.rx1 - wire.rx2) + fabs(wire.ry1 - wire.ry2);
      netLengths[currentNet] += wireLength;
    }
  }

  for (const auto& net : netLengths) {
    // std::cout << "Net " << net.first << " wire length: " << net.second << std::endl;
    egr_wl += net.second;
  }

  // std::cout << "Total wire length for all nets: " << egr_wl << std::endl;
  return egr_wl;
}

float InitEGR::parseNetEGRWL(std::string guide_path, std::string net_name)
{
  float net_egr_wl = 0;

  std::ifstream file(guide_path);
  std::string line;
  std::string currentNet;

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
      iss >> currentNet;
    } else if (type == "wire" && currentNet == net_name) {
      Wire wire;
      iss >> wire.gx1 >> wire.gy1 >> wire.gx2 >> wire.gy2 >> wire.rx1 >> wire.ry1 >> wire.rx2 >> wire.ry2 >> wire.layer;
      float wireLength = fabs(wire.rx1 - wire.rx2) + fabs(wire.ry1 - wire.ry2);
      net_egr_wl += wireLength;
    }
  }
  return net_egr_wl;
}

float InitEGR::parsePathEGRWL(std::string guide_path, std::string net_name, std::string load_name)
{
  float path_egr_wl = 0;

  std::ifstream file(guide_path);
  std::string line;
  std::string targetNet = net_name;
  bool isTargetNet = false;
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
  Pin drivenPin;
  Pin loadPin;
  std::stack<Wire> wireStack;

  for (int i = 0; i < 4; ++i) {
    std::getline(file, line);
  }

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string type;
    iss >> type;

    if (type == "guide") {
      std::string netName;
      iss >> netName;
      isTargetNet = (netName == targetNet);
    } else if (isTargetNet) {
      if (type == "pin") {
        Pin pin;
        iss >> pin.gx >> pin.gy >> pin.rx >> pin.ry >> pin.layer >> pin.energy >> pin.name;
        if (pin.energy == "driven") {
          drivenPin = pin;
        } else if (pin.energy == "load" && pin.name == load_name) {
          loadPin = pin;
        }
      } else if (type == "wire") {
        Wire wire;
        iss >> wire.gx1 >> wire.gy1 >> wire.gx2 >> wire.gy2 >> wire.rx1 >> wire.ry1 >> wire.rx2 >> wire.ry2 >> wire.layer;
        wireStack.push(wire);
      }
    }
  }

  int currentX = drivenPin.gx, currentY = drivenPin.gy;
  std::stack<Wire> tempStack;

  while (!wireStack.empty()) {
    bool wireFound = false;
    while (!wireStack.empty()) {
      Wire wire = wireStack.top();
      wireStack.pop();

      if (wire.gx1 == currentX && wire.gy1 == currentY) {
        path_egr_wl += std::fabs(wire.rx1 - wire.rx2) + std::fabs(wire.ry1 - wire.ry2);
        currentX = wire.gx2;
        currentY = wire.gy2;
        wireFound = true;

        if (currentX == loadPin.gx && currentY == loadPin.gy) {
          return path_egr_wl;
        }
        break;
      } else {
        tempStack.push(wire);
      }
    }

    if (!wireFound) {
      std::cout << "Error: Path interrupted. Unable to reach load pin." << std::endl;
      return -1;
    }

    while (!tempStack.empty()) {
      wireStack.push(tempStack.top());
      tempStack.pop();
    }
  }

  std::cout << "Error: Reached end of wires without finding load pin." << std::endl;
  return -1;
}

std::unordered_map<std::string, LayerDirection> InitEGR::parseLayerDirection(std::string guide_path)
{
  std::ifstream file(guide_path);
  std::string line;
  std::unordered_map<std::string, LayerDirection> layerDirections;

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
      if (layerDirections.find(wire.layer) == layerDirections.end()) {
        if (wire.gx1 == wire.gx2) {
          layerDirections[wire.layer] = LayerDirection::Vertical;
        } else if (wire.gy1 == wire.gy2) {
          layerDirections[wire.layer] = LayerDirection::Horizontal;
        }
      }
    }
  }
  return layerDirections;
}

}  // namespace ieval
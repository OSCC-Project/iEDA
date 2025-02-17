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
#include "Master.h"
#include "Placer.h"
#include "Reporter.h"
#include "ToConfig.h"
#include "ViolationOptimizer.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "timing_engine.h"

namespace ito {

void ViolationOptimizer::optimizeViolationNet(ista::Net* net, double cap_load_allowed_max)
{
  if (_max_buf_load_cap != 0.0) {
    cap_load_allowed_max = min(_max_buf_load_cap, cap_load_allowed_max);
  }

  auto driver_pin_port = net->getDriver();

  TimingIDBAdapter* idb_adapter = timingEngine->get_sta_adapter();
  IdbCoordinate<int32_t>* loc = idb_adapter->idbLocation(driver_pin_port);
  Point driver_pt = Point(loc->get_x(), loc->get_y());

  int wire_length;
  float pin_cap;
  TODesignObjSeq pins_loaded;

  TreeBuild* tree = new TreeBuild();
  bool make_tree = tree->makeRoutingTree(net, toConfig->get_routing_tree());
  if (!make_tree) {
    return;
  }
  int root_id = tree->get_root()->get_id();
  tree->updateBranch();
  repairViolationNetByDP(tree, root_id, TreeBuild::_null_pt, net, cap_load_allowed_max, wire_length, pin_cap, pins_loaded);

  delete tree;

  std::optional<double> width = std::nullopt;
  double wire_length_cap = dynamic_cast<TimingIDBAdapter*>(timingEngine->get_sta_adapter())
                               ->getCapacitance(1, (double) wire_length / toDmInst->get_dbu(), width);
  double driver_load_cap = pin_cap + wire_length_cap;

  if (driver_load_cap > cap_load_allowed_max) {
    printf("ViolationFixer | func  insertBuffer direction{driver}\n");
    ista::LibCell* insert_buffer_cell = _insert_buffer_cell;
    insertBuffer(driver_pt.get_x(), driver_pt.get_y(), net, insert_buffer_cell, wire_length, pin_cap, pins_loaded);
  }
}

void ViolationOptimizer::repairViolationNetByDP(TreeBuild* tree, int curr_pt, int father_pt, ista::Net* net, float cap_load_allowed_max,
                                                int& wire_length, float& pin_cap, TODesignObjSeq& pins_loaded)
{
  int left_branch = tree->left(curr_pt);
  int left_wire_length = 0;
  float left_pin_cap = 0.0;
  TODesignObjSeq left_pins_loaded;
  if (left_branch != TreeBuild::_null_pt) {
    repairViolationNetByDP(tree, left_branch, curr_pt, net, cap_load_allowed_max, left_wire_length, left_pin_cap, left_pins_loaded);
  }

  int middle_branch = tree->middle(curr_pt);
  int middle_wire_length = 0;
  float middle_pin_cap = 0.0;
  TODesignObjSeq middle_pins_loaded;
  if (middle_branch != TreeBuild::_null_pt) {
    repairViolationNetByDP(tree, middle_branch, curr_pt, net, cap_load_allowed_max, middle_wire_length, middle_pin_cap, middle_pins_loaded);
  }

  int right_branch = tree->right(curr_pt);
  int right_wire_length = 0;
  float right_pin_cap = 0.0;
  TODesignObjSeq right_pins_loaded;
  if (right_branch != TreeBuild::_null_pt) {
    repairViolationNetByDP(tree, right_branch, curr_pt, net, cap_load_allowed_max, right_wire_length, right_pin_cap, right_pins_loaded);
  }

  bool fix_left = false;
  bool fix_middle = false;
  bool fix_right = false;

  auto calcTotalLoadCap = [&](float len, float total_pin_cap) {
    std::optional<double> width = std::nullopt;
    auto total_wire_cap = timingEngine->get_sta_adapter()->getCapacitance(1, len, width);
    return (total_pin_cap + total_wire_cap);
  };

  double total_left_cap = calcTotalLoadCap((float) left_wire_length / toDmInst->get_dbu(), left_pin_cap);
  double total_middle_cap = calcTotalLoadCap((float) middle_wire_length / toDmInst->get_dbu(), middle_pin_cap);
  double total_right_cap = calcTotalLoadCap((float) right_wire_length / toDmInst->get_dbu(), right_pin_cap);

  bool cap_violation = (total_left_cap + total_middle_cap + total_right_cap) > cap_load_allowed_max;
  if (cap_violation) {
    determineFixSide(cap_load_allowed_max, total_left_cap, total_middle_cap, total_right_cap, fix_left, fix_middle, fix_right);
  }

  Point buffer_loc = tree->get_location(curr_pt);
  ista::LibCell* insert_buffer_cell = _insert_buffer_cell;

  if (fix_left) {
    insertBuffer(buffer_loc.get_x(), buffer_loc.get_y(), net, insert_buffer_cell, left_wire_length, left_pin_cap, left_pins_loaded);
  }
  if (fix_middle) {
    insertBuffer(buffer_loc.get_x(), buffer_loc.get_y(), net, insert_buffer_cell, middle_wire_length, middle_pin_cap, middle_pins_loaded);
  }
  if (fix_right) {
    insertBuffer(buffer_loc.get_x(), buffer_loc.get_y(), net, insert_buffer_cell, right_wire_length, right_pin_cap, right_pins_loaded);
  }

  wire_length = left_wire_length + middle_wire_length + right_wire_length;
  pin_cap = left_pin_cap + middle_pin_cap + right_pin_cap;

  for (DesignObject* load_pin : left_pins_loaded) {
    pins_loaded.push_back(load_pin);
  }
  for (DesignObject* load_pin : middle_pins_loaded) {
    pins_loaded.push_back(load_pin);
  }
  for (DesignObject* load_pin : right_pins_loaded) {
    pins_loaded.push_back(load_pin);
  }

  if (father_pt == TreeBuild::_null_pt) {
    return;
  }

  DesignObject* load_pin = tree->get_pin(curr_pt);
  Point curr_pt_loc = tree->get_location(curr_pt);

  if (load_pin) {
    if (load_pin->isPin()) {
      LibPort* load_port = dynamic_cast<Pin*>(load_pin)->get_cell_port();
      if (load_port) {
        pin_cap += load_pin->cap(AnalysisMode::kMax, TransType::kRise);
        // The constructed tree structure may have the case that the children
        // of a pin are also a pin. This results in pin_cap >= cap_load_allowed_max, which
        // should be separated.
        int left_branch = tree->left(curr_pt);
        // FIXME: The pin may not have left branch
        if (pin_cap >= cap_load_allowed_max && left_branch != TreeBuild::_null_pt) {
          Point left_loc = tree->get_location(left_branch);

          auto buf_loc = (curr_pt_loc + left_loc) / 2;
          insertBuffer(buf_loc.get_x(), buf_loc.get_y(), net, insert_buffer_cell, wire_length, pin_cap, pins_loaded);

          wire_length += Point::manhattanDistance(left_loc, curr_pt_loc) / 2;
          pin_cap += load_pin->cap(AnalysisMode::kMax, TransType::kRise);
        }
      }
    } else {
      pin_cap += load_pin->cap(AnalysisMode::kMax, TransType::kRise);
    }
    pins_loaded.push_back(load_pin);
  }

  Point father_loc = tree->get_location(father_pt);

  int length = Point::manhattanDistance(father_loc, curr_pt_loc);
  if (length == 0)
    return;
  wire_length += length;

  std::optional<double> width = std::nullopt;
  double wire_length_cap = timingEngine->get_sta_adapter()->getCapacitance(1, (double) wire_length / toDmInst->get_dbu(), width);
  double pin_wire_length_cap = pin_cap + wire_length_cap;

  while ((pin_cap < cap_load_allowed_max && pin_wire_length_cap > cap_load_allowed_max)) {
    std::optional<double> width = std::nullopt;
    int cap_length = timingEngine->get_sta_adapter()->capToLength(1, abs(cap_load_allowed_max - pin_cap), width) * toDmInst->get_dbu();
    // Make the wire slightly shorter than needed to accommodate the offset from the
    // instance origin to the pin and to allow for detailed placement adjustments.
    double deleta_dist = length - (wire_length - cap_length * (1.0 - 0.2));
    deleta_dist = max(0.0, deleta_dist);

    double delta = deleta_dist / length;
    Point buf_loc = curr_pt_loc + (father_loc - curr_pt_loc) * delta;
    curr_pt_loc = buf_loc;

    insertBuffer(buf_loc.get_x(), buf_loc.get_y(), net, insert_buffer_cell, wire_length, pin_cap, pins_loaded);

    wire_length = length - deleta_dist;

    // update the sum of pin_cap and wire_length_cap
    double wire_length_cap = timingEngine->get_sta_adapter()->getCapacitance(1, (double) wire_length / toDmInst->get_dbu(), width);
    pin_wire_length_cap = pin_cap + wire_length_cap;
  }
}

template <class T1, class T2>
void ViolationOptimizer::determineFixSide(T1 max_numb, T2 left, T2 middle, T2 right, bool& fix_left, bool& fix_middle, bool& fix_right)
{
  // case 1: fix all
  if (min(min(left, middle), right) >= max_numb) {
    fix_left = true;
    fix_middle = true;
    fix_right = true;
  } else if (left >= max(middle, right)) {
    fix_left = true;                   // case 2: fix one
    if (middle + right >= max_numb) {  // case 3: fix two
      if (middle > right) {
        fix_middle = true;
      } else {
        fix_right = true;
      }
    }
  } else if (middle >= max(left, right)) {
    fix_middle = true;
    if (left + right >= max_numb) {
      if (left > right) {
        fix_left = true;
      } else {
        fix_right = true;
      }
    }
  } else {
    fix_right = true;
    if (left + middle >= max_numb) {
      if (left > middle) {
        fix_left = true;
      } else {
        fix_middle = true;
      }
    }
  }
}

}  // namespace ito
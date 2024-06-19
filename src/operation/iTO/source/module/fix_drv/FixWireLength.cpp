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
#include "FixWireLength.h"
// iSTA
#include "Type.hh"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"

using namespace std;

namespace ito {

FixWireLength::FixWireLength(ista::TimingEngine *timing) {
  _timing_engine = timing;
  _db_adapter = _timing_engine->get_db_adapter();
  _insert_buffer_cell = _timing_engine->findLibertyCell("LVT_BUFHDV8");
  LOG_ERROR_IF(!_insert_buffer_cell) << "Can not found specified buffer.\n";
}

void FixWireLength::set_insert_buffer(LibertyCell *insert_buf) {
  _insert_buffer_cell = insert_buf;
}

void FixWireLength::fixMaxLength(int max_length) {
  auto *netlist = _timing_engine->get_netlist();
  Net  *net;
  FOREACH_NET(netlist, net) {
    if (net->isClockNet()) {
      continue;
    }
    fixMaxLength(net, max_length);
  }
}

void FixWireLength::fixMaxLength(Net *net, int max_length, bool fix) {
  RoutingTree *tree;
  tree = makeRoutingTree(net, _db_adapter, RoutingType::kSteiner);

  bool length_violation = false;
  if (fix) {
    length_violation = true;
  } else {
    // check max length
    if (tree) {
      int                 max_wire_length = 0;
      vector<vector<int>> ans;
      // max wire length from drvr pin to load pins
      findMaxDistfromDrvr(tree->get_root(), 0, max_wire_length);
      // tree->findPathOverMaxLength(tree->get_Root(), 0, max_wire_length, ans);
      length_violation = max_wire_length > max_length;
    }
  }

  if (length_violation) {
    int          wire_length;
    TODesignObjSeq load_pins;

    int root_id = tree->get_root()->get_id();
    tree->updateBranch();
    fixMaxLength(tree, root_id, RoutingTree::_null_pt, net, max_length, 0, wire_length,
                 load_pins);
  }
}

void FixWireLength::fixMaxLength(RoutingTree *tree, int curr_pt, int prev_pt, Net *net,
                                 int max_length, int level,
                                 int &wire_length, TODesignObjSeq &load_pins) {
  int          left_branch = tree->left(curr_pt);
  int          left_wire_length = 0;
  TODesignObjSeq left_load_pins;
  if (left_branch != RoutingTree::_null_pt) {
    fixMaxLength(tree, left_branch, curr_pt, net, max_length, level + 1, left_wire_length,
                 left_load_pins);
  }

  int          middle_branch = tree->middle(curr_pt);
  int          middle_wire_length = 0;
  TODesignObjSeq middle_load_pins;
  if (middle_branch != RoutingTree::_null_pt) {
    fixMaxLength(tree, middle_branch, curr_pt, net, max_length, level + 1,
                 middle_wire_length, middle_load_pins);
  }

  int          right_branch = tree->right(curr_pt);
  int          right_wire_length = 0;
  TODesignObjSeq right_load_pins;
  if (right_branch != RoutingTree::_null_pt) {
    fixMaxLength(tree, right_branch, curr_pt, net, max_length, level + 1,
                 right_wire_length, right_load_pins);
  }
  // Add up to three buffers to left/middle/right branch to stay under the max
  // cap/length/fanout.
  bool fix_left = false;
  bool fix_middle = false;
  bool fix_right = false;

  // length violation
  bool length_violation =
      (left_wire_length + middle_wire_length + right_wire_length) > max_length;
  if (length_violation) {
    determineFixSide(max_length, left_wire_length, middle_wire_length, right_wire_length,
                     fix_left, fix_middle, fix_right);
  }

  Point        buffer_loc = tree->get_location(curr_pt);
  LibertyCell *insert_buffer_cell = _insert_buffer_cell;

  if (fix_left) {
    printf("repair_net | level{%d}  direction{left}\n", level);
    insertBuffer(buffer_loc.get_x(), buffer_loc.get_y(), net, insert_buffer_cell, level,
                 left_wire_length, left_load_pins);
  }
  if (fix_middle) {
    printf("repair_net | level{%d}  direction{middle}\n", level);
    insertBuffer(buffer_loc.get_x(), buffer_loc.get_y(), net, insert_buffer_cell, level,
                 middle_wire_length, middle_load_pins);
  }
  if (fix_right) {
    printf("repair_net | level{%d}  direction{right}\n", level);
    insertBuffer(buffer_loc.get_x(), buffer_loc.get_y(), net, insert_buffer_cell, level,
                 right_wire_length, right_load_pins);
  }

  wire_length = left_wire_length + middle_wire_length + right_wire_length;
  for (auto load_pin : left_load_pins) {
    load_pins.push_back(load_pin);
  }
  for (auto load_pin : middle_load_pins) {
    load_pins.push_back(load_pin);
  }
  for (auto load_pin : right_load_pins) {
    load_pins.push_back(load_pin);
  }

  Point prev_loc = tree->get_location(prev_pt);
  Point curr_pt_loc = tree->get_location(curr_pt);

  int length = Point::manhattanDistance(prev_loc, curr_pt_loc);
  wire_length += length;
  // Back up from pt to prev_pt adding repeaters every max_length.
  int prev_x = prev_loc.get_x();
  int prev_y = prev_loc.get_y();
  int curr_pt_x = curr_pt_loc.get_x();
  int curr_pt_y = curr_pt_loc.get_y();

  int wire_insert_count = 0; // number of buffer insert in wire // max 10
  while (wire_length > max_length) {
    double length_margin_for_placement_remove = .2;
    // Distance from pt to repeater backward toward prev_pt.
    double buf_dist;
    buf_dist = length - (wire_length - max_length * (1.0 - length_margin_for_placement_remove));
    buf_dist = max(0.0, buf_dist);

    double dx = prev_x - curr_pt_x;
    double dy = prev_y - curr_pt_y;
    double d = buf_dist / length;
    int    buf_x = curr_pt_x + d * dx;
    int    buf_y = curr_pt_y + d * dy;

    insertBuffer(buf_x, buf_y, net, insert_buffer_cell, level, wire_length, load_pins);
    wire_insert_count++;

    // Update for the next round.
    length -= buf_dist;
    wire_length = length;
    curr_pt_x = buf_x;
    curr_pt_y = buf_y;
  }
}

template <class T1, class T2>
void FixWireLength::determineFixSide(T1 max_numb, T2 left, T2 middle, T2 right,
                                     bool &fix_left, bool &fix_middle, bool &fix_right) {
  // case 1: fix all
  if (min(min(left, middle), right) >= max_numb) {
    fix_left = true;
    fix_middle = true;
    fix_right = true;
  } else if (left >= max(middle, right)) {
    fix_left = true;                  // case 2: fix one
    if (middle + right >= max_numb) { // case 3: fix two
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

void FixWireLength::insertBuffer(int x, int y, Net *net, LibertyCell *insert_buf_cell,
                                 int level, int &wire_length, TODesignObjSeq &load_pins) {

  LibertyPort *buffer_input_port, *buffer_output_port;
  insert_buf_cell->bufferPorts(buffer_input_port, buffer_output_port);

  // make instance name
  std::string buffer_name = ("length_buffer_" + to_string(_insert_instance_index));
  _insert_instance_index++;

  TimingIDBAdapter *db_adapter = dynamic_cast<TimingIDBAdapter *>(_db_adapter);

  Net *in_net, *out_net;
  in_net = net;
  // make net name
  std::string net_name = ("length_net_" + to_string(_make_net_index));
  _make_net_index++;
  out_net = db_adapter->createNet(net_name.c_str(), nullptr);

  idb::IdbNet *out_net_db = db_adapter->staToDb(out_net);
  idb::IdbNet *in_net_db = db_adapter->staToDb(in_net);
  out_net_db->set_connect_type(in_net_db->get_connect_type());

  // Re-connect the load_pins to out_net.
  for (auto *pin_port : load_pins) {
    if (pin_port->isPin()) {
      Pin      *pin = dynamic_cast<Pin *>(pin_port);
      Instance *inst = pin->get_own_instance();

      db_adapter->disattachPin(pin);
      auto debug = db_adapter->attach(inst, pin->get_name(), out_net);
      LOG_ERROR_IF(!debug);
    }
  }
  Instance *buffer = db_adapter->createInstance(insert_buf_cell, buffer_name.c_str());

  auto debug_buf_in =
      db_adapter->attach(buffer, buffer_input_port->get_port_name(), in_net);
  auto debug_buf_out =
      db_adapter->attach(buffer, buffer_output_port->get_port_name(), out_net);
  LOG_ERROR_IF(!debug_buf_in);
  LOG_ERROR_IF(!debug_buf_out);

  _timing_engine->insertBuffer(buffer->get_name());

  setLocation(buffer, x, y);

  insert_buf_cell = buffer->get_inst_cell();
  insert_buf_cell->bufferPorts(buffer_input_port, buffer_output_port);
  Pin *buf_in_pin = buffer->findPin(buffer_input_port);
  if (buf_in_pin) {
    wire_length = 0;
  }
}

void FixWireLength::setLocation(Instance *inst, int x, int y) {
  TimingIDBAdapter *idb_adapter = dynamic_cast<TimingIDBAdapter *>(_db_adapter);
  idb::IdbInstance *idb_inst = idb_adapter->staToDb(inst);

  idb_inst->set_status_placed();
  idb_inst->set_coodinate(x, y);
  idb_inst->set_orient(idb::IdbOrient::kN_R0);
}

} // namespace ito

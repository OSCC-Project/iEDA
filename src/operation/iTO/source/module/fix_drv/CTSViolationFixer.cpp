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
#include "CTSViolationFixer.h"
// iSTA
#include "Type.hh"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"

using namespace std;

namespace ito {

int Tree::_null_pt = -1;

void Tree::add_edge(Edge edg) {
  int father = edg._father;
  int child = edg._child;
  if (_left.at(father) == _null_pt) {
    _left.at(father) = child;
  } else {
    if (_middle.at(father) == _null_pt) {
      _middle.at(father) = child;
    } else {
      _right.at(father) = child;
    }
  }
}

DesignObject *Tree::get_pin(int id) {
  auto iter = _id_to_design_obj.find(id);
  if (iter != _id_to_design_obj.end()) {
    return iter->second;
  }
  return nullptr;
}

Point Tree::get_location(int id) {
  auto iter = _id_to_location.find(id);
  if (iter != _id_to_location.end()) {
    return iter->second;
  }
  LOG_INFO << "Can't find location";
  exit(1);
  return Point(-1, -1);
}

void Tree::printTree() {
  int size = _left.size();
  for (int i = 0; i != size; i++) {
    cout << _left[i] << "  " << _middle[i] << "  " << _right[i] << endl;
  }
  cout << endl;
}

////////////////////////////////       CTSViolationFixer

CTSViolationFixer *CTSViolationFixer::_cts_drv_fix = nullptr;
int                CTSViolationFixer::_rise = (int)TransType::kRise - 1;
int                CTSViolationFixer::_fall = (int)TransType::kFall - 1;

CTSViolationFixer *CTSViolationFixer::get_cts_violation_fixer(DbInterface *dbInterface) {
  if (_cts_drv_fix == nullptr) {
    _cts_drv_fix = new CTSViolationFixer(dbInterface);
  }
  return _cts_drv_fix;
}

CTSViolationFixer *
CTSViolationFixer::get_cts_violation_fixer(idb::IdbBuilder *   idb,
                                           ista::TimingEngine *timing) {
  if (_cts_drv_fix == nullptr) {
    LOG_ERROR_IF(!idb) << "[ERROR] Function loss parameter idb::IdbBuilder.";
    LOG_ERROR_IF(!timing) << "[ERROR] Function loss parameter ista::TimingEngine.";
    _cts_drv_fix = new CTSViolationFixer(idb, timing);
  }
  return _cts_drv_fix;
}

void CTSViolationFixer::destroyCTSViolationFixer() {
  if (_cts_drv_fix != nullptr) {
    delete _cts_drv_fix;
    _cts_drv_fix = nullptr;
  }
}

CTSViolationFixer::CTSViolationFixer(DbInterface *dbInterface) {
  _db_interface = dbInterface;
  _timing_engine = dbInterface->get_timing_engine();
  _db_adapter = _timing_engine->get_db_adapter();
  _dbu = _db_interface->get_dbu();
  _parasitics_estimator = new EstimateParasitics(_timing_engine, _dbu);
  _placer = new Placer(dbInterface->get_idb());
}

CTSViolationFixer::CTSViolationFixer(idb::IdbBuilder *idb, ista::TimingEngine *timing) {
  _timing_engine = timing;
  _db_adapter = _timing_engine->get_db_adapter();

  IdbLefService *idb_lef_service = idb->get_lef_service();
  IdbLayout *    idb_layout = idb_lef_service->get_layout();
  _dbu = idb_layout->get_units()->get_micron_dbu();
  _parasitics_estimator = new EstimateParasitics(_timing_engine, _dbu);
  _placer = new Placer(idb);
}

std::vector<IdbNet *> CTSViolationFixer::fixTiming(IdbNet *idb_net, Tree *topo) {
  _net_for_estimate.clear();
  _net_for_estimate.emplace_back(idb_net);

  bool repair_cap = false;
  bool repair_slew = false;

  double max_drvr_cap = kInf;
  double max_fanout = kInf;

  string net_name = idb_net->get_net_name();
  Net *  sta_net = _timing_engine->get_netlist()->findNet(net_name.c_str());
  auto   drvr = sta_net->getDriver();

  // cap violation
  if (_check_cap) {
    checkCapacitanceViolation(drvr, max_drvr_cap, repair_cap);
  }
  // slew violation
  if (_check_slew) {
    checkSlewViolation(drvr, max_drvr_cap, repair_slew);
  }

  if (repair_slew || repair_cap) {
    double max_cap = max_drvr_cap;

    // Tree *tree = clkTreeToTree(sta_net, _cts_net);

    LibertyPort *buffer_input_port, *buffer_output_port;
    LibertyCell *insert_buffer_cell =
        _timing_engine->findLibertyCell(_insert_clk_buf.c_str());
    insert_buffer_cell->bufferPorts(buffer_input_port, buffer_output_port);

    float                 buf_cap_limit = 0.0;
    bool                  buf_cap_limit_exists = false;
    std::optional<double> cap_limit =
        buffer_output_port->get_port_cap_limit(AnalysisMode::kMax);
    if (cap_limit.has_value()) {
      buf_cap_limit_exists = true;
      buf_cap_limit = *cap_limit;
    }

    if ((repair_cap || repair_slew) && buf_cap_limit_exists) {
      max_cap = buf_cap_limit;
    }

    int          wire_length;
    float        pin_cap, fanout;
    DesignObjSeq load_pins;

    int  drvr_id = topo->get_drvr_id();
    auto drvr_pt = topo->get_location(drvr_id);

    topo->printTree();
    fixViolations(topo, drvr_id, Tree::_null_pt, sta_net, max_cap, max_fanout, 0,
                  wire_length, pin_cap, fanout, load_pins);

    std::optional<double> width = std::nullopt;
    double                wire_length_cap = dynamic_cast<TimingIDBAdapter *>(_db_adapter)
                                 ->getCapacitance(1, (double)wire_length / _dbu, width);
    double drvr_load_cap = pin_cap + wire_length_cap;

    if (drvr_load_cap > max_drvr_cap) {
      insertCLKBuffer(drvr_pt.get_x(), drvr_pt.get_y(), sta_net, insert_buffer_cell, 0,
                      wire_length, pin_cap, fanout, load_pins);
    }
  }

  return _net_for_estimate;
}

void CTSViolationFixer::fixViolations(Tree *tree, int curr_pt, int prev_pt, Net *net,
                                      float max_cap, float max_fanout, int level,
                                      // Return values.
                                      // Remaining parasiics after repeater insertion.
                                      int &  wire_length, // dbu
                                      float &pin_cap, float &fanout,
                                      DesignObjSeq &load_pins) {
  int          left_branch = tree->left(curr_pt);
  int          left_wire_length = 0;
  float        left_pin_cap = 0.0;
  float        left_fanout = 0.0;
  DesignObjSeq left_load_pins;
  if (left_branch != Tree::_null_pt) {
    fixViolations(tree, left_branch, curr_pt, net, max_cap, max_fanout, level + 1,
                  left_wire_length, left_pin_cap, left_fanout, left_load_pins);
  }

  int          middle_branch = tree->middle(curr_pt);
  int          middle_wire_length = 0;
  float        middle_pin_cap = 0.0;
  float        middle_fanout = 0.0;
  DesignObjSeq middle_load_pins;
  if (middle_branch != Tree::_null_pt) {
    fixViolations(tree, middle_branch, curr_pt, net, max_cap, max_fanout, level + 1,
                  middle_wire_length, middle_pin_cap, middle_fanout, middle_load_pins);
  }

  int          right_branch = tree->right(curr_pt);
  int          right_wire_length = 0;
  float        right_pin_cap = 0.0;
  float        right_fanout = 0.0;
  DesignObjSeq right_load_pins;
  if (right_branch != Tree::_null_pt) {
    fixViolations(tree, right_branch, curr_pt, net, max_cap, max_fanout, level + 1,
                  right_wire_length, right_pin_cap, right_fanout, right_load_pins);
  }

  // Add up to three buffers to left/middle/right branch to stay under the max
  // cap/length/fanout.
  bool fix_left = false;
  bool fix_middle = false;
  bool fix_right = false;

  std::optional<double> width = std::nullopt;

  double left_wire_length_cap =
      dynamic_cast<TimingIDBAdapter *>(_db_adapter)
          ->getCapacitance(1, (double)left_wire_length / _dbu, width);
  double middle_wire_length_cap =
      dynamic_cast<TimingIDBAdapter *>(_db_adapter)
          ->getCapacitance(1, (double)middle_wire_length / _dbu, width);
  double right_wire_length_cap =
      dynamic_cast<TimingIDBAdapter *>(_db_adapter)
          ->getCapacitance(1, (double)right_wire_length / _dbu, width);

  double left_cap = left_pin_cap + left_wire_length_cap;
  double middle_cap = middle_pin_cap + middle_wire_length_cap;
  double right_cap = right_pin_cap + right_wire_length_cap;

  bool cap_violation = (left_cap + middle_cap + right_cap) > max_cap;
  if (cap_violation) {
    determineFixSide(max_cap, left_cap, middle_cap, right_cap, fix_left, fix_middle,
                     fix_right);
  }
  // fanout violation
  bool fanout_violation = (left_fanout + middle_fanout + right_fanout) > max_fanout;
  if (fanout_violation) {
    determineFixSide(max_fanout, left_fanout, middle_fanout, right_fanout, fix_left,
                     fix_middle, fix_right);
  }

  auto         buffer_loc = tree->get_location(curr_pt);
  LibertyCell *insert_buffer_cell =
      _timing_engine->findLibertyCell(_insert_clk_buf.c_str());

  if (fix_left) {
    insertCLKBuffer(buffer_loc.get_x(), buffer_loc.get_y(), net, insert_buffer_cell,
                    level, left_wire_length, left_pin_cap, left_fanout, left_load_pins);
  }
  if (fix_middle) {
    insertCLKBuffer(buffer_loc.get_x(), buffer_loc.get_y(), net, insert_buffer_cell,
                    level, middle_wire_length, middle_pin_cap, middle_fanout,
                    middle_load_pins);
  }
  if (fix_right) {
    insertCLKBuffer(buffer_loc.get_x(), buffer_loc.get_y(), net, insert_buffer_cell,
                    level, right_wire_length, right_pin_cap, right_fanout,
                    right_load_pins);
  }

  pin_cap = left_pin_cap + middle_pin_cap + right_pin_cap;
  fanout = left_fanout + middle_fanout + right_fanout;
  wire_length = left_wire_length + middle_wire_length + right_wire_length;

  for (DesignObject *load_pin : left_load_pins) {
    load_pins.push_back(load_pin);
  }
  for (DesignObject *load_pin : middle_load_pins) {
    load_pins.push_back(load_pin);
  }
  for (DesignObject *load_pin : right_load_pins) {
    load_pins.push_back(load_pin);
  }

  // Steiner pt pin is the net driver if prev_pt is null.
  if (prev_pt != Tree::_null_pt) {
    DesignObject *load_pin = tree->get_pin(curr_pt);
    auto          curr_pt_loc = tree->get_location(curr_pt);
    int           curr_pt_x = curr_pt_loc.get_x();
    int           curr_pt_y = curr_pt_loc.get_y();

    if (load_pin) {
      if (load_pin->isPin()) {
        LibertyPort *load_port = dynamic_cast<Pin *>(load_pin)->get_cell_port();
        if (load_port) {
          pin_cap += load_pin->cap(AnalysisMode::kMax, TransType::kRise);

          if (pin_cap >= max_cap) {
            int  left_branch = tree->left(curr_pt);
            auto left_loc = tree->get_location(left_branch);
            int  buf_x = (left_loc.get_x() + curr_pt_x) / 2;
            int  buf_y = (left_loc.get_y() + curr_pt_y) / 2;
            insertCLKBuffer(buf_x, buf_y, net, insert_buffer_cell, level, wire_length,
                            pin_cap, fanout, load_pins);
            wire_length +=
                (abs(left_loc.get_x() - curr_pt_x) + abs(left_loc.get_y() - curr_pt_y)) /
                2;
            pin_cap += load_pin->cap(AnalysisMode::kMax, TransType::kRise);
          }
          fanout += portFanoutLoadNum(load_port);
        } else {
          fanout += 1;
        }
      } else {
        pin_cap += load_pin->cap(AnalysisMode::kMax, TransType::kRise);
        fanout += 1;
      }
      load_pins.push_back(load_pin);
    }

    auto prev_loc = tree->get_location(prev_pt);
    int  length = Point::manhattanDistance(prev_loc, curr_pt_loc);
    wire_length += length;

    std::deque<Point> segments;
    segments.push_back(prev_loc);
    if (prev_loc.get_x() == curr_pt_loc.get_x() ||
        prev_loc.get_y() == curr_pt_loc.get_y()) {
      segments.push_back(curr_pt_loc);
    } else {
      segments.push_back(Point(prev_loc.get_x(), curr_pt_loc.get_y()));
      segments.push_back(curr_pt_loc);
    }

    std::optional<double> width = std::nullopt;
    double                wire_length_cap = dynamic_cast<TimingIDBAdapter *>(_db_adapter)
                                 ->getCapacitance(1, (double)wire_length / _dbu, width);
    double pin_wire_length_cap = pin_cap + wire_length_cap;

    int wire_insert_count = 0; // number of buffer insert in wire // max 10
    while (pin_cap < max_cap && pin_wire_length_cap > max_cap) {
      // Make the wire a bit shorter than necessary to allow for
      // offset from instance origin to pin and detailed placement movement.
      double length_margin = .2;
      // Distance from pt to repeater backward toward prev_pt.
      double buf_dist;

      std::optional<double> width = std::nullopt;

      int cap_length = dynamic_cast<TimingIDBAdapter *>(_db_adapter)
                           ->capToLength(1, abs(max_cap - pin_cap), width) *
                       _dbu;
      buf_dist = length - (wire_length - cap_length * (1.0 - length_margin));
      if (buf_dist < 0.0) {
        buf_dist = 0;
      }

      // select location
      auto buf_loc =
          selectBufferLocation(segments, curr_pt_loc, curr_pt_loc, buf_dist, length);
      int buf_x = buf_loc.first;
      int buf_y = buf_loc.second;

      insertCLKBuffer(buf_x, buf_y, net, insert_buffer_cell, level, wire_length, pin_cap,
                      fanout, load_pins);
      wire_insert_count++;

      // Update for the next round.
      length -= buf_dist;
      wire_length = length;
      curr_pt_x = buf_x;
      curr_pt_y = buf_y;

      double wire_length_cap = dynamic_cast<TimingIDBAdapter *>(_db_adapter)
                                   ->getCapacitance(1, (double)wire_length / _dbu, width);
      pin_wire_length_cap = pin_cap + wire_length_cap;
    }
  }
}

pair<int, int> CTSViolationFixer::selectBufferLocation(std::deque<Point> &segments,
                                                       Point current_loc, Point prev_loc,
                                                       double buf_dist, int length) {
  int prev_x = prev_loc.get_x();
  int prev_y = prev_loc.get_y();
  int curr_pt_x = current_loc.get_x();
  int curr_pt_y = current_loc.get_y();
  int buf_x, buf_y;
  if (segments.size() == 2) {
    double dx = prev_x - curr_pt_x;
    double dy = prev_y - curr_pt_y;
    double d = buf_dist / length;
    buf_x = curr_pt_x + d * dx;
    buf_y = curr_pt_y + d * dy;
  } else {
    assert(segments.size() == 3);
    auto inter_pt = segments[1];
    int  length_between_curr_and_inter;
    if (current_loc.get_x() == inter_pt.get_x()) {
      length_between_curr_and_inter = abs(current_loc.get_y() - inter_pt.get_y());
    } else {
      length_between_curr_and_inter = abs(current_loc.get_x() - inter_pt.get_x());
    }

    if (length_between_curr_and_inter > buf_dist) {
      double dx = inter_pt.get_x() - curr_pt_x;
      double dy = inter_pt.get_y() - curr_pt_y;
      double d = buf_dist / length_between_curr_and_inter;
      buf_x = curr_pt_x + d * dx;
      buf_y = curr_pt_y + d * dy;
      segments.pop_back();
      segments.push_back(Point(buf_x, buf_y));
    } else {
      int remain_length = length - length_between_curr_and_inter;
      buf_dist -= length_between_curr_and_inter;
      double dx = prev_x - inter_pt.get_x(); // - curr_pt_x;
      double dy = prev_y - inter_pt.get_y(); // - curr_pt_y;
      double d = buf_dist / remain_length;
      buf_x = inter_pt.get_x() + d * dx;
      buf_y = inter_pt.get_y() + d * dy;
      segments.pop_back();
      segments.pop_back();
      segments.push_back(Point(buf_x, buf_y));
    }
  }

  return make_pair(buf_x, buf_y);
}

void CTSViolationFixer::insertCLKBuffer(int x, int y, Net *net,
                                        LibertyCell *insert_buf_cell, int level,
                                        int &wire_length, float &cap, float &fanout,
                                        DesignObjSeq &load_pins) {
  LibertyPort *buffer_input_port, *buffer_output_port;
  insert_buf_cell->bufferPorts(buffer_input_port, buffer_output_port);

  // make instance name
  std::string buffer_name = ("clk_buffer_" + to_string(_insert_instance_index));
  _insert_instance_index++;

  ista::TimingDBAdapter *timing_db_adapter = _timing_engine->get_db_adapter();
  TimingIDBAdapter *     db_adapter = dynamic_cast<TimingIDBAdapter *>(timing_db_adapter);

  Net *in_net, *out_net;
  in_net = net;
  // make net name
  std::string net_name = ("clk_buffer_net_" + to_string(_make_net_index));
  _make_net_index++;
  out_net = db_adapter->makeNet(net_name.c_str(), nullptr);
  auto *idb_net = db_adapter->staToDb(out_net);
  idb_net->set_connect_type(idb::IdbConnectType::kClock);

  idb::IdbNet *out_net_db = db_adapter->staToDb(out_net);
  idb::IdbNet *in_net_db = db_adapter->staToDb(in_net);
  out_net_db->set_connect_type(in_net_db->get_connect_type());

  // Move load pins to out_net.
  for (auto *pin_port : load_pins) {
    if (pin_port->isPin()) {
      Pin *     pin = dynamic_cast<Pin *>(pin_port);
      Instance *inst = pin->get_own_instance();
      db_adapter->disconnectPin(pin);
      auto debug = db_adapter->connect(inst, pin->get_name(), out_net);
      LOG_ERROR_IF(!debug);
    }
  }

  Instance *buffer = db_adapter->makeInstance(insert_buf_cell, buffer_name.c_str());
  setLocation(buffer, x, y);

  _inserted_buffer_count++;

  // connect to CLKBuffer
  auto debug_buf_in =
      db_adapter->connect(buffer, buffer_input_port->get_port_name(), in_net);
  auto debug_buf_out =
      db_adapter->connect(buffer, buffer_output_port->get_port_name(), out_net);
  LOG_ERROR_IF(!debug_buf_in);
  LOG_ERROR_IF(!debug_buf_out);

  _parasitics_estimator->estimateNetParasitics(in_net);
  _parasitics_estimator->estimateNetParasitics(out_net);

  _net_for_estimate.emplace_back(out_net_db);

  _timing_engine->insertBuffer(buffer->get_name());

  insert_buf_cell = buffer->get_inst_cell();
  insert_buf_cell->bufferPorts(buffer_input_port, buffer_output_port);
  Pin *buf_in_pin = buffer->findPin(buffer_input_port);
  if (buf_in_pin) {
    load_pins.clear();
    load_pins.push_back(dynamic_cast<DesignObject *>(buf_in_pin));
    wire_length = 0;
    auto input_cap =
        buffer_input_port->get_port_cap(AnalysisMode::kMax, TransType::kRise);
    if (input_cap) {
      cap = *input_cap;
    } else {
      cap = buffer_input_port->get_port_cap();
    }
    fanout = portFanoutLoadNum(buffer_input_port);
  }
}

template <class T1, class T2>
void CTSViolationFixer::determineFixSide(T1 max_numb, T2 left, T2 middle, T2 right,
                                         bool &fix_left, bool &fix_middle,
                                         bool &fix_right) {
  // case 1: fix all
  if (min(min(left, middle), right) > max_numb) {
    fix_left = true;
    fix_middle = true;
    fix_right = true;
  } else if (left >= max(middle, right)) {
    fix_left = true;                 // case 2: fix one
    if (middle + right > max_numb) { // case 3: fix two
      if (middle > right) {
        fix_middle = true;
      } else {
        fix_right = true;
      }
    }
  } else if (middle >= max(left, right)) {
    fix_middle = true;
    if (left + right > max_numb) {
      if (left > right) {
        fix_left = true;
      } else {
        fix_right = true;
      }
    }
  } else {
    fix_right = true;
    if (left + middle > max_numb) {
      if (left > middle) {
        fix_left = true;
      } else {
        fix_middle = true;
      }
    }
  }
}

void CTSViolationFixer::setLocation(Instance *inst, int x, int y) {
  ista::TimingDBAdapter *timing_db_adapter = _timing_engine->get_db_adapter();
  TimingIDBAdapter *idb_adapter = dynamic_cast<TimingIDBAdapter *>(timing_db_adapter);
  idb::IdbInstance *idb_inst = idb_adapter->staToDb(inst);
  unsigned          master_width = idb_inst->get_cell_master()->get_width();

  pair<int, int> loc = _placer->findNearestSpace(master_width, x, y);
  idb_inst->set_status_placed();
  idb_inst->set_coodinate(loc.first, loc.second);
  // set orient
  auto row = _db_interface->placer()->findRow(loc.second);
  if (row) {
    idb_inst->set_orient(row->get_site()->get_orient());
  } else {
    idb_inst->set_orient(idb::IdbOrient::kN_R0);
  }
  _placer->updateRow(master_width, loc.first, loc.second);
}

void CTSViolationFixer::checkFanoutViolation(DesignObject *drvr_pin,
                                             // return values
                                             double &max_fanout, bool &repair_fanout) {
  double                fanout, fanout_slack;
  std::optional<double> max_fanout_tmp;
  _timing_engine->checkFanout(drvr_pin->getFullName().c_str(), ista::AnalysisMode::kMax,
                              fanout, max_fanout_tmp, fanout_slack);

  if (max_fanout_tmp) {
    if (fanout_slack < 0.0) {
      max_fanout = *max_fanout_tmp;
      repair_fanout = true;
    }
  }
}

void CTSViolationFixer::checkCapacitanceViolation(DesignObject *drvr_pin,
                                                  // return values
                                                  double &max_drvr_cap,
                                                  bool &  repair_cap) {
  double                cap1;
  std::optional<double> max_cap1;
  double                cap_slack1;
  ista::TransType       rf = TransType::kRise;
  _timing_engine->checkCapacitance(drvr_pin->getFullName().c_str(), AnalysisMode::kMax,
                                   rf, cap1, max_cap1, cap_slack1);
  if (max_cap1) {
    max_drvr_cap = *max_cap1;
    if (cap_slack1 < 0) {
      repair_cap = true;
    }
  }
}

void CTSViolationFixer::checkSlewViolation(DesignObject *drvr_pin,
                                           // return values
                                           double &max_drvr_cap, bool &repair_slew) {
  float slew_slack1 = kInf;
  float max_slew1{0.f};

  Net *         net = drvr_pin->get_net();
  DesignObject *pin;
  FOREACH_NET_PIN(net, pin) {
    // Slew slew_tmp;
    double                slew_tmp;
    std::optional<double> limit_tmp;
    double                slack_tmp;
    TransType             rf = TransType::kRise;

    _timing_engine->checkSlew(pin->getFullName().c_str(), AnalysisMode::kMax, rf,
                              slew_tmp, limit_tmp, slack_tmp);
    if (limit_tmp) {
      if (slack_tmp < slew_slack1) {
        slew_slack1 = slack_tmp;
        max_slew1 = *limit_tmp;
      }
    }
  }
  if (slew_slack1 < 0.0) {
    _store.push_back(make_pair(net->get_name(), slew_slack1));
    LibertyPort *drvr_port = nullptr;
    if (drvr_pin->isPin()) {
      drvr_port = dynamic_cast<Pin *>(drvr_pin)->get_cell_port();
    }

    if (drvr_port) {
      // Find max load cap that corresponds to max_slew.
      double max_cap1 = calcLoadCap(drvr_port, max_slew1); // 2
      max_drvr_cap = min(max_drvr_cap, max_cap1);          // 3
      repair_slew = true;
    }
  }
}

double CTSViolationFixer::calcLoadCap(LibertyPort *drvr_port, double slew) {
  double lower_cap = 0.0;
  double upper_cap = slew / drvr_port->driveResistance() * 2;
  double tol = .01;

  double slew_diff_upper_cap = calcSlewDiff(drvr_port, slew, upper_cap);

  while (abs(lower_cap - upper_cap) > tol * max(lower_cap, upper_cap)) {
    if (slew_diff_upper_cap < 0.0) {
      lower_cap = upper_cap;
      upper_cap *= 2;
      slew_diff_upper_cap = calcSlewDiff(drvr_port, slew, upper_cap);
    } else {
      double cap3 = (lower_cap + upper_cap) / 2.0;
      double diff3 = calcSlewDiff(drvr_port, slew, cap3);
      if (diff3 < 0.0) {
        lower_cap = cap3;
      } else {
        upper_cap = cap3;
        slew_diff_upper_cap = diff3;
      }
    }
  }
  return lower_cap;
}

/**
 * @brief Calculate the difference between gate slew and target slew.
 *
 * @param drvr_port
 * @param target_slew
 * @param load_cap
 * @return double
 */
double CTSViolationFixer::calcSlewDiff(LibertyPort *drvr_port, double target_slew,
                                       double load_cap) {
  Delay delays[2];
  Slew  slews[2];
  calcGateRiseFallDelays(drvr_port, load_cap, delays, slews);
  int  fall = (int)TransType::kFall - 1;
  int  rise = (int)TransType::kRise - 1;
  Slew gate_slew = max(slews[fall], slews[rise]);
  return gate_slew - target_slew;
}

/**
 * @brief Rise/fall delays across all timing arcs into drvr_port.
  Uses target slew for input slew.
 *
 * @param drvr_port
 * @param load_cap
 * @param delays
 * @param slews
 */
void CTSViolationFixer::calcGateRiseFallDelays(LibertyPort *drvr_port, float load_cap,
                                               // return values.
                                               Delay delays[], Slew slews[]) {
  for (int rf_index = 0; rf_index < 2; rf_index++) {
    delays[rf_index] = -kInf;
    slews[rf_index] = -kInf;
  }

  LibertyCell *cell = drvr_port->get_ower_cell();
  // get all cell arcset
  auto &cell_arcset = cell->get_cell_arcs();
  for (auto &arcset : cell_arcset) {
    ieda::Vector<std::unique_ptr<ista::LibertyArc>> &arcs = arcset->get_arcs();
    for (auto &arc : arcs) {
      if (arc->isDelayArc()) {
        if ((arc->get_timing_type() == LibertyArc::TimingType::kComb) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kCombRise) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kRisingEdge) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kDefault)) {
          gateRiseFallDelay(TransType::kRise, arc.get(), load_cap, delays, slews);
        }

        if ((arc->get_timing_type() == LibertyArc::TimingType::kComb) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kCombFall) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kFallingEdge) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kDefault)) {
          gateRiseFallDelay(TransType::kFall, arc.get(), load_cap, delays, slews);
        }
      }
    }
  }
}

void CTSViolationFixer::gateRiseFallDelay(TransType rf, LibertyArc *arc, float load_cap,
                                          // return values.
                                          Delay delays[], Slew slews[]) {
  int   rise_fall = (int)rf - 1;
  float in_slew = _targ_slews[rise_fall];
  Delay gate_delay;
  Slew  drvr_slew;
  gate_delay = arc->getDelayOrConstrainCheckNs(rf, in_slew, load_cap);
  drvr_slew = arc->getSlewNs(rf, in_slew, load_cap);
  delays[rise_fall] = max(delays[rise_fall], gate_delay);
  slews[rise_fall] = max(slews[rise_fall], drvr_slew);
}

int CTSViolationFixer::portFanoutLoadNum(LibertyPort *port) {
  auto &fanout_load = port->get_fanout_load();
  if (!fanout_load.has_value()) {
    LibertyCell *   cell = port->get_ower_cell();
    LibertyLibrary *lib = cell->get_owner_lib();
    fanout_load = lib->get_default_fanout_load();
  }
  if (fanout_load) {
    return *fanout_load;
  } else {
    return 1;
  }
}
} // namespace ito
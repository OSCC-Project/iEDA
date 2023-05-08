#include "ViolationOptimizer.h"

#include "api/TimingIDBAdapter.hh"
#include "api/TimingEngine.hh"

using namespace std;

namespace ito {

int ViolationOptimizer::_rise = (int)TransType::kRise - 1;
int ViolationOptimizer::_fall = (int)TransType::kFall - 1;

ViolationOptimizer::ViolationOptimizer(DbInterface *dbinterface)
    : _db_interface(dbinterface) {
  _dbu = _db_interface->get_dbu();
  _timing_engine = _db_interface->get_timing_engine();
  _db_adapter = _timing_engine->get_db_adapter();
  _parasitics_estimator = new EstimateParasitics(_db_interface);
}

void ViolationOptimizer::initBuffer() {
  bool not_specified_buffer = _db_interface->get_drv_insert_buffers().empty();
  if (not_specified_buffer) {
    _insert_buffer_cell = _db_interface->get_lowest_drive_buffer();
    return;
  }

  float low_drive = -kInf;
  auto bufs = _db_interface->get_drv_insert_buffers();
  for (auto buf : bufs) {
    auto buffer = _timing_engine->findLibertyCell(buf.c_str());
    LibertyPort *in_port;
    LibertyPort *out_port;
    buffer->bufferPorts(in_port, out_port);
    float drvr_res = out_port->driveResistance();
    if (drvr_res > low_drive) {
      low_drive = drvr_res;
      _insert_buffer_cell = buffer;
    }
  }
}

void ViolationOptimizer::fixViolations() {
  initBuffer();
  LOG_ERROR_IF(!_insert_buffer_cell) << "Can not found specified buffer.\n";
  int repair_count = 0;
  int slew_violations = 0;
  int length_violations = 0;
  int cap_violations = 0;
  // int fanout_violations = 0;
  _inserted_buffer_count = 0;
  _resize_instance_count = 0;

  //////////////////for test ////////////////////////////////////////////
  int slew_violations_after = 0;
  int length_violations_after = 0;
  int cap_violations_after = 0;
  // int fanout_violations_after = 0;
  //////////////////////////////////////////////////////////////////////

  _parasitics_estimator->estimateAllNetParasitics();
  _timing_engine->updateTiming();
  // _timing_engine->reportTiming();

StaSeqPathData *worst_path_rise =
    _timing_engine->vertexWorstRequiredPath(AnalysisMode::kMax, TransType::kRise);
StaSeqPathData *worst_path_fall =
    _timing_engine->vertexWorstRequiredPath(AnalysisMode::kMax, TransType::kFall);

Slack worst_slack_rise = worst_path_rise->getSlackNs();
Slack worst_slack_fall = worst_path_fall->getSlackNs();
cout << "worst_slack_rise  " << worst_slack_rise << "  worst_slack_fall  " << worst_slack_fall << endl;

  StaSeqPathData *worst_path =
      _timing_engine->vertexWorstRequiredPath(AnalysisMode::kMax, TransType::kRise);
  Slack worst_slack = worst_path->getSlackNs();
  _db_interface->report()->get_ofstream() << "Worst Slack: " << worst_slack << endl;
  _db_interface->report()->get_ofstream().close();

  int       number_drvr_vertices = _db_interface->get_drvr_vertices().size();
  VertexSeq _level_drvr_vertices = _db_interface->get_drvr_vertices();

  int                net_connect_port = 0;

  for (int i = number_drvr_vertices - 1; i >= 0; --i) {
    StaVertex *drvr = _level_drvr_vertices[i];

    auto *design_obj = drvr->get_design_obj();
    auto *net = design_obj->get_net();

    // do not fix clock net
    if (net->isClockNet()) {
      continue;
    }

    if (netConnectToPort(net)) {
      net_connect_port++;
      continue;
    }

    if (!drvr->is_clock() && !drvr->is_const()) {
      repairViolations(net, drvr, _check_slew, _check_cap, repair_count, slew_violations,
                       cap_violations, length_violations);
    }
  }

#ifdef REPORT_TO_TXT
  _db_interface->report()->reportDRVResult(repair_count, slew_violations,
                                           length_violations, cap_violations, 0, true);
#endif
  _timing_engine->updateTiming();

  // ===================test if there are still violations===================
  checkViolations();

  if (!_still_violation_net.empty()) {
    // If there are still a violation nets, the secondary fix is performed.
    for (auto net : _still_violation_net) {
      DesignObject *driver = net->getDriver();
      StaVertex    *drvr = _timing_engine->findVertex(driver->getFullName().c_str());

      repairViolations(net, drvr, _check_slew, _check_cap, repair_count, slew_violations,
                       cap_violations, length_violations);
    }
    _still_violation_net.clear();
    _timing_engine->updateTiming();
    checkViolations();
  }
  for (auto net : _still_violation_net) {
    auto net_name = net->get_name();
    _db_interface->report()->report(net_name);
  }

#ifdef REPORT_TO_TXT
  _db_interface->report()->get_ofstream()
      << "Inserted " << _inserted_buffer_count << " buffers in " << repair_count
      << " nets.\n"
      << "Resized " << _resize_instance_count << " instances." << endl;
  _db_interface->report()->reportTime(false);
#endif

  printf("\t\t\t\t\t\tFound {%d} slew violations.\n", slew_violations);
  printf("\t\t\t\t\t\tFound {%d} capacitance violations.\n", cap_violations);
  printf("\t\t\t\t\t\tFound {%d} long wires.\n", length_violations);
  printf("\t\t\t\t\t\tInserted {%d} buffers in {%d} nets.\n", _inserted_buffer_count,
         repair_count);
  printf("\t\t\t\t\t\tResized {%d} instances.\n", _resize_instance_count);

  printf("Before ViolationFix | slew_vio:%d  cap_vio:%d  length_vio:%d\n",
         slew_violations, cap_violations, length_violations);

  printf("===============================test if there are still "
         "violations==================================\n");
  printf("After  ViolationFix | slew_vio:%d  cap_vio:%d  length_vio:%d\n",
         slew_violations_after, cap_violations_after, length_violations_after);
}

void ViolationOptimizer::fixViolations(const char *net_name) {
  initBuffer();

  _parasitics_estimator->estimateAllNetParasitics();
  _timing_engine->updateTiming();

  Netlist *design_nl = _timing_engine->get_netlist();
  Net *    net = design_nl->findNet(net_name);

  DesignObject *driver = net->getDriver();

  if (driver) {
    int repair_count = 0;
    int slew_violations = 0;
    int length_violations = 0;
    int cap_violations = 0;
    StaVertex *drvr_vertex = _timing_engine->findVertex(driver->getFullName().c_str());
    repairViolations(net, drvr_vertex, _check_slew, _check_cap, repair_count,
                     slew_violations, cap_violations, length_violations);

    _timing_engine->updateTiming();
    checkViolations();
    // write to def
    TimingIDBAdapter *idb_adapter = dynamic_cast<TimingIDBAdapter *>(_db_adapter);
    IdbBuilder *      idb_builder = idb_adapter->get_idb();
    string            defWritePath = string(_db_interface->get_output_def_file());
    idb_builder->saveDef(defWritePath);
  }
}

void ViolationOptimizer::checkViolations() {
  int slew_violations_after = 0;
  int cap_violations_after = 0;

  Netlist *design_nl = _timing_engine->get_netlist();
  Net     *net;
  // VertexSeq drvr_vertices;
  FOREACH_NET(design_nl, net) {
    if (net->isClockNet() || netConnectToPort(net)) {
      continue;
    }

    DesignObject *driver = net->getDriver();
    if (driver) {
      double max_drvr_cap = kInf;
      bool   repair_cap = false;
      bool   repair_slew = false;

      if (_check_cap) {
        checkCapacitanceViolation(driver, max_drvr_cap, cap_violations_after, repair_cap);
      }
      if (_check_slew) {
        checkSlewViolation(driver, max_drvr_cap, slew_violations_after, repair_slew);
      }
      if (repair_cap || repair_slew) {
        _still_violation_net.push_back(net);
      }
    }
  }
#ifdef REPORT_TO_TXT
  _db_interface->report()->reportDRVResult(0, slew_violations_after, 0,
                                           cap_violations_after, 0, false);
#endif
}

void ViolationOptimizer::repairViolations(Net *net, StaVertex *drvr, bool check_slew,
                                          bool check_cap, int &repair_count,
                                          int &slew_violations, int &cap_violations,
                                          int &length_violations) {
  DesignObject *drvr_pin_port = drvr->get_design_obj();

  RoutingTree *tree;
  tree = makeRoutingTree(net, _db_adapter, RoutingType::kSteiner);
  if (tree) {
    // printTree(tree->get_root());
    LibertyPort *buffer_input_port, *buffer_output_port;
    LibertyCell *buffer_cell = _insert_buffer_cell;
    buffer_cell->bufferPorts(buffer_input_port, buffer_output_port);

    float buf_cap_limit = 0.0;
    bool  buf_cap_limit_exists = false;

    std::optional<double> cap_limit =
        buffer_output_port->get_port_cap_limit(AnalysisMode::kMax);
    if (cap_limit.has_value()) {
      buf_cap_limit_exists = true;
      buf_cap_limit = *cap_limit;
    }

    double max_drvr_cap = kInf;
    // double max_fanout = kInf;
    bool repair_cap = false;
    bool repair_slew = false;
    // bool   repair_fanout = false;
    // bool repair_wire_length = false;

    // cap violation
    if (check_cap) {
      checkCapacitanceViolation(drvr_pin_port, max_drvr_cap, cap_violations, repair_cap);
    }
    // slew violation
    if (check_slew) {
      checkSlewViolation(drvr_pin_port, max_drvr_cap, slew_violations, repair_slew);
    }

    if (repair_slew || repair_cap) {
      double max_cap = max_drvr_cap;

      if (buf_cap_limit_exists) {
        max_cap = buf_cap_limit;
      }

      TimingIDBAdapter       *idb_adapter = dynamic_cast<TimingIDBAdapter *>(_db_adapter);
      IdbCoordinate<int32_t> *loc = idb_adapter->idbLocation(drvr_pin_port);
      Point                   drvr_pt = Point(loc->get_x(), loc->get_y());

      int          wire_length;
      float        pin_cap;
      DesignObjSeq load_pins;

      int root_id = tree->get_root()->get_id();
      tree->updateBranch();
      fixViolations(tree, root_id, RoutingTree::_null_pt, net, max_cap, 0, wire_length,
                    pin_cap, load_pins);

      repair_count++;

      std::optional<double> width = std::nullopt;
      double wire_length_cap = dynamic_cast<TimingIDBAdapter *>(_db_adapter)
                                   ->getCapacitance(1, (double)wire_length / _dbu, width);
      double drvr_load_cap = pin_cap + wire_length_cap;

      if (drvr_load_cap > max_drvr_cap) {
        printf("ViolationFixer | func  insertBuffer direction{drvr}\n");
        LibertyCell *insert_buffer_cell = _insert_buffer_cell;
        insertBuffer(drvr_pt.get_x(), drvr_pt.get_y(), net, insert_buffer_cell, 0,
                     wire_length, pin_cap, load_pins);
      }
    }
    delete tree;
  }
}

void ViolationOptimizer::fixViolations(RoutingTree *tree, int curr_pt, int prev_pt,
                                       Net *net, float max_cap, int level,
                                       // Return values.
                                       // Remaining parasiics after repeater insertion.
                                       int   &wire_length, // dbu
                                       float &pin_cap, DesignObjSeq &load_pins) {
  int   left_branch = tree->left(curr_pt);
  int   left_wire_length = 0;
  float left_pin_cap = 0.0;
  // float        left_fanout = 0.0;
  DesignObjSeq left_load_pins;
  if (left_branch != RoutingTree::_null_pt) {
    fixViolations(tree, left_branch, curr_pt, net, max_cap, level + 1, left_wire_length,
                  left_pin_cap, left_load_pins);
  }

  int          middle_branch = tree->middle(curr_pt);
  int          middle_wire_length = 0;
  float        middle_pin_cap = 0.0;
  DesignObjSeq middle_load_pins;
  if (middle_branch != RoutingTree::_null_pt) {
    fixViolations(tree, middle_branch, curr_pt, net, max_cap, level + 1,
                  middle_wire_length, middle_pin_cap, middle_load_pins);
  }

  int          right_branch = tree->right(curr_pt);
  int          right_wire_length = 0;
  float        right_pin_cap = 0.0;
  DesignObjSeq right_load_pins;
  if (right_branch != RoutingTree::_null_pt) {
    fixViolations(tree, right_branch, curr_pt, net, max_cap, level + 1, right_wire_length,
                  right_pin_cap, right_load_pins);
  }

  // Add up to three buffers to left/middle/right branch to stay under the max
  // cap/length/fanout.
  bool fix_left = false;
  bool fix_middle = false;
  bool fix_right = false;

  // cap violation
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

  Point        buffer_loc = tree->get_location(curr_pt);
  LibertyCell *insert_buffer_cell = _insert_buffer_cell;

  if (fix_left) {
    printf("repair_net | level{%d}  direction{left}\n", level);
    insertBuffer(buffer_loc.get_x(), buffer_loc.get_y(), net, insert_buffer_cell, level,
                 left_wire_length, left_pin_cap, left_load_pins);
  }
  if (fix_middle) {
    printf("repair_net | level{%d}  direction{middle}\n", level);
    insertBuffer(buffer_loc.get_x(), buffer_loc.get_y(), net, insert_buffer_cell, level,
                 middle_wire_length, middle_pin_cap, middle_load_pins);
  }
  if (fix_right) {
    printf("repair_net | level{%d}  direction{right}\n", level);
    insertBuffer(buffer_loc.get_x(), buffer_loc.get_y(), net, insert_buffer_cell, level,
                 right_wire_length, right_pin_cap, right_load_pins);
  }

  wire_length = left_wire_length + middle_wire_length + right_wire_length;
  pin_cap = left_pin_cap + middle_pin_cap + right_pin_cap;
  // fanout = left_fanout + middle_fanout + right_fanout;

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
  if (prev_pt != RoutingTree::_null_pt) {
    DesignObject *load_pin = tree->get_pin(curr_pt);
    Point         curr_pt_loc = tree->get_location(curr_pt);
    int           curr_pt_x = curr_pt_loc.get_x();
    int           curr_pt_y = curr_pt_loc.get_y();

    if (load_pin) {
      if (load_pin->isPin()) {
        LibertyPort *load_port = dynamic_cast<Pin *>(load_pin)->get_cell_port();
        if (load_port) {
          pin_cap += load_pin->cap(AnalysisMode::kMax, TransType::kRise);
          // The constructed tree structure may have the case that the children
          // of a pin are also a pin. This results in pin_cap >= max_cap, which
          // should be separated.
          if (pin_cap >= max_cap) {
            int   left_branch = tree->left(curr_pt);
            Point left_loc = tree->get_location(left_branch);
            int   buf_x = (left_loc.get_x() + curr_pt_x) / 2;
            int   buf_y = (left_loc.get_y() + curr_pt_y) / 2;
            insertBuffer(buf_x, buf_y, net, insert_buffer_cell, level, wire_length,
                         pin_cap, load_pins);
            wire_length +=
                (abs(left_loc.get_x() - curr_pt_x) + abs(left_loc.get_y() - curr_pt_y)) /
                2;
            pin_cap += load_pin->cap(AnalysisMode::kMax, TransType::kRise);
          }
        }
      } else {
        pin_cap += load_pin->cap(AnalysisMode::kMax, TransType::kRise);
      }
      load_pins.push_back(load_pin);
    }

    Point prev_loc = tree->get_location(prev_pt);

    int length = Point::manhattanDistance(prev_loc, curr_pt_loc);
    wire_length += length;
    // Back up from pt to prev_pt adding repeaters every max_length.
    int prev_x = prev_loc.get_x();
    int prev_y = prev_loc.get_y();

    std::optional<double> width = std::nullopt;
    double                wire_length_cap = dynamic_cast<TimingIDBAdapter *>(_db_adapter)
                                 ->getCapacitance(1, (double)wire_length / _dbu, width);
    double pin_wire_length_cap = pin_cap + wire_length_cap;

    int wire_insert_count = 0; // number of buffer insert in wire

    while ((pin_cap < max_cap && pin_wire_length_cap > max_cap)) {
      // Make the wire a bit shorter than necessary to allow for
      // offset from instance origin to pin and detailed placement movement.
      double length_margin = .2;
      double buf_dist;

      std::optional<double> width = std::nullopt;
      int                   cap_length = dynamic_cast<TimingIDBAdapter *>(_db_adapter)
                            ->capToLength(1, abs(max_cap - pin_cap), width) *
                        _dbu;
      buf_dist = length - (wire_length - cap_length * (1.0 - length_margin));

      buf_dist = max(0.0, buf_dist);

      double dx = prev_x - curr_pt_x;
      double dy = prev_y - curr_pt_y;
      double d = buf_dist / length;
      int    buf_x = curr_pt_x + d * dx;
      int    buf_y = curr_pt_y + d * dy;

      insertBuffer(buf_x, buf_y, net, insert_buffer_cell, level, wire_length, pin_cap,
                   load_pins);
      wire_insert_count++;

      // Update for the next round.
      length -= buf_dist;
      wire_length = length;
      curr_pt_x = buf_x;
      curr_pt_y = buf_y;

      // update the sum of pin_cap and wire_length_cap
      double wire_length_cap = dynamic_cast<TimingIDBAdapter *>(_db_adapter)
                                   ->getCapacitance(1, (double)wire_length / _dbu, width);
      pin_wire_length_cap = pin_cap + wire_length_cap;
    }
  }
}

template <class T1, class T2>
void ViolationOptimizer::determineFixSide(T1 max_numb, T2 left, T2 middle, T2 right,
                                          bool &fix_left, bool &fix_middle,
                                          bool &fix_right) {
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

void ViolationOptimizer::insertBuffer(int x, int y, Net *net,
                                      LibertyCell *insert_buf_cell, int level,
                                      int &wire_length, float &cap,
                                      DesignObjSeq &load_pins) {
  bool overlap = _db_interface->get_core().overlaps(x, y);
  if (overlap) {
    LibertyPort *buffer_input_port, *buffer_output_port;
    insert_buf_cell->bufferPorts(buffer_input_port, buffer_output_port);

    // make instance name
    std::string buffer_name = ("DRV_buffer_" + to_string(_insert_instance_index));
    _insert_instance_index++;

    TimingIDBAdapter *db_adapter = dynamic_cast<TimingIDBAdapter *>(_db_adapter);

    Net *in_net, *out_net;
    in_net = net;
    // make net name
    std::string net_name = ("DRV_net_" + to_string(_make_net_index));
    _make_net_index++;
    out_net = db_adapter->makeNet(net_name.c_str(), nullptr);
    // Copy signal type to new net.
    idb::IdbNet *out_net_db = db_adapter->staToDb(out_net);
    idb::IdbNet *in_net_db = db_adapter->staToDb(in_net);
    out_net_db->set_connect_type(in_net_db->get_connect_type());

    // Move load pins to out_net.
    for (auto *pin_port : load_pins) {
      if (pin_port->isPin()) {
        Pin         *pin = dynamic_cast<Pin *>(pin_port);
        Instance    *inst = pin->get_own_instance();

        db_adapter->disconnectPin(pin);
        auto debug = db_adapter->connect(inst, pin->get_name(), out_net);
        LOG_ERROR_IF(!debug);
      }
    }
    Instance *buffer = db_adapter->makeInstance(insert_buf_cell, buffer_name.c_str());

    idb::IdbCellMaster *idb_master = db_adapter->staToDb(insert_buf_cell);
    Master             *master = new Master(idb_master);

    float area = DesignCalculator::calcMasterArea(master, _dbu);
    increDesignArea(area);

    _inserted_buffer_count++;

    auto debug_buf_in = db_adapter->connect(buffer, buffer_input_port->get_port_name(), in_net);
    auto debug_buf_out = db_adapter->connect(buffer, buffer_output_port->get_port_name(), out_net);
    LOG_ERROR_IF(!debug_buf_in);
    LOG_ERROR_IF(!debug_buf_out);

    _timing_engine->insertBuffer(buffer->get_name());

    // Resize repeater as we back up by levels.
    Pin *drvr_pin = buffer->findPin(buffer_output_port);

    std::optional<double> width = std::nullopt;
    double                wire_cap = dynamic_cast<TimingIDBAdapter *>(_db_adapter)
                          ->getCapacitance(1, (double)wire_length / _dbu, width);
    std::optional<double> cap_limit =
        buffer_output_port->get_port_cap_limit(AnalysisMode::kMax);
    if (cap_limit.has_value()) {
      double buf_cap_limit = *cap_limit;
      if (buf_cap_limit < cap + wire_cap) {
        if (repowerInstance(drvr_pin)) {
          auto inst = drvr_pin->get_own_instance();
          Pin *rsz_pin;
          FOREACH_INSTANCE_PIN(inst, rsz_pin) {
            Net *rsz_net = rsz_pin->get_net();
            if (rsz_net) {
              _parasitics_estimator->parasiticsInvalid(rsz_net);
              _parasitics_estimator->estimateInvalidNetParasitics(rsz_net->getDriver(),
                                                                  rsz_net);
            }
          }
          _resize_instance_count++;
        }
      }
    }

    setLocation(buffer, x, y);

    _parasitics_estimator->parasiticsInvalid(in_net);
    _parasitics_estimator->parasiticsInvalid(out_net);

    _parasitics_estimator->estimateInvalidNetParasitics(in_net->getDriver(), in_net);
    _parasitics_estimator->estimateInvalidNetParasitics(out_net->getDriver(), out_net);

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
    }
  }
}

void ViolationOptimizer::setLocation(Instance *inst, int x, int y) {
  TimingIDBAdapter *idb_adapter = dynamic_cast<TimingIDBAdapter *>(_db_adapter);
  idb::IdbInstance *idb_inst = idb_adapter->staToDb(inst);
  unsigned          master_width = idb_inst->get_cell_master()->get_width();
  pair<int, int>    loc = _db_interface->placer()->findNearestSpace(master_width, x, y);
  idb_inst->set_status_placed();
  idb_inst->set_coodinate(loc.first, loc.second);
  // set orient
  auto row = _db_interface->placer()->findRow(loc.second);
  if (row) {
    idb_inst->set_orient(row->get_site()->get_orient());
  } else {
    idb_inst->set_orient(idb::IdbOrient::kN_R0);
  }
  _db_interface->placer()->updateRow(master_width, loc.first, loc.second);
}

bool ViolationOptimizer::repowerInstance(Pin *drvr_pin) {
  Instance    *inst = drvr_pin->get_own_instance();
  LibertyCell *cell = inst->get_inst_cell();
  if (cell) {
    Vector<LibertyCell *> *equiv_cells = _timing_engine->equivCells(cell);
    if (equiv_cells) {
      bool revisiting_inst = false;
      if (hasMultipleOutputs(inst)) {
        if (_resized_multi_output_insts.find(inst) != _resized_multi_output_insts.end()) {
          revisiting_inst = true;
        }
        _resized_multi_output_insts.insert(inst);
      }
      bool is_buf_inv = cell->isBuffer() || cell->isInverter();

      Net *net = drvr_pin->get_net();
      if (net) {
        _parasitics_estimator->estimateInvalidNetParasitics(drvr_pin, net);
      }

      // Includes net parasitic capacitance.
      double load = drvr_pin->get_net()->getLoad(AnalysisMode::kMax, TransType::kRise);
      if (load > 0.0) {
        LibertyCell *best_cell = cell;
        float        target_load = (*_db_interface->get_cell_target_load_map())[cell];
        float        best_load = target_load;
        float        best_dist = abs(load - target_load); //
        float        best_delay = is_buf_inv ? calcBufferDelay(cell, load) : 0.0;

        for (auto *target_cell : *equiv_cells) {
          //!_db_interface->dontUse(target_cell) // TODO:

          // Do not use clk buffer in signal net, and vice versa
          const char *buf_name = target_cell->get_cell_name();
          if (strstr(buf_name, "CLK") != NULL || strstr(buf_name, "24") != NULL ||
              strstr(buf_name, "20") != NULL) {
            continue;
          }

          if (_db_interface->isLinkCell(target_cell)) {
            float target_cell_load =
                (*_db_interface->get_cell_target_load_map())[target_cell];
            float delay = is_buf_inv ? calcBufferDelay(target_cell, load) : 0.0;
            float dist = abs(load - target_cell_load);

            if (is_buf_inv
                    ? ((delay < best_delay && dist < best_dist * 1.1) ||
                       (dist < best_dist && delay < best_delay * 1.1))
                    : dist < best_dist && (!revisiting_inst || target_load > best_load)) {
              best_cell = target_cell;
              best_dist = dist;
              best_load = target_load;
              best_delay = delay;
            }
          }
        }
        if (best_cell != cell) {
          return repowerInstance(inst, best_cell) && !revisiting_inst;
        }
      }
    }
  }
  return false;
}

bool ViolationOptimizer::repowerInstance(Instance *inst, LibertyCell *replace) {
  const char *replacement_name = replace->get_cell_name();

  TimingIDBAdapter   *idb_adapter = dynamic_cast<TimingIDBAdapter *>(_db_adapter);
  idb::IdbLayout     *layout = idb_adapter->get_idb()->get_def_service()->get_layout();
  idb::IdbCellMaster *replacement_master =
      layout->get_cell_master_list()->find_cell_master(replacement_name);
  if (replacement_master) {
    idb::IdbInstance   *dinst = idb_adapter->staToDb(inst);
    idb::IdbCellMaster *idb_master = dinst->get_cell_master();
    Master             *master = new Master(idb_master);
    float               area_master = DesignCalculator::calcMasterArea(master, _dbu);

    if (replacement_master->get_name() == idb_master->get_name()) {
      return false;
    }

    increDesignArea(-area_master);

    idb::IdbCellMaster *replace_master_idb = idb_adapter->staToDb(replace);

    Master *replace_master = new Master(replace_master_idb);
    float   area_replace_master = DesignCalculator::calcMasterArea(replace_master, _dbu);

    idb_adapter->replaceCell(inst, replace);
    _timing_engine->repowerInstance(inst->get_name(), replace->get_cell_name());
    increDesignArea(area_replace_master);

    return true;
  }
  return false;
}

bool ViolationOptimizer::hasMultipleOutputs(Instance *inst) {
  int  output_count = 0;
  Pin *pin;
  FOREACH_INSTANCE_PIN(inst, pin) {
    if (pin->isOutput()) {
      output_count++;
      if (output_count > 1) {
        return true;
      }
    }
  }
  return false;
}

double ViolationOptimizer::calcBufferDelay(LibertyCell *buffer_cell, float load) {
  LibertyPort *input, *output;
  buffer_cell->bufferPorts(input, output);
  Delay gate_delays[2];
  Slew  slews[2];
  calcGateRiseFallDelays(output, load, gate_delays, slews);
  return max(gate_delays[_rise], gate_delays[_fall]);
}

void ViolationOptimizer::checkCapacitanceViolation(DesignObject *drvr_pin,
                                                   // return values
                                                   double &max_drvr_cap,
                                                   int    &cap_violations,
                                                   bool   &repair_cap) {
  double                cap1;
  std::optional<double> max_cap1;
  double                cap_slack1;

  TransType rf = TransType::kRise;
  _timing_engine->checkCapacitance(drvr_pin->getFullName().c_str(), AnalysisMode::kMax,
                                   rf, cap1, max_cap1, cap_slack1);
  if (max_cap1) {
    max_drvr_cap = *max_cap1;
    if (cap_slack1 < 0) {
      repair_cap = true;
      cap_violations++;
    }
  }
}

void ViolationOptimizer::checkSlewViolation(DesignObject *drvr_pin,
                                            // return values
                                            double &max_drvr_cap, int &slew_violations,
                                            bool &repair_slew) {
  float slew_slack1 = kInf;
  float max_slew1 = kInf;

  Net          *net = drvr_pin->get_net();
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
    _slew_record.push_back(slew_slack1);
    slew_violations++;
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

/**
 * @brief Calculata the output port load capacitance that causes the current
 * slew.
 *
 * @param drvr_port
 * @param slew
 * @return double
 */
double ViolationOptimizer::calcLoadCap(LibertyPort *drvr_port, double slew) {
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
double ViolationOptimizer::calcSlewDiff(LibertyPort *drvr_port, double target_slew,
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
void ViolationOptimizer::calcGateRiseFallDelays(LibertyPort *drvr_port, float load_cap,
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

void ViolationOptimizer::gateRiseFallDelay(TransType rf, LibertyArc *arc, float load_cap,
                                           // return values.
                                           Delay delays[], Slew slews[]) {
  int   rise_fall = (int)rf - 1;
  float in_slew = _db_interface->get_target_slews()[rise_fall];
  Delay gate_delay;
  Slew  drvr_slew;
  gate_delay = arc->getDelayOrConstrainCheck(rf, in_slew, load_cap);
  drvr_slew = arc->getSlew(rf, in_slew, load_cap);
  delays[rise_fall] = max(delays[rise_fall], gate_delay);
  slews[rise_fall] = max(slews[rise_fall], drvr_slew);
}

/**
 * @brief net's loads contain a port.
 *
 * @param net
 * @return true
 * @return false
 */
bool ViolationOptimizer::netConnectToPort(Net *net) {
  auto load_pin_ports = net->getLoads();
  for (auto pin_port : load_pin_ports) {
    if (pin_port->isPort()) {
      return true;
    }
  }
  return false;
}

int ViolationOptimizer::portFanoutLoadNum(LibertyPort *port) {
  auto &fanout_load = port->get_fanout_load();
  if (!fanout_load.has_value()) {
    LibertyCell    *cell = port->get_ower_cell();
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
#include "HoldOptimizer.h"

#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"

using namespace std;

namespace ito {
int HoldOptimizer::_mode_max = (int)AnalysisMode::kMax - 1;
int HoldOptimizer::_mode_min = (int)AnalysisMode::kMin - 1;
int HoldOptimizer::_rise = (int)TransType::kRise - 1;
int HoldOptimizer::_fall = (int)TransType::kFall - 1;

HoldOptimizer::HoldOptimizer(DbInterface *dbinterface) : _db_interface(dbinterface) {
  _timing_engine = _db_interface->get_timing_engine();
  _db_adapter = _timing_engine->get_db_adapter();
  _dbu = _db_interface->get_dbu();

  _parasitics_estimator = new EstimateParasitics(_db_interface);
  _violation_fixer = new ViolationOptimizer(_db_interface);

  _slack_margin = _db_interface->get_hold_slack_margin();
}

void HoldOptimizer::optimizeHold() {
  _parasitics_estimator->estimateAllNetParasitics();

  _timing_engine->updateTiming();
  _timing_engine->reportTiming();

  reportWNSAndTNS();

  initBufferCell();
  LOG_ERROR_IF(_buffer_cells.empty()) << "Can not found specified buffers.\n";
  calcBufferCap();

  // Preparation
  float max_buffer_percent = _db_interface->get_max_buffer_percent();
  int   instance_num = _timing_engine->get_netlist()->getInstanceNum();
  _max_numb_insert_buf = max_buffer_percent * instance_num;
  LibertyCell *insert_buf_cell = findBufferWithMaxDelay();

  // get end points
  VertexSet end_points = getEndPoints();

  Slack worst_slack;
  // endpoints with hold violation.
  VertexSet end_pts_hold_violation;

  Slack worst_hold_slack = getWorstSlack(AnalysisMode::kMin);
  int   iteration = 1;
  int   insert_buf_count = 1;
  while (insert_buf_count > 0 && worst_hold_slack < _slack_margin) {
    insert_buf_count = checkAndOptimizeHold(end_points, insert_buf_cell);
    _parasitics_estimator->estimateAllNetParasitics();
    _timing_engine->updateTiming();
    worst_hold_slack = getWorstSlack(AnalysisMode::kMin);
    _db_interface->report()->get_ofstream()
        << "\nThe " << iteration << "-th timing check." << endl
        << "\tworst hold slack: " << worst_hold_slack << endl;
    iteration++;
  }

  if (fuzzyLess(worst_hold_slack, _slack_margin)) {
    findEndpointsWithHoldViolation(end_points, worst_slack, end_pts_hold_violation);
    _db_interface->report()->get_ofstream()
        << "Unable to repair all hold violations. There are still "
        << end_pts_hold_violation.size() << " endpoints with hold violation." << endl;
  }

  if (_inserted_buffer_count > _max_numb_insert_buf) {
    printf("Max buffer count reached.\n");
    _db_interface->report()->report("Max buffer count reached.\n");
  }
  if (_db_interface->overMaxArea()) {
    printf("Max utilization reached.\n");
    _db_interface->report()->report("Max utilization reached.\n");
  }

  _db_interface->report()->get_ofstream()
      << "\nFinish hold optimization!\n"
      << "Total inserted " << _inserted_buffer_count << " hold buffers and "
      << _inserted_load_buffer_count << " load buffers.\n";

  reportWNSAndTNS();

  _db_interface->report()->reportTime(false);

  // _timing_engine->reportTiming();
}

int HoldOptimizer::checkAndOptimizeHold(VertexSet    end_points,
                                        LibertyCell *insert_buf_cell) {
  // store worst hold slack
  vector<Slack> hold_slacks;
  // store end's number with hold violation
  vector<int> hold_vio_num;
  // store inserted hold buffer number
  vector<int> insert_buf_num;

  int insert_buf_count = 0;

  Slack worst_slack;
  // endpoints with hold violation.
  VertexSet end_pts_hold_violation;
  findEndpointsWithHoldViolation(end_points, worst_slack, end_pts_hold_violation);
  hold_slacks.push_back(worst_slack);
  hold_vio_num.push_back(end_pts_hold_violation.size());

  _db_interface->report()->get_ofstream()
      << "\nBeign hold optimization! Hold target slack -> " << _slack_margin << endl;
  if (!end_pts_hold_violation.empty()) {
    _db_interface->report()->get_ofstream() << "\nFound " << end_pts_hold_violation.size()
                                            << " endpoints with hold violations.\n";
    _db_interface->report()->get_ofstream().close();
    int repair_count = 1;
    int pass = 1;

    while (!end_pts_hold_violation.empty() &&
           _inserted_buffer_count < _max_numb_insert_buf &&
           !_db_interface->overMaxArea() && repair_count > 0) {
      VertexSet fanins = getFanins(end_pts_hold_violation);

      VertexSeq sorted_fanins = sortFanins(fanins);

      repair_count = fixHoldPass(sorted_fanins, insert_buf_cell);

      insert_buf_count += repair_count;
      insert_buf_num.push_back(repair_count);
      if (repair_count > 0) {
        _parasitics_estimator->excuteParasiticsEstimate();
        _timing_engine->updateTiming();
      }

      findEndpointsWithHoldViolation(end_points, worst_slack, end_pts_hold_violation);
      hold_slacks.push_back(worst_slack);
      hold_vio_num.push_back(end_pts_hold_violation.size());
      pass++;
    }
  } else {
    _db_interface->report()->get_ofstream() << "No hold violations found.\n";
  }
  _db_interface->report()->reportHoldResult(hold_slacks, hold_vio_num, insert_buf_num,
                                            worst_slack, _inserted_buffer_count);

  _db_interface->report()->get_ofstream()
      << "Inserted " << insert_buf_count << " hold buffers.\n";
  _db_interface->report()->get_ofstream().close();
  return insert_buf_count;
}

void HoldOptimizer::initBufferCell() {
  bool not_specified_buffer = _db_interface->get_hold_insert_buffers().empty();
  if (not_specified_buffer) {
    LibertyCellSeq buf_cells = _db_interface->get_buffer_cells();
    _buffer_cells = std::move(buf_cells);
    return;
  }

  auto bufs = _db_interface->get_hold_insert_buffers();
  for (auto buf : bufs) {
    auto buffer = _timing_engine->findLibertyCell(buf.c_str());
    _buffer_cells.emplace_back(buffer);
  }
}

void HoldOptimizer::calcBufferCap() {
  for (auto buf : _buffer_cells) {
    LibertyPort *input, *output;
    buf->bufferPorts(input, output);
    double buf_in_cap = input->get_port_cap();
    _buffer_cap_pair.emplace_back(make_pair(buf_in_cap, buf));
  }
  sort(_buffer_cap_pair.begin(), _buffer_cap_pair.end(),
       [=, this](std::pair<double, ista::LibertyCell *> v1,
                 std::pair<double, ista::LibertyCell *> v2) {
         return v1.first > v2.first;
       });
}

void HoldOptimizer::insertLoadBuffer(VertexSeq fanins) {
  for (auto vertex : fanins) {
    auto rise_cap = _timing_engine->get_ista()->getVertexCapacitanceLimit(
        vertex, AnalysisMode::kMax, TransType::kRise);
    auto fall_cap = _timing_engine->get_ista()->getVertexCapacitanceLimit(
        vertex, AnalysisMode::kMax, TransType::kFall);
    double rise_cap_limit = rise_cap ? (*rise_cap) : 0.0;
    double fall_cap_limit = fall_cap ? (*fall_cap) : 0.0;

    double cap_limit = min(rise_cap_limit, fall_cap_limit);

    double load_cap_rise = _timing_engine->get_ista()->getVertexCapacitance(
        vertex, AnalysisMode::kMax, TransType::kRise);
    double load_cap_fall = _timing_engine->get_ista()->getVertexCapacitance(
        vertex, AnalysisMode::kMax, TransType::kFall);
    double load_cap = max(load_cap_rise, load_cap_fall);

    auto   max_fanout1 = vertex->getMaxFanout();
    double max_fanout = max_fanout1 ? *(max_fanout1) : 10;
    auto   fanout = vertex->get_design_obj()->get_net()->getFanouts();

    LibertyCell *insert_load_buffer = nullptr;
    auto         buffer_cells = _db_interface->get_buffer_cells();
    if (load_cap < cap_limit && fanout < max_fanout) {
      double cap_slack = cap_limit - load_cap;

      auto iter = find_if(_buffer_cap_pair.begin(), _buffer_cap_pair.end(),
                          [&cap_slack](std::pair<double, ista::LibertyCell *> item) {
                            return item.first <= cap_slack;
                          });

      insert_load_buffer = (*iter).second;
      double load_cap = (*iter).first;

      int insert_number_cap = cap_slack / (2 * load_cap);
      int insert_number_fanout = max_fanout - fanout;

      int insert_number = min(insert_number_cap, insert_number_fanout);
      // insert_number = min(insert_number, 1);

      if (insert_load_buffer && insert_number > 0) {
        insertLoadBuffer(insert_load_buffer, vertex, insert_number);
        _inserted_load_buffer_count += insert_number;
      }
    }
  }
}

int HoldOptimizer::fixHoldPass(VertexSeq fanins, LibertyCell *insert_buffer_cell) {
  int repair_pass_count = 0;
  int insert_buffer_count = 0;
  for (size_t i = 0; i < fanins.size(); i++) {
    StaVertex *vertex = fanins[i];

    // loads with hold violation.
    DesignObjSeq load_pins;

    Delay max_insert_delay = 1e+30;

    DesignObjSeq loads = vertex->get_design_obj()->get_net()->getLoads();
    for (auto load_obj : loads) {
      StaVertex *fanout_vertex =
          _timing_engine->findVertex(load_obj->getFullName().c_str());

      if (!fanout_vertex) {
        continue;
      }

      Slack hold_slack = getWorstSlack(fanout_vertex, AnalysisMode::kMin);
      Slack setup_slack = getWorstSlack(fanout_vertex, AnalysisMode::kMax);
      if (hold_slack < _slack_margin) {
        Delay delay = _allow_setup_violation
                          ? _slack_margin - hold_slack
                          : min(_slack_margin - hold_slack, setup_slack);

        if (delay > 0.0) {
          max_insert_delay = min(max_insert_delay, delay);
          // add load with hold violation.
          load_pins.push_back(fanout_vertex->get_design_obj());
        }
      }
    } // for all loads

    if (!load_pins.empty()) {
      Delay buffer_hold_delay = getBufferHoldDelay(insert_buffer_cell);
      int   insert_number = std::ceil(max_insert_delay / buffer_hold_delay);

      repair_pass_count += 1;
      insert_buffer_count += insert_number;
      insertBufferDelay(vertex, insert_number, load_pins, insert_buffer_cell);

      if (_inserted_buffer_count > _max_numb_insert_buf || _db_interface->overMaxArea()) {
        // return repair_pass_count;
        return insert_buffer_count;
      }
    }
  }
  return insert_buffer_count;
}

void HoldOptimizer::insertLoadBuffer(LibertyCell *load_buffer, StaVertex *drvr_vtx,
                                     int insert_num) {
  auto              drvr = drvr_vtx->get_design_obj();
  TimingIDBAdapter *idb_adapter = dynamic_cast<TimingIDBAdapter *>(_db_adapter);

  DesignObjSeq loads = drvr_vtx->get_design_obj()->get_net()->getLoads();
  if (loads.empty()) {
    return;
  }
  auto cap_FF = loads[0];

  IdbCoordinate<int32_t> *loc = idb_adapter->idbLocation(cap_FF);
  Point                   drvr_pin_loc = Point(loc->get_x(), loc->get_y());

  for (int i = 0; i < insert_num; i++) {
    std::string buffer_name = ("hold_load_buf_" + to_string(_insert_instance_index));
    _insert_instance_index++;
    auto buffer = idb_adapter->makeInstance(load_buffer, buffer_name.c_str());

    LibertyPort *input, *output;
    load_buffer->bufferPorts(input, output);
    auto debug_buf_in =
        idb_adapter->connect(buffer, input->get_port_name(), drvr->get_net());
    LOG_ERROR_IF(!debug_buf_in);

    int loc_x = drvr_pin_loc.get_x();
    int loc_y = drvr_pin_loc.get_y();

    if (!_db_interface->get_core().overlaps(loc_x, loc_y)) {
      if (loc_x > _db_interface->get_core().get_x_max()) {
        loc_x = _db_interface->get_core().get_x_max() - 1;
      } else if (loc_x < _db_interface->get_core().get_x_min()) {
        loc_x = _db_interface->get_core().get_x_min() + 1;
      }

      if (loc_y > _db_interface->get_core().get_y_max()) {
        loc_y = _db_interface->get_core().get_y_max() - 1;
      } else if (loc_y < _db_interface->get_core().get_y_min()) {
        loc_y = _db_interface->get_core().get_y_min() + 1;
      }
    }
    setLocation(buffer, loc_x, loc_y);

    _timing_engine->insertBuffer(buffer->get_name());
  }

  _parasitics_estimator->parasiticsInvalid(drvr->get_net());
}

void HoldOptimizer::insertBufferDelay(StaVertex *drvr_vertex, int insert_number,
                                      DesignObjSeq &load_pins,
                                      LibertyCell  *insert_buffer_cell) {
  auto *drvr_obj = drvr_vertex->get_design_obj();
  Net  *drvr_net = drvr_obj->get_net();

  TimingIDBAdapter *idb_adapter = dynamic_cast<TimingIDBAdapter *>(_db_adapter);

  // make net
  Net *in_net, *out_net;

  in_net = drvr_net;

  std::string net_name = ("hold_net_" + to_string(_make_net_index));
  _make_net_index++;

  out_net = idb_adapter->makeNet(net_name.c_str(), nullptr);
  // Copy signal type to new net.
  idb::IdbNet *out_net_db = idb_adapter->staToDb(out_net);
  idb::IdbNet *in_net_db = idb_adapter->staToDb(in_net);
  out_net_db->set_connect_type(in_net_db->get_connect_type());

  for (auto *pin_port : load_pins) {
    if (pin_port->isPin()) {
      Pin      *load_pin = dynamic_cast<Pin *>(pin_port);
      Instance *load_inst = load_pin->get_own_instance();
      // idb_adapter->disconnectPin(load_pin);
      idb_adapter->disconnectPinPort(pin_port);
      auto debug = idb_adapter->connect(load_inst, load_pin->get_name(), out_net);
      LOG_ERROR_IF(!debug);
    } else if (pin_port->isPort()) {
      Port *load_port = dynamic_cast<Port *>(pin_port);
      idb_adapter->disconnectPinPort(pin_port);
      auto debug = idb_adapter->connect(load_port, load_port->get_name(), out_net);
      LOG_ERROR_IF(!debug);
    }
  }

  _parasitics_estimator->parasiticsInvalid(in_net);
  // Spread buffers between driver and load center.
  IdbCoordinate<int32_t> *loc = idb_adapter->idbLocation(drvr_obj);
  Point                   drvr_pin_loc = Point(loc->get_x(), loc->get_y());

  // calculate center of "load_pins"
  long long int sum_x = 0;
  long long int sum_y = 0;
  for (auto *pin_port : load_pins) {
    IdbCoordinate<int32_t> *idb_loc = idb_adapter->idbLocation(pin_port);
    Point                   loc = Point(idb_loc->get_x(), idb_loc->get_y());
    sum_x += loc.get_x();
    sum_y += loc.get_y();
  }
  int center_x = sum_x / load_pins.size();
  int center_y = sum_y / load_pins.size();

  int dx = (drvr_pin_loc.get_x() - center_x) / (insert_number + 1);
  int dy = (drvr_pin_loc.get_y() - center_y) / (insert_number + 1);

  Net         *buf_in_net = in_net;
  Instance    *buffer = nullptr;
  LibertyPort *input, *output;
  insert_buffer_cell->bufferPorts(input, output);

  std::vector<const char *> insert_inst_name;
  // drvr_pin->in_net->hold_buffer1->net2->hold_buffer2->out_net...->load_pins
  for (int i = 0; i < insert_number; i++) {
    Net *buf_out_net;
    if (i == insert_number - 1) {
      buf_out_net = out_net;
    } else {
      std::string net_name = ("hold_net_" + to_string(_make_net_index));
      _make_net_index++;

      buf_out_net = idb_adapter->makeNet(net_name.c_str(), nullptr);
    }

    // Copy signal type to new net.
    idb::IdbNet *buf_out_net_db = idb_adapter->staToDb(buf_out_net);
    buf_out_net_db->set_connect_type(in_net_db->get_connect_type());

    std::string buffer_name = ("hold_buf_" + to_string(_insert_instance_index));
    _insert_instance_index++;
    buffer = idb_adapter->makeInstance(insert_buffer_cell, buffer_name.c_str());

    _inserted_buffer_count++;

    auto debug_buf_in = idb_adapter->connect(buffer, input->get_port_name(), buf_in_net);
    auto debug_buf_out =
        idb_adapter->connect(buffer, output->get_port_name(), buf_out_net);
    LOG_ERROR_IF(!debug_buf_in);
    LOG_ERROR_IF(!debug_buf_out);

    int loc_x = drvr_pin_loc.get_x() - dx * i;
    int loc_y = drvr_pin_loc.get_y() - dy * i;
    if (!_db_interface->get_core().overlaps(loc_x, loc_y)) {
      if (loc_x > _db_interface->get_core().get_x_max()) {
        loc_x = _db_interface->get_core().get_x_max() - 1;
      } else if (loc_x < _db_interface->get_core().get_x_min()) {
        loc_x = _db_interface->get_core().get_x_min() + 1;
      }

      if (loc_y > _db_interface->get_core().get_y_max()) {
        loc_y = _db_interface->get_core().get_y_max() - 1;
      } else if (loc_y < _db_interface->get_core().get_y_min()) {
        loc_y = _db_interface->get_core().get_y_min() + 1;
      }
    }
    setLocation(buffer, loc_x, loc_y);

    // update in net
    buf_in_net = buf_out_net;

    _parasitics_estimator->parasiticsInvalid(buf_out_net);

    _timing_engine->insertBuffer(buffer->get_name());
    insert_inst_name.push_back(buffer->get_name());

    // increase design area
    idb::IdbCellMaster *idb_master = idb_adapter->staToDb(insert_buffer_cell);
    Master             *master = new Master(idb_master);

    float area = DesignCalculator::calcMasterArea(master, _dbu);
    _violation_fixer->increDesignArea(area);
  }
}

LibertyCell *HoldOptimizer::findBufferWithMaxDelay() {
  LibertyCell *max_delay_buf = nullptr;
  float        max_delay = 0.0;

  for (LibertyCell *buffer : _buffer_cells) {
    if (strstr(buffer->get_cell_name(), "CLK") != NULL) {
      continue;
    }
    float buffer_delay = getBufferHoldDelay(buffer);
    if (max_delay_buf == nullptr || buffer_delay > max_delay) {
      max_delay_buf = buffer;
      max_delay = buffer_delay;
    }
  }
  return max_delay_buf;
}

float HoldOptimizer::getBufferHoldDelay(LibertyCell *buffer) {
  Delay delays[2] = {kInf, kInf};

  LibertyPort *input_port, *output_port;
  buffer->bufferPorts(input_port, output_port);

  float load_cap = input_port->get_port_cap();
  Delay gate_delays[2];
  Slew  slews[2];
  _violation_fixer->calcGateRiseFallDelays(output_port, load_cap, gate_delays, slews);
  for (int rf_index = 0; rf_index < 2; rf_index++) {
    delays[rf_index] = min(delays[rf_index], gate_delays[rf_index]);
  }

  return min(delays[_rise], delays[_fall]);
}

void HoldOptimizer::findEndpointsWithHoldViolation(VertexSet end_points,
                                                   // return values
                                                   Slack     &worst_slack,
                                                   VertexSet &hold_violations) {
  worst_slack = kInf;
  hold_violations.clear();

  for (auto *end : end_points) {
    Slack slack = getWorstSlack(end, AnalysisMode::kMin);
    worst_slack = min(worst_slack, slack);
    if (fuzzyLess(slack, _slack_margin)) {
      hold_violations.insert(end);
    }
  }
}

/**
 * @brief calc vertex max/min, rise/fall slack
 *
 * @param vertex
 * @param slacks
 */
void HoldOptimizer::vertexSlacks(StaVertex *vertex,
                                 // return value
                                 Slacks slacks) {
  vector<AnalysisMode> analy_mode = {AnalysisMode::kMax, AnalysisMode::kMin};
  vector<TransType>    rise_fall = {TransType::kRise, TransType::kFall};

  for (auto mode : analy_mode) {
    for (auto rf : rise_fall) {
      auto  pin_slack = vertex->getSlackNs(mode, rf);
      Slack slack = pin_slack ? *pin_slack : kInf;
      int   mode_idx = (int)mode - 1;
      int   rf_idx = (int)rf - 1;
      slacks[rf_idx][mode_idx] = slack;
    }
  }
}

Slack HoldOptimizer::calcSlackGap(StaVertex *vertex) {
  Slacks slacks;
  vertexSlacks(vertex, slacks);

  return min(slacks[_rise][_mode_max] - slacks[_rise][_mode_min],
             slacks[_fall][_mode_max] - slacks[_fall][_mode_min]);
}

void HoldOptimizer::setLocation(Instance *inst, int x, int y) {
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

VertexSet HoldOptimizer::getEndPoints() {
  VertexSet  end_points;
  auto      *ista = _timing_engine->get_ista();
  StaGraph  *the_graph = &(ista->get_graph());
  StaVertex *vertex;
  FOREACH_END_VERTEX(the_graph, vertex) { end_points.insert(vertex); }
  return end_points;
}

VertexSet HoldOptimizer::getFanins(VertexSet end_points) {
  VertexSet fanins;
  fanins.clear();
  for (auto *end_point : end_points) {
    auto net = end_point->get_design_obj()->get_net();
    auto drvr = net->getDriver();
    auto vertex = _timing_engine->findVertex(drvr->getFullName().c_str());
    fanins.insert(vertex);
  }
  return fanins;
};

VertexSeq HoldOptimizer::sortFanins(VertexSet fanins) {
  // sort fanins
  VertexSeq sorted_fanins;
  for (auto *vertex : fanins) {
    sorted_fanins.push_back(vertex);
  }
  sort(sorted_fanins.begin(), sorted_fanins.end(),
       [=, this](StaVertex *v1, StaVertex *v2) {
         auto  v1_slack = v1->getSlack(AnalysisMode::kMin, TransType::kRise);
         auto  v2_slack = v2->getSlack(AnalysisMode::kMin, TransType::kRise);
         Slack s1 = v1_slack ? *v1_slack : kInf;
         Slack s2 = v2_slack ? *v2_slack : kInf;
         if (fuzzyEqual(s1, s2)) {
           float gap1 = calcSlackGap(v1);
           float gap2 = calcSlackGap(v2);
           // Break ties based on the hold/setup gap.
           if (fuzzyEqual(gap1, gap2))
             return v1->get_level() > v2->get_level();
           else
             return gap1 > gap2;
         } else
           return s1 < s2;
       });
  return sorted_fanins;
}

Slack HoldOptimizer::getWorstSlack(AnalysisMode mode) {
  StaSeqPathData *worst_path_rise =
      _timing_engine->vertexWorstRequiredPath(mode, TransType::kRise);
  StaSeqPathData *worst_path_fall =
      _timing_engine->vertexWorstRequiredPath(mode, TransType::kFall);
  Slack worst_slack_rise = worst_path_rise->getSlackNs();
  Slack worst_slack_fall = worst_path_fall->getSlackNs();
  Slack slack = min(worst_slack_rise, worst_slack_fall);

  StaSeqPathData *worst_path =
      worst_slack_rise > worst_slack_fall ? worst_path_fall : worst_path_rise;
  string capture_name = worst_path->get_capture_clock_data()->get_own_vertex()->getName();
  string launch_name = worst_path->get_launch_clock_data()->get_own_vertex()->getName();
  if (mode == AnalysisMode::kMin) {
    _db_interface->report()->report("\nWorst Hold Path Launch : " + launch_name);
    _db_interface->report()->report("Worst Hold Path Capture: " + capture_name);
  } else {
    _db_interface->report()->report("\nWorst Setup Path Launch : " + launch_name);
    _db_interface->report()->report("Worst Setup Path Capture: " + capture_name);
  }

  return slack;
}

Slack HoldOptimizer::getWorstSlack(StaVertex *vertex, AnalysisMode mode) {
  auto  rise_slack = vertex->getSlackNs(mode, TransType::kRise);
  Slack rise = rise_slack ? *rise_slack : kInf;
  auto  fall_slack = vertex->getSlackNs(mode, TransType::kFall);
  Slack fall = fall_slack ? *fall_slack : kInf;
  Slack slack = min(rise, fall);
  return slack;
}

void HoldOptimizer::insertHoldDelay(string insert_buf_name, string pin_name,
                                    int insert_number) {
  ista::Sta   *ista = _timing_engine->get_ista();
  LibertyCell *insert_buffer_cell = ista->findLibertyCell(insert_buf_name.c_str());
  LibertyPort *input, *output;
  insert_buffer_cell->bufferPorts(input, output);

  // iSta ->findPin
  StaVertex    *vertex = ista->findVertex(pin_name.c_str());
  DesignObject *load_obj = vertex->get_design_obj();
  Net          *net = load_obj->get_net();

  DesignObject *drvr_obj = net->getDriver();

  Net              *drvr_net = net;
  TimingIDBAdapter *idb_adapter = dynamic_cast<TimingIDBAdapter *>(_db_adapter);

  // make net
  Net *in_net, *out_net;
  in_net = drvr_net;
  std::string net_name =
      ("hold_net_byhand_" + to_string(_db_interface->make_net_index()));
  _db_interface->make_net_index()++;
  out_net = idb_adapter->makeNet(net_name.c_str(), nullptr);

  // Copy signal type to new net.
  idb::IdbNet *out_net_db = idb_adapter->staToDb(out_net);
  idb::IdbNet *in_net_db = idb_adapter->staToDb(in_net);
  out_net_db->set_connect_type(in_net_db->get_connect_type());

  /**
   * @brief move load pin to outnet
   */
  if (load_obj->isPin()) {
    Pin      *load_pin = dynamic_cast<Pin *>(load_obj);
    Instance *load_inst = load_pin->get_own_instance();
    idb_adapter->disconnectPinPort(load_pin);
    auto debug = idb_adapter->connect(load_inst, load_pin->get_name(), out_net);
    LOG_ERROR_IF(!debug);
  } else if (load_obj->isPort()) {
    Port *load_port = dynamic_cast<Port *>(load_obj);
    idb_adapter->disconnectPinPort(load_port);
    auto debug = idb_adapter->connect(load_port, load_port->get_name(), out_net);
    LOG_ERROR_IF(!debug);
  }

  // get drvr location
  IdbCoordinate<int32_t> *drvr_loc = idb_adapter->idbLocation(drvr_obj);
  Point                   drvr_pin_loc = Point(drvr_loc->get_x(), drvr_loc->get_y());
  // get load location
  IdbCoordinate<int32_t> *load_loc = idb_adapter->idbLocation(load_obj);
  Point                   load_pin_loc = Point(load_loc->get_x(), load_loc->get_y());

  int dx = (drvr_pin_loc.get_x() - load_pin_loc.get_x()) / (insert_number + 1);
  int dy = (drvr_pin_loc.get_y() - load_pin_loc.get_y()) / (insert_number + 1);

  Net *buf_in_net = in_net;
  // drvr_pin->in_net->hold_buffer1->net2->hold_buffer2->out_net...->load_pins
  for (int i = 0; i < insert_number; i++) {
    Net *buf_out_net = nullptr;
    if (i == insert_number - 1) {
      buf_out_net = out_net;
    } else {
      std::string net_name =
          ("hold_net_byhand_" + to_string(_db_interface->make_net_index()));
      _db_interface->make_net_index()++;

      buf_out_net = idb_adapter->makeNet(net_name.c_str(), nullptr);
    }

    // Copy signal type to new net.
    idb::IdbNet *buf_out_net_db = idb_adapter->staToDb(buf_out_net);
    buf_out_net_db->set_connect_type(in_net_db->get_connect_type());

    // drvr_pin->drvr_net->hold_buffer->net2->load_pins
    std::string buffer_name =
        ("hold_buf_byhand_" + to_string(_db_interface->make_instance_index()));
    _db_interface->make_instance_index()++;
    Instance *buffer = idb_adapter->makeInstance(insert_buffer_cell, buffer_name.c_str());

    _inserted_buffer_count++;

    auto debug_buf_in = idb_adapter->connect(buffer, input->get_port_name(), buf_in_net);
    auto debug_buf_out =
        idb_adapter->connect(buffer, output->get_port_name(), buf_out_net);
    LOG_ERROR_IF(!debug_buf_in);
    LOG_ERROR_IF(!debug_buf_out);

    int loc_x = drvr_pin_loc.get_x() - dx * i;
    int loc_y = drvr_pin_loc.get_y() - dy * i;
    setLocation(buffer, loc_x, loc_y);

    buf_in_net = buf_out_net;
    _timing_engine->insertBuffer(buffer->get_name());
  }

  IdbBuilder *idb_builder = idb_adapter->get_idb();
  string      defWritePath = string(_db_interface->get_output_def_file());
  idb_builder->saveDef(defWritePath);
}

void HoldOptimizer::reportWNSAndTNS() {
  _db_interface->report()->get_ofstream()
      << "\n---------------------------------------------------------------------------\n"
      << setiosflags(ios::left) << setw(35) << "Clock Group" << resetiosflags(ios::left)
      << setiosflags(ios::right) << setw(20) << "Hold TNS" << setw(20) << "Hold WNS"
      << resetiosflags(ios::right) << endl
      << "---------------------------------------------------------------------------"
      << endl;
  // _db_interface->report()->get_ofstream().close();
  auto clk_list = _timing_engine->getClockList();
  for (auto clk : clk_list) {
    auto clk_name = clk->get_clock_name();
    auto tns1 = _timing_engine->reportTNS(clk_name, AnalysisMode::kMin);
    auto wns1 = _timing_engine->reportWNS(clk_name, AnalysisMode::kMin);
    _db_interface->report()->get_ofstream()
        << setiosflags(ios::left) << setw(35) << clk_name << resetiosflags(ios::left)
        << setiosflags(ios::right) << setw(20) << tns1 << setw(20) << wns1
        << resetiosflags(ios::right) << endl;
  }
  _db_interface->report()->get_ofstream()
      << "---------------------------------------------------------------------------"
      << endl;
  _db_interface->report()->get_ofstream().close();
}

} // namespace ito

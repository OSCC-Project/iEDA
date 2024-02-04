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
#include "SetupOptimizer.h"

#include "api/TimingIDBAdapter.hh"
#include "api/TimingEngine.hh"
#include "liberty/Liberty.hh"

using namespace std;

namespace ito {

int SetupOptimizer::_rise = (int)TransType::kRise - 1;
int SetupOptimizer::_fall = (int)TransType::kFall - 1;

SetupOptimizer::SetupOptimizer(DbInterface *dbinterface)
    : _db_interface(dbinterface), _resize_instance_count(0), _inserted_buffer_count(0),
      _insert_instance_index(1), _make_net_index(1) {
  _timing_engine = _db_interface->get_timing_engine();
  _dbu = _db_interface->get_dbu();
  _db_adapter = _timing_engine->get_db_adapter();
  _parasitics_estimator = new EstimateParasitics(_db_interface);
  _violation_fixer = new ViolationOptimizer(_db_interface);
}

void SetupOptimizer::initBufferCell() {
  bool not_specified_buffer = _db_interface->get_setup_insert_buffers().empty();
  if (not_specified_buffer) {
    LibertyCellSeq buf_cells = _db_interface->get_buffer_cells();
    for (auto buf : buf_cells) {
      if (strstr(buf->get_cell_name(), "BUF") != NULL &&
          strstr(buf->get_cell_name(), "CLK") == NULL) {
        _buf_cells.emplace_back(buf);
      }
    }
    return;
  }

  auto bufs = _db_interface->get_setup_insert_buffers();
  for (auto buf : bufs) {
    auto buffer = _timing_engine->findLibertyCell(buf.c_str());
    _buf_cells.emplace_back(buffer);
  }
}

void SetupOptimizer::optimizeSetup() {
  // to store slack each time
  vector<Slack> slack_store;

  _parasitics_estimator->estimateAllNetParasitics();
  _timing_engine->updateTiming();
  _timing_engine->reportTiming();
  _db_interface->set_eval_data();

  initBufferCell();
  LOG_ERROR_IF(_buf_cells.empty()) << "Can not found specified buffers.\n";

  Slack prev_worst_slack = -kInf;
  int   pass = 1;
  int   decreasing_slack_passes = 0;

  float slack_margin = _db_interface->get_setup_slack_margin();
  int   _number_passes_allowed_decreasing_slack =
      _db_interface->get_number_passes_allowed_decreasing_slack();

  StaSeqPathData *worst_path = worstRequiredPath();
  Slack worst_slack = worst_path->getSlackNs();
  slack_store.push_back(worst_slack);

  auto end_points = getEndPoints();
  // endpoints with setup violation.
  VertexSeq end_pts_setup_violation;
  findEndpointsWithSetupViolation(end_points, slack_margin, end_pts_setup_violation);
  // 根据slack进行升序排序
  sort(end_pts_setup_violation.begin(), end_pts_setup_violation.end(),
       [](StaVertex *end1, StaVertex *end2) {
        return end1->getWorstSlackNs(AnalysisMode::kMax) < end2->getWorstSlackNs(AnalysisMode::kMax);
       });

  _db_interface->report()->get_ofstream()
      << "Find " << (int)end_pts_setup_violation.size() << " endpoints with setup violation.\n";
  _db_interface->report()->get_ofstream().close();

  // slack violation
  for (auto node : end_pts_setup_violation) {

    prev_worst_slack = -kInf;
    while (worst_slack < slack_margin) {
      optimizeSetup(node, worst_slack);

      // _parasitics_estimator->excuteParasiticsEstimate();
      // _timing_engine->updateTiming();

      auto nets_for_update = _parasitics_estimator->get_parasitics_invalid_net();
      for (auto net_up : nets_for_update) {
        auto net_pins = net_up->get_pin_ports();
        for (auto pin_port : net_pins) {
          if (pin_port->isPort()) {
            continue;
          }
          auto inst_name = pin_port->get_own_instance()->getFullName();
          _timing_engine->moveInstance(inst_name.c_str(), 20);
        }
      }
      _parasitics_estimator->excuteParasiticsEstimate();
      _timing_engine->incrUpdateTiming();

      StaSeqPathData *worst_path_rise = _timing_engine->vertexWorstRequiredPath(
          node, AnalysisMode::kMax, TransType::kRise);
      StaSeqPathData *worst_path_fall = _timing_engine->vertexWorstRequiredPath(
          node, AnalysisMode::kMax, TransType::kFall);
      if (!worst_path_fall || !worst_path_rise) {
        break;
      }
      Slack           worst_slack_rise = worst_path_rise->getSlackNs();
      Slack           worst_slack_fall = worst_path_fall->getSlackNs();
      StaSeqPathData *worst_path =
          worst_slack_rise > worst_slack_fall ? worst_path_fall : worst_path_rise;
      worst_slack = worst_path->getSlackNs();
      slack_store.push_back(worst_slack);

      if (fuzzyLessEqual(worst_slack, prev_worst_slack)) {
        // excessive slack increase is prohibited
        float diff = prev_worst_slack - worst_slack;
        if (diff > 0.02 * abs(prev_worst_slack)) {
          break;
        }

        // Allow slack to increase a few passes to get out of local minima.
        // Do not update prev_worst_slack so it saves the high water mark.
        decreasing_slack_passes++;
        if (decreasing_slack_passes > _number_passes_allowed_decreasing_slack) {
          break;
        }
      } else {
        prev_worst_slack = worst_slack;
        decreasing_slack_passes = 0;
      }
    }
    // if (_db_interface->overMaxArea()) {
    //   break;
    // }
    pass++;
  }
  _db_interface->report()->reportSetupResult(slack_store);

  _parasitics_estimator->estimateAllNetParasitics();
  _timing_engine->updateTiming();
  _timing_engine->reportTiming();

  printf("Inserted {%d} buffers.\n", _inserted_buffer_count);
  printf("Resized {%d} instances.\n", _resize_instance_count);
  _db_interface->report()->get_ofstream()
      << "Inserted " << _inserted_buffer_count << " buffers."
      << "\nResized " << _resize_instance_count << " instances.\n";
  _db_interface->report()->get_ofstream().close();

  if (worst_slack < slack_margin) {
    printf("Unable to repair all setup violations.\n");
    _db_interface->report()->get_ofstream() << "Unable to repair all setup violations.\n";
    _db_interface->report()->get_ofstream().close();
  }
  if (_db_interface->overMaxArea()) {
    printf("max utilization reached.\n");
  }

  _db_interface->report()->reportSetupResult(slack_store);
  _db_interface->report()->reportTime(false);
}

void SetupOptimizer::optimizeSetup(StaSeqPathData *worst_path, Slack path_slack) {
  vector<TimingEngine::PathNet> path_driver_vertexs =
      _timing_engine->getPathDriverVertexs(worst_path);
  int path_length = path_driver_vertexs.size();

  vector<TimingEngine::PathNet> sorted_path_driver_vertexs;
  for (int i = 0; i < path_length; i++) {
    auto path = path_driver_vertexs[i];
    sorted_path_driver_vertexs.push_back(path);
  }
  sort(sorted_path_driver_vertexs.begin(), sorted_path_driver_vertexs.end(),
       [](TimingEngine::PathNet n1, TimingEngine::PathNet n2) {
         return n1.delay > n2.delay;
       });

  if (path_length > 1) {
    for (int i = 0; i < (int)path_length; i++) {
      auto       path = sorted_path_driver_vertexs[i];
      StaVertex *drvr_vertex = path.driver;
      auto      *obj = drvr_vertex->get_design_obj();
      Pin       *drvr_pin = dynamic_cast<Pin *>(obj);

      int fanout = getFanoutNumber(drvr_pin);
      int _rebuffer_max_fanout = _db_interface->get_rebuffer_max_fanout();

      LibertyPort *drvr_port = drvr_pin->get_cell_port();
      float load_cap = drvr_pin->get_net()->getLoad(AnalysisMode::kMax, TransType::kRise);

      vector<TimingEngine::PathNet>::iterator itr =
          find(path_driver_vertexs.begin(), path_driver_vertexs.end(), path);
      int drvr_idx = distance(path_driver_vertexs.begin(), itr);
      if (drvr_idx >= 1) {
        auto         in_path = path_driver_vertexs[drvr_idx - 1];
        StaVertex   *in_vertex = in_path.load;
        auto        *in_obj = in_vertex->get_design_obj();
        Pin         *in_pin = dynamic_cast<Pin *>(in_obj);
        LibertyPort *in_port = in_pin->get_cell_port();

        float        prev_drive;
        auto        *prev_drvr_vertex = in_path.driver;
        auto        *prev_drvr_obj = prev_drvr_vertex->get_design_obj();
        Pin         *prev_drvr_pin = dynamic_cast<Pin *>(prev_drvr_obj);
        LibertyPort *prev_drvr_port = prev_drvr_pin->get_cell_port();
        prev_drive = prev_drvr_port->driveResistance();

        LibertyCell *upsize = upsizeCell(in_port, drvr_port, load_cap, prev_drive);
        if (upsize) {
          Instance *drvr_inst = drvr_pin->get_own_instance();
          if (_violation_fixer->repowerInstance(drvr_inst, upsize)) {
            _resize_instance_count++;
            _parasitics_estimator->estimateNetParasitics(drvr_pin->get_net());
          }
          break;
        }
      }

      if (fanout > 1
          // Rebuffer blows up on large fanout nets.
          && fanout < _rebuffer_max_fanout) {
        int count_before = _inserted_buffer_count;
        buffering(drvr_pin); // _inserted_buffer_count++
        int insert_count = _inserted_buffer_count - count_before;

        if (insert_count > 0) {
          break;
        }
      }

      // Don't split loads on low fanout nets.
      int split_load_min_fanout = _db_interface->get_split_load_min_fanout();
      if (fanout > split_load_min_fanout) {
        insertBufferSeparateLoads(drvr_vertex, path_slack);
        break;
      }
    }
  }
}

void SetupOptimizer::optimizeSetup(StaVertex *vertex, Slack path_slack) {
  StaSeqPathData *worst_path_rise = _timing_engine->vertexWorstRequiredPath(
      vertex, AnalysisMode::kMax, TransType::kRise);
  StaSeqPathData *worst_path_fall = _timing_engine->vertexWorstRequiredPath(
      vertex, AnalysisMode::kMax, TransType::kFall);
  Slack           worst_slack_rise = worst_path_rise->getSlackNs();
  Slack           worst_slack_fall = worst_path_fall->getSlackNs();
  StaSeqPathData *worst_path =
      worst_slack_rise > worst_slack_fall ? worst_path_fall : worst_path_rise;

  vector<TimingEngine::PathNet> path_driver_vertexs =
      _timing_engine->getPathDriverVertexs(worst_path);
  int path_length = path_driver_vertexs.size();

  vector<TimingEngine::PathNet> sorted_path_driver_vertexs;
  for (int i = 0; i < path_length; i++) {
    auto path = path_driver_vertexs[i];
    sorted_path_driver_vertexs.push_back(path);
  }
  sort(sorted_path_driver_vertexs.begin(), sorted_path_driver_vertexs.end(),
       [](TimingEngine::PathNet n1, TimingEngine::PathNet n2) {
         return n1.delay > n2.delay;
       });

  if (path_length > 1) {
    for (int i = 0; i < (int)path_length; i++) {
      auto       path = sorted_path_driver_vertexs[i];
      StaVertex *drvr_vertex = path.driver;
      auto      *obj = drvr_vertex->get_design_obj();
      Pin       *drvr_pin = dynamic_cast<Pin *>(obj);

      int fanout = getFanoutNumber(drvr_pin);
      int _rebuffer_max_fanout = _db_interface->get_rebuffer_max_fanout();

      LibertyPort *drvr_port = drvr_pin->get_cell_port();
      float load_cap = drvr_pin->get_net()->getLoad(AnalysisMode::kMax, TransType::kRise);

      vector<TimingEngine::PathNet>::iterator itr =
          find(path_driver_vertexs.begin(), path_driver_vertexs.end(), path);
      int drvr_idx = distance(path_driver_vertexs.begin(), itr);
      if (drvr_idx >= 1) {
        auto         in_path = path_driver_vertexs[drvr_idx - 1];
        StaVertex   *in_vertex = in_path.load;
        auto        *in_obj = in_vertex->get_design_obj();
        Pin         *in_pin = dynamic_cast<Pin *>(in_obj);
        LibertyPort *in_port = in_pin->get_cell_port();

        float        prev_drive;
        auto        *prev_drvr_vertex = in_path.driver;
        auto        *prev_drvr_obj = prev_drvr_vertex->get_design_obj();
        Pin         *prev_drvr_pin = dynamic_cast<Pin *>(prev_drvr_obj);
        LibertyPort *prev_drvr_port = prev_drvr_pin->get_cell_port();
        prev_drive = prev_drvr_port->driveResistance();

        LibertyCell *upsize = upsizeCell(in_port, drvr_port, load_cap, prev_drive);
        if (upsize) {
          Instance *drvr_inst = drvr_pin->get_own_instance();
          if (_violation_fixer->repowerInstance(drvr_inst, upsize)) {
            _resize_instance_count++;
            _parasitics_estimator->parasiticsInvalid(drvr_pin->get_net());
          }
          break;
        }
      }

      if (fanout > 1
          // Rebuffer blows up on large fanout nets.
          && fanout < _rebuffer_max_fanout) {
        int count_before = _inserted_buffer_count;
        buffering(drvr_pin); // _inserted_buffer_count++
        int insert_count = _inserted_buffer_count - count_before;

        if (insert_count > 0) {
          break;
        }
      }

      // Don't split loads on low fanout nets.
      int split_load_min_fanout = _db_interface->get_split_load_min_fanout();
      if (fanout > split_load_min_fanout) {
        insertBufferSeparateLoads(drvr_vertex, path_slack);
        break;
      }
    }
  }
}

void SetupOptimizer::buffering(Pin *pin) {
  Net         *net = pin->get_net();
  LibertyPort *drvr_port = pin->get_cell_port();

  if (netConnectToPort(net)) {
    return;
  }

  if (drvr_port && net) {
    RoutingTree *tree = makeRoutingTree(net, _db_adapter, RoutingType::kSteiner);
    if (tree) {
      int drvr_id = tree->get_root()->get_id();
      tree->updateBranch();
      BufferedOptionSeq buf_opts =
          bottomUpBuffering(tree, tree->left(drvr_id), drvr_id, 1);

      Required        best_slack = -kInf;
      BufferedOption *best_option = nullptr;
      for (BufferedOption *opt : buf_opts) {
        Slack slack = opt->get_required_arrival_time();
        if (slack > best_slack) {
          best_slack = slack;
          best_option = opt;
        }
      }
      if (best_option) {
        // for DEBUG
        best_option->printBuffered(0);
        topDownImplementBuffering(best_option, net, 1);
      }
    }
    delete tree;
  }
}

BufferedOptionSeq SetupOptimizer::bottomUpBuffering(RoutingTree *tree, int curr_id,
                                                    int prev_id, int level) {
  if (curr_id != RoutingTree::_null_pt) {
    auto *obj_pin = tree->get_pin(curr_id);
    Point curr_loc = tree->get_location(curr_id);
    Point prev_loc = tree->get_location(prev_id);

    if (obj_pin) {
      auto *pin = dynamic_cast<Pin *>(obj_pin);

      if (_timing_engine->isLoad(pin->getFullName().c_str())) {
        StaVertex      *vertex = _timing_engine->findVertex(pin->getFullName().c_str());
        auto   req_ns_r = vertex->getReqTimeNs(AnalysisMode::kMax, TransType::kRise);
        double req_r = req_ns_r ? *req_ns_r : 0.0;
        auto   req_ns_f = vertex->getReqTimeNs(AnalysisMode::kMax, TransType::kFall);
        double req_f = req_ns_f ? *req_ns_f : 0.0;
        double req = min(req_r, req_f);

        BufferedOption *buffered_option = new BufferedOption(
            BufferedOptionType::kLoad,
            curr_loc,   // load pin loc
            pin->cap(), // load pin cap
            pin,        // load pin
            0.0, nullptr, nullptr, nullptr, req);
        BufferedOptionSeq buf_option_seq;
        buf_option_seq.emplace_back(buffered_option);

        return addWireAndBuffer(buf_option_seq, curr_loc, prev_loc, level);
      }
    }
    // curr -> steiner point
    else if (obj_pin == nullptr) {
      BufferedOptionSeq buf_opt_left =
          bottomUpBuffering(tree, tree->left(curr_id), curr_id, level + 1);
      BufferedOptionSeq buf_opt_mid =
          bottomUpBuffering(tree, tree->middle(curr_id), curr_id, level + 1);

      BufferedOptionSeq buf_opt_merger = mergeBranch(buf_opt_left, buf_opt_mid, curr_loc);

      return addWireAndBuffer(buf_opt_merger, curr_loc, prev_loc, level);
    }
  }
  return BufferedOptionSeq();
}

BufferedOptionSeq SetupOptimizer::mergeBranch(BufferedOptionSeq buf_opt_left,
                                              BufferedOptionSeq buf_opt_right,
                                              Point             curr_loc) {
  BufferedOptionSeq buf_opt_merger;
  for (auto left : buf_opt_left) {
    for (auto right : buf_opt_right) {
      Required left_req = left->get_required_arrival_time();
      float    left_cap = left->get_cap();
      Required right_req = right->get_required_arrival_time();
      float    right_cap = right->get_cap();

      BufferedOption *min_opt = fuzzyLess(left_req, right_req) ? left : right;

      BufferedOption *buffered_option =
          new BufferedOption(BufferedOptionType::kJunction,
                             curr_loc,             // load pin loc
                             left_cap + right_cap, // load pin cap
                             nullptr,              // load pin
                             min_opt->get_required_delay(),
                             nullptr, left, right, min_opt->get_req());
      buf_opt_merger.emplace_back(buffered_option);
    }
  }
  sort(buf_opt_merger.begin(), buf_opt_merger.end(),
       [=](BufferedOption *opt1, BufferedOption *opt2) {
         return fuzzyGreater(opt1->get_required_arrival_time(),
                             opt2->get_required_arrival_time());
       });
  int si = 0;
  for (size_t pi = 0; pi < buf_opt_merger.size(); pi++) {
    BufferedOption *p = buf_opt_merger[pi];
    float           Lp = p->get_cap();
    // Remove options by shifting down with index si.
    si = pi + 1;
    // Because the options are sorted we don't have to look
    // beyond the first option.
    for (size_t qi = pi + 1; qi < buf_opt_merger.size(); qi++) {
      BufferedOption *q = buf_opt_merger[qi];
      float           Lq = q->get_cap();
      if (fuzzyLess(Lq, Lp)) {
        buf_opt_merger[si++] = q;
      }
    }
    buf_opt_merger.resize(si);
  }
  return buf_opt_merger;
}

BufferedOptionSeq SetupOptimizer::addWireAndBuffer(BufferedOptionSeq buf_opt_seq,
                                                   Point curr_loc, Point prev_loc,
                                                   int level) {
  int wire_length_dbu =
      abs(curr_loc.get_x() - prev_loc.get_x()) + abs(curr_loc.get_y() - prev_loc.get_y());

  BufferedOptionSeq buf_option_seq;
  for (BufferedOption *buf_opt : buf_opt_seq) {
    std::optional<double> width = std::nullopt;
    double                wire_cap = dynamic_cast<TimingIDBAdapter *>(_db_adapter)
                          ->getCapacitance(1, (double)wire_length_dbu / _dbu, width);
    double wire_res = dynamic_cast<TimingIDBAdapter *>(_db_adapter)
                          ->getResistance(1, (double)wire_length_dbu / _dbu, width);
    // double wire_delay = wire_res * wire_cap;
    double wire_delay = wire_res * (wire_cap / 2 + buf_opt->get_cap());

    float update_cap = buf_opt->get_cap() + wire_cap;
    Delay update_req_delay = buf_opt->get_required_delay() + wire_delay;

    BufferedOption *buffered_option =
        new BufferedOption(BufferedOptionType::kWire,
                           prev_loc,   // load pin loc
                           update_cap, // load pin cap
                           nullptr,    // load pin
                           update_req_delay, // plus wire delay
                           nullptr,          // no buf
                           buf_opt, nullptr, buf_opt->get_req());

    buf_option_seq.push_back(buffered_option);
  }
  // Determine if a buffer needs to be inserted after adding wire
  if (!buf_option_seq.empty()) {
    BufferedOptionSeq buf_options = addBuffer(buf_option_seq, prev_loc);
    for (BufferedOption *buf_opt : buf_options) {
      buf_option_seq.push_back(buf_opt);
    }
  }
  return buf_option_seq;
}

BufferedOptionSeq SetupOptimizer::addBuffer(BufferedOptionSeq buf_opt_seq,
                                            Point             prev_loc) {
  BufferedOptionSeq buf_option_seq;

  for (LibertyCell *buf_cell : _buf_cells) {
    Required better_req_time = -kInf;
    // "wire" option
    BufferedOption *better_buf_option = nullptr;

    // "wire" option
    for (BufferedOption *buf_opt : buf_opt_seq) {
      Required        req_time = kInf;
      // buffer delay, related to load capacitance
      Delay buffer_delay = calcBufferDelay(buf_cell, buf_opt->get_cap());

      req_time = buf_opt->get_required_arrival_time() - buffer_delay;

      // Find greater required arrival time and it's corresponding "wire" option
      if (fuzzyGreater(req_time, better_req_time)) {
      // if (req_time > better_req_time) {
        better_req_time = req_time;
        better_buf_option = buf_opt;
      }
    }
    if (better_buf_option) {
      Required        req_time = kInf;
      float           buffer_cap = 0.0;
      Delay           buffer_delay = 0.0;

      LibertyPort *intput_port, *output_port;
      buf_cell->bufferPorts(intput_port, output_port);
      buffer_cap = intput_port->get_port_cap();

      // buffer delay, related to load capacitance
      buffer_delay = calcBufferDelay(buf_cell, better_buf_option->get_cap());

      req_time = better_buf_option->get_required_arrival_time() - buffer_delay;

      // Don't add this buffer option if it has bigger input cap and smaller req
      // than another existing buffer option.
      bool prune = false;
      for (BufferedOption *buffer_option : buf_option_seq) {
        if (buffer_option->get_cap() <= buffer_cap &&
            buffer_option->get_required_arrival_time() >= req_time) {
          prune = true;
          break;
        }
      }
      if (!prune) {
        Delay update_req_delay = better_buf_option->get_required_delay() + buffer_delay;
        BufferedOption *buffered_option = new BufferedOption(
            BufferedOptionType::kBuffer,
            prev_loc,   // load pin loc
            buffer_cap, // load pin cap
            nullptr,    // load pin
            update_req_delay, // plus buf delay
            buf_cell, better_buf_option, nullptr, better_buf_option->get_req());

        buf_option_seq.push_back(buffered_option);
      }
    }
  }
  return buf_option_seq;
}

void SetupOptimizer::topDownImplementBuffering(BufferedOption *buf_opt, Net *net,
                                               int level) {
  switch (buf_opt->get_type()) {
  case BufferedOptionType::kBuffer: {
    // step 1: make instance
    std::string buffer_name = ("setup_buffer_" + to_string(_insert_instance_index));
    _insert_instance_index++;
    LibertyCell *insert_buf_cell = buf_opt->get_buffer_cell();

    // step 2: make net
    std::string net_name = ("setup_net_" + to_string(_make_net_index));
    _make_net_index++;
    LibertyPort *input, *output;
    insert_buf_cell->bufferPorts(input, output);

    // step 3: connect
    TimingIDBAdapter *idb_adapter = dynamic_cast<TimingIDBAdapter *>(_db_adapter);
    Instance *buffer = idb_adapter->makeInstance(insert_buf_cell, buffer_name.c_str());

    Net *net2 = idb_adapter->makeNet(net_name.c_str(), nullptr);

    auto debug_buf_in = idb_adapter->connect(buffer, input->get_port_name(), net);
    auto debug_buf_out = idb_adapter->connect(buffer, output->get_port_name(), net2);
    LOG_ERROR_IF(!debug_buf_in);
    LOG_ERROR_IF(!debug_buf_out);

    _timing_engine->insertBuffer(buffer->get_name());
    // _timing_engine->updateTiming();

    Point loc = buf_opt->get_location();
    setLocation(buffer, loc.get_x(), loc.get_y());

    _parasitics_estimator->parasiticsInvalid(net);
    _parasitics_estimator->parasiticsInvalid(net2);

    // _parasitics_estimator->estimateInvalidNetParasitics(net->getDriver(), net);
    // _parasitics_estimator->estimateInvalidNetParasitics(net2->getDriver(), net2);

    _inserted_buffer_count++;

    // increase design area
    idb::IdbCellMaster *idb_master = idb_adapter->staToDb(insert_buf_cell);
    Master             *master = new Master(idb_master);

    float area = DesignCalculator::calcMasterArea(master, _dbu);
    _violation_fixer->increDesignArea(area);

    topDownImplementBuffering(buf_opt->get_left(), net2, level + 1);
    break;
  }
  case BufferedOptionType::kJunction: {
    if (buf_opt->get_left()) {
      topDownImplementBuffering(buf_opt->get_left(), net, level + 1);
    }
    if (buf_opt->get_right()) {
      topDownImplementBuffering(buf_opt->get_right(), net, level + 1);
    }
    break;
  }
  case BufferedOptionType::kLoad: {
    TimingIDBAdapter *idb_adapter = dynamic_cast<TimingIDBAdapter *>(_db_adapter);
    Pin              *load_pin = buf_opt->get_load_pin();
    Net              *load_net = load_pin->get_net();
    if (load_net != net) {
      Instance    *load_inst = load_pin->get_own_instance();
      idb_adapter->disconnectPin(load_pin);
      auto debug = idb_adapter->connect(load_inst, load_pin->get_name(), net);
      LOG_ERROR_IF(!debug);

      _timing_engine->insertBuffer(load_inst->get_name());
      // _timing_engine->updateTiming();

      _parasitics_estimator->parasiticsInvalid(load_net);
      _parasitics_estimator->parasiticsInvalid(net);

      // _parasitics_estimator->estimateInvalidNetParasitics(net->getDriver(), net);
      // _parasitics_estimator->estimateInvalidNetParasitics(load_net->getDriver(),
      //                                                     load_net);
    }
    break;
  }
  case BufferedOptionType::kWire: {
    if (buf_opt->get_left()) {
      topDownImplementBuffering(buf_opt->get_left(), net, level + 1);
    }
    break;
  }
  }
}

void SetupOptimizer::insertBufferSeparateLoads(StaVertex *drvr_vertex, Slack drvr_slack) {
  // Sort fanouts of the drvr on the critical path by slack margin
  // wrt the critical path slack.
  vector<pair<StaVertex *, Slack>> fanout_slacks;
  VertexSeq fanout_vertexes = _timing_engine->getFanoutVertexs(drvr_vertex);
  for (auto fanout_vertex : fanout_vertexes) {
    auto fanout_vertex_slack =
        fanout_vertex->getSlackNs(AnalysisMode::kMax, TransType::kRise);
    Slack fanout_slack = fanout_vertex_slack ? *fanout_vertex_slack : kInf;
    Slack slack_margin = fanout_slack - drvr_slack;
    fanout_slacks.push_back(pair<StaVertex *, Slack>(fanout_vertex, slack_margin));
  }

  sort(fanout_slacks.begin(), fanout_slacks.end(),
       [](pair<StaVertex *, Slack> pair1, pair<StaVertex *, Slack> pair2) {
         return pair1.second > pair2.second;
       });

  DesignObject *drvr_pin = drvr_vertex->get_design_obj();
  Net          *net = drvr_pin->get_net();

  // make instance name
  std::string buffer_name = ("setup_split_buffer_" + to_string(_insert_instance_index));
  _insert_instance_index++;

  TimingIDBAdapter *idb_adapter = dynamic_cast<TimingIDBAdapter *>(_db_adapter);
  LibertyCell      *insert_buf_cell = _db_interface->get_lowest_drive_buffer();

  Instance    *buffer = idb_adapter->makeInstance(insert_buf_cell, buffer_name.c_str());
  LibertyPort *input, *output;
  insert_buf_cell->bufferPorts(input, output);

  // make net
  std::string net_name = ("setup_split_net_" + to_string(_make_net_index));
  _make_net_index++;
  Net *out_net = idb_adapter->makeNet(net_name.c_str(), nullptr);

  // before
  // drvr_pin -> net -> load_pins
  // after
  // drvr_pin -> net -> load_pins with low slack
  //                 -> buffer_in -> setup_split_net_ -> rest of loads
  auto debug_buf_in = idb_adapter->connect(buffer, input->get_port_name(), net);
  auto debug_buf_out = idb_adapter->connect(buffer, output->get_port_name(), out_net);
  LOG_ERROR_IF(!debug_buf_in);
  LOG_ERROR_IF(!debug_buf_out);

  int split_index = fanout_slacks.size() / 2;
  for (int i = 0; i < split_index; i++) {
    pair<StaVertex *, Slack> fanout_slack = fanout_slacks[i];
    StaVertex               *load_vertex = fanout_slack.first;
    auto                    *load_obj = load_vertex->get_design_obj();

    if (load_obj->isPin()) {
      Pin         *load_pin = dynamic_cast<Pin *>(load_obj);
      Instance    *load_inst = load_pin->get_own_instance();

      idb_adapter->disconnectPin(load_pin);
      auto debug = idb_adapter->connect(load_inst, load_pin->get_name(), out_net);
      LOG_ERROR_IF(!debug);
    }
  }

  IdbCoordinate<int32_t> *idb_loc = idb_adapter->idbLocation(drvr_pin);
  Point                   drvr_loc = Point(idb_loc->get_x(), idb_loc->get_y());

  _timing_engine->insertBuffer(buffer->get_name());
  Pin *buffer_out_pin = buffer->findPin(output);
  _violation_fixer->repowerInstance(buffer_out_pin);
  setLocation(buffer, drvr_loc.get_x(), drvr_loc.get_y());

  _parasitics_estimator->parasiticsInvalid(net);
  // _parasitics_estimator->estimateInvalidNetParasitics(net->getDriver(), net);

  _parasitics_estimator->parasiticsInvalid(out_net);
  // _parasitics_estimator->estimateInvalidNetParasitics(out_net->getDriver(), out_net);
}

float SetupOptimizer::calcBufferDelay(LibertyCell *buffer_cell, float load_cap) {
  auto delay_rise = calcBufferDelay(buffer_cell, load_cap, TransType::kRise);
  auto delay_fall = calcBufferDelay(buffer_cell, load_cap, TransType::kFall);
  return max(delay_rise, delay_fall);
}

float SetupOptimizer::calcBufferDelay(LibertyCell *buffer_cell, float load_cap,
                                      TransType rf) {
  LibertyPort *input, *output;
  buffer_cell->bufferPorts(input, output);
  Delay delays[2];
  Slew  slews[2];
  _violation_fixer->calcGateRiseFallDelays(output, load_cap, delays, slews);
  int rise_fall = (int)rf - 1;
  return delays[rise_fall];
}

float SetupOptimizer::calcGateDelay(LibertyPort *drvr_port, float load_cap,
                                    TransType rf) {
  Delay delays[2];
  Slew  slews[2];
  _violation_fixer->calcGateRiseFallDelays(drvr_port, load_cap, delays, slews);
  int rise_fall = (int)rf - 1;
  return delays[rise_fall];
}

float SetupOptimizer::calcGateDelay(LibertyPort *drvr_port, float load_cap) {
  Delay delays[2];
  Slew  slews[2];
  _violation_fixer->calcGateRiseFallDelays(drvr_port, load_cap, delays, slews);
  return max(delays[_fall], delays[_rise]);
}

int SetupOptimizer::getFanoutNumber(Pin *pin) {
  auto *net = pin->get_net();
  return net->getFanouts();
}

bool SetupOptimizer::hasTopLevelOutputPort(Net *net) {
  DesignObject *pin;
  FOREACH_NET_PIN(net, pin) {
    if (pin->isOutput()) {
      return true;
    }
  }
  return false;
}

void SetupOptimizer::setLocation(Instance *inst, int x, int y) {

  TimingIDBAdapter *idb_adapter = dynamic_cast<TimingIDBAdapter *>(_db_adapter);
  idb::IdbInstance *idb_inst = idb_adapter->staToDb(inst);
  idb_inst->set_status_placed();
  idb_inst->set_coodinate(x, y);
  unsigned       master_width = idb_inst->get_cell_master()->get_width();
  pair<int, int> loc = _db_interface->placer()->findNearestSpace(master_width, x, y);
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

LibertyCell *SetupOptimizer::upsizeCell(LibertyPort *in_port, LibertyPort *drvr_port,
                                        float load_cap, float prev_drive) {
  LibertyCell           *cell = drvr_port->get_ower_cell();
  Vector<LibertyCell *> *equiv_cells = _timing_engine->equivCells(cell);
  if (equiv_cells) {
    const char *in_port_name = in_port->get_port_name();
    const char *drvr_port_name = drvr_port->get_port_name();

    sort(equiv_cells->begin(), equiv_cells->end(),
         [=](LibertyCell *cell1, LibertyCell *cell2) {
           LibertyPort *port1 = cell1->get_cell_port_or_port_bus(drvr_port_name);
           LibertyPort *port2 = cell2->get_cell_port_or_port_bus(drvr_port_name);
           return (port1->driveResistance() > port2->driveResistance());
         });

    // float drive = drvr_port->driveResistance();
    float delay =
        calcGateDelay(drvr_port, load_cap) + prev_drive * in_port->get_port_cap();

    for (LibertyCell *equiv : *equiv_cells) {
      if (strstr(equiv->get_cell_name(), "CLK") != NULL) {
        continue;
      }
      LibertyPort *equiv_drvr = equiv->get_cell_port_or_port_bus(drvr_port_name);
      LibertyPort *equiv_input = equiv->get_cell_port_or_port_bus(in_port_name);

      // Include delay of previous driver into equiv gate.
      float equiv_delay;
      if (equiv_input) {
        equiv_delay = calcGateDelay(equiv_drvr, load_cap);// +
                      // prev_drive * equiv_input->get_port_cap();
      } else {
        equiv_delay =
            calcGateDelay(equiv_drvr, load_cap);// + prev_drive * in_port->get_port_cap();
      }

      if (equiv_delay < 0.5*delay) {
        return equiv;
      }
    }
  }
  return nullptr;
}

StaSeqPathData *SetupOptimizer::worstRequiredPath() {
  vector<TransType> rise_fall = {TransType::kRise, TransType::kFall};
  StaSeqPathData *worst_path = nullptr;
  Slack wns = kInf;
  for (auto rf : rise_fall) {
    auto path = _timing_engine->vertexWorstRequiredPath(AnalysisMode::kMax, rf);
    if (path->getSlackNs() < wns) {
      wns = path->getSlackNs();
      worst_path = path;
    }
  }
  return worst_path;
}

bool SetupOptimizer::netConnectToPort(Net *net) {
  auto load_pin_ports = net->getLoads();
  for (auto pin_port : load_pin_ports) {
    if (pin_port->isPort()) {
      return true;
    }
  }
  return false;
}

Slack SetupOptimizer::getWorstSlack(StaVertex *vertex, AnalysisMode mode) {
  auto  rise_slack = vertex->getSlackNs(mode, TransType::kRise);
  Slack rise = rise_slack ? *rise_slack : kInf;
  auto  fall_slack = vertex->getSlackNs(mode, TransType::kFall);
  Slack fall = fall_slack ? *fall_slack : kInf;
  Slack slack = min(rise, fall);
  return slack;
}

VertexSet SetupOptimizer::getEndPoints() {
  VertexSet  end_points;
  auto      *ista = _timing_engine->get_ista();
  StaGraph  *the_graph = &(ista->get_graph());
  StaVertex *vertex;
  FOREACH_END_VERTEX(the_graph, vertex) { end_points.insert(vertex); }
  return end_points;
}

void SetupOptimizer::findEndpointsWithSetupViolation(VertexSet end_points, Slack slack_margin,
                                                   // return values
                                                   VertexSeq &setup_violations) {
  setup_violations.clear();

  for (auto *end : end_points) {
    Slack slack = getWorstSlack(end, AnalysisMode::kMax);
    if (slack < slack_margin) {
      setup_violations.emplace_back(end);
    }
  }
}
} // namespace ito
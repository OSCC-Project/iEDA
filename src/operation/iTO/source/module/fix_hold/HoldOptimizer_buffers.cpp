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
#include "../../config/ToConfig.h"
#include "EstimateParasitics.h"
#include "HoldOptimizer.h"
#include "Master.h"
#include "Placer.h"
#include "Point.h"
#include "Reporter.h"
#include "data_manager.h"
#include "timing_engine.h"

using namespace std;

namespace ito {

void HoldOptimizer::insertHoldDelay(string insert_buf_name, string pin_name, int insert_number)
{
  ista::Sta* ista = timingEngine->get_sta_engine()->get_ista();
  LibCell* insert_buffer_cell = ista->findLibertyCell(insert_buf_name.c_str());
  LibPort *input_buffer_port, *output_buffer_port;
  insert_buffer_cell->bufferPorts(input_buffer_port, output_buffer_port);

  // iSta ->findPin
  StaVertex* vertex = ista->findVertex(pin_name.c_str());
  DesignObject* load_obj = vertex->get_design_obj();
  Net* net = load_obj->get_net();

  DesignObject* driver_obj = net->getDriver();

  Net* driver_net = net;
  TimingIDBAdapter* idb_adapter = timingEngine->get_sta_adapter();

  // make net
  Net *net_signal_input, *net_signal_output;
  net_signal_input = driver_net;
  std::string net_created_name = (toConfig->get_hold_net_prefix() + "byhand_" + to_string(toDmInst->add_net_num()));
  net_signal_output = idb_adapter->createNet(net_created_name.c_str(), nullptr);

  idb::IdbNet* db_net_signal_output = idb_adapter->staToDb(net_signal_output);
  idb::IdbNet* db_net_signal_input = idb_adapter->staToDb(net_signal_input);
  db_net_signal_output->set_connect_type(db_net_signal_input->get_connect_type());

  /**
   * @brief re-connect load pin to outnet
   */
  if (load_obj->isPin()) {
    Pin* load_pin = dynamic_cast<Pin*>(load_obj);
    Instance* load_inst = load_pin->get_own_instance();
    idb_adapter->disattachPinPort(load_pin);
    auto debug = idb_adapter->attach(load_inst, load_pin->get_name(), net_signal_output);
    LOG_ERROR_IF(!debug);
  } else if (load_obj->isPort()) {
    Port* load_port = dynamic_cast<Port*>(load_obj);
    idb_adapter->disattachPinPort(load_port);
    auto debug = idb_adapter->attach(load_port, load_port->get_name(), net_signal_output);
    LOG_ERROR_IF(!debug);
  }

  // get driver location
  IdbCoordinate<int32_t>* driver_loc = idb_adapter->idbLocation(driver_obj);
  Point driver_pin_loc = Point(driver_loc->get_x(), driver_loc->get_y());
  // get load location
  IdbCoordinate<int32_t>* load_loc = idb_adapter->idbLocation(load_obj);
  Point load_pin_loc = Point(load_loc->get_x(), load_loc->get_y());

  int dx = (driver_pin_loc.get_x() - load_pin_loc.get_x()) / (insert_number + 1);
  int dy = (driver_pin_loc.get_y() - load_pin_loc.get_y()) / (insert_number + 1);

  Net* buf_in_net = net_signal_input;

  for (int i = 0; i < insert_number; i++) {
    Net* buf_out_net = nullptr;
    if (i == insert_number - 1) {
      buf_out_net = net_signal_output;
    } else {
      std::string net_name = (toConfig->get_hold_net_prefix() + "byhand_" + to_string(toDmInst->add_net_num()));

      buf_out_net = idb_adapter->createNet(net_name.c_str(), nullptr);
    }

    idb::IdbNet* buf_out_net_db = idb_adapter->staToDb(buf_out_net);
    buf_out_net_db->set_connect_type(db_net_signal_input->get_connect_type());

    std::string buffer_created_name = (toConfig->get_hold_buffer_prefix() + "byhand_" + to_string(toDmInst->add_buffer_num()));
    Instance* buffer = idb_adapter->createInstance(insert_buffer_cell, buffer_created_name.c_str());

    auto debug_buf_in = idb_adapter->attach(buffer, input_buffer_port->get_port_name(), buf_in_net);
    auto debug_buf_out = idb_adapter->attach(buffer, output_buffer_port->get_port_name(), buf_out_net);
    LOG_ERROR_IF(!debug_buf_in);
    LOG_ERROR_IF(!debug_buf_out);

    int loc_x = driver_pin_loc.get_x() - dx * i;
    int loc_y = driver_pin_loc.get_y() - dy * i;
    timingEngine->placeInstance(loc_x, loc_y, buffer);

    buf_in_net = buf_out_net;
    timingEngine->get_sta_engine()->insertBuffer(buffer->get_name());
  }

  IdbBuilder* idb_builder = idb_adapter->get_idb();
  string defWritePath = string(toConfig->get_output_def_file());
  idb_builder->saveDef(defWritePath);
}

void HoldOptimizer::initBufferCell()
{
  bool not_specified_buffer = toConfig->get_hold_insert_buffers().empty();
  if (not_specified_buffer) {
    TOLibertyCellSeq& buf_cells = timingEngine->get_buffer_cells();
    _available_buffer_cells = buf_cells;
    return;
  }

  auto bufs = toConfig->get_hold_insert_buffers();
  for (auto buf : bufs) {
    auto buffer = timingEngine->get_sta_engine()->findLibertyCell(buf.c_str());
    if (!buffer) {
      LOG_INFO << "Buffer cell " << buf.c_str() << " not found" << endl;
    } else {
      _available_buffer_cells.emplace_back(buffer);
    }
  }
}

void HoldOptimizer::calcBufferCap()
{
  for (auto buf : _available_buffer_cells) {
    LibPort *input, *output;
    buf->bufferPorts(input, output);
    double buf_in_cap = input->get_port_cap();
    _buffer_cap_pair.emplace_back(make_pair(buf_in_cap, buf));
  }
  sort(_buffer_cap_pair.begin(), _buffer_cap_pair.end(),
       [=, this](std::pair<double, ista::LibCell*> v1, std::pair<double, ista::LibCell*> v2) { return v1.first > v2.first; });
}

LibCell* HoldOptimizer::ensureInsertBufferSize()
{
  LibCell* max_delay_buf = nullptr;

  for (LibCell* buffer : _available_buffer_cells) {
    if (strstr(buffer->get_cell_name(), "CLK") != NULL) {
      continue;
    }
    float delay_value_buffer = calcHoldDelayOfBuffer(buffer);
    if (max_delay_buf == nullptr || delay_value_buffer > _hold_insert_buf_cell_delay) {
      max_delay_buf = buffer;
      _hold_insert_buf_cell_delay = delay_value_buffer;
    }
  }
  return max_delay_buf;
}

void HoldOptimizer::insertLoadBuffer(TOVertexSeq fanins)
{
  for (auto vertex : fanins) {
    auto rise_cap = timingEngine->get_sta_engine()->get_ista()->getVertexCapacitanceLimit(vertex, AnalysisMode::kMax, TransType::kRise);
    auto fall_cap = timingEngine->get_sta_engine()->get_ista()->getVertexCapacitanceLimit(vertex, AnalysisMode::kMax, TransType::kFall);
    double rise_cap_limit = rise_cap ? (*rise_cap) : 0.0;
    double fall_cap_limit = fall_cap ? (*fall_cap) : 0.0;

    double cap_limit = min(rise_cap_limit, fall_cap_limit);

    double load_cap_rise = timingEngine->get_sta_engine()->get_ista()->getVertexCapacitance(vertex, AnalysisMode::kMax, TransType::kRise);
    double load_cap_fall = timingEngine->get_sta_engine()->get_ista()->getVertexCapacitance(vertex, AnalysisMode::kMax, TransType::kFall);
    double cap_load = max(load_cap_rise, load_cap_fall);

    auto max_fanout1 = vertex->getMaxFanout();
    double max_fanout = max_fanout1 ? *(max_fanout1) : 10;
    auto fanout = vertex->get_design_obj()->get_net()->getFanouts();

    LibCell* insert_load_buffer = nullptr;
    auto buffer_cells = timingEngine->get_buffer_cells();
    if (cap_load < cap_limit && fanout < max_fanout) {
      double cap_slack = cap_limit - cap_load;

      auto iter = find_if(_buffer_cap_pair.begin(), _buffer_cap_pair.end(),
                          [&cap_slack](std::pair<double, ista::LibCell*> item) { return item.first <= cap_slack; });

      insert_load_buffer = (*iter).second;
      double cap_load = (*iter).first;

      int insert_number_cap = cap_slack / (2 * cap_load);
      int insert_number_fanout = max_fanout - fanout;

      int insert_number = min(insert_number_cap, insert_number_fanout);
      if (insert_load_buffer && insert_number > 0) {
        insertLoadBuffer(insert_load_buffer, vertex, insert_number);
      }
    }
  }
}

void HoldOptimizer::insertLoadBuffer(LibCell* load_buffer, StaVertex* driver_vtx, int insert_num)
{
  auto driver = driver_vtx->get_design_obj();
  TimingIDBAdapter* idb_adapter = timingEngine->get_sta_adapter();

  TODesignObjSeq loads = driver_vtx->get_design_obj()->get_net()->getLoads();
  if (loads.empty()) {
    return;
  }
  auto cap_FF = loads[0];

  IdbCoordinate<int32_t>* loc = idb_adapter->idbLocation(cap_FF);
  Point driver_pin_loc = Point(loc->get_x(), loc->get_y());

  for (int i = 0; i < insert_num; i++) {
    std::string buffer_created_name = ("hold_load_buf_" + to_string(toDmInst->add_buffer_num()));
    auto buffer = idb_adapter->createInstance(load_buffer, buffer_created_name.c_str());

    LibPort *input, *output;
    load_buffer->bufferPorts(input, output);
    auto debug_buf_in = idb_adapter->attach(buffer, input->get_port_name(), driver->get_net());
    LOG_ERROR_IF(!debug_buf_in);

    int loc_x = driver_pin_loc.get_x();
    int loc_y = driver_pin_loc.get_y();

    if (!toDmInst->get_core().overlaps(loc_x, loc_y)) {
      if (loc_x > toDmInst->get_core().get_x_max()) {
        loc_x = toDmInst->get_core().get_x_max() - 1;
      } else if (loc_x < toDmInst->get_core().get_x_min()) {
        loc_x = toDmInst->get_core().get_x_min() + 1;
      }

      if (loc_y > toDmInst->get_core().get_y_max()) {
        loc_y = toDmInst->get_core().get_y_max() - 1;
      } else if (loc_y < toDmInst->get_core().get_y_min()) {
        loc_y = toDmInst->get_core().get_y_min() + 1;
      }
    }
    timingEngine->placeInstance(loc_x, loc_y, buffer);

    timingEngine->get_sta_engine()->insertBuffer(buffer->get_name());
  }

  toEvalInst->invalidNetRC(driver->get_net());
}

void HoldOptimizer::insertBufferOptHold(StaVertex* driver_vertex, int insert_number, TODesignObjSeq& pins_loaded)
{
  auto createBufferInstance = [&](LibCell *buf_cell, const std::string &name) {
    return timingEngine->get_sta_adapter()
        ->createInstance(buf_cell, name.c_str());
  };

  auto createBufferNet = [&](const std::string &name) {
    return timingEngine->get_sta_adapter()
        ->createNet(name.c_str(), nullptr);
  };

  auto* driver_obj = driver_vertex->get_design_obj();
  Net* driver_net = driver_obj->get_net();

  TimingIDBAdapter* idb_adapter = timingEngine->get_sta_adapter();

  // make net
  Net *net_signal_input, *net_signal_output;

  net_signal_input = driver_net;

  std::string net_name = (toConfig->get_hold_net_prefix() + to_string(toDmInst->add_net_num()));
  net_signal_output = createBufferNet(net_name);

  idb::IdbNet* db_net_signal_output = idb_adapter->staToDb(net_signal_output);
  idb::IdbNet* db_net_signal_input = idb_adapter->staToDb(net_signal_input);
  db_net_signal_output->set_connect_type(db_net_signal_input->get_connect_type());

  for (auto* pin_port : pins_loaded) {
    if (pin_port->isPin()) {
      Pin* load_pin = dynamic_cast<Pin*>(pin_port);
      Instance* load_inst = load_pin->get_own_instance();
      // idb_adapter->disattachPin(load_pin);
      idb_adapter->disattachPinPort(pin_port);
      auto debug = idb_adapter->attach(load_inst, load_pin->get_name(), net_signal_output);
      LOG_ERROR_IF(!debug);
    } else if (pin_port->isPort()) {
      Port* load_port = dynamic_cast<Port*>(pin_port);
      idb_adapter->disattachPinPort(pin_port);
      auto debug = idb_adapter->attach(load_port, load_port->get_name(), net_signal_output);
      LOG_ERROR_IF(!debug);
    }
  }

  toEvalInst->invalidNetRC(net_signal_input);
  // Spread buffers between driver and load center.
  IdbCoordinate<int32_t>* loc = idb_adapter->idbLocation(driver_obj);
  Point driver_pin_loc = Point(loc->get_x(), loc->get_y());

  // calculate center of "pins_loaded"
  long long int sum_x = 0;
  long long int sum_y = 0;
  for (auto* pin_port : pins_loaded) {
    IdbCoordinate<int32_t>* idb_loc = idb_adapter->idbLocation(pin_port);
    Point loc = Point(idb_loc->get_x(), idb_loc->get_y());
    sum_x += loc.get_x();
    sum_y += loc.get_y();
  }
  int center_x = sum_x / pins_loaded.size();
  int center_y = sum_y / pins_loaded.size();

  int dx = (driver_pin_loc.get_x() - center_x) / (insert_number + 1);
  int dy = (driver_pin_loc.get_y() - center_y) / (insert_number + 1);

  Net* buf_in_net = net_signal_input;
  Instance* buffer = nullptr;
  LibPort *input, *output;
  _hold_insert_buf_cell->bufferPorts(input, output);

  std::vector<const char*> insert_inst_name;

  for (int i = 0; i < insert_number; i++) {
    Net* buf_out_net;
    if (i == insert_number - 1) {
      buf_out_net = net_signal_output;
    } else {
      std::string net_name = (toConfig->get_hold_net_prefix() + to_string(toDmInst->add_net_num()));
      buf_out_net = createBufferNet(net_name);
    }

    idb::IdbNet* buf_out_net_db = idb_adapter->staToDb(buf_out_net);
    buf_out_net_db->set_connect_type(db_net_signal_input->get_connect_type());

    std::string buffer_created_name = (toConfig->get_hold_buffer_prefix() + to_string(toDmInst->add_buffer_num()));
    buffer = createBufferInstance(_hold_insert_buf_cell, buffer_created_name);

    auto debug_buf_in = idb_adapter->attach(buffer, input->get_port_name(), buf_in_net);
    auto debug_buf_out = idb_adapter->attach(buffer, output->get_port_name(), buf_out_net);
    LOG_ERROR_IF(!debug_buf_in);
    LOG_ERROR_IF(!debug_buf_out);

    int loc_x = driver_pin_loc.get_x() - dx * i;
    int loc_y = driver_pin_loc.get_y() - dy * i;
    if (!toDmInst->get_core().overlaps(loc_x, loc_y)) {
      if (loc_x > toDmInst->get_core().get_x_max()) {
        loc_x = toDmInst->get_core().get_x_max() - 1;
      } else if (loc_x < toDmInst->get_core().get_x_min()) {
        loc_x = toDmInst->get_core().get_x_min() + 1;
      }

      if (loc_y > toDmInst->get_core().get_y_max()) {
        loc_y = toDmInst->get_core().get_y_max() - 1;
      } else if (loc_y < toDmInst->get_core().get_y_min()) {
        loc_y = toDmInst->get_core().get_y_min() + 1;
      }
    }
    timingEngine->placeInstance(loc_x, loc_y, buffer);

    // update in net
    buf_in_net = buf_out_net;

    toEvalInst->invalidNetRC(buf_out_net);

    timingEngine->get_sta_engine()->insertBuffer(buffer->get_name());
    insert_inst_name.push_back(buffer->get_name());

    // increase design area
    idb::IdbCellMaster* idb_master = idb_adapter->staToDb(_hold_insert_buf_cell);
    Master* master = new Master(idb_master);

    float area = toDmInst->calcMasterArea(master, toDmInst->get_dbu());
    toDmInst->increDesignArea(area);
  }
}

float HoldOptimizer::calcHoldDelayOfBuffer(LibCell* buffer)
{
  LibPort *input_buffer_port, *output_buffer_port;
  buffer->bufferPorts(input_buffer_port, output_buffer_port);

  float cap_load = input_buffer_port->get_port_cap();
  TODelay gate_delays[2];

  timingEngine->calcGateRiseFallDelays(gate_delays, cap_load, output_buffer_port);

  return min(gate_delays[TYPE_RISE], gate_delays[TYPE_FALL]);
}

}  // namespace ito

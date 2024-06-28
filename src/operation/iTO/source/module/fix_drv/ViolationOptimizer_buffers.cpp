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

bool ViolationOptimizer::initBuffer()
{
  bool b_init = false;

  if (toConfig->get_drv_insert_buffers().empty()) {
    _insert_buffer_cell = timingEngine->get_buf_lowest_driver_res();
    b_init = true;
  } else {
    float low_drive = -kInf;
    for (auto config_buf : toConfig->get_drv_insert_buffers()) {
      auto buffer = timingEngine->get_sta_engine()->findLibertyCell(config_buf.c_str());
      if (!buffer) {
        _insert_buffer_cell = timingEngine->get_buf_lowest_driver_res();
        b_init = true;
        break;
      }

      ista::LibPort* intput_buffer_port;
      ista::LibPort* output_buffer_port;
      buffer->bufferPorts(intput_buffer_port, output_buffer_port);
      float driver_res = output_buffer_port->driveResistance();
      if (driver_res > low_drive) {
        low_drive = driver_res;
        _insert_buffer_cell = buffer;
        b_init = true;
      }
    }
  }

  if (true == b_init) {
    ista::LibPort *intput_buffer_port, *output_buffer_port;
    _insert_buffer_cell->bufferPorts(intput_buffer_port, output_buffer_port);

    std::optional<double> cap_limit = output_buffer_port->get_port_cap_limit(AnalysisMode::kMax);
    if (cap_limit.has_value()) {
      _max_buf_load_cap = *cap_limit;
    }
  }

  return b_init;
}

void ViolationOptimizer::insertBuffer(int x, int y, ista::Net* net, ista::LibCell* insert_buf_cell, int& wire_length, float& cap,
                                      TODesignObjSeq& pins_loaded)
{
  // make ista::Instance name
  std::string buffer_created_name = (toConfig->get_drv_buffer_prefix() + to_string(toDmInst->add_buffer_num()));

  TimingIDBAdapter* db_adapter = timingEngine->get_sta_adapter();

  ista::Net *net_signal_input, *net_signal_output;
  net_signal_input = net;
  // make net name
  std::string net_created_name = (toConfig->get_drv_net_prefix() + to_string(toDmInst->add_net_num()));
  net_signal_output = db_adapter->createNet(net_created_name.c_str(), nullptr);

  idb::IdbNet* db_net_signal_output = db_adapter->staToDb(net_signal_output);
  idb::IdbNet* db_net_signal_input = db_adapter->staToDb(net_signal_input);
  db_net_signal_output->set_connect_type(db_net_signal_input->get_connect_type());

  // Re-connect the pins_loaded to net_signal_output.
  for (auto* pin_port : pins_loaded) {
    if (pin_port->isPin()) {
      Pin* pin = dynamic_cast<Pin*>(pin_port);
      ista::Instance* inst = pin->get_own_instance();

      db_adapter->disattachPin(pin);
      auto debug = db_adapter->attach(inst, pin->get_name(), net_signal_output);
      LOG_ERROR_IF(!debug);
    }
  }
  ista::Instance* buffer = db_adapter->createInstance(insert_buf_cell, buffer_created_name.c_str());

  idb::IdbCellMaster* idb_master = db_adapter->staToDb(insert_buf_cell);
  Master* master = new Master(idb_master);

  float area = toDmInst->calcMasterArea(master, toDmInst->get_dbu());
  toDmInst->increDesignArea(area);

  ista::LibPort *intput_buffer_port, *output_buffer_port;
  insert_buf_cell->bufferPorts(intput_buffer_port, output_buffer_port);
  auto debug_buf_in = db_adapter->attach(buffer, intput_buffer_port->get_port_name(), net_signal_input);
  auto debug_buf_out = db_adapter->attach(buffer, output_buffer_port->get_port_name(), net_signal_output);
  LOG_ERROR_IF(!debug_buf_in);
  LOG_ERROR_IF(!debug_buf_out);

  timingEngine->get_sta_engine()->insertBuffer(buffer->get_name());

  Pin* driver_pin = buffer->findPin(output_buffer_port);

  std::optional<double> width = std::nullopt;
  double wire_length_cap = timingEngine->get_sta_adapter()->getCapacitance(1, (double) wire_length / toDmInst->get_dbu(), width);
  std::optional<double> cap_limit = output_buffer_port->get_port_cap_limit(AnalysisMode::kMax);
  if (cap_limit.has_value()) {
    double buf_cap_limit = *cap_limit;
    if (buf_cap_limit < cap + wire_length_cap) {
      if (timingEngine->repowerInstance(driver_pin)) {
        auto inst = driver_pin->get_own_instance();
        Pin* int_pin;
        FOREACH_INSTANCE_PIN(inst, int_pin)
        {
          ista::Net* ins_net = int_pin->get_net();
          if (ins_net) {
            toEvalInst->invalidNetRC(ins_net);
            toEvalInst->estimateInvalidNetParasitics(ins_net, ins_net->getDriver());
          }
        }
        toDmInst->add_resize_instance_num();
      }
    }
  }

  insert_buf_cell = buffer->get_inst_cell();
  insert_buf_cell->bufferPorts(intput_buffer_port, output_buffer_port);
  Pin* buf_in_pin = buffer->findPin(intput_buffer_port);
  if (buf_in_pin) {
    pins_loaded.clear();
    pins_loaded.push_back(dynamic_cast<DesignObject*>(buf_in_pin));
    wire_length = 0;
    auto input_cap = intput_buffer_port->get_port_cap(AnalysisMode::kMax, TransType::kRise);
    if (input_cap) {
      cap = *input_cap;
    } else {
      cap = intput_buffer_port->get_port_cap();
    }
  }

  timingEngine->placeInstance(x, y, buffer);

  toEvalInst->invalidNetRC(net_signal_input);
  toEvalInst->invalidNetRC(net_signal_output);
  toEvalInst->estimateInvalidNetParasitics(net_signal_input, net_signal_input->getDriver());
  toEvalInst->estimateInvalidNetParasitics(net_signal_output, net_signal_output->getDriver());
}

}  // namespace ito
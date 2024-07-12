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
#include "EstimateParasitics.h"
#include "Placer.h"
#include "Reporter.h"
#include "SetupOptimizer.h"
#include "ToConfig.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "data_manager.h"
#include "liberty/Lib.hh"
#include "timing_engine.h"
#include "VGBuffer.h"

using namespace std;

namespace ito {

bool SetupOptimizer::performBufferingIfNecessary(Pin* driver_pin, int fanout)
{
  int max_fanout = toConfig->get_max_allowed_buffering_fanout();
  if (fanout > 1 && fanout < max_fanout) {
    int num_insert_buf = 0;
    performVGBuffering(driver_pin, num_insert_buf);
    return num_insert_buf > 0;
  }
  return false;
}

void SetupOptimizer::performVGBuffering(Pin* pin, int& num_insert_buf)
{
  Net* net = pin->get_net();

  if (netConnectToPort(net) || !net) {
    return;
  }

  TreeBuild* tree = new TreeBuild();
  bool make_tree = tree->makeRoutingTree(net, toConfig->get_routing_tree());
  if (!make_tree) {
    return;
  }

  VGBuffer vg_buffer(_available_lib_cell_sizes);
  BufferedOptionSeq buf_solotions = vg_buffer.VGBuffering(tree);

  TORequired best_slack = -kInf;
  BufferedOption* best_option = nullptr;
  for (BufferedOption* opt : buf_solotions) {
    TOSlack slack = opt->get_required_arrival_time();
    if (slack > best_slack) {
      best_slack = slack;
      best_option = opt;
    }
  }
  if (best_option) {
    // for DEBUG
    // best_option->printBuffered(0);
    int old = toDmInst->get_buffer_num();
    implementVGSolution(best_option, net);
    num_insert_buf = toDmInst->get_buffer_num() - old;
  }

  delete tree;
}

void SetupOptimizer::implementVGSolution(BufferedOption* buf_opt, Net* net)
{
  if (!buf_opt) {
    return;
  }

  TimingIDBAdapter* idb_adapter = timingEngine->get_sta_adapter();
  auto handleBuffer = [&](LibCell* insert_buf_cell, Net* current_net, Point loc) {
    std::string buffer_created_name = toConfig->get_setup_buffer_prefix() + std::to_string(toDmInst->add_buffer_num());
    Instance* buffer = idb_adapter->createInstance(insert_buf_cell, buffer_created_name.c_str());

    std::string net_created_name = toConfig->get_setup_net_prefix() + std::to_string(toDmInst->add_net_num());
    Net* net_out = idb_adapter->createNet(net_created_name.c_str(), nullptr);

    LibPort *input, *output;
    insert_buf_cell->bufferPorts(input, output);
    if (!idb_adapter->attach(buffer, input->get_port_name(), current_net) || !idb_adapter->attach(buffer, output->get_port_name(), net_out)) {
      LOG_ERROR << "Failed to attach buffer ports.";
    }

    timingEngine->get_sta_engine()->insertBuffer(buffer->get_name());
    timingEngine->placeInstance(loc.get_x(), loc.get_y(), buffer);

    toEvalInst->invalidNetRC(current_net);
    toEvalInst->invalidNetRC(net_out);

    idb::IdbCellMaster* idb_master = idb_adapter->staToDb(insert_buf_cell);
    Master* master = new Master(idb_master);
    float area = toDmInst->calcMasterArea(master, toDmInst->get_dbu());
    toDmInst->increDesignArea(area);

    implementVGSolution(buf_opt->get_left(), net_out);
  };

  switch (buf_opt->get_type()) {
    case BufferedOptionType::kBuffer:
      handleBuffer(buf_opt->get_lib_cell_size(), net, buf_opt->get_location());
      break;

    case BufferedOptionType::kBranch:
      implementVGSolution(buf_opt->get_left(), net);
      implementVGSolution(buf_opt->get_right(), net);
      break;

    case BufferedOptionType::kSink: {
      Pin* load_pin = buf_opt->get_pin_loaded();
      Net* load_net = load_pin->get_net();
      if (load_net != net) {
        Instance* load_inst = load_pin->get_own_instance();
        idb_adapter->disattachPin(load_pin);
        if (!idb_adapter->attach(load_inst, load_pin->get_name(), net)) {
          LOG_ERROR << "Failed to attach load pin.";
        }

        timingEngine->get_sta_engine()->insertBuffer(load_inst->get_name());
        toEvalInst->invalidNetRC(load_net);
        toEvalInst->invalidNetRC(net);
      }
      break;
    }

    case BufferedOptionType::kWire:
      implementVGSolution(buf_opt->get_left(), net);
      break;

    default:
      LOG_ERROR << "Unknown BufferedOptionType.";
      break;
  }
}

/////////////////////////////////////////////////////////////////////////////////////

bool SetupOptimizer::performSplitBufferingIfNecessary(StaVertex *driver_vertex,
                                                      int        fanout) {
  int min_divide_fanout = toConfig->get_min_divide_fanout();
  if (fanout > min_divide_fanout) {
    insertBufferDivideFanout(driver_vertex);
    return true;
  }
  return false;
}

void SetupOptimizer::insertBufferDivideFanout(StaVertex *driver_vertex) {
  // 获取并排序驱动器的扇出顶点，根据它们的松弛进行排序
  auto fanout_vertexes = timingEngine->get_sta_engine()->getFanoutVertexs(driver_vertex);
  std::vector<std::pair<StaVertex *, TOSlack>> fanout_slacks;
  for (auto fanout_vertex : fanout_vertexes) {
    auto    slack_ns = fanout_vertex->getSlackNs(AnalysisMode::kMax, TransType::kRise);
    TOSlack slack = slack_ns ? *slack_ns : kInf;
    fanout_slacks.emplace_back(fanout_vertex, slack);
  }

  std::sort(fanout_slacks.begin(), fanout_slacks.end(),
            [](auto &a, auto &b) { return a.second > b.second; });

  auto *driver_pin = driver_vertex->get_design_obj();
  auto *net = driver_pin->get_net();

  // 创建缓冲器和线网
  auto createBufferInstance = [&](LibCell *buf_cell, const std::string &name) {
    return timingEngine->get_sta_adapter()
        ->createInstance(buf_cell, name.c_str());
  };

  auto createBufferNet = [&](const std::string &name) {
    return timingEngine->get_sta_adapter()
        ->createNet(name.c_str(), nullptr);
  };

  std::string buffer_created_name = toConfig->get_setup_buffer_prefix() + std::to_string(toDmInst->add_buffer_num());
  std::string net_name = toConfig->get_setup_net_prefix() + std::to_string(toDmInst->add_net_num());

  auto insert_buf_cell = timingEngine->get_buf_lowest_driver_res();
  auto buffer = createBufferInstance(insert_buf_cell, buffer_created_name);
  auto net_signal_output = createBufferNet(net_name);

  LibPort *input, *output;
  insert_buf_cell->bufferPorts(input, output);

  auto *idb_adapter = timingEngine->get_sta_adapter();
  if (!idb_adapter->attach(buffer, input->get_port_name(), net) ||
      !idb_adapter->attach(buffer, output->get_port_name(), net_signal_output)) {
    LOG_ERROR << "Failed to attach buffer ports.";
  }

  // 将一半的负载从原线网移到新线网
  int split_index = fanout_slacks.size() / 2;
  for (int i = 0; i < split_index; ++i) {
    auto *load_vertex = fanout_slacks[i].first;
    auto *load_obj = load_vertex->get_design_obj();

    if (load_obj->isPin()) {
      auto *load_pin = dynamic_cast<Pin *>(load_obj);
      auto *load_inst = load_pin->get_own_instance();

      idb_adapter->disattachPin(load_pin);
      if (!idb_adapter->attach(load_inst, load_pin->get_name(), net_signal_output)) {
        LOG_ERROR << "Failed to reattach load pin.";
      }
    }
  }

  // 放置缓冲器并更新相关信息
  auto *idb_loc = idb_adapter->idbLocation(driver_pin);
  Point driver_loc(idb_loc->get_x(), idb_loc->get_y());

  timingEngine->get_sta_engine()->insertBuffer(buffer->get_name());
  // _number_insert_buffer++;
  auto *buffer_out_pin = buffer->findPin(output);
  timingEngine->repowerInstance(buffer_out_pin);
  timingEngine->placeInstance(driver_loc.get_x(), driver_loc.get_y(), buffer);

  toEvalInst->invalidNetRC(net);
  toEvalInst->invalidNetRC(net_signal_output);
}

}  // namespace ito
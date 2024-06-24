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

#include "timing_engine.h"

#include "ToConfig.h"
#include "data_manager.h"
#include "timing_engine_builder.h"
#include "timing_engine_util.h"

namespace ito {

ToTimingEngine* ToTimingEngine::_instance = nullptr;

ToTimingEngine::ToTimingEngine()
{
  _target_map = new TOLibCellLoadMap();
}

ToTimingEngine::~ToTimingEngine()
{
}

TOLibCellLoadMap* ToTimingEngine::get_target_map()
{
  return _target_map;
}

void ToTimingEngine::new_target_map()
{
  _target_map->clear();
}

void ToTimingEngine::initEngine()
{
  TimingEngineBuilder builder;
  builder.buildEngine();
}

bool ToTimingEngine::canFindLibertyCell(LibCell* cell)
{
  return _timing_engine->findLibertyCell(cell->get_cell_name()) == cell;
}

ista::LibCell* ToTimingEngine::get_drv_buffer()
{
  bool not_specified_buffer = toConfig->get_drv_insert_buffers().empty();
  if (not_specified_buffer) {
    return get_buf_lowest_driver_res();
  }

  float low_drive = -kInf;
  auto bufs = toConfig->get_drv_insert_buffers();
  for (auto buf : bufs) {
    auto buffer = _timing_engine->findLibertyCell(buf.c_str());
    if (!buffer) {
      return get_buf_lowest_driver_res();
    }
    ista::LibPort* in_port;
    ista::LibPort* out_port;
    buffer->bufferPorts(in_port, out_port);
    float driver_res = out_port->driveResistance();
    if (driver_res > low_drive) {
      low_drive = driver_res;
      return buffer;
    }
  }

  return nullptr;
}

void ToTimingEngine::set_eval_data()
{
  if (!_eval_data.empty()) {
    _eval_data.clear();
  }
  auto clk_list = timingEngine->get_sta_engine()->getClockList();
  for (auto clk : clk_list) {
    auto clk_name = clk->get_clock_name();
    auto wns = timingEngine->get_sta_engine()->getWNS(clk_name, AnalysisMode::kMax);
    auto tns = timingEngine->get_sta_engine()->getTNS(clk_name, AnalysisMode::kMax);
    auto freq = 1000.0 / (clk->getPeriodNs() - wns);
    _eval_data.push_back({clk_name, wns, tns, freq});
  }
}

double ToTimingEngine::getWNS()
{
  double wns = kInf;
  auto clk_list = timingEngine->get_sta_engine()->getClockList();
  for (auto clk : clk_list) {
    auto clk_name = clk->get_clock_name();
    auto wns1 = timingEngine->get_sta_engine()->getWNS(clk_name, AnalysisMode::kMax);
    wns = min(wns, wns1);
  }
  return wns;
}

void ToTimingEngine::refineRes(RctNode* node1, RctNode* node2, Net* net, double res, bool b_incre, double incre_cap)
{
  double current_cap = incre_cap / 2.0;

  /// incremental operation
  if (b_incre) {
    _timing_engine->incrCap(node1, current_cap, b_incre);
  }

  /// res operation
  _timing_engine->makeResistor(net, node1, node2, res);

  /// incremental operation
  if (b_incre) {
    _timing_engine->incrCap(node2, current_cap, b_incre);
  }
}

}  // namespace ito
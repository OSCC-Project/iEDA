// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file PowerEngine.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief  The power engine for provide power and timing analysis api.
 * @version 0.1
 * @date 2024-02-26
 *
 */
#include "PowerEngine.hh"

namespace ipower {
PowerEngine* PowerEngine::_power_engine = nullptr;

PowerEngine::PowerEngine() {
  _timing_engine = TimingEngine::getOrCreateTimingEngine();
  _ipower = Power::getOrCreatePower(&(_timing_engine->get_ista()->get_graph()));
}

PowerEngine::~PowerEngine() {
  Power::destroyPower();
  TimingEngine::destroyTimingEngine();
}

/**
 * @brief create dataflow for macro placer.
 * To create dataflow, firstly we build seq graph, the seq vertex is instance or
 * port.
 *
 *
 * @return unsigned
 */
unsigned PowerEngine::creatDataflow() {
  // build timing graph.
  if (!_timing_engine->isBuildGraph()) {
    _timing_engine->buildGraph();
  }

  // build power graph & sequential graph.
  if (!_ipower->isBuildGraph()) {
    // build power graph.
    _ipower->buildGraph();

    // build seq graph
    _ipower->buildSeqGraph();
  }

  return 1;
}

}  // namespace ipower
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
 * @file StaArc.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of timing arc.
 * @version 0.1
 * @date 2021-02-10
 */
#include "StaArc.hh"

#include "StaDump.hh"
#include "StaFunc.hh"
#include "StaVertex.hh"
#include "propagation-cuda/propagation.cuh"

namespace ista {

StaArc::StaArc(StaVertex* src, StaVertex* snk) : _src(src), _snk(snk) {}

/**
 * @brief Add arc delay data.
 *
 * @param arc_delay_data
 */
void StaArc::addData(StaArcDelayData* arc_delay_data) {
  if (arc_delay_data) {
    _arc_delay_bucket.addData(arc_delay_data, 0);
  }
}

/**
 * @brief Get arc delay.
 *
 * @param analysis_mode
 * @param trans_type
 * @return int
 */
int StaArc::get_arc_delay(AnalysisMode analysis_mode, TransType trans_type) {
  StaData* data;
  FOREACH_ARC_DELAY_DATA(this, data) {
    if (data->get_delay_type() == analysis_mode &&
        data->get_trans_type() == trans_type) {
      auto* arc_delay = dynamic_cast<StaArcDelayData*>(data);
      return arc_delay->get_arc_delay();
    }
  }

  return 0;
}

/**
 * @brief init arc delay data.
 * 
 */
void StaArc::initArcDelayData() {
  auto& delay_bucket = getDataBucket();
  if (!delay_bucket.empty()) {
    return;
  }

  auto construct_delay_data = [this](AnalysisMode delay_type,
                                     TransType trans_type, StaArc* own_arc,
                                     int delay) {
    StaArcDelayData* arc_delay =
        new StaArcDelayData(delay_type, trans_type, own_arc, delay);
    ;
    own_arc->addData(arc_delay);

    arc_delay->set_arc_delay(delay);
  };

  /*if not, create default zero slew.*/
  construct_delay_data(AnalysisMode::kMax, TransType::kRise, this, 0);
  construct_delay_data(AnalysisMode::kMax, TransType::kFall, this, 0);
  construct_delay_data(AnalysisMode::kMin, TransType::kRise, this, 0);
  construct_delay_data(AnalysisMode::kMin, TransType::kFall, this, 0);
}

/**
 * @brief Get arc delay data.
 *
 * @param analysis_mode
 * @param trans_type
 * @return StaArcDelayData*
 */
StaArcDelayData* StaArc::getArcDelayData(AnalysisMode analysis_mode,
                                         TransType trans_type) {
  StaData* data;
  FOREACH_ARC_DELAY_DATA(this, data) {
    if (data->get_delay_type() == analysis_mode &&
        data->get_trans_type() == trans_type) {
      auto* arc_delay = dynamic_cast<StaArcDelayData*>(data);
      return arc_delay;
    }
  }

  return nullptr;
}

unsigned StaArc::exec(StaFunc& func) { return func(this); }

/**
 * @brief dump arc info for debug.
 *
 */
void StaArc::dump() {
  StaDumpYaml dump_data;
  dump_data(this);
  dump_data.printText("arc.txt");
}

StaNetArc::StaNetArc(StaVertex* driver, StaVertex* load, Net* net)
    : StaArc(driver, load), _net(net) {}

StaInstArc::StaInstArc(StaVertex* src, StaVertex* snk, LibArc* lib_arc,
                       Instance* inst)
    : StaArc(src, snk),
      _lib_arc(lib_arc),
      _inst(inst){}

// for debug by printLIBTableGPU.(to be deleted)
void printLibTableGPU(const Lib_Table_GPU& gpu_table) {
  // print x axis
  std::cout << "index_1(";
  for (unsigned i = 0; i < gpu_table._num_x; ++i) {
    std::cout << std::fixed << std::setprecision(8) << gpu_table._x[i];
    if (i < gpu_table._num_x - 1) {
      std::cout << ",";
    }
  }
  std::cout << ");" << std::endl;

  // print y axis
  std::cout << "index_2(";
  for (unsigned i = 0; i < gpu_table._num_y; ++i) {
    std::cout << std::fixed << std::setprecision(8) << gpu_table._y[i];
    if (i < gpu_table._num_y - 1) {
      std::cout << ",";
    }
  }
  std::cout << ");" << std::endl;

  // print values
  std::cout << "values (";
  for (unsigned i = 0; i < gpu_table._num_values / gpu_table._num_y; ++i) {
    std::cout << "\"";
    for (unsigned j = 0; j < gpu_table._num_y; ++j) {
      std::cout << std::fixed << std::setprecision(8)
                << gpu_table._values[i * gpu_table._num_y + j];
      if (j < gpu_table._num_y - 1) {
        std::cout << ",";
      }
    }
    std::cout << "\"";
    if (i < gpu_table._num_values - 1) {
      std::cout << ",";
    }
  }
  std::cout << ");" << std::endl;
}



}  // namespace ista

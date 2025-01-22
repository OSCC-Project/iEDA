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
      _inst(inst),
      _lib_gpu_arc(new LibArcGPU()) {}

// for debug by printLIBTableGPU.(to be deleted)
void printLibTableGPU(const LibTableGPU& gpu_table) {
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

/**
 * @brief build gpu lib arc(axes and values) according to the lib arc.
 */
void StaInstArc::buildLibArcsGPU() {
  auto* table_model = _lib_arc->get_table_model();
  if (isDelayArc()) {
    auto* delay_table_model = dynamic_cast<LibDelayTableModel*>(table_model);
    _lib_gpu_arc->_num_table = delay_table_model->kTableNum;
    // int num_table = delay_table_model->kTableNum;
    _lib_gpu_arc->_table = new LibTableGPU[_lib_gpu_arc->_num_table];

    for (size_t index = 0; index < delay_table_model->kTableNum; index++) {
      auto* table = delay_table_model->getTable(index);

      LibTableGPU gpu_table;
      // set the x axis.
      auto& x_axis = table->getAxis(0);
      auto& x_axis_values = x_axis.get_axis_values();
      gpu_table._num_x = static_cast<unsigned>(x_axis_values.size());
      gpu_table._x = new double[gpu_table._num_x];
      for (unsigned i = 0; i < x_axis_values.size(); ++i) {
        gpu_table._x[i] = x_axis_values[i]->getFloatValue();
      }

      auto axes_size = table->get_axes().size();
      LOG_FATAL_IF(axes_size > 2);

      // set the y axis.
      if (axes_size > 1) {
        auto& y_axis = table->getAxis(1);
        auto& y_axis_values = y_axis.get_axis_values();
        gpu_table._num_y = static_cast<unsigned>(y_axis_values.size());
        gpu_table._y = new double[gpu_table._num_y];
        for (unsigned i = 0; i < y_axis_values.size(); ++i) {
          gpu_table._y[i] = y_axis_values[i]->getFloatValue();
        }
      }

      auto* table_template = table->get_table_template();
      if (axes_size == 1) {
        if (*(table_template->get_template_variable1()) ==
                LibLutTableTemplate::Variable::INPUT_NET_TRANSITION ||
            *(table_template->get_template_variable1()) ==
                LibLutTableTemplate::Variable::RELATED_PIN_TRANSITION ||
            *(table_template->get_template_variable1()) ==
                LibLutTableTemplate::Variable::INPUT_TRANSITION_TIME) {
          gpu_table._type = 0;  //(x axis denotes slew.)
        } else {
          gpu_table._type = 1;  //(x axis denotes constrain_slew_or_load.)
        }
      } else {
        if (*(table_template->get_template_variable1()) ==
                LibLutTableTemplate::Variable::INPUT_NET_TRANSITION ||
            *(table_template->get_template_variable1()) ==
                LibLutTableTemplate::Variable::RELATED_PIN_TRANSITION ||
            *(table_template->get_template_variable1()) ==
                LibLutTableTemplate::Variable::INPUT_TRANSITION_TIME) {
          gpu_table._type = 2;  // (x axis denotes slew, y axis denotes
                                // constrain_slew_or_load.)
        } else {
          gpu_table._type = 3;  //(x axis denotes constrain_slew_or_load, y axis
                                // denotes slew.)
        }
      }

      // set the values.
      auto& table_values = table->get_table_values();
      gpu_table._num_values = static_cast<unsigned>(table_values.size());
      gpu_table._values = new double[gpu_table._num_values];
      for (unsigned i = 0; i < table_values.size(); ++i) {
        gpu_table._values[i] = table_values[i]->getFloatValue();
      }

      // printLibTableGPU(gpu_table);
      // set the gpu table to the arc.(cpu index is the same as gpu index)
      _lib_gpu_arc->_table[index] = gpu_table;
    }
  }
}

}  // namespace ista

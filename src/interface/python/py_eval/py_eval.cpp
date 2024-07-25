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
#include "py_eval.h"

#include "EvalAPI.hpp"
#include "EvalType.hpp"

using namespace eval;

namespace python_interface {

// wirelength evaluation
void init_wirelength_eval()
{
  EvalAPI& eval_api = EvalAPI::initInst();
  eval_api.initWLDataFromIDB();
}

int64_t eval_total_wirelength(int wirelength_type)
{
  EvalAPI& eval_api = EvalAPI::getInst();
  return eval_api.evalTotalWL(WIRELENGTH_TYPE(wirelength_type));
}

// congestion evaluation
void init_cong_eval(int bin_cnt_x, int bin_cnt_y)
{
  std::cout << "bin_cnt_x=" << bin_cnt_x << " bin_cnt_y=" << bin_cnt_y << std::endl;
  EvalAPI& eval_api = EvalAPI::initInst();
  eval_api.initCongDataFromIDB(bin_cnt_x, bin_cnt_y);
}

void eval_macro_density()
{
  EvalAPI& eval_api = EvalAPI::getInst();
  eval_api.evalMacroDens();
}

void eval_macro_pin_density()
{
  EvalAPI& eval_api = EvalAPI::getInst();
  eval_api.evalMacroPinDens();
}

void eval_cell_pin_density()
{
  EvalAPI& eval_api = EvalAPI::getInst();
  eval_api.evalCellPinDens();
}

void eval_macro_margin()
{
  EvalAPI& eval_api = EvalAPI::getInst();
  eval_api.evalMacroMargin();
}

void eval_continuous_white_space()
{
  EvalAPI& eval_api = EvalAPI::getInst();
  double unused_value = eval_api.evalMaxContinuousSpace();
}

void eval_macro_channel(float die_size_ratio)
{
  EvalAPI& eval_api = EvalAPI::getInst();
  eval_api.evalMacroChannel(die_size_ratio);
}

void eval_cell_hierarchy(const std::string& plot_path, int level, int forward)
{
  EvalAPI& eval_api = EvalAPI::getInst();
  eval_api.evalCellHierarchy(plot_path, level, forward);
}

void eval_macro_hierarchy(const std::string& plot_path, int level, int forward)
{
  EvalAPI& eval_api = EvalAPI::getInst();
  eval_api.evalMacroHierarchy(plot_path, level, forward);
}

void eval_macro_connection(const std::string& plot_path, int level, int forward)
{
  EvalAPI& eval_api = EvalAPI::getInst();
  eval_api.evalMacroConnection(plot_path, level, forward);
}

void eval_macro_pin_connection(const std::string& plot_path, int level, int forward)
{
  EvalAPI& eval_api = EvalAPI::getInst();
  eval_api.evalMacroPinConnection(plot_path, level, forward);
}

void eval_macro_io_pin_connection(const std::string& plot_path, int level, int forward)
{
  EvalAPI& eval_api = EvalAPI::getInst();
  eval_api.evalMacroIOPinConnection(plot_path, level, forward);
}

void eval_inst_density(int inst_status, int eval_flip_flop)
{
  EvalAPI& eval_api = EvalAPI::getInst();
  eval_api.evalInstDens(INSTANCE_STATUS(inst_status), eval_flip_flop);
}

void eval_pin_density(int inst_status, int level)
{
  EvalAPI& eval_api = EvalAPI::getInst();
  eval_api.evalPinDens(INSTANCE_STATUS(inst_status), level);
}

void eval_rudy_cong(int rudy_type, int direction)
{
  EvalAPI& eval_api = EvalAPI::getInst();
  eval_api.evalNetCong(RUDY_TYPE(rudy_type), DIRECTION(direction));
}

std::vector<float> eval_overflow()
{
  EvalAPI& eval_api = EvalAPI::getInst();
  return eval_api.evalGRCong();
}

// timing evaluation
void init_timing_eval()
{
  EvalAPI& eval_api = EvalAPI::getInst();
  eval_api.initTimingDataFromIDB();
}

// plot API
void plot_bin_value(const std::string& plot_path, const std::string& file_name, int value_type)
{
  EvalAPI& eval_api = EvalAPI::getInst();
  eval_api.plotBinValue(plot_path, file_name, CONGESTION_TYPE(value_type));
}
void plot_tile_value(const string& plot_path, const string& file_name)
{
  EvalAPI& eval_api = EvalAPI::getInst();
  eval_api.plotTileValue(plot_path, file_name);
}
void plot_flow_value(const std::string& plot_path, const std::string& file_name, const std::string& step, const std::string& value)
{
  EvalAPI& eval_api = EvalAPI::getInst();
  eval_api.plotFlowValue(plot_path, file_name, step, value);
}

// void eval_net_density(int inst_status)
// {
//   EvalAPI& eval_api = EvalAPI::getInst();
//   eval_api.evalNetDens(INSTANCE_STATUS(inst_status));
// }
// void eval_local_net_density()
// {
//   EvalAPI& eval_api = EvalAPI::getInst();
//   eval_api.evalLocalNetDens();
// }
// void eval_global_net_density()
// {
//   EvalAPI& eval_api = EvalAPI::getInst();
//   eval_api.evalGlobalNetDens();
// }

// int32_t eval_inst_num(int inst_status)
// {
//   EvalAPI& eval_api = EvalAPI::getInst();
//   return eval_api.evalInstNum(INSTANCE_STATUS(inst_status));
// }
// int32_t eval_net_num(int net_type)
// {
//   EvalAPI& eval_api = EvalAPI::getInst();
//   return eval_api.evalNetNum(NET_CONNECT_TYPE(net_type));
// }
// int32_t eval_pin_num(int inst_status)
// {
//   EvalAPI& eval_api = EvalAPI::getInst();
//   return eval_api.evalPinNum(INSTANCE_STATUS(inst_status));
// }
// int32_t eval_routing_layer_num()
// {
//   EvalAPI& eval_api = EvalAPI::getInst();
//   return eval_api.evalRoutingLayerNum();
// }
// int32_t eval_track_num(int direction = 0)
// {
// }
// int32_t eval_track_remain_num()
// {
// }
// int32_t eval_track_overflow_num()
// {
// }

// std::vector<int64_t> eval_chip_size(int region_type)
// {
// }
// std::vector<std::pair<string, std::pair<int32_t, int32_t>>> eval_inst_size(int inst_status)
// {
// }
// std::vector<std::pair<string, std::pair<int32_t, int32_t>>> eval_net_size()
// {
// }
// int64_t eval_area(int inst_status)
// {
// }
// std::vector<int64_t> eval_macro_peri_area()
// {
// }

// float eval_area_util(int inst_status)
// {
// }
// double eval_macro_channel_util(float dist_ratio)
// {
// }
// double eval_macro_channel_pin_util(float dist_ratio)
// {
// }

}  // namespace python_interface
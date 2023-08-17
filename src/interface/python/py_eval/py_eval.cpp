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

namespace python_interface {

void init_cong_eval(const int bin_cnt_x, const int bin_cnt_y)
{
  eval::EvalAPI& eval_api = eval::EvalAPI::initInst();
  eval_api.initCongDataFromIDB(bin_cnt_x, bin_cnt_y);
}

void eval_inst_density(INSTANCE_STATUS inst_status, bool eval_flip_flop = false)
{
  eval::EvalAPI& eval_api = eval::EvalAPI::getInst();
  eval_api.evalInstDens(inst_status);
}
void eval_pin_density(INSTANCE_STATUS inst_status, int level = 0)
{
}
void eval_net_density(INSTANCE_STATUS inst_status)
{
}
void eval_local_net_density()
{
}
void eval_global_net_density()
{
}

int32_t eval_inst_num(INSTANCE_STATUS inst_status)
{
}
int32_t eval_net_num(NET_CONNECT_TYPE net_type)
{
}
int32_t eval_pin_num(INSTANCE_STATUS inst_status = INSTANCE_STATUS::kNone)
{
}
int32_t eval_routing_layer_num()
{
  eval::EvalAPI& eval_api = eval::EvalAPI::getInst();
  return eval_api.evalRoutingLayerNum();
}
int32_t eval_track_num(DIRECTION direction = DIRECTION::kNone)
{
}
int32_t eval_track_remain_num()
{
}
int32_t eval_track_overflow_num()
{
}

std::vector<int64_t> eval_chip_size(CHIP_REGION_TYPE region_type)
{
}
std::vector<std::pair<string, std::pair<int32_t, int32_t>>> eval_inst_size(INSTANCE_STATUS inst_status)
{
}
std::vector<std::pair<string, std::pair<int32_t, int32_t>>> eval_net_size()
{
}

void eval_rudy_cong(RUDY_TYPE rudy_type, DIRECTION direction = DIRECTION::kNone)
{
}
std::vector<float> eval_egr_cong()
{
}

int64_t eval_area(INSTANCE_STATUS inst_status)
{
}
std::vector<int64_t> eval_macro_peri_area()
{
}

float eval_area_util(INSTANCE_STATUS inst_status)
{
}
double eval_macro_channel_util(float dist_ratio)
{
}
double eval_macro_channel_pin_util(float dist_ratio)
{
}

void plot_bin_value(const string& plot_path, const string& file_name, CONGESTION_TYPE value_type)
{
  eval::EvalAPI& eval_api = eval::EvalAPI::getInst();
  eval_api.plotBinValue(plot_path, file_name, value_type);
}
void plot_tile_value(const string& plot_path, const string& file_name)
{
}
}  // namespace python_interface
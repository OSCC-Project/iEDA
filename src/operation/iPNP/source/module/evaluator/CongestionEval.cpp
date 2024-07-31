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
/**
 * @file CongestionEval.cpp
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

#include "CongestionEval.hh"

namespace ipnp {
void CongestionEval::rudy_routing()
{
  auto& eval_api = eval::EvalAPI::initInst();

  int32_t bin_cnt_x = 512;  // Grid size
  int32_t bin_cnt_y = 512;
  eval_api.initCongDataFromIDB(bin_cnt_x, bin_cnt_y);

  string eval_method = "RUDY";
  _net_cong_rudy = eval_api.evalNetCong(eval_method);  // using RUDY
}

void CongestionEval::global_routing()
{
}
}  // namespace ipnp
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
#include <cstdlib>
#include <iostream>
#include <string>

#include "Config.hpp"
#include "EvalAPI.hpp"
#include "EvalLog.hpp"
#include "gtest/gtest.h"
#include "idm.h"
#include "manager.hpp"
#include "usage/usage.hh"

namespace eval {

class CongAPITest : public testing::Test
{
  void SetUp()
  {  // Read Def, Lef
    std::string idb_json_file = "/home/yhqiu/iEDA/bin/db_ispd19.json";
    dmInst->init(idb_json_file);
  }
  void TearDown() final {}
};

TEST_F(CongAPITest, sample)
{
  EvalAPI& eval_api = EvalAPI::initInst();
  int32_t bin_cnt_x = 256;
  int32_t bin_cnt_y = 256;
  eval_api.initCongDataFromIDB(bin_cnt_x, bin_cnt_y);
  std::string plot_path = "/home/yhqiu/iEDA/bin/csv/";

  auto inst_status = INSTANCE_STATUS::kFixed;
  eval_api.evalInstDens(inst_status);
  eval_api.plotBinValue(plot_path, "macro_density", CONGESTION_TYPE::kInstDens);
  eval_api.evalPinDens(inst_status);
  eval_api.plotBinValue(plot_path, "macro_pin_density", CONGESTION_TYPE::kPinDens);
  eval_api.evalNetDens(inst_status);
  eval_api.plotBinValue(plot_path, "macro_net_density", CONGESTION_TYPE::kNetCong);

  inst_status = INSTANCE_STATUS::kPlaced;
  eval_api.evalInstDens(inst_status);
  eval_api.plotBinValue(plot_path, "stdcell_density", CONGESTION_TYPE::kInstDens);
  // eval_api.evalInstDens(inst_status, true);
  // eval_api.plotBinValue(plot_path, "flipflop_density", CONGESTION_TYPE::kInstDens);
  eval_api.evalPinDens(inst_status);
  eval_api.plotBinValue(plot_path, "stdcell_pin_density", CONGESTION_TYPE::kPinDens);
  eval_api.evalPinDens(inst_status, 1);
  eval_api.plotBinValue(plot_path, "stdcell_level_pin_density", CONGESTION_TYPE::kPinDens);

  eval_api.evalLocalNetDens();
  eval_api.plotBinValue(plot_path, "local_net_density", CONGESTION_TYPE::kNetCong);
  eval_api.evalGlobalNetDens();
  eval_api.plotBinValue(plot_path, "global_net_density", CONGESTION_TYPE::kNetCong);

  eval_api.evalNetCong(RUDY_TYPE::kRUDY);
  eval_api.plotBinValue(plot_path, "RUDY", CONGESTION_TYPE::kNetCong);
  eval_api.evalNetCong(RUDY_TYPE::kPinRUDY);
  eval_api.plotBinValue(plot_path, "PinRUDY", CONGESTION_TYPE::kNetCong);
  eval_api.evalNetCong(RUDY_TYPE::kLUTRUDY);
  eval_api.plotBinValue(plot_path, "LUTRUDY", CONGESTION_TYPE::kNetCong);
  eval_api.evalNetCong(RUDY_TYPE::kRUDY, DIRECTION::kH);
  eval_api.plotBinValue(plot_path, "RUDY_Hori", CONGESTION_TYPE::kNetCong);
  eval_api.evalNetCong(RUDY_TYPE::kRUDY, DIRECTION::kV);
  eval_api.plotBinValue(plot_path, "RUDY_Verti", CONGESTION_TYPE::kNetCong);

  std::vector<int64_t> die_info_list = eval_api.evalChipWidthHeightArea(CHIP_REGION_TYPE::kDie);
  std::vector<int64_t> core_info_list = eval_api.evalChipWidthHeightArea(CHIP_REGION_TYPE::kCore);
  LOG_INFO << "die width is " << die_info_list[0] << "; height is " << die_info_list[1] << "; area is " << die_info_list[2];
  LOG_INFO << "core width is " << core_info_list[0] << "; height is " << core_info_list[1] << "; area is " << core_info_list[2];
  inst_status = INSTANCE_STATUS::kFixed;
  LOG_INFO << "macro number is " << eval_api.evalInstNum(inst_status);
  LOG_INFO << "routing layer number is " << eval_api.evalRoutingLayerNum();
  std::vector<std::pair<string, std::pair<int32_t, int32_t>>> macro_info_list = eval_api.evalInstSize(inst_status);
  for (auto& macro_info : macro_info_list) {
    LOG_INFO << "macro name is " << macro_info.first << " ; width is " << macro_info.second.first << " ; height is "
             << macro_info.second.second;
  }
  std::vector<std::pair<string, std::pair<int32_t, int32_t>>> net_info_list = eval_api.evalNetSize();
  for (auto& net_info : net_info_list) {
    LOG_INFO << "net name is " << net_info.first << " ; width is " << net_info.second.first << " ; height is " << net_info.second.second;
  }
  inst_status = INSTANCE_STATUS::kPlaced;
  LOG_INFO << "standard cell number is " << eval_api.evalInstNum(inst_status);
  LOG_INFO << "pin number is " << eval_api.evalPinNum();
  LOG_INFO << "logical net number is " << eval_api.evalNetNum(NET_CONNECT_TYPE::kSignal);
  LOG_INFO << "power net number is " << eval_api.evalNetNum(NET_CONNECT_TYPE::kPower);
  LOG_INFO << "track Horizontal number is " << eval_api.evalTrackNum(DIRECTION::kH);
  LOG_INFO << "track Vertical number is " << eval_api.evalTrackNum(DIRECTION::kV);
  LOG_INFO << "track Total number is " << eval_api.evalTrackNum();

  auto congestion_metrics = eval_api.evalGRCong();
  LOG_INFO << "ACE is  " << congestion_metrics[0] << " TOF is " << congestion_metrics[1] << " MOF is " << congestion_metrics[2];
  eval_api.plotTileValue(plot_path, "congestion_map_");

  std::string py_command = "python plot.py";
  int result = std::system(py_command.c_str());
  if (result == 0) {
    std::cout << "success" << std::endl;
  } else {
    std::cout << "failed" << std::endl;
  }
  EvalAPI::destroyInst();
}

}  // namespace eval

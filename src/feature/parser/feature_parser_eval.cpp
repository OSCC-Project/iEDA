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
 * @project		iEDA
 * @file		feature_parser.cpp
 * @author		Yell
 * @date		10/08/2023
 * @version		0.1
 * @description


        feature parser
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "EvalAPI.hpp"
#include "Evaluator.hh"
#include "feature_parser.h"
#include "feature_summary.h"
#include "flow_config.h"
#include "idm.h"
#include "iomanip"
#include "json_parser.h"
// #include "report_evaluator.h"

namespace ieda_feature {

bool FeatureParser::buildSummaryMap(std::string csv_path, int bin_cnt_x, int bin_cnt_y)
{
  eval::EvalAPI& eval_api = eval::EvalAPI::initInst();
  eval_api.initCongDataFromIDB(bin_cnt_x, bin_cnt_y);

  auto inst_status = eval::INSTANCE_STATUS::kFixed;
  eval_api.evalInstDens(inst_status);
  eval_api.plotBinValue(csv_path, "macro_density", eval::CONGESTION_TYPE::kInstDens);
  eval_api.evalPinDens(inst_status);
  eval_api.plotBinValue(csv_path, "macro_pin_density", eval::CONGESTION_TYPE::kPinDens);
  eval_api.evalNetDens(inst_status);
  eval_api.plotBinValue(csv_path, "macro_net_density", eval::CONGESTION_TYPE::kNetCong);

  eval_api.plotMacroChannel(0.5, csv_path + "macro_channel.csv");
  eval_api.evalMacroMargin();
  eval_api.plotBinValue(csv_path, "macro_margin_h", eval::CONGESTION_TYPE::kMacroMarginH);
  eval_api.plotBinValue(csv_path, "macro_margin_v", eval::CONGESTION_TYPE::kMacroMarginV);
  double space_ratio = eval_api.evalMaxContinuousSpace();
  eval_api.plotBinValue(csv_path, "macro_continuous_white_space", eval::CONGESTION_TYPE::kContinuousWS);
  eval_api.evalIOPinAccess(csv_path + "io_pin_access.csv");

  std::cout << std::endl << "Save feature map success, path = " << csv_path << std::endl;
  return true;
}

}  // namespace ieda_feature

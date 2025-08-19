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
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#include "CongestionEval.hh"

#include "PNPConfig.hh"
#include "congestion_api.h"
#include "idm.h"

namespace ipnp {

void CongestionEval::evalEGR()
{
  std::string map_path;
  std::string stage = "place";

  if (!pnpConfig->get_egr_map_path().empty()) {
    map_path = pnpConfig->get_egr_map_path();
  } else {
    map_path = "../src/operation/iPNP/example";
  }

  ieval::CongestionAPI congestion_api;
  ieval::OverflowSummary overflow_summary;

  std::string temp = congestion_api.egrUnionMap(stage, map_path);
  overflow_summary = congestion_api.egrOverflow(stage, temp);
  _total_overflow_union = overflow_summary.total_overflow_union;
}
}  // namespace ipnp
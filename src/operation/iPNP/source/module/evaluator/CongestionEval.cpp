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
#include "idm.h"
#include "congestion_api.h"
#include "PNPConfig.hh"

namespace ipnp {

  void CongestionEval::evalEGR(idb::IdbBuilder* idb_builder)
  {
    dmInst->set_idb_builder(idb_builder);
    dmInst->set_idb_def_service(idb_builder->get_def_service());
    dmInst->set_idb_lef_service(idb_builder->get_lef_service());

    std::string map_path;
    std::string stage = "place";

    if (_config != nullptr && !_config->get_egr_map_path().empty()) {
      map_path = _config->get_egr_map_path();
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
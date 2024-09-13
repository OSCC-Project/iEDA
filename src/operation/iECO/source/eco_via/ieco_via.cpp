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
#include "ieco_via.h"

#include "ieco_dm.h"
#include "ieco_via_init.h"
#include "ieco_via_repair.h"

namespace ieco {

ECOVia::ECOVia(EcoDataManager* data_manager)
{
  _data_manager = data_manager;
}

ECOVia::~ECOVia()
{
}

void ECOVia::init()
{
  ECOViaInit via_init(_data_manager);
  via_init.initData();
}

int ECOVia::repair(std::string type)
{
  ECOViaType enum_type;
  if (type == eco_repair_via_by_shape) {
    enum_type = ECOViaType::kECOViaByShape;
  } else if (type == eco_repair_via_by_pattern) {
    enum_type = ECOViaType::kECOViaByPattern;
  } else {
    enum_type = ECOViaType::kECOViaByShape;
  }

  return repair(enum_type);
}

int ECOVia::repair(ECOViaType type)
{
  int repair_num = 0;

  ECOViaRepair via_repair(_data_manager);
  switch (type) {
    case ECOViaType::kECOViaByShape:
      repair_num = via_repair.repairByShape();
      break;
    case ECOViaType::kECOViaByPattern:
      repair_num = via_repair.repairByPattern();
      break;
    default:
      break;
  }

  return repair_num;
}

}  // namespace ieco
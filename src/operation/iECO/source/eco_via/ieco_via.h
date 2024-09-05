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
#pragma once
#include <map>
#include <string>

namespace ieco {
enum ECOViaType
{
  kECONone = 0,
  kECOViaByShape = 1,
  kECOViaByPattern = 2,
  kECOViaMax
};

#define eco_repair_via_by_shape "shape"
#define eco_repair_via_by_pattern "pattern"

class EcoDataManager;

class ECOVia
{
 public:
  ECOVia(EcoDataManager* data_manager);
  ~ECOVia();

  void init();
  int repair(std::string type);

 private:
  EcoDataManager* _data_manager;
  int repair(ECOViaType type = ECOViaType::kECOViaByShape);
};

}  // namespace ieco
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
#include <set>
#include <string>
#include <vector>

#include "DRCViolationType.h"
#include "IdbLayer.h"
#include "boost_definition.h"
#include "idm.h"
#include "idrc_dm.h"
#include "idrc_region_query.h"
#include "idrc_violation.h"
#include "tech_rules.h"

namespace idrc {
class DrcEngineManager;

class DrcViolationManager
{
 public:
  DrcViolationManager(){};
  ~DrcViolationManager();

  std::map<ViolationEnumType, std::vector<DrcViolation*>> get_violation_map(DrcEngineManager* engine_manager);

  std::vector<DrcViolation*>& get_violation_list(ViolationEnumType type);

  void addViolation(int llx, int lly, int urx, int ury, ViolationEnumType type, std::set<int> net_id, std::string layer_name);

 private:
  std::map<ViolationEnumType, std::vector<DrcViolation*>> _violation_list;

  void set_net_ids(DrcEngineManager* engine_manager);
};

}  // namespace idrc
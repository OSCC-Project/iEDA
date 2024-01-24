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

#include <list>
#include <map>
#include <string>
#include <vector>

#include "condition.h"
#include "condition_record.h"
#include "idrc_violation_manager.h"
#include "tech_rules.h"

namespace idb {
class IdbLayer;
}

namespace idrc {

/**
 * rule conditions are concepts built from tech lef drc rules, it contains a condition matrix to guide condition check orders, the rule
 * matrix index indicates the checking order,
 *
 */

class DrcConditionManager
{
 public:
  DrcConditionManager(DrcViolationManager* violation_manager) : _violation_manager(violation_manager) {}
  ~DrcConditionManager() {}

  void deliverSequence(idb::IdbLayer* layer, uint64_t recognize_code, ConditionSequence::SequenceType sequence,
                       std::vector<DrcBasicPoint*>& points)
  {
    // refresh condition record
    auto record_list = _condition_recording_map[layer][recognize_code];
    for (auto record_it = record_list.begin(); record_it != record_list.end(); ++record_it) {
      auto record = *record_it;
      if (refreshConditionRecord(record, sequence, points)) {
        record_list.erase(record_it);
        delete record;
      }
    }

    // create new condition records
    auto& condition_list = DrcTechRuleInst->get_condition_routing_layers(layer)[sequence];
    for (auto* condition : condition_list) {
      auto record = new ConditionRecord(condition);
      if (refreshConditionRecord(record, sequence, points)) {
        delete record;
      } else {
        record_list.push_back(record);
      }
    }
  }

 private:
  DrcViolationManager* _violation_manager;

  std::map<idb::IdbLayer*, std::map<uint64_t, std::list<ConditionRecord*>>> _condition_recording_map;  // TODO: use object pool

  bool refreshConditionRecord(ConditionRecord* record, ConditionSequence::SequenceType sequence, std::vector<DrcBasicPoint*>& points)
  {
    auto state = record->record(sequence, points);
    switch (state) {
      case ConditionSequence::State::kFail:
      case ConditionSequence::State::kSuccess:
        return true;
      default:
        return false;
    }
  }
};

}  // namespace idrc
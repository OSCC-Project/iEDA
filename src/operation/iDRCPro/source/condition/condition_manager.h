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

  void deliverSequence(idb::IdbLayer* layer, ConditionSequence::SequenceType sequence, std::vector<DrcBasicPoint*> points)
  {
    // TODO: find record in condition running, put data to record object
    // TODO: if condition sequence returns fail or success, put record object to object pool

    // TODO: find condition in tech rules
    // TODO: iterate all conditions match sequence, find if there is a record in condition pool, move it to condition running
    // TODO: otherwize create a new record and put it to condition running
    // TODO: put data to record object
    // TODO: if fail ..., if success ...
  }

 private:
  DrcViolationManager* _violation_manager;

  std::map<idb::IdbLayer*, std::map<std::string, ConditionRecord*>> _condition_running_map;
  std::map<idb::IdbLayer*, std::map<uint64_t, std::list<ConditionRecord*>>> _condition_pool_map;

  uint64_t sequenceToIndex(idb::IdbLayer* layer, ConditionSequence::SequenceType sequence)
  {
    auto condition_routing_layers = DrcTechRuleInst->get_condition_routing_layers(layer);
    for (auto& [trigger_sequence, condition_list] : condition_routing_layers) {
      if (trigger_sequence & sequence) {
        return trigger_sequence;
      }
    }
    return 0;
  }

  std::string recordingIndex(ConditionSequence::SequenceType sequence, std::vector<DrcBasicPoint*> points)
  {
    // TODO: use forward/backward and polygon id map to get index
    // TODO: index type
    return "none";
  }
};

}  // namespace idrc
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
  DrcConditionManager(DrcViolationManager* violation_manager) : _violation_manager(violation_manager)
  {
    _record_new = new ConditionRecord(nullptr);
  }
  ~DrcConditionManager() {}

  bool isSequenceNeedDeliver(idb::IdbLayer* layer, uint64_t recognize_code, ConditionSequence::SequenceType sequence)
  {
    if (sequence == ConditionSequence::SequenceType::kNone) {
      return false;
    }

    bool need_deliver = false;

    // refresh condition record
    auto& record_list = _condition_recording_map[layer][recognize_code];
    auto record_it = record_list.begin();
    while (record_it != record_list.end()) {
      auto record = *record_it;
      auto state = record->transferState(sequence);
      if (state == ConditionSequence::State::kFail) {
        record_it = record_list.erase(record_it);
        delete record;
      } else {
        need_deliver = true;
        ++record_it;
      }
    }

    // create new condition records
    auto& condition_list = DrcTechRuleInst->get_condition_trigger(layer, sequence);
    for (auto* condition : condition_list) {
      _record_new->clear();
      _record_new->set_condition(condition);
      auto state = _record_new->transferState(sequence);
      switch (state) {
        case ConditionSequence::State::kTrigger:
        case ConditionSequence::State::kRecording:
        case ConditionSequence::State::kSuccess:
          record_list.push_back(_record_new);
          need_deliver = true;
          _record_new = new ConditionRecord(nullptr);
          break;
        default:
          break;
      }
    }

    // if recognize code is not found, erase it
    if (record_list.empty()) {
      _condition_recording_map[layer].erase(recognize_code);
    }

    return need_deliver;
  }

  void recordRegion(idb::IdbLayer* layer, uint64_t recognize_code, ConditionSequence::SequenceType sequence,
                    std::vector<DrcBasicPoint*>& points)
  {
    auto record_list = _condition_recording_map[layer][recognize_code];
    auto record_it = record_list.begin();
    while (record_it != record_list.end()) {
      auto record = *record_it;
      auto state = record->record(sequence, points);
      if (state == ConditionSequence::State::kSuccess) {
        record_it = record_list.erase(record_it);
        delete record;
      } else {
        ++record_it;
      }
    }

    // if recognize code is not found, erase it
    if (record_list.empty()) {
      _condition_recording_map[layer].erase(recognize_code);
    }
  }

 private:
  uint64_t debug_code = 0;
  DrcViolationManager* _violation_manager;

  std::map<idb::IdbLayer*, std::map<uint64_t, std::list<ConditionRecord*>>> _condition_recording_map;

  ConditionRecord* _record_new = nullptr;  // TODO: use object pool
};

}  // namespace idrc
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

#include <vector>

#include "condition.h"
#include "idrc_engine.h"

namespace idrc {

class ConditionRecord
{
 public:
  ConditionRecord(Condition* condition) : _condition(condition) {}
  ~ConditionRecord() = default;

  ConditionSequence::State record(ConditionSequence::SequenceType sequence, std::vector<DrcBasicPoint*> points)
  {
    _region_record.push_back(std::make_pair(sequence, points));
    if (_state == ConditionSequence::State::kSuccess) {
      _condition->get_detail()->apply(_region_record);  // TODO: use thread pool
    } else if (_state == ConditionSequence::State::kTrigger || _state == ConditionSequence::State::kRecording) {
      _condition->get_sequence()->applyValue(_state, sequence, points);
    }
    return _state;
  }

  ConditionSequence::State transferState(ConditionSequence::SequenceType sequence)
  {
    _condition->get_sequence()->applySequence(_state, sequence);
    return _state;
  }

  void set_condition(Condition* condition) { _condition = condition; }

  Condition* get_condition() { return _condition; }

  void clear()
  {
    _condition = nullptr;
    _state = ConditionSequence::State::kNone;
    _region_record.clear();
  }

 private:
  Condition* _condition;

  ConditionSequence::State _state = ConditionSequence::State::kNone;

  std::vector<std::pair<ConditionSequence::SequenceType, std::vector<DrcBasicPoint*>>> _region_record;
};
}  // namespace idrc
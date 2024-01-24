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

namespace idrc {

class ConditionRecord
{
 public:
  ConditionRecord(Condition* condition) : _condition(condition) {}
  ~ConditionRecord() = default;

  ConditionSequence::State record(ConditionSequence::SequenceType sequence, std::vector<DrcBasicPoint*> points)
  {
    _region_record.push_back(std::make_pair(sequence, points));
    _state = _condition->get_sequence()->apply(sequence, points, _state);
    return _state;
  }

 private:
  Condition* _condition;

  ConditionSequence::State _state;

  std::vector<std::pair<ConditionSequence::SequenceType, std::vector<DrcBasicPoint*>>> _region_record;
};
}  // namespace idrc
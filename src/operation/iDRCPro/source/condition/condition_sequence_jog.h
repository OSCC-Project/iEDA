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

#include <stdint.h>

#include <vector>

#include "condition_sequence.h"
#include "drc_basic_point.h"

namespace idrc {

class ConditionSequenceJog : public ConditionSequence
{
 public:
  ConditionSequenceJog(int filter_value, uint64_t trigger_sequence, uint64_t middle_sequence, uint64_t success_sequence)
      : ConditionSequence(filter_value, trigger_sequence), _middle_sequence(middle_sequence), _success_sequence(success_sequence)
  {
  }
  ~ConditionSequenceJog() override {}

  ConditionSequence::State apply(ConditionSequence::SequenceType sequence, std::vector<DrcBasicPoint*> points, State state) override
  {
    // TODO: use sequence and filter value to create state machine
    // TODO: if state machine is kNone and sequence is ESW or ESE (_trigger_sequence), then return kTrigger
    // TODO: if state machine is kTrigger and sequence is WES (_middle_sequence), then return kRecording
    // TODO: is state machine is kRecording and sequence is ESW or ESE (_success_sequence), then return kSuccess
    // TODO: if state machine is kRecording and sequence is _middle_sequence, if within > _filter_value, then return kFail
    // TODO: if state machine is kTrigger and sequence is not _middle_sequence, then return kFail
    // TODO: if state machine is kRecording and sequence is not _success_sequence, then return kFail
    return ConditionSequence::State::kNone;
  }

 private:
  uint64_t _middle_sequence;
  uint64_t _success_sequence;
};

}  // namespace idrc
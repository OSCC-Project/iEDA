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
  ConditionSequenceJog(uint64_t start_sequence, uint64_t middle_sequence, uint64_t end_sequence)
      : _start_sequence(start_sequence), _middle_sequence(middle_sequence), _end_sequence(end_sequence)
  {
  }
  ~ConditionSequenceJog() override {}

  // prototype pattern
  ConditionSequence* clone() override { return new ConditionSequenceJog(_start_sequence, _middle_sequence, _end_sequence); }

  ConditionSequence::State apply(ConditionSequenceEnum sequence) override
  {
    // TODO: use sequence to create state machine
    // TODO: if state turns to stop, create check item and push to check list
    return ConditionSequence::State::kNone;
  }

 private:
  uint64_t _start_sequence;
  uint64_t _middle_sequence;
  uint64_t _end_sequence;

  std::vector<std::pair<ConditionSequenceEnum, std::vector<DrcBasicPoint*>>> _region_record;
};
// condition manager
}  // namespace idrc
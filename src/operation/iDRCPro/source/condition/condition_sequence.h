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

namespace idrc {

class DrcBasicPoint;

class ConditionSequence
{
 public:
  /*
    SE : Starting Edge
    TE : Turning Edge
    EE : Ending Edge
    MS : Mutual Spacing
    SS : Self Spacing
    W : Width
  */
  enum SequenceType : uint64_t
  {
    kNone = 0,
    kSE = 1,
    kTE = 2,
    kEE = 4,
    kSE_MS_W = 8,
    kSE_MS_SE = 16,
    kSE_MS_TE = 32,
    kTE_MS_W = 64,
    kTE_MS_TE = 128,
    kEE_MS_W = 256,
    kEE_MS_EE = 512,
    kEE_MS_TE = 1024
  };

  enum class State
  {
    kNone,
    kTrigger,
    kRecording,
    kSuccess,
    kFail
  };

  ConditionSequence(int filter_value, uint64_t trigger_sequence) : _filter_value(filter_value), _trigger_sequence(trigger_sequence) {}
  virtual ~ConditionSequence() {}

  virtual void applySequence(State& state, SequenceType condition_sequence_enum) = 0;

  virtual void applyValue(State& state, int& value, SequenceType condition_sequence_enum, std::vector<DrcBasicPoint*>& points) = 0;

  bool match(SequenceType condition_sequence) { return _trigger_sequence & condition_sequence; }

 protected:
  int _filter_value;
  uint64_t _trigger_sequence;

 private:
};

}  // namespace idrc
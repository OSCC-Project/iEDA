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

namespace idrc {

class ConditionSequence
{
 public:
  /*
    W - Width
    E - Edge
    N - None
    S - Spacing
  */
  enum ConditionSequenceEnum : uint64_t
  {
    kWEW = 1,
    kNEW_WEN = 2,
    kSEW_WES = 4,
    kSES = 8,
    kNES_SEN = 16,
    kNEN = 32,
    kESW_WSE = 64,
    kESE = 128
  };

  ConditionSequence(uint64_t sequence_pattern) : _sequence_pattern(sequence_pattern) {}
  ~ConditionSequence() {}

 private:
  uint64_t _sequence_pattern;
};

}  // namespace idrc
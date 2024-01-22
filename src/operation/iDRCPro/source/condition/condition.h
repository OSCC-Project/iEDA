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

#include "condition_detail.h"
#include "condition_sequence.h"
#include "condition_value.h"

namespace idrc {

class Condition
{
 public:
  Condition(Condition* base_condition, ConditionSequence* sequence, ConditionValue* value, ConditionDetail* detail)
      : _base_condition(base_condition), _sequence(sequence), _value(value), _detail(detail)
  {
  }
  ~Condition()
  {
    delete _sequence;
    delete _value;
    delete _detail;
  }

  // getter
  Condition* get_base_condition() { return _base_condition; }
  ConditionSequence* get_sequence() { return _sequence; }
  ConditionValue* get_value() { return _value; }
  ConditionDetail* get_detail() { return _detail; }

  // prototype pattern
  Condition* clone()
  {
    Condition* base = _base_condition ? _base_condition : this;
    Condition* condition = new Condition(base, _sequence->clone(), _value->clone(), _detail->clone());
    return condition;
  }

 private:
  Condition* _base_condition;

  ConditionSequence* _sequence;
  ConditionValue* _value;
  ConditionDetail* _detail;
};

}  // namespace idrc
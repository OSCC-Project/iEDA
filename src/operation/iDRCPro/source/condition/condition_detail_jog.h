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

#include "check_item.h"
#include "condition_detail.h"
#include "drc_basic_point.h"
#include "idm.h"

namespace idrc {

class ConditionJogCheckItem : CheckItem
{
 public:
  ConditionJogCheckItem() {}
  ~ConditionJogCheckItem() override {}

 private:
  std::vector<DrcBasicPoint*> _condition_region;
  int _wire_width;
};

class ConditionDetailJog : public ConditionDetail
{
 public:
  ConditionDetailJog(idb::routinglayer::Lef58SpacingTableJogToJog* rule) : _jog_to_jog(rule) {}
  ~ConditionDetailJog() override {}

  bool apply(std::vector<std::pair<ConditionSequence::SequenceType, std::vector<DrcBasicPoint*>>>& check_region) override;

  // bool apply(CheckItem* item) override;

 private:
  idb::routinglayer::Lef58SpacingTableJogToJog* _jog_to_jog;
};

}  // namespace idrc
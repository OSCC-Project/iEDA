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

#include <map>
#include <set>

#include "IdbLayer.h"
#include "rule_basic.h"
#include "rule_condition_map.h"

namespace idrc {

class ConditionRuleCutSpacing : public ConditionRule
{
 public:
  ConditionRuleCutSpacing(RuleType type, int spacing, idb::IdbLayerCutSpacing* cut_spacing)
      : ConditionRule(type, spacing), _cut_spacing(cut_spacing)
  {
  }
  ~ConditionRuleCutSpacing() {}

  idb::IdbLayerCutSpacing* get_cut_spacing() { return _cut_spacing; }

 private:
  idb::IdbLayerCutSpacing* _cut_spacing;
};

class ConditionRuleCutArraySpacing : public ConditionRule
{
 public:
  ConditionRuleCutArraySpacing(RuleType type, int spacing, idb::IdbLayerCutArraySpacing* cut_array_spacing)
      : ConditionRule(type, spacing), _cut_array_spacing(cut_array_spacing)
  {
  }
  ~ConditionRuleCutArraySpacing() {}

  idb::IdbLayerCutArraySpacing* get_cut_array_spacing() { return _cut_array_spacing; }

 private:
  idb::IdbLayerCutArraySpacing* _cut_array_spacing;
};

class ConditionRuleCutEnclosure : public ConditionRule
{
 public:
  ConditionRuleCutEnclosure(RuleType type, int spacing, idb::IdbLayerCutEnclosure* above_enclosure,
                            idb::IdbLayerCutEnclosure* below_enclosure)
      : ConditionRule(type, spacing), _above_enclosure(above_enclosure), _below_enclosure(below_enclosure)
  {
  }
  ~ConditionRuleCutEnclosure() {}

  idb::IdbLayerCutEnclosure* get_above_enclosure() { return _above_enclosure; }
  idb::IdbLayerCutEnclosure* get_below_enclosure() { return _below_enclosure; }

 private:
  idb::IdbLayerCutEnclosure* _above_enclosure;
  idb::IdbLayerCutEnclosure* _below_enclosure;
};

class ConditionRuleCutWidth : public ConditionRule
{
 public:
  ConditionRuleCutWidth(RuleType type, int spacing, int cut_width) : ConditionRule(type, spacing), _cut_width(cut_width) {}
  ~ConditionRuleCutWidth() {}

  int get_cut_width() { return _cut_width; }

 private:
  int _cut_width;
};

}  // namespace idrc
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
/**
 * @project		iDB
 * @file		IdbViaMaster.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Via master information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbViaRule.h"

namespace idb {

IdbViaRuleGenerate::IdbViaRuleGenerate()
{
  _name = "";
  _layer_bottom = nullptr;
  _enclosure_bottom = new IdbLayerCutEnclosure();
  _layer_cut = nullptr;
  // _enclosure_cut = new IdbLayerCutEnclosure();
  _cut_rect = new IdbRect();
  _spacing_x = -1;
  _spacing_y = -1;
  _layer_top = nullptr;
  _enclosure_top = new IdbLayerCutEnclosure();
}

IdbViaRuleGenerate::~IdbViaRuleGenerate()
{
  if (_enclosure_bottom) {
    delete _enclosure_bottom;
    _enclosure_bottom = nullptr;
  }

  if (_cut_rect) {
    delete _cut_rect;
    _cut_rect = nullptr;
  }

  if (_enclosure_top) {
    delete _enclosure_top;
    _enclosure_top = nullptr;
  }
}

void IdbViaRuleGenerate::set_cut_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y)
{
  _cut_rect->set_rect(ll_x, ll_y, ur_x, ur_y);
}

void IdbViaRuleGenerate::swap_routing_layer()
{
  if (_layer_bottom != nullptr && _layer_top != nullptr) {
    if (_layer_bottom->get_order() > _layer_top->get_order()) {
      std::swap(_layer_bottom, _layer_top);
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbViaRuleList::IdbViaRuleList()
{
  _num_rule_generate = 0;
}

IdbViaRuleList::~IdbViaRuleList()
{
  for (IdbViaRuleGenerate* via_rule : _via_rule_generate_list) {
    if (via_rule != nullptr) {
      delete via_rule;
      via_rule = nullptr;
    }
  }
}

IdbViaRuleGenerate* IdbViaRuleList::find_via_rule_generate(string name)
{
  for (IdbViaRuleGenerate* via_rule : _via_rule_generate_list) {
    if (via_rule->get_name() == name) {
      return via_rule;
    }
  }

  return nullptr;
}

IdbViaRuleGenerate* IdbViaRuleList::find_via_rule_generate(int32_t index)
{
  if (_num_rule_generate > index) {
    return _via_rule_generate_list.at(index);
  }

  return nullptr;
}

IdbViaRuleGenerate* IdbViaRuleList::add_via_rule_generate(IdbViaRuleGenerate* via_rule)
{
  IdbViaRuleGenerate* pRule = via_rule;
  if (pRule == nullptr) {
    pRule = new IdbViaRuleGenerate();
  }
  _via_rule_generate_list.emplace_back(pRule);
  _num_rule_generate++;

  return pRule;
}

IdbViaRuleGenerate* IdbViaRuleList::add_via_rule_generate(string name)
{
  IdbViaRuleGenerate* pRule = find_via_rule_generate(name);
  if (pRule == nullptr) {
    pRule = new IdbViaRuleGenerate();
    pRule->set_name(name);
    _via_rule_generate_list.emplace_back(pRule);
    _num_rule_generate++;
  }

  return pRule;
}

}  // namespace idb

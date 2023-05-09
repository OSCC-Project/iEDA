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
/**
 * @project		iDB
 * @file		IdbRuleSpacing.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe PROPERTY LEF58_SPACING for SPACING module.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>

// #include "IdbGeometry.h"
#include "IdbObject.h"

namespace idb {

using std::vector;

class IdbLayer;

class IdbPropertyCutSpacing
{
 public:
  enum CutSpacingStatus : int8_t
  {
    kNone,
    kSameNet,
    kSameMetal,
    kSameVia,
    kMax
  };

 public:
  IdbPropertyCutSpacing()
  {
    _status = CutSpacingStatus::kNone;
    _center_to_center = false;
  }
  ~IdbPropertyCutSpacing() = default;

  // getter
  bool is_center_to_center() { return _center_to_center; }
  bool is_same_net() { return _status == CutSpacingStatus::kSameNet ? true : false; }
  bool is_same_metal() { return _status == CutSpacingStatus::kSameMetal ? true : false; }
  bool is_same_via() { return _status == CutSpacingStatus::kSameVia ? true : false; }
  int32_t get_cut_spacing() { return _cut_spacing; }

  // setter
  void set_center_to_center(bool center_to_center) { _center_to_center = center_to_center; }
  void set_same_net() { _status = CutSpacingStatus::kSameNet; }
  void set_same_metal() { _status = CutSpacingStatus::kSameMetal; }
  void set_same_Via() { _status = CutSpacingStatus::kSameVia; }
  void set_cut_spacing(int32_t cut_spacing) { _cut_spacing = cut_spacing; }

  // operator

 private:
  bool _center_to_center;
  CutSpacingStatus _status;
  int32_t _cut_spacing;
};

class IdbRuleCutSpacingList
{
 public:
  IdbRuleCutSpacingList() {}
  ~IdbRuleCutSpacingList() { reset(); }

 public:
  // getter
  const int32_t get_num() const { return _spacing_list.size(); };
  vector<IdbPropertyCutSpacing*>& get_spacing_list() { return _spacing_list; }
  IdbPropertyCutSpacing* get_spacing(int i)
  {
    if (i > 0 && i < (int) _spacing_list.size()) {
      return _spacing_list[i];
    }

    return nullptr;
  }

  // setter
  IdbPropertyCutSpacing* add_spacing()
  {
    IdbPropertyCutSpacing* spacing = new IdbPropertyCutSpacing();

    _spacing_list.emplace_back(spacing);

    return spacing;
  }

  void reset()
  {
    for (auto& spacing : _spacing_list) {
      if (spacing != nullptr) {
        delete spacing;
        spacing = nullptr;
      }
    }

    _spacing_list.clear();
  }

  // operator

 private:
  vector<IdbPropertyCutSpacing*> _spacing_list;
};

}  // namespace idb

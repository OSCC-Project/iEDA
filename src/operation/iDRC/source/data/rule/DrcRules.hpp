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
// #pragma once

// #include "DrcEnum.h"

// namespace idrc {

// class DrcRule
// {
//  public:
//   virtual RuleTypeEnum getType() const = 0;
// };

// class EnclosureRule : public DrcRule
// {
//  public:
//   EnclosureRule()
//       : _required_dir(EnclosureDirEnum::Default), _overhang1(-1), _overhang2(-1), _min_length(-1), _min_width(-1), _cut_within(-1)
//   {
//   }
//   // getter
//   int getOverhang1() const { return _overhang1; }
//   int getOverhang2() const { return _overhang2; }
//   int getMinWidth() const { return _min_width; }
//   int getCutWithin() const { return _cut_within; }
//   int getMinLength() const { return _min_length; }
//   EnclosureDirEnum getRequiredDir() const { return _required_dir; }

//   // setter
//   void setRequiredDir(EnclosureDirEnum in) { _required_dir = in; }
//   void setOverhang1(int in) { _overhang1 = in; }
//   void setOverhang2(int in) { _overhang2 = in; }
//   void setMinWidth(int in) { _min_width = in; }
//   void setCutWithin(int in) { _cut_within = in; }
//   void setMinLength(int in) { _min_length = in; }
//   // func
//   bool hasRequiredDir() const { return _required_dir != EnclosureDirEnum::Default; }
//   bool hasWidth() const { return _min_width != -1; }
//   bool hasExceptExtraCut() const { return _cut_within != -1; }
//   bool hasLength() const { return _min_length != -1; }
//   RuleTypeEnum getType() const override { return RuleTypeEnum::kEnclosureRule; }

//  private:
//   EnclosureDirEnum _required_dir;
//   int _overhang1;
//   int _overhang2;
//   int _min_width;
//   int _cut_within;
//   int _min_length;
// };

// class MinimumCutRule : public DrcRule
// {
//  public:
//   MinimumCutRule() : _num_cuts(-1), _width(-1), _cut_distance(-1), _connection(MinimumCutDirEnum::Default), _length(-1), _distance(-1) {}
//   // getters
//   int getNumCuts() const { return _num_cuts; }
//   int getWidth() const { return _width; }
//   bool hasWithin() const { return !(_cut_distance == -1); }
//   int getCutDistance() const { return _cut_distance; }
//   bool hasConnection() const { return !(_connection == MinimumCutDirEnum::Default); }
//   MinimumCutDirEnum getConnection() const { return _connection; }
//   bool hasLength() const { return !(_length == -1); }
//   int getLength() const { return _length; }
//   int getDistance() const { return _distance; }
//   // setters
//   void setNumCuts(int in) { _num_cuts = in; }
//   void setWidth(int in) { _width = in; }
//   void setWithin(int in) { _cut_distance = in; }
//   void setConnection(MinimumCutDirEnum in) { _connection = in; }
//   void setLength(int in1, int in2)
//   {
//     _length = in1;
//     _distance = in2;
//   }
//   // func
//   RuleTypeEnum getType() const override { return RuleTypeEnum::kMinimumCutRule; }

//  private:
//   int _num_cuts;
//   int _width;
//   int _cut_distance;
//   MinimumCutDirEnum _connection;
//   int _length;
//   int _distance;
// };

// class LEF58EOLSpacingRule : public DrcRule
// {
//  public:
//   LEF58EOLSpacingRule() = default;
//   ~LEF58EOLSpacingRule() = default;
//   // getter
//   int getSpacing() const { return _spacing; }
//   int getEOLMaxLength() const { return _eol_max_length; }
//   int getWithin() const { return _within; }
//   int getEnd2EndSpacing() const { return _end2end_spacing; }
//   int getAdjEdgeMinLength() const { return _adj_edge_min_length; }
//   int getPrlSpace() const { return _prl_space; }
//   int getPrlWithin() const { return _prl_within; }
//   int getPrlEOLMinLength() const { return _prl_eol_min_length; }
//   int getCutEncloseDist() const { return _cut_enclose_dist; }
//   int getCutToMetalSpacing() const { return _cut_to_metal_spacing; }
//   bool getSubstractTag() const { return _has_substract_eol_width; }
//   // setter
//   void setSpacing(int in) { _spacing = in; }
//   void setEOLMaxLength(int in) { _eol_max_length = in; }
//   void setWithin(int in) { _within = in; }
//   void setEnd2EndSpacing(int in) { _end2end_spacing = in; }
//   void setAdjEdgeMinLength(int in) { _adj_edge_min_length = in; }
//   void setPrlSpace(int in) { _prl_space = in; }
//   void setPrlWithin(int in) { _prl_within = in; }
//   void setPrlEOLMinLength(int in) { _prl_eol_min_length = in; }
//   void setCutEncloseDist(int in) { _cut_enclose_dist = in; }
//   void setCutToMetalSpacing(int in) { _cut_to_metal_spacing = in; }
//   void setSubstractTag(int in) { _has_substract_eol_width = in; }

//   // func
//   bool hasAdjEdgeCons() { return !(_adj_edge_min_length == -1); }
//   bool hasPrlCons() { return !(_prl_space == -1); }
//   bool hasCutCons() { return !(_cut_enclose_dist == -1); }
//   bool hasEnd2End() { return !(_end2end_spacing == -1); }
//   bool hasSubtractTag() { return !(_has_substract_eol_width == false); }
//   RuleTypeEnum getType() const override { return RuleTypeEnum::kLEF58SpacingEOLRule; }

//  private:
//   int _spacing = -1;
//   int _eol_max_length = -1;
//   int _within = -1;
//   int _end2end_spacing = -1;
//   int _adj_edge_min_length = -1;
//   int _prl_space = -1;
//   bool _has_substract_eol_width = false;
//   int _prl_within = -1;
//   int _prl_eol_min_length = -1;
//   int _cut_enclose_dist = -1;
//   int _cut_to_metal_spacing = -1;
// };

// }  // namespace idrc
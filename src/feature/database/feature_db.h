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
 * @file		summary_db.h
 * @date		13/05/2024
 * @version		0.1
 * @description


        summary data
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace ieda_feature {

struct SummaryInfo
{
  std::string eda_tool;
  std::string eda_version;
  std::string design_name;
  std::string design_version;
  std::string flow_stage;
  std::string flow_runtime;
  std::string flow_memory;
};

struct SummaryLayout
{
  int design_dbu;
  double die_area;
  double die_usage;
  double die_bounding_width;
  double die_bounding_height;
  double core_area;
  double core_usage;
  double core_bounding_width;
  double core_bounding_height;
};

struct SummaryStatis
{
  uint num_layers;
  uint num_layers_routing;
  uint num_layers_cut;
  uint num_iopins;
  uint num_instances;
  uint num_nets;
  uint num_pdn;
};

struct SummaryInstance
{
  uint num;
  double num_ratio;
  double area;
  double area_ratio;
  double die_usage;
  double core_usage;
  uint pin_num;
  double pin_ratio;
};

struct SummaryInstances
{
  SummaryInstance total;
  SummaryInstance iopads;
  SummaryInstance macros;
  SummaryInstance logic;
  SummaryInstance clock;
};

struct SummaryNets
{
  uint64_t num_total;
  uint64_t num_signal;
  uint64_t num_clock;
  uint64_t num_pins;
  uint64_t num_segment;
  uint64_t num_via;
  uint64_t num_wire;
  uint64_t num_patch;

  double wire_len;
  double wire_len_signal;
  double ratio_signal;
  double wire_len_clock;
  double ratio_clock;
};

struct SummaryLayerRouting
{
  std::string layer_name;
  int32_t layer_order;
  double wire_len;
  double wire_ratio;
  uint64_t wire_num;
  uint64_t patch_num;
};

struct SummaryLayerCut
{
  std::string layer_name;
  int32_t layer_order;
  uint64_t via_num;
  double via_ratio;
};

struct SummaryLayers
{
  int32_t num_layers;
  int32_t num_layers_routing;
  int32_t num_layers_cut;
  std::vector<SummaryLayerRouting> routing_layers;
  std::vector<SummaryLayerCut> cut_layers;
};

struct SummaryPin
{
  uint64_t pin_num;
  uint64_t net_num;
  double net_ratio;
  uint64_t inst_num;
  double inst_ratio;
};

struct SummaryPins
{
  int max_fanout;
  std::vector<SummaryPin> pin_distribution;
};

struct DBSummary
{
  SummaryInfo info;
  SummaryLayout layout;
  SummaryStatis statis;
  SummaryInstances instances;
  SummaryNets nets;
  SummaryLayers layers;
  SummaryPins pins;
};

}  // namespace ieda_feature

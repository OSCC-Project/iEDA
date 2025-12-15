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

#include "RTHeader.hpp"

namespace irt {

class PASummary
{
 public:
  PASummary() = default;
  ~PASummary() = default;
  std::map<int32_t, double> routing_wire_length_map;
  double total_wire_length = 0;
  std::map<int32_t, int32_t> cut_via_num_map;
  int32_t total_via_num = 0;
  std::map<int32_t, int32_t> routing_patch_num_map;
  int32_t total_patch_num = 0;
  std::map<int32_t, int32_t> routing_violation_num_map;
  int32_t total_violation_num = 0;
};

class SASummary
{
 public:
  SASummary() = default;
  ~SASummary() = default;
  std::map<int32_t, int32_t> routing_supply_map;
  int32_t total_supply = 0;
};

class TGSummary
{
 public:
  TGSummary() = default;
  ~TGSummary() = default;
  double total_demand = 0;
  double total_overflow = 0;
  double total_wire_length = 0;
  std::map<std::string, std::map<std::string, double>> clock_timing_map;
  std::map<std::string, double> type_power_map;
};

class LASummary
{
 public:
  LASummary() = default;
  ~LASummary() = default;
  std::map<int32_t, double> routing_demand_map;
  double total_demand = 0;
  std::map<int32_t, double> routing_overflow_map;
  double total_overflow = 0;
  std::map<int32_t, double> routing_wire_length_map;
  double total_wire_length = 0;
  std::map<int32_t, int32_t> cut_via_num_map;
  int32_t total_via_num = 0;
  std::map<std::string, std::map<std::string, double>> clock_timing_map;
  std::map<std::string, double> type_power_map;
};

class SRSummary
{
 public:
  SRSummary() = default;
  ~SRSummary() = default;
  std::map<int32_t, double> routing_demand_map;
  double total_demand = 0;
  std::map<int32_t, double> routing_overflow_map;
  double total_overflow = 0;
  std::map<int32_t, double> routing_wire_length_map;
  double total_wire_length = 0;
  std::map<int32_t, int32_t> cut_via_num_map;
  int32_t total_via_num = 0;
  std::map<std::string, std::map<std::string, double>> clock_timing_map;
  std::map<std::string, double> type_power_map;
};

class TASummary
{
 public:
  TASummary() = default;
  ~TASummary() = default;
  std::map<int32_t, double> routing_wire_length_map;
  double total_wire_length = 0;
  std::map<int32_t, int32_t> routing_violation_num_map;
  int32_t total_violation_num = 0;
};

class DRSummary
{
 public:
  DRSummary() = default;
  ~DRSummary() = default;
  std::map<int32_t, double> routing_wire_length_map;
  double total_wire_length = 0;
  std::map<int32_t, int32_t> cut_via_num_map;
  int32_t total_via_num = 0;
  std::map<int32_t, int32_t> routing_patch_num_map;
  int32_t total_patch_num = 0;
  std::map<int32_t, int32_t> routing_violation_num_map;
  int32_t total_violation_num = 0;
  std::map<std::string, std::map<std::string, double>> clock_timing_map;
  std::map<std::string, double> type_power_map;
};

class VRSummary
{
 public:
  VRSummary() = default;
  ~VRSummary() = default;
  std::map<int32_t, double> routing_wire_length_map;
  double total_wire_length = 0;
  std::map<int32_t, int32_t> cut_via_num_map;
  int32_t total_via_num = 0;
  std::map<int32_t, int32_t> routing_patch_num_map;
  int32_t total_patch_num = 0;
  std::map<int32_t, std::map<std::string, int32_t>> within_net_routing_violation_type_num_map;
  std::map<std::string, int32_t> within_net_violation_type_num_map;
  std::map<int32_t, int32_t> within_net_routing_violation_num_map;
  int32_t within_net_total_violation_num = 0;
  std::map<int32_t, std::map<std::string, int32_t>> among_net_routing_violation_type_num_map;
  std::map<std::string, int32_t> among_net_violation_type_num_map;
  std::map<int32_t, int32_t> among_net_routing_violation_num_map;
  int32_t among_net_total_violation_num = 0;
  std::map<std::string, std::map<std::string, double>> clock_timing_map;
  std::map<std::string, double> type_power_map;
};

class Summary
{
 public:
  Summary() = default;
  ~Summary() = default;
  std::map<int32_t, PASummary> iter_pa_summary_map;
  SASummary sa_summary;
  TGSummary tg_summary;
  LASummary la_summary;
  std::map<int32_t, SRSummary> iter_sr_summary_map;
  TASummary ta_summary;
  std::map<int32_t, DRSummary> iter_dr_summary_map;
  VRSummary vr_summary;
};

}  // namespace irt

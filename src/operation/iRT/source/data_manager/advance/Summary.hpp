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
  std::map<int32_t, int32_t> routing_access_point_num_map;
  std::map<AccessPointType, int32_t> type_access_point_num_map;
  int32_t total_access_point_num = 0;
};

class SASummary
{
 public:
  SASummary() = default;
  ~SASummary() = default;
  std::map<int32_t, int32_t> routing_supply_map;
  int32_t total_supply = 0;
};

class IRSummary
{
 public:
  IRSummary() = default;
  ~IRSummary() = default;
  std::map<int32_t, int32_t> routing_demand_map;
  int32_t total_demand = 0;
  std::map<int32_t, int32_t> routing_overflow_map;
  int32_t total_overflow = 0;
  std::map<int32_t, double> routing_wire_length_map;
  double total_wire_length = 0;
  std::map<int32_t, int32_t> cut_via_num_map;
  int32_t total_via_num = 0;
  std::map<std::string, std::vector<double>> timing;
};

class GRSummary
{
 public:
  GRSummary() = default;
  ~GRSummary() = default;
  std::map<int32_t, int32_t> routing_demand_map;
  int32_t total_demand = 0;
  std::map<int32_t, int32_t> routing_overflow_map;
  int32_t total_overflow = 0;
  std::map<int32_t, double> routing_wire_length_map;
  double total_wire_length = 0;
  std::map<int32_t, int32_t> cut_via_num_map;
  int32_t total_via_num = 0;
  std::map<std::string, std::vector<double>> timing;
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
  std::map<std::string, std::vector<double>> timing;
};

class Summary
{
 public:
  Summary() = default;
  ~Summary() = default;
  PASummary pa_summary;
  SASummary sa_summary;
  IRSummary ir_summary;
  GRSummary gr_summary;
  TASummary ta_summary;
  std::map<int32_t, DRSummary> iter_dr_summary_map;
};

}  // namespace irt

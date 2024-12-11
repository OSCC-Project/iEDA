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
#include <string>
#include <vector>

#include "feature_ipw.h"
#include "feature_ista.h"
namespace ieda_feature {
/// ###################################################################################///
///  summary
/// ###################################################################################///
struct PASummary
{
  std::map<int32_t, int32_t> routing_access_point_num_map;
  std::map<std::string, int32_t> type_access_point_num_map;
  int32_t total_access_point_num = 0;
};

struct SASummary
{
  std::map<int32_t, int32_t> routing_supply_map;
  int32_t total_supply = 0;
};

struct TGSummary
{
  int32_t total_demand = 0;
  int32_t total_overflow = 0;
  double total_wire_length = 0;
  std::vector<ClockTiming> clocks_timing;
  PowerInfo power_info;
};

struct LASummary
{
  std::map<int32_t, int32_t> routing_demand_map;
  int32_t total_demand = 0;
  std::map<int32_t, int32_t> routing_overflow_map;
  int32_t total_overflow = 0;
  std::map<int32_t, double> routing_wire_length_map;
  double total_wire_length = 0;
  std::map<int32_t, int32_t> cut_via_num_map;
  int32_t total_via_num = 0;
  std::vector<ClockTiming> clocks_timing;
  PowerInfo power_info;
};

struct ERSummary
{
  std::map<int32_t, int32_t> routing_demand_map;
  int32_t total_demand = 0;
  std::map<int32_t, int32_t> routing_overflow_map;
  int32_t total_overflow = 0;
  std::map<int32_t, double> routing_wire_length_map;
  double total_wire_length = 0;
  std::map<int32_t, int32_t> cut_via_num_map;
  int32_t total_via_num = 0;
  std::vector<ClockTiming> clocks_timing;
  PowerInfo power_info;
};

struct GRSummary
{
  std::map<int32_t, int32_t> routing_demand_map;
  int32_t total_demand = 0;
  std::map<int32_t, int32_t> routing_overflow_map;
  int32_t total_overflow = 0;
  std::map<int32_t, double> routing_wire_length_map;
  double total_wire_length = 0;
  std::map<int32_t, int32_t> cut_via_num_map;
  int32_t total_via_num = 0;
  std::vector<ClockTiming> clocks_timing;
  PowerInfo power_info;
};

struct TASummary
{
  std::map<int32_t, double> routing_wire_length_map;
  double total_wire_length = 0;
  std::map<int32_t, int32_t> routing_violation_num_map;
  int32_t total_violation_num = 0;
};

struct DRSummary
{
  std::map<int32_t, double> routing_wire_length_map;
  double total_wire_length = 0;
  std::map<int32_t, int32_t> cut_via_num_map;
  int32_t total_via_num = 0;
  std::map<int32_t, int32_t> routing_patch_num_map;
  int32_t total_patch_num = 0;
  std::map<int32_t, int32_t> routing_violation_num_map;
  int32_t total_violation_num = 0;
  std::vector<ClockTiming> clocks_timing;
  PowerInfo power_info;
};

struct RTSummary
{
  PASummary pa_summary;
  SASummary sa_summary;
  TGSummary tg_summary;
  LASummary la_summary;
  ERSummary er_summary;
  std::map<int32_t, GRSummary> iter_gr_summary_map;
  TASummary ta_summary;
  std::map<int32_t, DRSummary> iter_dr_summary_map;
};

/// ###################################################################################///
///  net feature
/// ###################################################################################///
///  pin access distribution
struct DbPinAccess
{
  std::string layer;
  int x = 0;
  int y = 0;
  int number = 0;
};

struct TermPA
{
  std::vector<DbPinAccess> pa_list;
};

struct CellMasterPA
{
  std::string name;                         // cell master name
  std::map<std::string, TermPA> term_list;  // 1 term name, 2 pa list for a term
};

struct RouteAnalyseData
{
  std::map<std::string, CellMasterPA> cell_master_list;
};

}  // namespace ieda_feature
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
#pragma once

#include <map>
#include <string>
#include <vector>

namespace idb {

struct access_point_num_map
{
  std::map<int32_t, int32_t> routing_access_point_num_map;
  std::map<std::string, int32_t> type_access_point_num_map;
  int32_t total_access_point_num = 0;
};

struct routing_supply_map
{
  std::map<int32_t, int32_t> routing_supply_map;
  int32_t total_supply = 0;
};

struct demand_map
{
  std::map<int32_t, int32_t> routing_demand_map;
  int32_t total_demand = 0;
};

struct overflow_map
{
  std::map<int32_t, int32_t> routing_overflow_map;
  int32_t total_overflow = 0;
};

struct wire_length_map
{
  std::map<int32_t, double> routing_wire_length_map;
  double total_wire_length = 0;
};

struct cut_via_num_map
{
  std::map<int32_t, int32_t> cut_via_num_map;
  int32_t total_via_num = 0;
};

struct timing_map
{
  std::map<std::string, std::vector<double>> timing;
};

struct violation_num_map
{
  std::map<int32_t, int32_t> routing_violation_num_map;
  int32_t total_violation_num = 0;
};

struct patch_num_map
{
  std::map<int32_t, int32_t> routing_patch_num_map;
  int32_t total_patch_num = 0;
};

// PA
class PASummary
{
 private:
  access_point_num_map access_point_num;

 public:
  PASummary() = default;
  ~PASummary() = default;
  access_point_num_map& getAccessPointNum() { return access_point_num; }
  // setter
  void setAccessPointNum(access_point_num_map&& access_point_num) { this->access_point_num = std::move(access_point_num); }
  void setRoutingAccessPointNumMap(std::map<int32_t, int32_t>&& routing_access_point_num_map)
  {
    access_point_num.routing_access_point_num_map = std::move(routing_access_point_num_map);
  }
  void setRoutingAccessPointNumMap(const int32_t key, const int32_t value)
  {
    access_point_num.routing_access_point_num_map[key] = value;
  }
  void setTypeAccessPointNumMap(std::map<std::string, int32_t>&& type_access_point_num_map)
  {
    access_point_num.type_access_point_num_map = std::move(type_access_point_num_map);
  }
  void setTypeAccessPointNumMap(const std::string key, const int32_t value)
  {
    access_point_num.type_access_point_num_map[key] = value;
  }
  void setTotalAccessPointNum(int32_t num) { access_point_num.total_access_point_num = num; }
};

// SA
class SASummary
{
 private:
  routing_supply_map supply;

 public:
  SASummary() = default;
  ~SASummary() = default;
  routing_supply_map& getSupply() { return supply; }
  // setter
  void setSupply(routing_supply_map&& supply) { this->supply = std::move(supply); }
  void setRoutingSupplyMap(std::map<int32_t, int32_t>&& routing_supply_map)
  {
    supply.routing_supply_map = std::move(routing_supply_map);
  }
  void setRoutingSupplyMap(const int32_t key, const int32_t value)
  {
    supply.routing_supply_map[key] = value;
  }
  void setTotalSupply(int32_t num) { supply.total_supply = num; }
};

// IR
class IRSummary
{
 private:
  demand_map demand;
  overflow_map overflow;
  wire_length_map wire_length;
  cut_via_num_map cut_via_num;
  timing_map timing;

 public:
  IRSummary() = default;
  ~IRSummary() = default;
  demand_map& getDemand() { return demand; }
  overflow_map& getOverflow() { return overflow; }
  wire_length_map& getWireLength() { return wire_length; }
  cut_via_num_map& getCutViaNum() { return cut_via_num; }
  timing_map& getTiming() { return timing; }
  // setter
  void setDemand(demand_map&& demand) { this->demand = std::move(demand); }
  void setRoutingDemandMap(std::map<int32_t, int32_t>&& routing_demand_map)
  {
    demand.routing_demand_map = std::move(routing_demand_map);
  }
  void setRoutingDemandMap(const int32_t key, const int32_t value)
  {
    demand.routing_demand_map[key] = value;
  }
  void setRoutingDemandNum(int32_t num) { demand.total_demand = num; }

  void setOverflow(overflow_map&& overflow) { this->overflow = std::move(overflow); }
  void setRoutingOverflowMap(std::map<int32_t, int32_t>&& routing_overflow_map)
  {
    overflow.routing_overflow_map = std::move(routing_overflow_map);
  }
  void setRoutingOverflowMap(const int32_t key, const int32_t value)
  {
    overflow.routing_overflow_map[key] = value;
  }
  void setRoutingOverflowNum(int32_t num) { overflow.total_overflow = num; }

  void setWireLength(wire_length_map&& wire_length) { this->wire_length = std::move(wire_length); }
  void setRoutingWireLengthMap(std::map<int32_t, double>&& routing_wire_length_map)
  {
    wire_length.routing_wire_length_map = std::move(routing_wire_length_map);
  }
  void setRoutingWireLengthMap(const int32_t key, const double value)
  {
    wire_length.routing_wire_length_map[key] = value;
  }
  void setRoutingWireLengthNum(int32_t num) { wire_length.total_wire_length = num; }

  void setCutViaNum(cut_via_num_map&& via_num) { this->cut_via_num = std::move(via_num); }
  void setCutViaNumMap(std::map<int32_t, int32_t>&& cut_via_num_map)
  {
    cut_via_num.cut_via_num_map = std::move(cut_via_num_map);
  }
  void setCutViaNumMap(const int32_t key, const int32_t value)
  {
    cut_via_num.cut_via_num_map[key] = value;
  }
  void setCutViaNumNum(int32_t num) { cut_via_num.total_via_num = num; }

  void setTiming(timing_map&& timing) { this->timing = std::move(timing); }
  void setTimingMap(std::map<std::string, std::vector<double>>&& timing) { this->timing.timing = std::move(timing); }
  void setTimingMap(const std::string key, const std::vector<double> value) { timing.timing[key] = value; }
};

// GR
class GRSummary
{
 private:
  demand_map demand;
  overflow_map overflow;
  wire_length_map wire_length;
  cut_via_num_map cut_via_num;
  timing_map timing;

 public:
  GRSummary() = default;
  ~GRSummary() = default;
  demand_map& getDemand() { return demand; }
  overflow_map& getOverflow() { return overflow; }
  wire_length_map& getWireLength() { return wire_length; }
  cut_via_num_map& getCutViaNum() { return cut_via_num; }
  timing_map& getTiming() { return timing; }
  // setter
  void setDemand(demand_map&& demand) { this->demand = std::move(demand); }
  void setRoutingDemandMap(std::map<int32_t, int32_t>&& routing_demand_map)
  {
    demand.routing_demand_map = std::move(routing_demand_map);
  }
  void setRoutingDemandMap(const int32_t key, const int32_t value)
  {
    demand.routing_demand_map[key] = value;
  }
  void setRoutingDemandNum(int32_t num) { demand.total_demand = num; }

  void setOverflow(overflow_map&& overflow) { this->overflow = std::move(overflow); }
  void setRoutingOverflowMap(std::map<int32_t, int32_t>&& routing_overflow_map)
  {
    overflow.routing_overflow_map = std::move(routing_overflow_map);
  }
  void setRoutingOverflowMap(const int32_t key, const int32_t value)
  {
    overflow.routing_overflow_map[key] = value;
  }
  void setRoutingOverflowNum(int32_t num) { overflow.total_overflow = num; }

  void setWireLength(wire_length_map&& wire_length) { this->wire_length = std::move(wire_length); }
  void setRoutingWireLengthMap(std::map<int32_t, double>&& routing_wire_length_map)
  {
    wire_length.routing_wire_length_map = std::move(routing_wire_length_map);
  }
  void setRoutingWireLengthMap(const int32_t key, const double value)
  {
    wire_length.routing_wire_length_map[key] = value;
  }
  void setRoutingWireLengthNum(int32_t num) { wire_length.total_wire_length = num; }

  void setCutViaNum(cut_via_num_map&& via_num) { this->cut_via_num = std::move(via_num); }
  void setCutViaNumMap(std::map<int32_t, int32_t>&& cut_via_num_map)
  {
    cut_via_num.cut_via_num_map = std::move(cut_via_num_map);
  }
  void setCutViaNumMap(const int32_t key, const int32_t value)
  {
    cut_via_num.cut_via_num_map[key] = value;
  }
  void setCutViaNumNum(int32_t num) { cut_via_num.total_via_num = num; }

  void setTiming(timing_map&& timing) { this->timing = std::move(timing); }
  void setTimingMap(std::map<std::string, std::vector<double>>&& timing) { this->timing.timing = std::move(timing); }
  void setTimingMap(const std::string key, const std::vector<double> value) { timing.timing[key] = value; }
};

// TA
class TASummary
{
 private:
  wire_length_map wire_length;
  violation_num_map violation_num;

 public:
  TASummary() = default;
  ~TASummary() = default;
  wire_length_map& getWireLength() { return wire_length; }
  violation_num_map& getViolationNum() { return violation_num; }
  // setter
  void setWireLength(wire_length_map&& wire_length) { this->wire_length = std::move(wire_length); }
  void setRoutingWireLengthMap(std::map<int32_t, double>&& routing_wire_length_map)
  {
    wire_length.routing_wire_length_map = std::move(routing_wire_length_map);
  }
  void setRoutingWireLengthMap(const int32_t key, const double value)
  {
    wire_length.routing_wire_length_map[key] = value;
  }
  void setRoutingWireLengthNum(int32_t num) { wire_length.total_wire_length = num; }

  void setViolationNum(violation_num_map&& violation_num) { this->violation_num = std::move(violation_num); }
  void setRoutingViolationNumMap(std::map<int32_t, int32_t>&& routing_violation_num_map)
  {
    violation_num.routing_violation_num_map = std::move(routing_violation_num_map);
  }
  void setRoutingViolationNumMap(const int32_t key, const int32_t value)
  {
    violation_num.routing_violation_num_map[key] = value;
  }
  void setRoutingViolationNumNum(int32_t num) { violation_num.total_violation_num = num;}
};

// DR
class DRSummary
{
 private:
  wire_length_map wire_length;
  cut_via_num_map cut_via_num;
  violation_num_map violation_num;
  patch_num_map patch_num;
  timing_map timing;

 public:
  DRSummary() = default;
  ~DRSummary() = default;
  cut_via_num_map& getCutViaNum() { return cut_via_num; }
  wire_length_map& getWireLength() { return wire_length; }
  violation_num_map& getViolationNum() { return violation_num; }
  patch_num_map& getPatchNum() { return patch_num; }
  timing_map& getTiming() { return timing; }
  // setter
  void setWireLength(wire_length_map&& wire_length) { this->wire_length = std::move(wire_length); }
  void setRoutingWireLengthMap(std::map<int32_t, double>&& routing_wire_length_map)
  {
    wire_length.routing_wire_length_map = std::move(routing_wire_length_map);
  }
  void setRoutingWireLengthMap(const int32_t key, const double value)
  {
    wire_length.routing_wire_length_map[key] = value;
  }
  void setRoutingWireLengthNum(int32_t num) { wire_length.total_wire_length = num; }

  void setCutViaNum(cut_via_num_map&& cut_via_num) { this->cut_via_num = std::move(cut_via_num); }
  void setCutViaNumMap(std::map<int32_t, int32_t>&& cut_via_num_map)
  {
    cut_via_num.cut_via_num_map = std::move(cut_via_num_map);
  }
  void setCutViaNumMap(const int32_t key, const int32_t value)
  {
    cut_via_num.cut_via_num_map[key] = value;
  }
  void setCutViaNumNum(int32_t num) { cut_via_num.total_via_num = num; }

  void setViolationNum(violation_num_map&& violation_num) { this->violation_num = std::move(violation_num); }
  void setRoutingViolationNumMap(std::map<int32_t, int32_t>&& routing_violation_num_map)
  {
    violation_num.routing_violation_num_map = std::move(routing_violation_num_map);
  }
  void setRoutingViolationNumMap(const int32_t key, const int32_t value)
  {
    violation_num.routing_violation_num_map[key] = value;
  }
  void setRoutingViolationNumNum(int32_t num) { violation_num.total_violation_num = num; }

  void setPatchNum(patch_num_map&& patch_num) { this->patch_num = std::move(patch_num); }
  void setRoutingPatchNumMap(std::map<int32_t, int32_t>&& routing_patch_num_map)
  {
    patch_num.routing_patch_num_map = std::move(routing_patch_num_map);
  }
  void setRoutingPatchNumMap(const int32_t key, const int32_t value)
  {
    patch_num.routing_patch_num_map[key] = value;
  }
  void setRoutingPatchNumNum(int32_t num) { patch_num.total_patch_num = num; }

  void setTiming(timing_map&& timing) { this->timing = std::move(timing); }
  void setTimingMap(std::map<std::string, std::vector<double>>&& timing) { this->timing.timing = std::move(timing); }
  void setTimingMap(const std::string key, const std::vector<double> value) { timing.timing[key] = value; }
};

// RT
class RTSummary
{
 private:
  PASummary pa_summary;
  SASummary sa_summary;
  IRSummary ir_summary;
  GRSummary gr_summary;
  TASummary ta_summary;
  DRSummary dr_summary;
  std::map<int32_t, DRSummary> iter_dr_summary_map;

 public:
  RTSummary() = default;
  ~RTSummary() = default;
  PASummary& get_pa_summary() { return pa_summary; }
  SASummary& get_sa_summary() { return sa_summary; }
  IRSummary& get_ir_summary() { return ir_summary; }
  GRSummary& get_gr_summary() { return gr_summary; }
  TASummary& get_ta_summary() { return ta_summary; }
  DRSummary& get_dr_summary() { return dr_summary; }
  std::map<int32_t, DRSummary>& get_iter_dr_summary_map() { return iter_dr_summary_map; }
  // setter
  void setPASummary(const PASummary& pa_summary) { this->pa_summary = pa_summary; }
  void setSASummary(const SASummary& sa_summary) { this->sa_summary = sa_summary; }
  void setIRSummary(const IRSummary& ir_summary) { this->ir_summary = ir_summary; }
  void setGRSummary(const GRSummary& gr_summary) { this->gr_summary = gr_summary; }
  void setTASummary(const TASummary& ta_summary) { this->ta_summary = ta_summary; }
  void setDRSummary(const DRSummary& dr_summary) { this->dr_summary = dr_summary; }
  void setIterDRSummaryMap(const std::map<int32_t, DRSummary>& iter_dr_summary_map) { this->iter_dr_summary_map = iter_dr_summary_map; }
  void setIterDRSummaryMap(const int32_t key, const DRSummary value) { iter_dr_summary_map[key] = value; }
};

}  // namespace idb
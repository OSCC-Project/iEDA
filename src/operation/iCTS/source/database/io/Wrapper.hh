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
 * @file Wrapper.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-11
 * @brief DB wrapper for iCTS
 */

#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "spatial/Point.hh"

namespace idb {
class IdbBuilder;
class IdbDesign;
class IdbInstance;
class IdbLayout;
class IdbNet;
class IdbPin;
template <typename T>
class IdbCoordinate;
}  // namespace idb

namespace idb {
class LibCell;
class LibLibrary;
}  // namespace idb

namespace icts {

class Clock;
class Config;
class Design;
class Inst;
class Net;
class Pin;
class SchemaWriter;
struct SdcClockTraceInput;
struct ClockTraceBuild;
struct ClockTraceClockTarget;
struct ClockRouteSegmentRc;
struct WrapperCellGeometry
{
  std::string name;
  std::string cell_master;
  Point<int> origin = Point<int>(-1, -1);
  int32_t width_dbu = 0;
  int32_t height_dbu = 0;
};

struct WrapperWriteSummary
{
  bool success = false;
  std::string failed_clock;
  std::string failed_net;
  bool idb_clock_tree_restored = false;
  std::string reason;
};

class Wrapper
{
 public:
  Wrapper();
  ~Wrapper();

  // Delete copy and move constructors
  Wrapper(const Wrapper& rhs) = delete;
  Wrapper(Wrapper&& rhs) = delete;
  auto operator=(const Wrapper& rhs) -> Wrapper& = delete;
  auto operator=(Wrapper&& rhs) -> Wrapper& = delete;

  // Initialize with idb builder
  auto init(idb::IdbBuilder* idb) -> void;

  // Reset wrapper
  auto reset() -> void;

  struct RootDriverCost
  {
    bool valid = false;
    std::string method;
    std::string cell_master;
    double input_slew_ns = 0.0;
    double output_load_pf = 0.0;
    double cell_delay_ns = 0.0;
    double output_slew_ns = 0.0;
    double internal_power_w = 0.0;
    double leakage_power_w = 0.0;
    double cell_power_w = 0.0;
  };

  struct ClockSourceDriveCapLimitInput
  {
    const Pin* clock_source = nullptr;
    std::optional<double> configured_max_cap_pf = std::nullopt;
  };

  struct PinSlewLimitInput
  {
    const Pin* pin = nullptr;
    double configured_max_sink_tran_ns = 0.0;
  };

  auto queryDbUnit() const -> int32_t;
  auto is_design_ready() const -> bool { return _idb_design != nullptr; }
  auto is_layout_ready() const -> bool { return _idb != nullptr && _idb_layout != nullptr; }
  auto queryWireResistance(int routing_layer, double length_um, std::optional<double> wire_width_um = std::nullopt) const -> double;
  auto queryWireCapacitance(int routing_layer, double length_um, std::optional<double> wire_width_um = std::nullopt) const -> double;
  auto queryRequiredWireResistance(int routing_layer, double length_um, std::optional<double> wire_width_um = std::nullopt) const -> double;
  auto queryRequiredWireCapacitance(int routing_layer, double length_um, std::optional<double> wire_width_um = std::nullopt) const
      -> double;
  auto queryConfiguredClockRouteSegmentRc(const Config& config) const -> ClockRouteSegmentRc;
  auto queryCellOutPinCapLimit(const std::string& cell_master) const -> double;
  auto queryCellOutPinCapTableAxisMax(const std::string& cell_master) const -> double;
  auto queryClockSourceDriveCapLimit(const ClockSourceDriveCapLimitInput& input) const -> double;
  auto queryClockSourceDriveCapLimit(const Config& config, const Pin* clock_source) const -> double;
  auto queryCellInPinSlewLimit(const std::string& cell_master) const -> double;
  auto queryCellInPinSlewTableAxisMax(const std::string& cell_master) const -> double;
  auto queryCellHeightUm(const std::string& cell_master) const -> double;
  auto queryCellAreaUm2(const std::string& cell_master) const -> double;
  auto queryCharInputPinCap(const std::string& cell_master) const -> double;
  auto queryPinCapacitance(const Pin* pin) const -> double;
  auto queryPinSlewLimit(const PinSlewLimitInput& input) const -> double;
  auto queryPinSlewLimit(const Config& config, const Pin* pin) const -> double;
  auto queryRootDriverCostDirect(const std::string& cell_master, double input_slew_ns, double output_load_pf, double clock_period_ns) const
      -> RootDriverCost;
  auto queryBufferPorts(const std::string& cell_master) const -> std::pair<std::string, std::string>;
  auto emitConfiguredUnitWireRcReport(SchemaWriter& reporter, const Config& config, const std::string& title) const -> void;
  auto findLibertyCell(const std::string& cell_master) const -> idb::LibCell*;

  // Setter
  auto set_idb_design(idb::IdbDesign* design) -> void { _idb_design = design; }
  auto set_idb_layout(idb::IdbLayout* layout) -> void { _idb_layout = layout; }

  // Interface
  auto read(Design& design, SchemaWriter& reporter) -> void;
  auto readClocks(Design& design, SchemaWriter& reporter, const std::vector<std::pair<std::string, std::string>>& clock_net_pairs) -> bool;
  auto readTraceClockTargets(Design& design, SchemaWriter& reporter, const std::vector<ClockTraceClockTarget>& clock_targets) -> bool;
  auto traceSdcClocks(const SdcClockTraceInput& input) const -> ClockTraceBuild;
  auto writeClock(Design& design, SchemaWriter& reporter, Clock& clock) -> bool;
  auto writeClocksDetailed(Design& design, SchemaWriter& reporter, const std::vector<Clock*>& clocks) -> WrapperWriteSummary;
  auto writeClocks(Design& design, SchemaWriter& reporter, const std::vector<Clock*>& clocks) -> bool;
  auto collectLogicCellGeometries() const -> std::vector<WrapperCellGeometry>;
  auto queryInstGeometry(const std::string& inst_name) const -> std::optional<WrapperCellGeometry>;
  auto withinCore(int32_t point_x, int32_t point_y) const -> bool;

 private:
  class CtsClockReader;
  class CtsClockIdbWriter;
  friend class CtsClockReader;
  friend class CtsClockIdbWriter;

  // DB to CTS
  static auto idbToCts(idb::IdbCoordinate<int32_t>& coord) -> Point<int>;

  // CTS to DB
  static auto ctsToIdb(const Point<int>& loc) -> idb::IdbCoordinate<int32_t>;

  // cross reference
  auto crossRef(idb::IdbPin* idb_pin, Pin* cts_pin) -> void
  {
    _idb2cts_pin_map[idb_pin] = cts_pin;
    _cts2idb_pin_map[cts_pin] = idb_pin;
  }
  auto crossRef(idb::IdbInstance* idb_inst, Inst* cts_inst) -> void
  {
    _idb2cts_inst_map[idb_inst] = cts_inst;
    _cts2idb_inst_map[cts_inst] = idb_inst;
  }
  auto crossRef(idb::IdbNet* idb_net, Net* cts_net) -> void
  {
    _idb2cts_net_map[idb_net] = cts_net;
    _cts2idb_net_map[cts_net] = idb_net;
  }

  auto loadLibertyIfNeeded() const -> void;

  idb::IdbBuilder* _idb = nullptr;
  idb::IdbDesign* _idb_design = nullptr;
  idb::IdbLayout* _idb_layout = nullptr;
  mutable bool _liberty_loaded = false;
  mutable std::vector<std::unique_ptr<idb::LibLibrary>> _lib_libraries;
  mutable std::unordered_map<std::string, idb::LibCell*> _lib_cell_by_master;

  std::unordered_map<Inst*, idb::IdbInstance*> _cts2idb_inst_map;
  std::unordered_map<idb::IdbInstance*, Inst*> _idb2cts_inst_map;

  std::unordered_map<Net*, idb::IdbNet*> _cts2idb_net_map;
  std::unordered_map<idb::IdbNet*, Net*> _idb2cts_net_map;

  std::unordered_map<Pin*, idb::IdbPin*> _cts2idb_pin_map;
  std::unordered_map<idb::IdbPin*, Pin*> _idb2cts_pin_map;
};

}  // namespace icts

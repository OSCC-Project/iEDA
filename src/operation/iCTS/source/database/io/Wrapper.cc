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
 * @file Wrapper.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-11
 * @brief DB wrapper for iCTS
 */
#include "Wrapper.hh"

#include <glog/logging.h>

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "IdbCellMaster.h"
#include "IdbCore.h"
#include "IdbDesign.h"
#include "IdbGeometry.h"
#include "IdbInstance.h"
#include "IdbLayout.h"
#include "IdbUnits.h"
#include "Log.hh"
#include "adapter/sdc/SdcClockReader.hh"
#include "builder.h"
#include "def_service.h"
#include "design/Clock.hh"
#include "design/Design.hh"
#include "lef_service.h"
#include "liberty/Lib.hh"
#include "spatial/Point.hh"

namespace icts {
namespace {

auto BuildCellGeometry(idb::IdbInstance* idb_inst) -> WrapperCellGeometry
{
  WrapperCellGeometry geometry;
  if (idb_inst == nullptr) {
    return geometry;
  }

  auto* cell_master = idb_inst->get_cell_master();
  auto* bbox = idb_inst->get_bounding_box();
  auto* coordinate = idb_inst->get_coordinate();
  geometry.name = idb_inst->get_name();
  geometry.cell_master = cell_master == nullptr ? std::string{} : cell_master->get_name();
  if (bbox != nullptr) {
    geometry.origin = Point<int>(bbox->get_low_x(), bbox->get_low_y());
    geometry.width_dbu = bbox->get_width();
    geometry.height_dbu = bbox->get_height();
  } else if (coordinate != nullptr) {
    geometry.origin = Point<int>(coordinate->get_x(), coordinate->get_y());
    if (cell_master != nullptr) {
      geometry.width_dbu = static_cast<int32_t>(cell_master->get_width());
      geometry.height_dbu = static_cast<int32_t>(cell_master->get_height());
    }
  }
  return geometry;
}

}  // namespace

Wrapper::Wrapper() = default;

Wrapper::~Wrapper() = default;

auto Wrapper::init(idb::IdbBuilder* idb) -> void
{
  _idb = idb;
  _idb_design = _idb->get_def_service()->get_design();
  _idb_layout = _idb->get_lef_service()->get_layout();
}

auto Wrapper::reset() -> void
{
  _idb = nullptr;
  _idb_design = nullptr;
  _idb_layout = nullptr;
  _liberty_loaded = false;
  _lib_libraries.clear();
  _lib_cell_by_master.clear();
  _cts2idb_inst_map.clear();
  _idb2cts_inst_map.clear();
  _cts2idb_net_map.clear();
  _idb2cts_net_map.clear();
  _cts2idb_pin_map.clear();
  _idb2cts_pin_map.clear();
}

auto Wrapper::queryDbUnit() const -> int32_t
{
  if (_idb_design == nullptr || _idb_design->get_units() == nullptr) {
    LOG_ERROR << "iDB design units are not ready in Wrapper.";
    return 0;
  }
  return _idb_design->get_units()->get_micron_dbu();
}

auto Wrapper::withinCore(int32_t point_x, int32_t point_y) const -> bool
{
  if (_idb_layout == nullptr) {
    LOG_WARNING << "iDB layout is null when checking core boundary; treating point as inside core.";
    return true;
  }
  auto* core = _idb_layout->get_core();
  if (core == nullptr || core->get_bounding_box() == nullptr) {
    LOG_WARNING << "iDB core boundary is unavailable; treating point as inside core.";
    return true;
  }
  auto* core_box = core->get_bounding_box();
  return point_x >= core_box->get_low_x() && point_x <= core_box->get_high_x() && point_y >= core_box->get_low_y()
         && point_y <= core_box->get_high_y();
}

auto Wrapper::read(Design& design, SchemaWriter& reporter) -> void
{
  std::vector<std::pair<std::string, std::string>> clock_net_pairs;
  for (const auto* clock : design.get_clocks()) {
    if (clock == nullptr) {
      continue;
    }
    clock_net_pairs.emplace_back(clock->get_clock_name(), clock->get_clock_net_name());
  }
  if (!readClocks(design, reporter, clock_net_pairs)) {
    LOG_WARNING << "Wrapper read failed while materializing predeclared CTS clocks.";
  }
}

auto Wrapper::idbToCts(idb::IdbCoordinate<int32_t>& coord) -> Point<int>
{
  return {coord.get_x(), coord.get_y()};
}

auto Wrapper::ctsToIdb(const Point<int>& loc) -> idb::IdbCoordinate<int32_t>
{
  return {loc.get_x(), loc.get_y()};
}

auto Wrapper::traceSdcClocks(const SdcClockTraceInput& input) const -> ClockTraceBuild
{
  LOG_FATAL_IF(input.clock_data == nullptr) << "Wrapper: SDC clock trace requires clock data.";
  LOG_FATAL_IF(input.reporter == nullptr) << "Wrapper: SDC clock trace requires reporter.";
  auto* idb_design = _idb_design;
  if (idb_design == nullptr && _idb != nullptr && _idb->get_def_service() != nullptr) {
    idb_design = _idb->get_def_service()->get_design();
  }
  const SdcLibertyCellLookup liberty_cell_lookup
      = [this](const std::string& cell_master) -> idb::LibCell* { return findLibertyCell(cell_master); };
  return SdcClockReader::traceClockTargets(*input.clock_data, idb_design, liberty_cell_lookup, input.max_fanout, *input.reporter);
}

auto Wrapper::collectLogicCellGeometries() const -> std::vector<WrapperCellGeometry>
{
  std::vector<WrapperCellGeometry> geometries;
  if (_idb_design == nullptr || _idb_design->get_instance_list() == nullptr) {
    LOG_WARNING << "Cannot collect iDB logic-cell geometry: iDB design or instance list is not ready.";
    return geometries;
  }

  const auto& idb_insts = _idb_design->get_instance_list()->get_instance_list();
  geometries.reserve(idb_insts.size());
  for (auto* idb_inst : idb_insts) {
    if (idb_inst == nullptr || idb_inst->is_physical()) {
      continue;
    }
    auto* cell_master = idb_inst->get_cell_master();
    if (cell_master == nullptr || !cell_master->is_logic()) {
      continue;
    }
    if (idb_inst->is_clock_instance()) {
      continue;
    }
    geometries.push_back(BuildCellGeometry(idb_inst));
  }
  return geometries;
}

auto Wrapper::queryInstGeometry(const std::string& inst_name) const -> std::optional<WrapperCellGeometry>
{
  if (inst_name.empty()) {
    return std::nullopt;
  }
  if (_idb_design == nullptr || _idb_design->get_instance_list() == nullptr) {
    LOG_WARNING << "Cannot query iDB inst geometry: iDB design or instance list is not ready.";
    return std::nullopt;
  }

  auto* idb_inst = _idb_design->get_instance_list()->find_instance(inst_name);
  if (idb_inst == nullptr) {
    return std::nullopt;
  }
  return BuildCellGeometry(idb_inst);
}

}  // namespace icts

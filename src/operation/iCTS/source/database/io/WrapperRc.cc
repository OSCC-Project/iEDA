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
 * @file WrapperRc.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-29
 * @brief Wrapper-backed wire RC queries for iCTS.
 */

#include <glog/logging.h>

#include <cmath>
#include <cstddef>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "IdbLayer.h"
#include "IdbLayout.h"
#include "IdbUnits.h"
#include "Log.hh"
#include "Wrapper.hh"
#include "config/Config.hh"
#include "logger/LogFormat.hh"
#include "logger/Schema.hh"
#include "routing/ClockRouteSegmentRc.hh"

namespace icts {
namespace {

constexpr double kMilliOhmPerOhm = 1000.0;

struct WireRcProbe
{
  int routing_layer = 0;
  std::optional<double> wire_width_um = std::nullopt;
  double query_length_um = 1.0;
  double resistance_per_um_ohm = 0.0;
  double capacitance_per_um_pf = 0.0;
  bool queried = false;
  bool has_diagnostic = false;
  DiagnosticLevel diagnostic_level = DiagnosticLevel::kInfo;
  std::string diagnostic_summary;
  std::string status = "not_queried";
  std::string detail;
};

auto formatOptionalWireWidth(std::optional<double> wire_width_um) -> std::string
{
  return wire_width_um.has_value() ? logformat::FormatWithUnit(*wire_width_um, "um") : "library_default";
}

auto requireLayout(idb::IdbLayout* layout, const char* metric) -> idb::IdbLayout*
{
  LOG_FATAL_IF(layout == nullptr) << "Wrapper: iDB layout is unavailable for required wire " << metric << " query.";
  LOG_FATAL_IF(layout->get_units() == nullptr) << "Wrapper: iDB units are unavailable for required wire " << metric << " query.";
  LOG_FATAL_IF(layout->get_layers() == nullptr) << "Wrapper: iDB layers are unavailable for required wire " << metric << " query.";
  return layout;
}

auto requireWireRcQuery(const char* metric, int routing_layer, double length_um) -> void
{
  LOG_FATAL_IF(routing_layer <= 0) << "Wrapper: required wire " << metric << " query needs a positive routing layer.";
  LOG_FATAL_IF(!std::isfinite(length_um) || length_um < 0.0)
      << "Wrapper: required wire " << metric << " query needs a finite non-negative length, got " << length_um << " um.";
}

auto resolveRoutingLayer(const Config& config) -> int
{
  const auto& routing_layers = config.get_routing_layers();
  if (routing_layers.empty()) {
    LOG_ERROR << "Wrapper: routing layer is not configured.";
    return 0;
  }
  return static_cast<int>(routing_layers.front());
}

auto requireRoutingLayer(const Config& config) -> int
{
  const auto& routing_layers = config.get_routing_layers();
  LOG_FATAL_IF(routing_layers.empty() || routing_layers.front() == 0U)
      << "Wrapper: routing layer is not configured for clock route segment RC.";
  return static_cast<int>(routing_layers.front());
}

auto resolveWireWidth(const Config& config) -> std::optional<double>
{
  const double wire_width_um = config.get_wire_width();
  return wire_width_um > 0.0 ? std::optional<double>{wire_width_um} : std::nullopt;
}

auto queryRoutingLayer(idb::IdbLayout* layout, int routing_layer, const char* metric) -> idb::IdbLayerRouting*
{
  layout = requireLayout(layout, metric);
  auto& routing_layers = layout->get_layers()->get_routing_layers();
  const int routing_layer_id = routing_layer - 1;
  LOG_FATAL_IF(routing_layer_id < 0) << "Wrapper: routing layer " << routing_layer << " is out of range for required wire " << metric
                                     << " query.";
  const auto layer_index = static_cast<std::size_t>(routing_layer_id);
  LOG_FATAL_IF(layer_index >= routing_layers.size())
      << "Wrapper: routing layer " << routing_layer << " is out of range for required wire " << metric << " query.";
  auto* layer = dynamic_cast<idb::IdbLayerRouting*>(routing_layers.at(layer_index));
  LOG_FATAL_IF(layer == nullptr) << "Wrapper: routing layer " << routing_layer << " is not a routing layer.";
  return layer;
}

auto resolveWidthUm(idb::IdbLayout* layout, idb::IdbLayerRouting* layer, std::optional<double> wire_width_um) -> double
{
  if (wire_width_um.has_value()) {
    return *wire_width_um;
  }
  layout = requireLayout(layout, "width");
  const auto dbu_per_um = layout->get_units()->get_micron_dbu();
  LOG_FATAL_IF(dbu_per_um <= 0) << "Wrapper: DBU-per-micron is invalid when resolving routing layer width.";
  return static_cast<double>(layer->get_width()) / static_cast<double>(dbu_per_um);
}

auto queryWireRcProbe(const Wrapper& wrapper, int routing_layer, std::optional<double> wire_width_um) -> WireRcProbe
{
  WireRcProbe probe;
  probe.routing_layer = routing_layer;
  probe.wire_width_um = wire_width_um;

  if (routing_layer <= 0) {
    probe.has_diagnostic = true;
    probe.diagnostic_level = DiagnosticLevel::kError;
    probe.diagnostic_summary = "effective routing layer is invalid for unit RC probing.";
    probe.status = "invalid_layer";
    probe.detail = "routing layer must be positive";
    return probe;
  }

  if (!wrapper.is_layout_ready()) {
    probe.has_diagnostic = true;
    probe.diagnostic_level = DiagnosticLevel::kError;
    probe.diagnostic_summary = "Wrapper iDB layout is not ready for unit RC probing.";
    probe.status = "adapter_unavailable";
    probe.detail = "Wrapper must be initialized before RC probing";
    return probe;
  }

  probe.queried = true;
  probe.resistance_per_um_ohm = wrapper.queryWireResistance(routing_layer, probe.query_length_um, wire_width_um) / kMilliOhmPerOhm;
  probe.capacitance_per_um_pf = wrapper.queryWireCapacitance(routing_layer, probe.query_length_um, wire_width_um);

  if (!std::isfinite(probe.resistance_per_um_ohm) || !std::isfinite(probe.capacitance_per_um_pf)) {
    probe.has_diagnostic = true;
    probe.diagnostic_level = DiagnosticLevel::kError;
    probe.diagnostic_summary = "unit wire RC query returned non-finite values.";
    probe.status = "invalid_nonfinite";
    probe.detail = "queried unit RC must be finite";
    return probe;
  }

  if (probe.resistance_per_um_ohm < 0.0 || probe.capacitance_per_um_pf < 0.0) {
    probe.has_diagnostic = true;
    probe.diagnostic_level = DiagnosticLevel::kError;
    probe.diagnostic_summary = "unit wire RC query returned negative values.";
    probe.status = "invalid_negative";
    probe.detail = "negative resistance/capacitance is physically invalid";
    return probe;
  }

  if (probe.resistance_per_um_ohm == 0.0 || probe.capacitance_per_um_pf == 0.0) {
    probe.has_diagnostic = true;
    probe.diagnostic_level = DiagnosticLevel::kWarning;
    probe.diagnostic_summary = "unit wire RC query returned zero on at least one metric.";
    probe.status = "warning_zero";
    probe.detail = "zero unit RC is suspicious; flow continues";
    return probe;
  }

  probe.status = "ok";
  probe.detail = "positive finite unit RC";
  return probe;
}

auto emitWireRcProbeDiagnostic(SchemaWriter& reporter, const WireRcProbe& probe) -> void
{
  if (!probe.has_diagnostic) {
    return;
  }

  if (probe.diagnostic_level == DiagnosticLevel::kError) {
    LOG_ERROR << "Wrapper: " << probe.diagnostic_summary << " layer=" << probe.routing_layer
              << ", wire_width=" << formatOptionalWireWidth(probe.wire_width_um) << ", resistance="
              << (probe.queried ? logformat::FormatEngineering(probe.resistance_per_um_ohm, "Ohm/um") : std::string{"n/a"})
              << ", capacitance=" << (probe.queried ? logformat::FormatWithUnit(probe.capacitance_per_um_pf, "pF/um") : std::string{"n/a"});
  } else {
    LOG_WARNING << "Wrapper: " << probe.diagnostic_summary << " layer=" << probe.routing_layer
                << ", wire_width=" << formatOptionalWireWidth(probe.wire_width_um) << ", resistance="
                << (probe.queried ? logformat::FormatEngineering(probe.resistance_per_um_ohm, "Ohm/um") : std::string{"n/a"})
                << ", capacitance="
                << (probe.queried ? logformat::FormatWithUnit(probe.capacitance_per_um_pf, "pF/um") : std::string{"n/a"});
  }

  EmitDiagnostic(
      reporter, probe.diagnostic_level, "Wrapper", probe.diagnostic_summary,
      {
          {"routing_setup_source", "Runtime Configuration"},
          {"query_length", logformat::FormatWithUnit(probe.query_length_um, "um")},
          {"unit_resistance", probe.queried ? logformat::FormatEngineering(probe.resistance_per_um_ohm, "Ohm/um") : std::string{"n/a"}},
          {"unit_capacitance", probe.queried ? logformat::FormatWithUnit(probe.capacitance_per_um_pf, "pF/um") : std::string{"n/a"}},
          {"status", probe.status},
      });
}

auto buildWireRcRows(const WireRcProbe& probe) -> logformat::TableRows
{
  return {
      {"routing_setup_source", "Runtime Configuration", "routing layer and wire width are reported once there"},
      {"query_length", logformat::FormatWithUnit(probe.query_length_um, "um"), "single-unit probe length"},
      {"unit_resistance", probe.queried ? logformat::FormatEngineering(probe.resistance_per_um_ohm, "Ohm/um") : std::string{"n/a"},
       "derived from Wrapper::queryWireResistance"},
      {"unit_capacitance", probe.queried ? logformat::FormatWithUnit(probe.capacitance_per_um_pf, "pF/um") : std::string{"n/a"},
       "derived from Wrapper::queryWireCapacitance"},
      {"status", probe.status, probe.detail},
  };
}

}  // namespace

auto Wrapper::queryWireResistance(int routing_layer, double length_um, std::optional<double> wire_width_um) const -> double
{
  if (!is_layout_ready()) {
    LOG_ERROR << "Wrapper: iDB layout is not ready for wire resistance query.";
    return 0.0;
  }
  if (routing_layer <= 0 || !std::isfinite(length_um) || length_um < 0.0) {
    LOG_ERROR << "Wrapper: invalid wire resistance query: layer=" << routing_layer << ", length=" << length_um << " um.";
    return 0.0;
  }
  auto* layer = queryRoutingLayer(_idb_layout, routing_layer, "resistance");
  const double width_um = resolveWidthUm(_idb_layout, layer, wire_width_um);
  return layer->get_resistance() * length_um / width_um;
}

auto Wrapper::queryWireCapacitance(int routing_layer, double length_um, std::optional<double> wire_width_um) const -> double
{
  if (!is_layout_ready()) {
    LOG_ERROR << "Wrapper: iDB layout is not ready for wire capacitance query.";
    return 0.0;
  }
  if (routing_layer <= 0 || !std::isfinite(length_um) || length_um < 0.0) {
    LOG_ERROR << "Wrapper: invalid wire capacitance query: layer=" << routing_layer << ", length=" << length_um << " um.";
    return 0.0;
  }
  auto* layer = queryRoutingLayer(_idb_layout, routing_layer, "capacitance");
  const double width_um = resolveWidthUm(_idb_layout, layer, wire_width_um);
  return (layer->get_capacitance() * length_um * width_um) + (layer->get_edge_capacitance() * 2.0 * (length_um + width_um));
}

auto Wrapper::queryRequiredWireResistance(int routing_layer, double length_um, std::optional<double> wire_width_um) const -> double
{
  requireWireRcQuery("resistance", routing_layer, length_um);
  auto* layer = queryRoutingLayer(_idb_layout, routing_layer, "resistance");
  const double width_um = resolveWidthUm(_idb_layout, layer, wire_width_um);
  return layer->get_resistance() * length_um / width_um;
}

auto Wrapper::queryRequiredWireCapacitance(int routing_layer, double length_um, std::optional<double> wire_width_um) const -> double
{
  requireWireRcQuery("capacitance", routing_layer, length_um);
  auto* layer = queryRoutingLayer(_idb_layout, routing_layer, "capacitance");
  const double width_um = resolveWidthUm(_idb_layout, layer, wire_width_um);
  return (layer->get_capacitance() * length_um * width_um) + (layer->get_edge_capacitance() * 2.0 * (length_um + width_um));
}

auto Wrapper::queryConfiguredClockRouteSegmentRc(const Config& config) const -> ClockRouteSegmentRc
{
  const auto dbu_per_um = queryDbUnit();
  LOG_FATAL_IF(dbu_per_um <= 0) << "Wrapper: DBU-per-micron is unavailable for configured clock route segment RC.";
  const auto routing_layer = requireRoutingLayer(config);
  const auto wire_width_um = resolveWireWidth(config);
  return ClockRouteSegmentRc{
      .dbu_per_um = dbu_per_um,
      .resistance_per_um_ohm = queryRequiredWireResistance(routing_layer, 1.0, wire_width_um) / kMilliOhmPerOhm,
      .capacitance_per_um_pf = queryRequiredWireCapacitance(routing_layer, 1.0, wire_width_um),
  };
}

auto Wrapper::emitConfiguredUnitWireRcReport(SchemaWriter& reporter, const Config& config, const std::string& title) const -> void
{
  const auto probe = queryWireRcProbe(*this, resolveRoutingLayer(config), resolveWireWidth(config));
  emitWireRcProbeDiagnostic(reporter, probe);
  EmitTable(reporter, title, {"Item", "Value", "Detail"}, buildWireRcRows(probe));
}

}  // namespace icts

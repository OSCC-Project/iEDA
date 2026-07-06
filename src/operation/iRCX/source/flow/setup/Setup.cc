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
#include "Setup.hh"

#include <cmath>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <utility>

#include "CapTable.hpp"
#include "ItfBuilder.hpp"
#include "LayerTable.hh"
#include "LayoutData.hh"
#include "RCXConfig.hh"
#include "IdbAdapter.hh"
#include "ProcessCorner.hpp"
#include "RCXData.hh"
#include "SpefContext.hh"
#include "PathUtils.hh"
#include "StageLog.hh"
#include "idm.h"
#include "log/Log.hh"

namespace ircx {

namespace {

auto temperatureName(F64 temperature) -> Str
{
  std::ostringstream oss;
  oss << std::setprecision(12) << temperature;

  Str name = oss.str();
  for (char& ch : name) {
    if (ch == '.') {
      ch = 'p';
    } else if (ch == '-') {
      ch = 'm';
    }
  }
  return name + "C";
}

auto makeTemperatureCornerName(const Str& corner_name, F64 temperature) -> Str
{
  return corner_name + "_" + temperatureName(temperature);
}

auto loadProcessCorner(const Str& corner_name,
                       const Str& itf_file) -> std::unique_ptr<::itf::ProcessCorner>
{
  if (!string::require_non_empty(corner_name, "corner name")) {
    return nullptr;
  }
  if (!path::file_exists(itf_file, "itf_file")) {
    return nullptr;
  }

  ::itf::ItfBuilder itf_builder;
  StageLog stage_log("read_itf corner=" + corner_name + " file=" + itf_file);
  if (!itf_builder.buildItf(itf_file)) {
    LOG_ERROR << "failed to parse ITF file for corner "
              << corner_name << ": " << itf_file;
    return nullptr;
  }

  auto* itf_service = itf_builder.get_itf_service();
  if (!itf_service) {
    LOG_ERROR << "ITF service is null for corner " << corner_name;
    return nullptr;
  }

  auto pcs = itf_service->take_process_corners();
  std::unique_ptr<::itf::ProcessCorner> loaded_corner;
  Size valid_corner_num = 0;
  for (auto& pc : pcs) {
    if (!pc || pc->get_technology().empty()) {
      continue;
    }
    ++valid_corner_num;
    if (!loaded_corner) {
      loaded_corner = std::move(pc);
    }
  }

  if (valid_corner_num != 1 || !loaded_corner) {
    LOG_ERROR << "read_corner expects exactly one process corner in ITF file "
              << itf_file << ", but got " << valid_corner_num;
    return nullptr;
  }

  const Str original_corner_name = loaded_corner->get_technology();
  if (original_corner_name != corner_name) {
    LOG_INFO << "rename process corner "
             << original_corner_name << " -> " << corner_name;
    loaded_corner->set_technology(corner_name);
  }

  stage_log.set_success();
  return loaded_corner;
}

auto loadCapTable(const Str& corner_name,
                  const Str& captab_file) -> std::optional<parser::CapTable>
{
  if (!string::require_non_empty(corner_name, "corner name")) {
    return std::nullopt;
  }
  if (!path::file_exists(captab_file, "captab_file")) {
    return std::nullopt;
  }

  StageLog stage_log("read_captab corner=" + corner_name + " file=" + captab_file);

  parser::CapTable cap_table;
  if (!cap_table.loadFromFile(captab_file)) {
    LOG_ERROR << "failed to load captab for corner "
              << corner_name << ": " << captab_file;
    return std::nullopt;
  }

  stage_log.set_success();
  return cap_table;
}

auto registerProcessLayers(const ::itf::ProcessCorner& pc) -> void
{
  RCXData& data = RCX_DATA_INST;
  if (data.processLayersRegistered()) {
    return;
  }

  auto& cond_layers = pc.get_layers()->get_conductor_layers();
  auto& via_layers = pc.get_layers()->get_via_layers();
  LayerTable& layer_table = data.layer_table();

  for (auto* layer : cond_layers)
    layer_table.registerProcessLayer(layer->get_id(), layer->get_name());
  for (auto* layer : via_layers)
    layer_table.registerProcessLayer(layer->get_id(), layer->get_name());

  data.setProcessLayersRegistered(true);
}

auto validateProcessLayers(const ::itf::ProcessCorner& pc) -> bool
{
  const auto& corner_data = RCX_DATA_INST.corner_data();
  if (corner_data.empty()) {
    return true;
  }

  const ::itf::ProcessCorner& ref = *corner_data.front().process_corner;
  const auto& ref_layers = ref.get_layers()->get_layers();
  const auto& cur_layers = pc.get_layers()->get_layers();

  if (ref_layers.size() != cur_layers.size()) {
    LOG_ERROR << "process layer count mismatch between corner "
              << pc.get_technology() << " and " << ref.get_technology()
              << ": " << cur_layers.size() << " vs " << ref_layers.size();
    return false;
  }

  for (Size idx = 0; idx < ref_layers.size(); ++idx) {
    const auto* ref_layer = ref_layers[idx];
    const auto* cur_layer = cur_layers[idx];

    if (ref_layer == nullptr || cur_layer == nullptr) {
      LOG_ERROR << "null process layer in corner layer list.";
      return false;
    }
    if (ref_layer->get_type() != cur_layer->get_type()) {
      LOG_ERROR << "process layer type mismatch at index " << idx
                << " between corner " << pc.get_technology()
                << " and " << ref.get_technology();
      return false;
    }
    if (ref_layer->get_id() != cur_layer->get_id()) {
      LOG_ERROR << "process layer id mismatch at index " << idx
                << " between corner " << pc.get_technology()
                << " and " << ref.get_technology()
                << ": " << cur_layer->get_id() << " vs " << ref_layer->get_id();
      return false;
    }
    if (ref_layer->get_order() != cur_layer->get_order()) {
      LOG_ERROR << "process layer order mismatch at index " << idx
                << " between corner " << pc.get_technology()
                << " and " << ref.get_technology();
      return false;
    }
    if (ref_layer->get_name() != cur_layer->get_name()) {
      LOG_ERROR << "process layer name mismatch at index " << idx
                << " between corner " << pc.get_technology()
                << " and " << ref.get_technology()
                << ": " << cur_layer->get_name() << " vs " << ref_layer->get_name();
      return false;
    }
  }

  return true;
}

}  // namespace

auto Setup::initialize(const std::string& config) -> bool
{
  RCXConfig& rcx_config = RCX_CONFIG_INST;
  if (!rcx_config.init(config)) {
    return false;
  }

  for (const auto& corner : rcx_config.get_corners()) {
    if (corner.temperatures.empty()) {
      LOG_ERROR << "corner temperature list is empty for " << corner.name << ".";
      return false;
    }

    for (F64 temperature : corner.temperatures) {
      const Str runtime_corner_name = makeTemperatureCornerName(corner.name, temperature);
      if (!readCorner(runtime_corner_name, temperature, corner.itf_file.c_str(), corner.captab_file.c_str())) {
        return false;
      }
    }
  }
  if (!readMapping(rcx_config.get_mapping_file().c_str())) {
    return false;
  }

  rcx_config.set_initialized(true);
  return true;
}

auto Setup::readCorner(const std::string& corner_name,
                       const char* itf_file,
                       const char* captab_file) -> bool
{
  return readCorner(corner_name, kDefaultOperatingTemperature, itf_file, captab_file);
}

auto Setup::readCorner(const std::string& corner_name,
                       F64 temperature,
                       const char* itf_file,
                       const char* captab_file) -> bool
{
  if (!string::require_non_empty(corner_name, "corner name")) {
    return false;
  }
  if (!std::isfinite(temperature)) {
    LOG_ERROR << "corner temperature is invalid for " << corner_name << ".";
    return false;
  }
  if (itf_file == nullptr) {
    LOG_ERROR << "itf_file is null.";
    return false;
  }
  if (captab_file == nullptr) {
    LOG_ERROR << "captab_file is null.";
    return false;
  }

  RCXData& data = RCX_DATA_INST;
  if (data.hasCorner(corner_name)) {
    LOG_ERROR << "duplicate corner: " << corner_name;
    return false;
  }

  RCXData::CornerData corner;
  corner.name = corner_name;
  corner.temperature = temperature;
  corner.itf_file = itf_file;
  corner.captab_file = captab_file;
  corner.process_corner = loadProcessCorner(corner.name, corner.itf_file);
  if (!corner.process_corner) {
    return false;
  }

  auto cap_table = loadCapTable(corner.name, corner.captab_file);
  if (!cap_table.has_value()) {
    return false;
  }
  corner.cap_table = std::move(cap_table.value());

  auto& corner_data = data.corner_data();
  if (corner_data.empty()) {
    registerProcessLayers(*corner.process_corner);
  } else if (!validateProcessLayers(*corner.process_corner)) {
    return false;
  }

  corner_data.push_back(std::move(corner));
  return true;
}

auto Setup::readMapping(const char* mapping_file) -> bool
{
  if (mapping_file == nullptr) {
    LOG_ERROR << "mapping_file is null.";
    return false;
  }
  if (!path::file_exists(mapping_file, "mapping_file")) {
    return false;
  }

  StageLog stage_log("read_mapping file=" + Str(mapping_file));

  RCXData& data = RCX_DATA_INST;
  LayerTable& layer_table = data.layer_table();
  auto& mapping_builder = data.mapping_builder();

  layer_table.clearMappings();
  if (!mapping_builder.read(mapping_file)) {
    return false;
  }

  for (const auto& [dn, pn] : mapping_builder.design_to_process_layer_names())
    layer_table.registerMapping(dn, pn);

  stage_log.set_success();
  return true;
}

auto Setup::adaptDB() -> bool
{
  if (!dmInst) {
    LOG_ERROR << "adapt db failed: dmInst is null.";
    return false;
  }

  auto* idb_builder = dmInst->get_idb_builder();
  if (!idb_builder) {
    LOG_ERROR << "adapt db failed: idb builder is null.";
    return false;
  }

  LayoutData layout_data;
  LayerTable design_layer_table;
  SpefContext spef_context;
  IdbAdapter adapter(idb_builder);

  if (!adapter.adapt(layout_data, design_layer_table, spef_context)) {
    LOG_ERROR << "adapt db failed.";
    return false;
  }

  RCX_DATA_INST.setDBData(std::move(layout_data),
                          design_layer_table,
                          std::move(spef_context));
  return true;
}

}  // namespace ircx

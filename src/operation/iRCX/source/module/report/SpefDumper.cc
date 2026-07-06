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
#include "SpefDumper.hh"

#include <algorithm>
#include <sstream>
#include <utility>

#include "LayerTable.hh"
#include "LayoutData.hh"
#include "ProcessCorner.hpp"
#include "RCXConfig.hh"
#include "SpefContext.hh"
#include "TopoPool.hh"
#include "log/Log.hh"

namespace ircx {

// ─────────────────────────────────────────────────────────────────────────────
// buildNameMaps
// ─────────────────────────────────────────────────────────────────────────────

void SpefDumper::buildNameMaps() const
{
  name_maps_ = NameMaps{};

  // 1. Regular net names
  for (const Str& name : spef_context_->net_names) {
    name_maps_.net_id_to_name[name_maps_.next_id] = name;
    name_maps_.net_name_to_id[name] = name_maps_.next_id;
    name_maps_.next_id++;
  }
  // 2. Ports (IO pins)
  for (const Str& name : spef_context_->port_names) {
    name_maps_.port_id_to_name[name_maps_.next_id] = name;
    name_maps_.port_name_to_id[name] = name_maps_.next_id;
    name_maps_.next_id++;
  }
  // 3. Instance names
  for (const Str& name : spef_context_->instance_names) {
    name_maps_.inst_id_to_name[name_maps_.next_id] = name;
    name_maps_.inst_name_to_id[name] = name_maps_.next_id;
    name_maps_.next_id++;
  }
}

void SpefDumper::buildPortIo() const
{
  port_io_.clear();
  port_io_.reserve(spef_context_->port_names.size());

  for (Size port_idx = 0; port_idx < spef_context_->port_names.size(); ++port_idx) {
    port_io_.emplace(spef_context_->port_names[port_idx], spef_context_->port_io[port_idx]);
  }
}

void SpefDumper::buildNodeSpefNames() const
{
  const auto& nodes = topo_pool_->node_pool();

  node_spef_names_.assign(nodes.size(), Str{});
  for (const TopoNode& node : nodes) {
    node_spef_names_[topo_pool_->node_index(node)] = nodeName(node);
  }
}

void SpefDumper::buildNetSpefNames() const
{
  const auto& net_name_to_id = name_maps_.net_name_to_id;
  const Size net_num = layout_data_->regular_net_count();

  net_spef_names_.assign(net_num, Str{});
  for (Size net_idx = 0; net_idx < net_num; ++net_idx) {
    const Str& net_name = layout_data_->net_vec[net_idx].name;
    auto it = net_name_to_id.find(net_name);
    if (it != net_name_to_id.end()) {
      net_spef_names_[net_idx] = "*" + std::to_string(it->second);
    } else {
      net_spef_names_[net_idx] = net_name;
    }
  }
}

void SpefDumper::buildCouplingRefs(Size corner_idx) const
{
  const auto& edge_pool = topo_pool_->edge_pool();
  const Size net_num = layout_data_->regular_net_count();
  std::vector<Size> coupling_counts(net_num, 0);

  auto for_each_coupling = [&](auto&& fn) {
    for (const auto& [key, cap_vec] : rc_table_->merged_ccap()) {
      if (corner_idx >= cap_vec.size())
        continue;

      const double cap_ff = static_cast<double>(cap_vec[corner_idx]);
      if (cap_ff <= 0.0)
        continue;

      const Size edge_a_id = key.first;
      const Size edge_b_id = key.second;
      if (edge_a_id >= edge_pool.size() || edge_b_id >= edge_pool.size())
        continue;

      const TopoEdge& edge_a = edge_pool[edge_a_id];
      const TopoEdge& edge_b = edge_pool[edge_b_id];
      if (edge_a.net_id() >= net_num || edge_b.net_id() >= net_num)
        continue;

      fn(edge_a.net_id(), edge_a_id, edge_b_id, cap_ff);
      if (edge_b.net_id() != edge_a.net_id()) {
        fn(edge_b.net_id(), edge_b_id, edge_a_id, cap_ff);
      }
    }
  };

  for_each_coupling([&](Size net_idx, Size, Size, double) { ++coupling_counts[net_idx]; });

  net_coupling_refs_.assign(net_num, {});
  for (Size net_idx = 0; net_idx < net_num; ++net_idx) {
    net_coupling_refs_[net_idx].reserve(coupling_counts[net_idx]);
  }

  for_each_coupling([&](Size net_idx, Size self_edge_id, Size other_edge_id, double cap_ff) {
    net_coupling_refs_[net_idx].push_back({self_edge_id, other_edge_id, cap_ff});
  });
}

void SpefDumper::buildReportLayerMap() const
{
  report_layers_.clear();
  design_to_report_layer_level_.clear();

  if (!RCX_CONFIG_INST.get_report_geometry() || layer_table_ == nullptr) {
    return;
  }

  auto layers = layer_table_->design_layers();
  std::sort(layers.begin(), layers.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  auto append_layer = [&](Size design_id, const Str& design_name, Str process_name) {
    const Size report_id = report_layers_.size();
    report_layers_.push_back({report_id, design_id, design_name, std::move(process_name)});
    design_to_report_layer_level_[design_id] = report_id;
  };

  for (const auto& [design_id, design_name] : layers) {
    if (design_id == 0) {
      append_layer(design_id, design_name, {});
      continue;
    }

    try {
      const Size process_id = layer_table_->design_to_process_id(design_id);
      const Str& process_name = layer_table_->process_name(process_id);
      if (!process_name.empty()) {
        append_layer(design_id, design_name, process_name);
      }
    } catch (...) {
      // Unmapped layers such as CT/ACT are internal design layers and are not
      // part of StarRC-style report_geometry layer levels.
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// nodeName
// ─────────────────────────────────────────────────────────────────────────────

Str SpefDumper::nodeName(const TopoNode& node) const
{
  if (node.is_pin_node()) {
    const Str& full_name = node.pin_name();
    // Port pin: no ':' separator
    if (full_name.find(':') == Str::npos) {
      auto it = name_maps_.port_name_to_id.find(full_name);
      if (it != name_maps_.port_name_to_id.end())
        return "*" + std::to_string(it->second);
      // Fallback: keep full name
      return full_name;
    } else {
      // Instance pin: "inst_name:pin_name"
      auto colon = full_name.find(':');
      Str inst_name = full_name.substr(0, colon);
      Str pin_name = full_name.substr(colon + 1);
      auto it = name_maps_.inst_name_to_id.find(inst_name);
      if (it != name_maps_.inst_name_to_id.end())
        return "*" + std::to_string(it->second) + ":" + pin_name;
      // Fallback: keep full name
      return full_name;
    }
  }

  // Internal node: *<net_spef_id>:<node_global_id + 1>
  Size net_id = node.net_id();
  Str net_name = layout_data_->net_vec[net_id].name;
  auto it = name_maps_.net_name_to_id.find(net_name);
  if (it != name_maps_.net_name_to_id.end())
    return "*" + std::to_string(it->second) + ":" + std::to_string(node.id() + 1);
  return "*" + net_name + ":" + std::to_string(node.id() + 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// dump
// ─────────────────────────────────────────────────────────────────────────────

auto SpefDumper::dump(const Str& output_dir) const -> bool
{
  if (corner_data_ == nullptr || corner_data_->empty()) {
    LOG_ERROR << "SpefDumper: process corners not set.";
    return false;
  }

  for (Size corner_idx = 0; corner_idx < corner_data_->size(); ++corner_idx) {
    if (!dumpCorner(output_dir, corner_idx)) {
      return false;
    }
  }

  return true;
}

auto SpefDumper::dumpCorner(const Str& output_dir, Size corner_idx) const -> bool
{
  const auto& corner_data = (*corner_data_)[corner_idx];
  if (corner_data.process_corner == nullptr) {
    LOG_ERROR << "SpefDumper: process corner missing for corner " << corner_data.name;
    return false;
  }

  Str corner_name = corner_data.process_corner->get_technology();
  Str filename = output_dir + "/" + layout_data_->design_name + "_" + corner_name + ".spef";
  std::ofstream ofs(filename);
  if (!ofs) {
    LOG_ERROR << "SpefDumper: cannot open output file: " << filename;
    return false;
  }

  const Size net_count = layout_data_->regular_net_count();

  buildNameMaps();
  buildPortIo();
  buildNodeSpefNames();
  buildNetSpefNames();
  buildCouplingRefs(corner_idx);
  buildReportLayerMap();
  net_str_buffer_.assign(net_count, Str{});

  writeHeader(ofs, corner_idx);
  writeNameMap(ofs);
  writePorts(ofs);
  writeLayerMap(ofs);

#pragma omp parallel for schedule(dynamic)
  for (Size net_idx = 0; net_idx < net_count; ++net_idx) {
    std::ostringstream net_os;
    writeDNet(net_os, corner_idx, net_idx);
    net_str_buffer_[net_idx] = net_os.str();
  }

  for (const Str& net_str : net_str_buffer_) {
    ofs << net_str;
  }

  LOG_INFO << "SpefDumper: wrote " << filename;
  return true;
}

}  // namespace ircx

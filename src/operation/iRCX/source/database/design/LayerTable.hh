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

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Types.hh"
namespace ircx {

// LayerTable
//
// Centralizes all design ↔ process layer name/id mappings.
//
// Populated in three stages matching the RCX pipeline:
//   1. readCorner()  → registerProcessLayer()
//   2. adaptDB()     → registerDesignLayer()
//   3. readMapping() → registerMapping()
//
// All query methods throw std::out_of_range on a missing key.
class LayerTable {
 public:
  // Registration
  void clear() {
    clearDesignLayers();
    clearProcessLayers();
    clearMappings();
  }

  void clearDesignLayers() {
    design_id_to_name_.clear();
    design_name_to_id_.clear();
  }

  void clearProcessLayers() {
    process_id_to_name_.clear();
    process_name_to_id_.clear();
  }

  void clearMappings() {
    design_to_process_name_.clear();
    process_to_design_name_.clear();
  }

  void registerDesignLayer(Size id, Str name) {
    design_id_to_name_[id] = name;
    design_name_to_id_[std::move(name)] = id;
  }

  void copyDesignLayersFrom(const LayerTable& other) {
    clearDesignLayers();
    for (const auto& [id, name] : other.design_layers()) {
      registerDesignLayer(id, name);
    }
  }

  void registerProcessLayer(Size id, Str name) {
    process_id_to_name_[id] = name;
    process_name_to_id_[std::move(name)] = id;
  }

  // Registers both directions of the design↔process name mapping.
  void registerMapping(const Str& design_name,
                       const Str& process_name) {
    design_to_process_name_[design_name] = process_name;
    process_to_design_name_[process_name] = design_name;
  }

  // Single-domain queries

  Size design_id(const Str& name) const {
    return design_name_to_id_.at(name);
  }
  const Str& design_name(Size id) const {
    return design_id_to_name_.at(id);
  }

  std::vector<std::pair<Size, Str>> design_layers() const {
    std::vector<std::pair<Size, Str>> layers;
    layers.reserve(design_id_to_name_.size());
    for (const auto& [id, name] : design_id_to_name_) {
      layers.emplace_back(id, name);
    }
    return layers;
  }

  Size process_id(const Str& name) const {
    return process_name_to_id_.at(name);
  }
  const Str& process_name(Size id) const {
    return process_id_to_name_.at(id);
  }

  // Cross-domain queries

  Size design_to_process_id(Size design_id) const {
    const Str& pname = design_to_process_name_.at(design_id_to_name_.at(design_id));
    return process_name_to_id_.at(pname);
  }

  Size process_to_design_id(Size process_id) const {
    const Str& dname = process_to_design_name_.at(process_id_to_name_.at(process_id));
    return design_name_to_id_.at(dname);
  }

 private:
  std::unordered_map<Size, Str> design_id_to_name_;
  std::unordered_map<Str, Size> design_name_to_id_;

  std::unordered_map<Size, Str> process_id_to_name_;
  std::unordered_map<Str, Size> process_name_to_id_;

  std::unordered_map<Str, Str> design_to_process_name_;
  std::unordered_map<Str, Str> process_to_design_name_;
};

}  // namespace ircx

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

#include <cassert>
#include <optional>

#include "Types.hh"
#include "builder.h"

namespace ircx {

class Net;
class Segment;
class Patch;
class Via;
class Pin;

class LayoutData;
class LayerTable;
class SpefContext;

class IdbAdapter
{
 public:
  explicit IdbAdapter(::idb::IdbBuilder* idb) : idb_(idb) { assert(idb_); }
  IdbAdapter() = delete;
  ~IdbAdapter() = default;

  auto adapt(LayoutData& layout_data,
             LayerTable& layer_table,
             SpefContext& spef_context) -> bool;

 private:
  auto adaptLayerTable(::idb::IdbLayers* idb_layers) -> void;
  auto adaptRoutingLayer(::idb::IdbLayers* idb_layers) -> void;

  auto adaptSpefContext(::idb::IdbDesign* idb_design) -> void;

  auto adaptNet(::idb::IdbNetList* idb_netlist) -> void;
  auto adaptPin(::idb::IdbPin* idb_pin, bool is_driving) -> Pin;
  auto adaptSegments(::idb::IdbRegularWireSegment* idb_seg) -> std::optional<Segment>;
  auto adaptPatch(::idb::IdbRegularWireSegment* idb_seg) -> std::optional<Patch>;
  auto adaptVia(::idb::IdbVia* idb_via) -> std::optional<Via>;

  auto adaptSpecialNet(::idb::IdbSpecialNetList* idb_special_netlist) -> void;

  // iDB source
  ::idb::IdbBuilder* idb_{nullptr};

  // RCX targets
  LayoutData* layout_data_{nullptr};
  LayerTable* layer_table_{nullptr};
  SpefContext* spef_context_{nullptr};

  auto idbRectToGtlRect(::idb::IdbRect* idb_rect) const -> GtlRectI;
  auto idbPointToGtlPoint(::idb::IdbCoordinate<int32_t>* idb_point) const -> GtlPointI;
};

}  // namespace ircx

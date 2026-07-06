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
#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "IdbAdapter.hh"
#include "LayerTable.hh"
#include "LayoutData.hh"
#include "StringUtils.hh"
#include "SpefContext.hh"
#include "log/Log.hh"

namespace ircx {

using namespace idb;

auto IdbAdapter::adapt(LayoutData& layout_data,
                       LayerTable& layer_table,
                       SpefContext& spef_context) -> bool
{
  layout_data_ = &layout_data;
  layer_table_ = &layer_table;
  spef_context_ = &spef_context;

  // idb
  auto* def_service = idb_->get_def_service();
  auto* lef_service = idb_->get_lef_service();
  if (!def_service || !lef_service) {
    LOG_ERROR_IF(!def_service) << "idb DEF service is null.";
    LOG_ERROR_IF(!lef_service) << "idb LEF service is null.";
    return false;
  }

  IdbDesign* idb_design = def_service->get_design();
  if (!idb_design) {
    LOG_ERROR << "idb::IdbDesign is null";
    return false;
  }

  IdbLayout* idb_layout = lef_service->get_layout();

  IdbNetList* idb_net_list = idb_design->get_net_list();
  IdbSpecialNetList* idb_special_net_list = idb_design->get_special_net_list();
  IdbUnits* idb_units = idb_design->get_units();
  if (!idb_net_list || !idb_layout) {
    LOG_ERROR_IF(!idb_net_list) << "idb::IdbNetList is null";
    LOG_ERROR_IF(!idb_layout) << "idb::IdbLayout is null";
    return false;
  }

  layout_data.clear();
  spef_context.clear();
  layer_table.clearDesignLayers();

  IdbLayers* idb_layers = idb_layout->get_layers();
  adaptLayerTable(idb_layers);
  adaptRoutingLayer(idb_layers);
  adaptSpefContext(idb_design);

  // set LayoutData : design_name die_shape dbu_per_micron
  IdbDie* idb_die = idb_layout->get_die();
  layout_data.design_name = idb_design->get_design_name();
  if (idb_die) {
    layout_data.die_shape = idbRectToGtlRect(idb_die->get_bounding_box());
  }
  if (idb_units) {
    layout_data.dbu_per_micron = idb_units->get_micron_dbu();
  }
  adaptNet(idb_net_list);

  adaptSpecialNet(idb_special_net_list);

  return true;
}

auto IdbAdapter::adaptLayerTable(IdbLayers* idb_layers) -> void
{
  layer_table_->registerDesignLayer(0, "SUBSTRATE");

  Size next_layer_id = 1;
  for (IdbLayer* idb_routing_layer : idb_layers->get_routing_layers()) {
    layer_table_->registerDesignLayer(next_layer_id, idb_routing_layer->get_name());
    ++next_layer_id;
  }
  for (IdbLayer* idb_cut_layer : idb_layers->get_cut_layers()) {
    layer_table_->registerDesignLayer(next_layer_id, idb_cut_layer->get_name());
    ++next_layer_id;
  }
}

auto IdbAdapter::adaptRoutingLayer(IdbLayers* idb_layers) -> void
{
  // set RoutingLayer
  std::map<Size, RoutingLayer>& routing_layers = layout_data_->routing_layers;

  std::vector<IdbLayer*>& routing_layer_defs = idb_layers->get_routing_layers();
  for (IdbLayer* idb_layer : routing_layer_defs) {
    auto* idb_routing_layer = dynamic_cast<IdbLayerRouting*>(idb_layer);
    if (!idb_routing_layer) {
      continue;
    }

    RoutingLayer routing_layer;

    const Size design_layer_id = layer_table_->design_id(idb_routing_layer->get_name());
    routing_layer.set_layer_id(design_layer_id);
    routing_layer.set_layer_name(idb_routing_layer->get_name());
    routing_layer.set_layer_width(idb_routing_layer->get_width());
    if (idb_routing_layer->is_horizontal()) {
      routing_layer.set_prefer_horz(true);
    } else if (idb_routing_layer->is_vertical()) {
      routing_layer.set_prefer_horz(false);
    }

    // track info
    RoutingLayer::TrackInfo track_info;
    for (IdbTrackGrid* track_grid : idb_routing_layer->get_track_grid_list()) {
      IdbTrack* track = track_grid->get_track();
      if (track->get_direction() == IdbTrackDirection::kDirectionX) {
        track_info.x0 = track->get_start();
        track_info.dx = track->get_pitch();
        track_info.nx = track_grid->get_track_num();
      } else if (track->get_direction() == IdbTrackDirection::kDirectionY) {
        track_info.y0 = track->get_start();
        track_info.dy = track->get_pitch();
        track_info.ny = track_grid->get_track_num();
      }
    }

    routing_layer.set_track_info(track_info);
    routing_layers.emplace(design_layer_id, std::move(routing_layer));
  }
}

auto IdbAdapter::adaptSpefContext(IdbDesign* idb_design) -> void
{
  const std::vector<IdbNet*>& idb_nets = idb_design->get_net_list()->get_net_list();
  spef_context_->net_names.reserve(idb_nets.size());
  for (IdbNet* idb_net : idb_nets) {
    if (idb_net->is_pdn()) {
      continue;
    }
    spef_context_->net_names.push_back(string::escape_spef_name(idb_net->get_net_name()));
  }

  const std::vector<IdbPin*>& io_pins = idb_design->get_io_pin_list()->get_pin_list();
  spef_context_->port_names.reserve(io_pins.size());
  spef_context_->port_io.reserve(io_pins.size());
  for (IdbPin* io_pin : io_pins) {
    if (io_pin->is_special_net_pin() || !io_pin->get_net()) {
      continue;
    }
    spef_context_->port_names.push_back(string::escape_spef_name(io_pin->get_pin_name()));

    if (io_pin->is_primary_input()) {
      spef_context_->port_io.emplace_back('I');
    } else if (io_pin->is_primary_output()) {
      spef_context_->port_io.emplace_back('O');
    } else {
      spef_context_->port_io.emplace_back('B');
    }
  }

  const std::vector<IdbInstance*>& instances =
      idb_design->get_instance_list()->get_instance_list();
  spef_context_->instance_names.reserve(instances.size());
  for (IdbInstance* instance : instances) {
    const Str instance_name = string::escape_spef_name(instance->get_name());
    spef_context_->instance_names.push_back(instance_name);
    spef_context_->instance_to_cell[instance_name] =
        string::escape_spef_name(instance->get_cell_master()->get_name());
  }
}


auto IdbAdapter::adaptNet(IdbNetList* idb_netlist) -> void
{
  std::vector<Net>& regular_nets = layout_data_->net_vec;
  std::vector<IdbNet*>& idb_nets = idb_netlist->get_net_list();
  const Size net_count = idb_nets.size();
  regular_nets.resize(net_count);
  for (Size net_idx = 0; net_idx < net_count; ++net_idx) {
    IdbNet* idb_net = idb_nets[net_idx];

    if (!idb_net) {
      continue;
    }

    Net& net = regular_nets[net_idx];
    net.id = net_idx;
    net.name = string::escape_spef_name(idb_net->get_net_name());

    auto* idb_wire_list = idb_net->get_wire_list();
    if (!idb_wire_list || idb_wire_list->get_num() == 0) {
      continue;
    }

    // adapt driving
    if (IdbPin* driver_pin = idb_net->get_driving_pin()) {
      Pin driver = adaptPin(driver_pin, true);
      net.pins.push_back(std::move(driver));
    }

    // adapt load
    for (IdbPin* load_pin : idb_net->get_load_pins()) {
      if (!load_pin) {
        continue;
      }

      Pin load = adaptPin(load_pin, false);
      net.pins.push_back(std::move(load));
    }

    // adapt segments
    for (auto* idb_wire : idb_wire_list->get_wire_list()) {
      if (!idb_wire) {
        continue;
      }

      for (auto* idb_segment : idb_wire->get_segment_list()) {
        if (!idb_segment) {
          continue;
        }

        // convert segments
        if (idb_segment->is_wire()) {
          if (auto segment = adaptSegments(idb_segment)) {
            net.segments.push_back(std::move(segment.value()));
          }
        }
        if (idb_segment->is_rect()) {
          if (auto patch = adaptPatch(idb_segment)) {
            net.patches.push_back(std::move(patch.value()));
          }
        }
        for (auto* idb_via : idb_segment->get_via_list()) {
          if (auto via = adaptVia(idb_via)) {
            net.vias.push_back(std::move(via.value()));
          }
        }
      }
    }
  }
}

auto IdbAdapter::adaptPin(IdbPin* idb_pin, bool is_driving) -> Pin
{
  Pin pin;

  if (idb_pin->is_io_pin()) {
    pin.name = string::escape_spef_name(idb_pin->get_pin_name());
  } else {
    pin.name = string::escape_spef_name(idb_pin->get_instance()->get_name())
               + ':'
               + string::escape_spef_name(idb_pin->get_pin_name());
  }

  pin.is_driver = is_driving;

  if (IdbTerm* idb_term = idb_pin->get_term()) {
    using Dir = IdbConnectDirection;
    const Dir pin_direction = idb_term->get_direction();
    switch (pin_direction) {
      case Dir::kInput:
        pin.is_input = true;
        break;
      case Dir::kOutput:
      case Dir::kOutputTriState:
        pin.is_output = true;
        break;
      case Dir::kInOut:
      case Dir::kFeedThru:
        pin.is_input = true;
        pin.is_output = true;
        break;
      default:
        break;
    }
  }

  for (IdbLayerShape* layer_shape : idb_pin->get_port_box_list()) {
    if (!layer_shape || !layer_shape->get_layer()) {
      continue;
    }
    IdbLayer* idb_layer = layer_shape->get_layer();
    const Size design_layer_id = layer_table_->design_id(idb_layer->get_name());
    for (IdbRect* idb_rect : layer_shape->get_rect_list()) {
      if (!idb_rect) {
        continue;
      }
      pin.layer_id_rects.emplace_back(design_layer_id, idbRectToGtlRect(idb_rect));
    }
  }

  return pin;
}

auto IdbAdapter::adaptSegments(IdbRegularWireSegment* idb_seg) -> std::optional<Segment>
{
  if (!idb_seg || !idb_seg->is_wire()) {
    LOG_ERROR << "skip invalid idb wire segment.";
    return std::nullopt;
  }

  auto* p1 = idb_seg->get_point_start();
  auto* p2 = idb_seg->get_point_end();
  if (!p1 || !p2) {
    LOG_ERROR << "skip idb wire segment without endpoint.";
    return std::nullopt;
  }

  IdbLayer* idb_layer = idb_seg->get_layer();
  if (!idb_layer) {
    LOG_ERROR << "skip idb wire segment without layer.";
    return std::nullopt;
  }

  IdbRect segment_rect = idb_seg->get_segment_rect();

  Segment segment;
  segment.p0 = GtlPointI(p1->get_x(), p1->get_y());
  segment.p1 = GtlPointI(p2->get_x(), p2->get_y());
  segment.layer_id = layer_table_->design_id(idb_layer->get_name());
  segment.rect = idbRectToGtlRect(&segment_rect);
  return segment;
}

auto IdbAdapter::adaptPatch(IdbRegularWireSegment* idb_seg) -> std::optional<Patch>
{
  if (!idb_seg || !idb_seg->is_rect()) {
    LOG_ERROR << "skip invalid idb patch segment.";
    return std::nullopt;
  }

  auto* delta_rect = idb_seg->get_delta_rect();
  auto* anchor_point = idb_seg->get_point(0);
  auto* idb_layer = idb_seg->get_layer();

  if (!delta_rect || !anchor_point || !idb_layer) {
    LOG_ERROR << "skip idb patch segment with incomplete geometry.";
    return std::nullopt;
  }

  const int lower_x = anchor_point->get_x() + delta_rect->get_low_x();
  const int lower_y = anchor_point->get_y() + delta_rect->get_low_y();
  const int upper_x = anchor_point->get_x() + delta_rect->get_high_x();
  const int upper_y = anchor_point->get_y() + delta_rect->get_high_y();

  Patch patch;
  patch.layer_id = layer_table_->design_id(idb_layer->get_name());
  patch.rect = GtlRectI(lower_x, lower_y, upper_x, upper_y);
  return patch;
}

auto IdbAdapter::adaptVia(IdbVia* idb_via) -> std::optional<Via>
{
  if (!idb_via) {
    LOG_ERROR << "skip null idb via.";
    return std::nullopt;
  }

  auto* center_point = idb_via->get_coordinate();
  if (!center_point) {
    LOG_ERROR << "skip idb via without coordinate.";
    return std::nullopt;
  }

  Via via;
  via.name = string::escape_spef_name(idb_via->get_name());
  via.point = GtlPointI(center_point->get_x(), center_point->get_y());

  auto read_layer_rect =
      [&](IdbLayerShape layer_shape,
          const char* multi_rect_error)
          -> std::optional<std::pair<Size, GtlRectI>> {
    IdbLayer* idb_layer = layer_shape.get_layer();
    if (!idb_layer) {
      return std::nullopt;
    }

    std::vector<IdbRect*>& layer_rects = layer_shape.get_rect_list();
    if (layer_rects.size() != 1) {
      LOG_ERROR << multi_rect_error;
      return std::nullopt;
    }

    const Size design_layer_id = layer_table_->design_id(idb_layer->get_name());
    return std::make_pair(design_layer_id, idbRectToGtlRect(layer_rects[0]));
  };

  if (auto top_layer_rect =
          read_layer_rect(idb_via->get_top_layer_shape(), "not support multirect for via top")) {
    via.layer_rect_top = *top_layer_rect;
  }

  if (auto bottom_layer_rect =
          read_layer_rect(idb_via->get_bottom_layer_shape(), "not support multirect for via bottom")) {
    via.layer_rect_btm = *bottom_layer_rect;
  }

  if (auto cut_layer_rect =
          read_layer_rect(idb_via->get_cut_layer_shape(), "not support multirect for via cut")) {
    via.layer_rect_cut = *cut_layer_rect;
  }

  if (via.layer_rect_top.first == kMaxSize ||
      via.layer_rect_btm.first == kMaxSize ||
      via.layer_rect_cut.first == kMaxSize) {
    LOG_ERROR << "skip idb via with incomplete layer rectangles: " << via.name;
    return std::nullopt;
  }

  return via;
}

auto IdbAdapter::adaptSpecialNet(IdbSpecialNetList* idb_special_net_list) -> void
{
  if (!idb_special_net_list) {
    return;
  }

  Net& layout_special_net = layout_data_->special_net;

  for (auto* special_net : idb_special_net_list->get_net_list()) {
    if (!special_net) {
      continue;
    }

    auto* special_wire_list = special_net->get_wire_list();
    if (!special_wire_list) {
      continue;
    }

    for (auto* special_wire : special_wire_list->get_wire_list()) {
      if (!special_wire) {
        continue;
      }

      for (auto* special_segment : special_wire->get_segment_list()) {
        if (!special_segment) {
          continue;
        }
        if (special_segment->is_via()) {
          continue;
        }

        auto* special_layer = special_segment->get_layer();
        auto* special_rect = special_segment->get_bounding_box();
        auto* start_point = special_segment->get_point_start();
        auto* end_point = special_segment->get_point_second();
        if (!special_layer || !special_rect || !start_point || !end_point) {
          continue;
        }

        Segment segment;
        segment.layer_id = layer_table_->design_id(special_layer->get_name());
        segment.rect = idbRectToGtlRect(special_rect);
        segment.p0 = idbPointToGtlPoint(start_point);
        segment.p1 = idbPointToGtlPoint(end_point);
        layout_special_net.segments.push_back(std::move(segment));
      }
    }
  }
}

auto IdbAdapter::idbRectToGtlRect(IdbRect* idb_rect) const -> GtlRectI
{
  return GtlRectI(idb_rect->get_low_x(), idb_rect->get_low_y(),
                  idb_rect->get_high_x(), idb_rect->get_high_y());
}

auto IdbAdapter::idbPointToGtlPoint(IdbCoordinate<int32_t>* idb_point) const -> GtlPointI
{
  return GtlPointI(idb_point->get_x(), idb_point->get_y());
}

}  // namespace ircx

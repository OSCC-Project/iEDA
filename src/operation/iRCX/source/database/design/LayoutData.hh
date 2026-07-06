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
#include <utility>
#include <vector>

#include "RoutingLayer.hh"
#include "Types.hh"
namespace ircx {

// ============================================================
// Segment
// ============================================================
struct Segment {
  Size layer_id{kMaxSize};
  GtlRectI  rect;

  GtlPointI p0, p1;
};

// ============================================================
// Patch
// ============================================================
struct Patch {
  Size layer_id{kMaxSize};
  GtlRectI  rect;
};

// ============================================================
// Pin
// ============================================================
struct Pin {
  Str name; // pin(inst:pin)/port name

  bool is_driver{false};
  bool is_input{false};
  bool is_output{false};

  std::vector<std::pair<Size, GtlRectI>> layer_id_rects;

  bool is_port() const {
    return name.find(':') == Str::npos;
  }

  Str instance_name() const {
    size_t pos = name.find(':');
    return name.substr(0, pos);
  }
  Str instance_pin_name() const {
    size_t pos = name.find(':');
    return name.substr(pos+1);
  }
  Str port_name() const {
    return name;
  }
};

// ============================================================
// Via
// ============================================================
struct Via {
  Str name;

  GtlPointI point;
  // Top/Cut/Bottom layer rect
  std::pair<Size, GtlRectI> layer_rect_top{kMaxSize, {}};
  std::pair<Size, GtlRectI> layer_rect_cut{kMaxSize, {}};
  std::pair<Size, GtlRectI> layer_rect_btm{kMaxSize, {}};
};

// ============================================================
// Net
// ============================================================
struct Net {
  Size id;
  Str name;
  std::vector<Segment> segments;
  std::vector<Patch> patches;
  std::vector<Via> vias;
  std::vector<Pin> pins;
};

struct LayoutData {
  void clear() {
    design_name.clear();
    die_shape = {};
    dbu_per_micron = 1;
    routing_layers.clear();
    net_vec.clear();
    special_net = {};
  }

  // Design metadata
  Str design_name;
  GtlRectI die_shape;
  Dbu dbu_per_micron{1};

  // Technology layers
  std::map<Size, RoutingLayer> routing_layers;

  // Net metadata
  // starting from 0.
  std::vector<Net> net_vec; // net id is index of net_vec

  // Special-net geometry (power/ground, no connectivity graph needed)
  Net special_net;

  // Helpers
  Size regular_net_count() const { return net_vec.size(); }
  bool empty() const
  {
    return net_vec.empty() &&
           special_net.segments.empty() &&
           special_net.patches.empty() &&
           special_net.vias.empty() &&
           special_net.pins.empty();
  }
};

}  // namespace ircx

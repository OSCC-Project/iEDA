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

#include "Types.hh"
namespace ircx {

class RoutingLayer {
 public:
  RoutingLayer() = default;
  RoutingLayer(const RoutingLayer&) = default;
  auto operator=(const RoutingLayer&) -> RoutingLayer& = default;
  RoutingLayer(RoutingLayer&&) = default;
  auto operator=(RoutingLayer&&) -> RoutingLayer& = default;
  ~RoutingLayer() = default;

  struct TrackInfo {
    Dbu x0{};
    Dbu y0{};
    Dbu dx{};
    Dbu dy{};
    Size nx{};
    Size ny{};
  };

  void set_layer_id(Size v) { layer_id_ = v; }
  Size layer_id() const { return layer_id_; }

  void set_layer_name(Str v) { layer_name_ = std::move(v); }
  const Str& layer_name() const { return layer_name_; }

  void set_layer_width(Dbu v) { layer_width_ = v; }
  Dbu layer_width() const { return layer_width_; }

  void set_prefer_horz(bool v) { prefer_horz_ = v; }
  bool is_prefer_horz() const { return prefer_horz_; }

  void set_track_info(const TrackInfo& v) { track_info_ = v; }
  const TrackInfo& track_info() const { return track_info_; }

 private:
  Size layer_id_{};
  Str layer_name_{};
  Dbu layer_width_{};

  bool prefer_horz_{false};
  TrackInfo track_info_{};
};

}  // namespace ircx

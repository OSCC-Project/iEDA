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

#include <unordered_map>
#include <vector>

#include "CornerNetPool.hh"
#include "NetEtchProfile.hh"
#include "LayoutData.hh"
#include "RCXData.hh"
#include "RCTable.hh"
#include "Types.hh"
namespace ircx {

class LayerTable;
class TopoEdge;
class TopoPool;
}

namespace itf {
class LayerConductor;
class LayerVia;
class ProcessCorner;
}

namespace ircx {

class ResistanceCalc
{
 public:
  ResistanceCalc() = default;
  ~ResistanceCalc() = default;

  void set_layout_data(const LayoutData* v) {
    layout_data_ = v;
    micron_per_dbu_ = unit::to_micron(1, v->dbu_per_micron);
  }
  void set_layer_table(const LayerTable* v) { layer_table_ = v; }
  void set_topo_pool(const TopoPool* v) { topo_pool_ = v; }
  void set_corner_net_etch_pools(const CornerNetPool<NetEtchProfile>* v)
  {
    corner_net_etch_pools_ = v;
  }
  void set_rc_table(RCTable* v) { rc_table_ = v; }
  void set_corner_data(const std::vector<RCXData::CornerData>* v) { corner_data_ = v; }

  bool calc();

  ResistanceCalc(const ResistanceCalc&) = delete;
  ResistanceCalc(ResistanceCalc&&) = delete;
  auto operator=(const ResistanceCalc&) -> ResistanceCalc& = delete;
  auto operator=(ResistanceCalc&&) -> ResistanceCalc& = delete;

 private:
  class ProcessLayerResolver
  {
   public:
    bool build(const LayerTable& layer_table,
               const itf::ProcessCorner& corner,
               const TopoPool& topo_pool,
               const Str& corner_name);

    const itf::LayerConductor* conductor(Size design_layer_id) const;
    const itf::LayerVia* via(Size design_layer_id) const;

   private:
    std::unordered_map<Size, const itf::LayerConductor*> conductors_;
    std::unordered_map<Size, const itf::LayerVia*> vias_;
  };

  struct CornerCalcView
  {
    Size idx{kMaxSize};
    const RCXData::CornerData* data{nullptr};
    const itf::ProcessCorner* process_corner{nullptr};
    F64 temperature{kDefaultOperatingTemperature};
    ProcessLayerResolver layers;
  };

  bool validateInputs() const;
  bool buildCornerViews(std::vector<CornerCalcView>& views) const;
  void calcCorner(const CornerCalcView& corner) const;
  void calcNet(const CornerCalcView& corner, Size net_idx) const;
  F64 calcEdgeResistance(const CornerCalcView& corner,
                         Size net_idx,
                         Size edge_idx,
                         const TopoEdge& edge,
                         const NetEtchProfile& etch_profile) const;
  F64 calcViaResistance(const CornerCalcView& corner,
                        const TopoEdge& edge) const;
  F64 calcConductorResistance(const CornerCalcView& corner,
                              const TopoEdge& edge,
                              std::span<const EdgeEtchInterval> edge_etch_intervals) const;
  LineSegment<Micron> edgeSegment(const TopoEdge& edge) const;

  Micron micron_per_dbu_{1};

  const LayoutData* layout_data_{nullptr};
  const LayerTable* layer_table_{nullptr};
  const TopoPool* topo_pool_{nullptr};
  const CornerNetPool<NetEtchProfile>* corner_net_etch_pools_{nullptr};
  const std::vector<RCXData::CornerData>* corner_data_{nullptr};

  RCTable* rc_table_{nullptr};
};

}  // namespace ircx

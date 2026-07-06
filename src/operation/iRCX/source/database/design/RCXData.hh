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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "CapTable.hpp"
#include "CornerNetPool.hh"
#include "RCXConfig.hh"
#include "NetEtchProfile.hh"
#include "NetEnvironment.hh"
#include "LayerTable.hh"
#include "LayoutData.hh"
#include "MappingBuilder.hpp"
#include "RCTable.hh"
#include "SpefContext.hh"
#include "TopoPool.hh"
#include "Types.hh"

namespace itf {
class ProcessCorner;
}  // namespace itf

namespace ircx {

namespace parser {
class CapTable;
class MappingBuilder;
}  // namespace parser

#define RCX_DATA_INST (ircx::RCXData::getInst())

class RCXData final {
 public:
  struct CornerData {
    CornerData();
    ~CornerData();
    CornerData(CornerData&&);
    CornerData& operator=(CornerData&&);
    CornerData(const CornerData&) = delete;
    CornerData& operator=(const CornerData&) = delete;

    Str name;
    F64 temperature{kDefaultOperatingTemperature};
    Str itf_file;
    Str captab_file;
    std::unique_ptr<::itf::ProcessCorner> process_corner;
    parser::CapTable cap_table;

    F64 halfNodeScaleFactor() const;
  };

  static RCXData& getInst() {
    static RCXData inst;
    return inst;
  }

  void reset();
  void setDBData(LayoutData layout_data,
                 const LayerTable& design_layer_table,
                 SpefContext spef_context);

  LayoutData& layout() { return layout_; }
  const LayoutData& layout() const { return layout_; }
  SpefContext& spef_context() { return spef_context_; }
  const SpefContext& spef_context() const { return spef_context_; }
  LayerTable& layer_table() { return layer_table_; }
  const LayerTable& layer_table() const { return layer_table_; }
  parser::MappingBuilder& mapping_builder() { return mapping_builder_; }
  const parser::MappingBuilder& mapping_builder() const { return mapping_builder_; }
  TopoPool& topo_pool() { return topo_pool_; }
  const TopoPool& topo_pool() const { return topo_pool_; }
  RCTable& rc_table() { return rc_table_; }
  const RCTable& rc_table() const { return rc_table_; }
  std::vector<NetEnvironment>& net_env_pools() { return net_env_pools_; }
  const std::vector<NetEnvironment>& net_env_pools() const { return net_env_pools_; }
  CornerNetPool<NetEtchProfile>& corner_net_etch_pools() { return corner_net_etch_pools_; }
  const CornerNetPool<NetEtchProfile>& corner_net_etch_pools() const { return corner_net_etch_pools_; }
  std::vector<CornerData>& corner_data() { return corners_; }
  const std::vector<CornerData>& corner_data() const { return corners_; }
  F64 halfNodeScaleFactor(Size corner_idx) const;

  bool hasCorner(const Str& corner_name) const;
  void setProcessLayersRegistered(bool value) { process_layers_registered_ = value; }
  bool processLayersRegistered() const { return process_layers_registered_; }

  RCXData(const RCXData&) = delete;
  RCXData(RCXData&&) = delete;
  auto operator=(const RCXData&) -> RCXData& = delete;
  auto operator=(RCXData&&) -> RCXData& = delete;

 private:
  RCXData() = default;
  ~RCXData();

  // Design data adapted from iDB.
  LayoutData layout_;
  SpefContext spef_context_;
  LayerTable layer_table_;

  // Technology and mapping data loaded during setup.
  std::vector<CornerData> corners_;
  parser::MappingBuilder mapping_builder_;

  // Extraction intermediate and result data.
  TopoPool topo_pool_;
  RCTable rc_table_;
  std::vector<NetEnvironment> net_env_pools_;
  CornerNetPool<NetEtchProfile> corner_net_etch_pools_;

  bool process_layers_registered_{false};
};

}  // namespace ircx

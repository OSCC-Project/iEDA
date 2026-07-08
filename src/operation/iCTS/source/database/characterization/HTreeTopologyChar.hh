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
 * @file HTreeTopologyChar.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-30
 * @brief H-tree topology characterization with levels.
 */

#pragma once

#include <utility>

#include "CharCore.hh"
#include "PatternId.hh"

namespace icts {

/**
 * @brief H-tree topology characterization entry.
 *
 * Combines CharCore (electrical boundaries + cost) with H-tree level count.
 * Supports composition for H-tree concatenation via Hash-Join.
 *
 * Key difference from SegmentChar:
 * - Uses load_cap/2 in probe key (binary fan-out)
 * - Power composition: merged.power = up.power + 2 * (down.power - down.source_boundary_net_switch_power)
 */
class HTreeTopologyChar
{
 public:
  HTreeTopologyChar() = default;

  HTreeTopologyChar(CharCore core, unsigned levels) : _core(std::move(core)), _levels(levels) {}

  // Forwarded getters from CharCore
  auto get_input_slew_idx() const -> unsigned { return _core.get_input_slew_idx(); }
  auto get_output_slew_idx() const -> unsigned { return _core.get_output_slew_idx(); }
  auto get_driven_cap_idx() const -> unsigned { return _core.get_driven_cap_idx(); }
  auto get_load_cap_idx() const -> unsigned { return _core.get_load_cap_idx(); }
  auto get_delay() const -> double { return _core.get_delay() + _root_driver_delay_ns; }
  auto get_power() const -> double { return _core.get_power() + _root_driver_power_w; }
  auto get_raw_delay() const -> double { return _core.get_delay(); }
  auto get_raw_power() const -> double { return _core.get_power(); }
  auto get_root_driver_delay() const -> double { return _root_driver_delay_ns; }
  auto get_root_driver_power() const -> double { return _root_driver_power_w; }
  auto get_pattern_id() const -> PatternId { return _core.get_pattern_id(); }
  // The leaf-side boundary capability is the downstream-most load-cap bin already stored in CharCore.
  auto get_leaf_load_cap_idx() const -> unsigned { return _core.get_load_cap_idx(); }
  auto get_source_boundary_net_switch_power() const -> double { return _core.get_source_boundary_net_switch_power(); }
  auto set_root_driver_compensation(double delay_ns, double power_w) -> void
  {
    _root_driver_delay_ns = delay_ns;
    _root_driver_power_w = power_w;
  }

  // H-tree specific getter
  auto get_levels() const -> unsigned { return _levels; }

  auto withPatternId(PatternId pattern_id) const -> HTreeTopologyChar
  {
    CharCore core(get_input_slew_idx(), get_output_slew_idx(), get_driven_cap_idx(), get_load_cap_idx(), get_raw_delay(), get_raw_power(),
                  pattern_id, get_source_boundary_net_switch_power());
    HTreeTopologyChar result(std::move(core), _levels);
    result.set_root_driver_compensation(_root_driver_delay_ns, _root_driver_power_w);
    return result;
  }

  /**
   * @brief Compose two H-tree topology characterizations.
   *
   * Composition rules:
   * - levels = upstream.levels + downstream.levels
   * - input_slew = upstream.input_slew
   * - output_slew = downstream.output_slew
   * - driven_cap = upstream.driven_cap
   * - load_cap = downstream.load_cap
   * - delay = upstream.delay + downstream.delay
   * - power = upstream.power + 2 * (downstream.power - downstream.source_boundary_net_switch_power)
   * - source_boundary_net_switch_power = upstream.source_boundary_net_switch_power
   *
   * The /2 cap relationship is enforced by HTreeTraits::probeKey,
   * not here in compose.
   *
   * @param upstream Upstream H-tree (closer to root)
   * @param downstream Downstream H-tree (closer to leaves)
   * @param merged_topo_pid Pattern ID for the composed topology
   */
  static auto compose(const HTreeTopologyChar& upstream, const HTreeTopologyChar& downstream, PatternId merged_topo_pid)
      -> HTreeTopologyChar
  {
    CharCore merged_core(upstream.get_input_slew_idx(),                          // input from upstream
                         downstream.get_output_slew_idx(),                       // output from downstream
                         upstream.get_driven_cap_idx(),                          // driven_cap from upstream
                         downstream.get_load_cap_idx(),                          // load_cap from downstream
                         upstream.get_raw_delay() + downstream.get_raw_delay(),  // additive delay
                         // Binary fan-out: downstream source-boundary switching is owned by the upstream branch point.
                         upstream.get_raw_power() + 2.0 * (downstream.get_raw_power() - downstream.get_source_boundary_net_switch_power()),
                         merged_topo_pid, upstream.get_source_boundary_net_switch_power());
    return HTreeTopologyChar(std::move(merged_core), upstream._levels + downstream._levels);
  }

 private:
  CharCore _core;
  unsigned _levels = 0;
  double _root_driver_delay_ns = 0.0;
  double _root_driver_power_w = 0.0;
};

}  // namespace icts

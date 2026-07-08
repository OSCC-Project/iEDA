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
 * @file CharCore.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-30
 * @brief Core electrical boundary and cost carrier for characterization.
 */

#pragma once

#include "PatternId.hh"

namespace icts {

/**
 * @brief Core characterization data: electrical boundaries + cost metrics.
 *
 * Reusable component for both SegmentChar and HTreeTopologyChar.
 * Join keys are NOT stored here - they are computed by traits externally.
 */
class CharCore
{
 public:
  CharCore() = default;

  CharCore(unsigned input_slew_idx, unsigned output_slew_idx, unsigned driven_cap_idx, unsigned load_cap_idx, double delay, double power,
           PatternId pattern_id, double source_boundary_net_switch_power = 0.0)
      : _input_slew_idx(input_slew_idx),
        _output_slew_idx(output_slew_idx),
        _driven_cap_idx(driven_cap_idx),
        _load_cap_idx(load_cap_idx),
        _delay(delay),
        _power(power),
        _pattern_id(pattern_id),
        _source_boundary_net_switch_power(source_boundary_net_switch_power)
  {
  }

  // Getters - following iCTS naming convention
  auto get_input_slew_idx() const -> unsigned { return _input_slew_idx; }
  auto get_output_slew_idx() const -> unsigned { return _output_slew_idx; }
  auto get_driven_cap_idx() const -> unsigned { return _driven_cap_idx; }
  auto get_load_cap_idx() const -> unsigned { return _load_cap_idx; }
  auto get_delay() const -> double { return _delay; }
  auto get_power() const -> double { return _power; }
  auto get_pattern_id() const -> PatternId { return _pattern_id; }
  auto get_source_boundary_net_switch_power() const -> double { return _source_boundary_net_switch_power; }

 private:
  unsigned _input_slew_idx = 0;
  unsigned _output_slew_idx = 0;
  unsigned _driven_cap_idx = 0;
  unsigned _load_cap_idx = 0;
  double _delay = 0.0;
  double _power = 0.0;
  PatternId _pattern_id{PatternDomain::kSegmentPattern, 0};
  double _source_boundary_net_switch_power = 0.0;
};

}  // namespace icts

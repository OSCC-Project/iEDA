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
 * @file CharacterizationUnitCaseData.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Shared constants and builders for characterization tests.
 */

#pragma once

#include "database/characterization/CharCore.hh"
#include "database/characterization/HTreeTopologyChar.hh"
#include "database/characterization/PatternId.hh"
#include "database/characterization/SegmentChar.hh"

namespace icts_test::characterization {

inline constexpr unsigned kSlew80 = 80;
inline constexpr unsigned kSlew90 = 90;
inline constexpr unsigned kSlew100 = 100;
inline constexpr unsigned kSlew110 = 110;
inline constexpr unsigned kSlew120 = 120;
inline constexpr unsigned kSlew130 = 130;

inline constexpr unsigned kCap40 = 40;
inline constexpr unsigned kCap45 = 45;
inline constexpr unsigned kCap50 = 50;
inline constexpr unsigned kCap51 = 51;
inline constexpr unsigned kCap55 = 55;
inline constexpr unsigned kCap60 = 60;
inline constexpr unsigned kCap70 = 70;
inline constexpr unsigned kCap100 = 100;
inline constexpr unsigned kCap101 = 101;

inline constexpr double kDelay1p0 = 1.0;
inline constexpr double kDelay1p5 = 1.5;
inline constexpr double kDelay2p0 = 2.0;
inline constexpr double kDelay3p0 = 3.0;

inline constexpr double kPower0p2 = 0.2;
inline constexpr double kPower0p3 = 0.3;
inline constexpr double kPower0p4 = 0.4;
inline constexpr double kPower0p5 = 0.5;
inline constexpr double kPower0p6 = 0.6;
inline constexpr double kPower5p0 = 5.0;
inline constexpr double kPower10p0 = 10.0;
inline constexpr double kMergedPower0p8 = 0.8;
inline constexpr double kMergedPower1p1 = 1.1;
inline constexpr double kMergedPower20p0 = 20.0;

inline constexpr unsigned kPattern1 = 1;
inline constexpr unsigned kPattern2 = 2;
inline constexpr unsigned kPattern3 = 3;
inline constexpr unsigned kLength1000 = 1000;
inline constexpr unsigned kLength1500 = 1500;
inline constexpr unsigned kLength2000 = 2000;
inline constexpr unsigned kLength3000 = 3000;
inline constexpr unsigned kLengthSum3000 = 3000;
inline constexpr unsigned kBoundaryKey = 100;
inline constexpr unsigned kPatternId42 = 42;
inline constexpr unsigned kPackedKey12345678 = 0x12345678U;
inline constexpr unsigned kPackedKeyFFFF0000 = 0xFFFF0000U;
inline constexpr unsigned kPackedKey0000FFFF = 0x0000FFFFU;
inline constexpr unsigned kPackHigh1234 = 0x1234U;
inline constexpr unsigned kPackLow5678 = 0x5678U;
inline constexpr unsigned kPackWordFFFF = 0xFFFFU;
inline constexpr unsigned kPackWordZero = 0x0000U;

struct SegmentShape
{
  unsigned pattern_id = 0;
  unsigned length_idx = 0;
};

inline auto MakeSegmentChar(unsigned input_slew, unsigned output_slew, unsigned driven_cap, unsigned load_cap, double delay, double power,
                            SegmentShape shape, double source_boundary_net_switch_power = 0.0) -> icts::SegmentChar
{
  const icts::CharCore core(input_slew, output_slew, driven_cap, load_cap, delay, power, icts::PatternId::segment(shape.pattern_id),
                            source_boundary_net_switch_power);
  return {core, shape.length_idx};
}

struct HTreeShape
{
  unsigned pattern_id = 0;
  unsigned levels = 0;
};

inline auto MakeHTreeChar(unsigned input_slew, unsigned output_slew, unsigned driven_cap, unsigned load_cap, double delay, double power,
                          HTreeShape shape, double source_boundary_net_switch_power = 0.0) -> icts::HTreeTopologyChar
{
  const icts::CharCore core(input_slew, output_slew, driven_cap, load_cap, delay, power, icts::PatternId::topology(shape.pattern_id),
                            source_boundary_net_switch_power);
  return {core, shape.levels};
}

}  // namespace icts_test::characterization

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
 * @file FastStaClockTiming.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief CTS fast STA slew, arrival, skew, and DMP timing records.
 */

#pragma once

#include <string>

#include "FastSta.hh"
#include "liberty/FastStaLibertyModel.hh"

namespace icts {

enum class FastStaDmpAlgorithm
{
  kCap,
  kPi,
  kZeroC2
};

struct FastStaTimingPoint
{
  double arrival_ns = 0.0;
  double slew_ns = 0.0;
  bool valid = false;
};

struct FastStaDmpDriverResult
{
  bool valid = false;
  bool driver_waveform_valid = false;
  FastStaDmpAlgorithm algorithm = FastStaDmpAlgorithm::kCap;
  FastStaTransition transition = FastStaTransition::kRise;
  std::string driver_cell_master;
  double ceff_pf = 0.0;
  double gate_delay_ns = 0.0;
  double driver_slew_ns = 0.0;
  double driver_waveform_delay_ns = 0.0;
  double t0_ns = 0.0;
  double dt_ns = 0.0;
  double near_cap_pf = 0.0;
  double far_cap_pf = 0.0;
  double rpi_ns_per_pf = 0.0;
  double rd_ns_per_pf = 0.0;
  double input_threshold = 0.5;
  double output_threshold = 0.5;
  double slew_lower_threshold = 0.3;
  double slew_upper_threshold = 0.7;
  double slew_derate = 1.0;
  double pole1_per_ns = 0.0;
  double pole2_per_ns = 0.0;
  double zero1_per_ns = 0.0;
  double k0 = 0.0;
  double k1 = 0.0;
  double k2 = 0.0;
  double k3 = 0.0;
  double k4 = 0.0;
};

struct FastStaDmpLoadResult
{
  bool valid = false;
  double wire_delay_ns = 0.0;
  double load_slew_ns = 0.0;
};

}  // namespace icts

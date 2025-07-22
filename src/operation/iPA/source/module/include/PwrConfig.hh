// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file PwrConfig.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The power tool global config.
 * @version 0.1
 * @date 2023-04-08
 */

#pragma once

constexpr unsigned c_num_threads = 48;

#ifndef COMPRESS_BIT
#define COMPRESS_BIT
#endif

constexpr bool c_zlib_compress = false;
constexpr unsigned c_compress_bit_size = 32;

constexpr bool c_is_debug = false;
constexpr double c_default_sp = 0.5;
constexpr double c_default_clock_toggle = 2.0;
constexpr double c_default_clock_sp = 0.5;
constexpr double c_switch_power_K = 0.5;

constexpr double c_default_toggle =
    0.02;  //  time unit: ns, set by tcl cmd, not used anymore.
constexpr double c_default_period = 10;                  // time unit: ns
constexpr double c_default_toggle_relative_clk = 0.125;  // time unit: period

// for estimate IR
constexpr double c_resistance_coef = 0.033;

namespace ista {}
using namespace ista;

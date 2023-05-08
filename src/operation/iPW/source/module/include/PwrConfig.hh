/**
 * @file PwrConfig.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The power tool global config.
 * @version 0.1
 * @date 2023-04-08
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include "DisallowCopyAssign.hh"

constexpr unsigned c_num_threads = 48;

#ifndef COMPRESS_BIT
#define COMPRESS_BIT
#endif

constexpr bool c_zlib_compress = false;
constexpr unsigned c_compress_bit_size = 32;

constexpr bool c_is_debug = false;
constexpr double c_default_toggle = 0.02;  // time unit :ns
constexpr double c_default_sp = 0.5;
constexpr double c_default_clock_toggle = 2.0;
constexpr double c_default_clock_sp = 0.5;
constexpr double c_switch_power_K = 0.5;

constexpr double c_default_period = 10;  // time unit :ns
constexpr double c_default_toggle_relative_clk =
    c_default_toggle / c_default_period;  // time unit :period

namespace ista {}
using namespace ista;

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
 * @file Config.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-07-21
 */
#pragma once

#define ZLIB_FOUND

#ifdef ZLIB_FOUND

#include <zlib.h>

#else  // ZLIB_FOUND

#include <stdio.h>

#define gzFile FILE
#define gzopen fopen
#define gzclose fclose
#define gzgets(stream, s, size) fgets(s, size, stream)
#define gzprintf fprintf
#define Z_NULL nullptr
#endif

#include "FlatMap.hh"
#include "FlatSet.hh"
#include "BTreeMap.hh"
#include "BTreeSet.hh"
#include "Vector.hh"
#include "log/Log.hh"
#include "string/Str.hh"
#include "string/StrMap.hh"
#include "tcl/ScriptEngine.hh"
#include "time/Time.hh"
#include "usage/usage.hh"

namespace ista {
using ieda::FlatMap;
using ieda::FlatSet;
using ieda::Log;
using ieda::BTreeMap;
using ieda::Multimap;
using ieda::BTreeSet;
using ieda::Str;
using ieda::StrMap;
using ieda::Time;
using ieda::Vector;

// slew, delay, arrive time together.
#define INTEGRATION_FWD 1
// use CUDA GPU Speed.
// #define CUDA_PROPAGATION 1
// use cpu to simulate the gpu fwd.
#define CPU_SIM 0

// slew and path delay bucket config.
constexpr unsigned c_vertex_slew_data_bucket_size = 1;
constexpr unsigned c_vertex_path_delay_data_bucket_size = 1;

constexpr bool c_print_delay_yaml = false;
constexpr bool c_print_net_yaml = false;
constexpr bool c_print_wire_yaml = true;

}  // namespace ista
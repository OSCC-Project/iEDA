/**
 * @file Config.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-07-21
 *
 * @copyright Copyright (c) 2021
 *
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

#include "HashMap.hh"
#include "HashSet.hh"
#include "Map.hh"
#include "Set.hh"
#include "Vector.hh"
#include "log/Log.hh"
#include "string/Str.hh"
#include "string/StrMap.hh"
#include "tcl/ScriptEngine.hh"
#include "time/Time.hh"
#include "usage/usage.hh"

namespace ista {
using ieda::HashMap;
using ieda::HashSet;
using ieda::Log;
using ieda::Map;
using ieda::Multimap;
using ieda::Set;
using ieda::Str;
using ieda::StrMap;
using ieda::Time;
using ieda::Vector;

// slew and path delay bucket config.
constexpr unsigned c_vertex_slew_data_bucket_size = 1;
constexpr unsigned c_vertex_path_delay_data_bucket_size = 1;

}  // namespace ista